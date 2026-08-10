"""
Backfill DINOv3 image conditioning (cond.pth) for the direct-ovoxel dataset,
built from the TexVerse THUMBNAIL rather than from a rendered input.glb.

Why this exists: dataset_direct (build_dataset_direct.py) never produced any
conditioning, so every run on it has trained with `--cond zero`, feeding the DiT
zeros where it expects a DINOv3 embedding. The TEXGen / TRELLIS.2 baselines all
receive the TexVerse thumbnail as their image input, so conditioning on the same
thumbnail is what makes the comparison like-for-like.

DIFFERENCE FROM PATH A (build_dataset.py, deliberate): Path A rendered
<sid>_input.glb with albedo appearance and NO emission, then embedded that render.
This script embeds the TexVerse thumbnail, which is a fully-lit render in which a
genuinely emissive region is typically blown out to pure white. The conditioning
therefore SHOWS the model where the emission is. That is the same input the
baselines get, but it is a materially easier signal than Path A's emission-free
render -- these two conditionings are NOT equivalent and must not be described as
such.

Preprocessing is Path A's, unchanged: this calls SegviGen's own
data_toolkit/img_to_cond.img_to_cond(), i.e. the same BiRefNet("briaai/RMBG-2.0")
background removal, the same alpha-bbox square crop and premultiply, the same
DinoV3FeatureExtractor("facebook/dinov3-vitl16-pretrain-lvd1689m") at 512px.
The only input that changed is the image. Note that the thumbnails are RGB with
no alpha channel, so BiRefNet actually runs on them; Path A's bpy renders are
RGBA, so preprocess_image took its has_alpha branch there and skipped BiRefNet.
That divergence is a property of the source images, not of a changed code path.

Output contract (verified against a Path A cond.pth, see --verify):
  cond.pth = {"cond": float32 (1, 1029, 1024), "neg_cond": zeros_like}
1029 = 1024 patch tokens at 512px/16 + 1 CLS + 4 register tokens. NOTE this is
NOT the (1, 1024, 1024) that train_emissive.py's COND_T/COND_D constants build
for `--cond zero`; those constants are only used on the zero path and the sparse
DiT's cross-attention is agnostic to the token count. Real cond has always been
1029 tokens, including for every Path A `--cond real` run.

Idempotent and resumable: a shape that already has cond.pth is skipped, and each
file is written to a temp path then os.replace'd, so a killed job never leaves a
truncated tensor behind. Nothing else in the shape directory is touched.

Usage:
  # one-time (or whenever TexVerse thumbnails change): cache the sid -> batch index
  python build_cond_thumbnail.py --build_index

  # verify a handful against a Path A cond.pth before committing to a full run
  python build_cond_thumbnail.py --split val_72k --limit 8 --verify

  # full backfill, sharded over a SLURM array (task i handles sids[i::nshards])
  python build_cond_thumbnail.py --split train_72k --shard $SLURM_ARRAY_TASK_ID --nshards 32

  # residual pass: whatever the verbatim preprocessing could not handle (it skips
  # every shape that already has cond.pth, so this targets exactly the failures)
  python build_cond_thumbnail.py --split train_72k --repair

About 0.8%% of TexVerse thumbnails cannot be preprocessed by Path A's code at all,
for two reasons visible in the images themselves: some are TexVerse's grayscale
PLACEHOLDER icon rather than a render (1-band, so `input.split()[3]` raises), and
a few frame the object as a few-pixel speck so BiRefNet returns no foreground and
the crop bbox is empty. --repair handles both with documented guards rather than
writing zeros; see preprocess_image_repair.
"""
import os
import sys
import json
import time
import argparse
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
SEGVIGEN = os.path.join(ROOT, "SegviGen")
sys.path.insert(0, SEGVIGEN)
sys.path.insert(0, os.path.join(SEGVIGEN, "data_toolkit"))

os.environ.setdefault("HF_HOME", "/3dlg-jupiter-project/lightgen/hf_cache")

import torch  # noqa: E402

DATASET_ROOT = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct"
THUMB_ROOT = "/3dlg-falas/datasets/TexVerse/thumbnails/thumbnails_batch"
INDEX_PATH = "/3dlg-jupiter-project/lightgen/segvigen_emissive/cond_thumb/thumb_index.json"
REPORT_DIR = "/3dlg-jupiter-project/lightgen/segvigen_emissive/cond_thumb"
# a Path A shape built with --real_cond, used as the reference contract by --verify
PATHA_REF = ("/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset/train/"
             "d81780bf692e45478ea9e10f0635e3f0/cond.pth")


def build_index(index_path=INDEX_PATH):
    """sid -> batch dir name, over all thumbnails_batch/batch_XXXXX/<sid>.png.
    Cached because a SLURM array of N tasks should not each walk 850k NFS entries."""
    idx = {}
    for b in sorted(os.listdir(THUMB_ROOT)):
        p = os.path.join(THUMB_ROOT, b)
        if not os.path.isdir(p):
            continue
        for f in os.scandir(p):
            if f.name.endswith(".png"):
                idx[f.name[:-4]] = b
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    tmp = index_path + ".tmp"
    json.dump(idx, open(tmp, "w"))
    os.replace(tmp, index_path)
    print(f"[index] {len(idx)} thumbnails -> {index_path}", flush=True)
    return idx


def load_index(index_path=INDEX_PATH):
    if not os.path.exists(index_path):
        return build_index(index_path)
    return json.load(open(index_path))


def thumb_path(sid, idx):
    b = idx.get(sid)
    if b is None:
        return None
    p = os.path.join(THUMB_ROOT, b, f"{sid}.png")
    return p if os.path.exists(p) else None


def preprocess_image_repair(rembg_model, image):
    """DEVIATION FROM PATH A, applied ONLY under --repair, only to the shapes whose
    thumbnails Path A's preprocess_image cannot process at all. Two guards, both
    measured against real failures on the TexVerse thumbnails; nothing else changes,
    and the guards are no-ops on any image the unmodified function already handles.

    Guard 1 (mode). Path A's images are RGB or RGBA renders, so `input.split()[3]`
    is always the alpha band there. Some TexVerse thumbnails are 1-band grayscale
    ('L', 1024x640) -- these are TexVerse's PLACEHOLDER image, a generic cube icon
    rather than a render of the shape -- and the indexing raises IndexError. A
    1-band image has no alpha to composite, so converting to RGB is what the
    original branch's "flatten onto white" intent reduces to.

    Guard 2 (empty foreground). A handful of thumbnails frame the object as a
    few-pixel speck on a flat background; BiRefNet then produces no alpha above
    0.8*255 anywhere, the bbox is empty and np.min raises. Fall back to a lower
    alpha threshold, then to the uncropped full frame. The object is genuinely
    tiny in these images, so the resulting embedding is weak conditioning -- but
    it is the real image, never zeros.

    Effect on outputs: for the placeholder shapes the conditioning carries no
    shape-specific information (a near-constant embedding); for the speck shapes
    it carries very little. Both are counted and named in the run report.
    """
    from PIL import Image
    import numpy as np
    if image.mode not in ("RGB", "RGBA"):
        # guard 1
        if "A" in image.getbands():
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
    if image.mode != "RGB":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    has_alpha = False
    if image.mode == "RGBA":
        alpha = np.array(image)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    max_size = max(image.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)),
                             Image.Resampling.LANCZOS)
    if has_alpha:
        output = image
    else:
        image = image.convert("RGB")
        output = rembg_model(image)
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    used = "alpha>0.8"
    bbox_idx = np.argwhere(alpha > 0.8 * 255)
    if len(bbox_idx) == 0:
        # guard 2
        bbox_idx = np.argwhere(alpha > 0.1 * 255)
        used = "alpha>0.1"
    if len(bbox_idx) == 0:
        bbox = (0, 0, output.width - 1, output.height - 1)
        used = "full_frame"
    else:
        bbox = (int(bbox_idx[:, 1].min()), int(bbox_idx[:, 0].min()),
                int(bbox_idx[:, 1].max()), int(bbox_idx[:, 0].max()))
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    bbox = (center[0] - size // 2, center[1] - size // 2,
            center[0] + size // 2, center[1] + size // 2)
    output = output.crop(bbox)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    return Image.fromarray((output * 255).astype(np.uint8)), used


def img_to_cond_repair(rembg_model, dino_model, image_path, save_path):
    """--repair twin of img_to_cond: same get_cond (DINOv3 @512, neg_cond=zeros),
    only preprocess_image is swapped for the guarded version above."""
    from PIL import Image
    from img_to_cond import get_cond
    image = Image.open(image_path)
    image, used = preprocess_image_repair(rembg_model, image)
    torch.save(get_cond(dino_model, [image]), save_path)
    return used


def verify_against_patha(cond_paths, ref_path=PATHA_REF):
    """Contract check: keys, shape, dtype and value range of the tensors we just
    wrote, against a Path A cond.pth. Raises on a mismatch that would change what
    the DiT sees; prints the distributional comparison (which is EXPECTED to
    differ -- different source images)."""
    ref = torch.load(ref_path, map_location="cpu")
    rc = ref["cond"]
    print(f"[verify] Path A reference {ref_path}", flush=True)
    print(f"[verify]   keys={sorted(ref.keys())} cond={tuple(rc.shape)} {rc.dtype} "
          f"min={rc.min():.4f} max={rc.max():.4f} mean={rc.mean():.3e} std={rc.std():.4f}",
          flush=True)
    for p in cond_paths:
        d = torch.load(p, map_location="cpu")
        c = d["cond"]
        print(f"[verify] {p}", flush=True)
        print(f"[verify]   keys={sorted(d.keys())} cond={tuple(c.shape)} {c.dtype} "
              f"min={c.min():.4f} max={c.max():.4f} mean={c.mean():.3e} std={c.std():.4f} "
              f"neg_cond={tuple(d['neg_cond'].shape)} allzero={bool((d['neg_cond'] == 0).all())}",
              flush=True)
        assert sorted(d.keys()) == sorted(ref.keys()), f"key mismatch in {p}"
        assert c.shape == rc.shape, f"shape {tuple(c.shape)} != Path A {tuple(rc.shape)} in {p}"
        assert c.dtype == rc.dtype, f"dtype {c.dtype} != Path A {rc.dtype} in {p}"
        assert d["neg_cond"].shape == c.shape and bool((d["neg_cond"] == 0).all())
        # a silent all-zero cond is indistinguishable from --cond zero, which is
        # exactly the confound this whole exercise removes. Hard-fail on it.
        assert float(c.abs().max()) > 0, f"cond is all zeros in {p}"
    print(f"[verify] OK: {len(cond_paths)} file(s) match the Path A contract", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default=DATASET_ROOT)
    ap.add_argument("--split", default=None, help="e.g. train_72k / val_72k")
    ap.add_argument("--shard", type=int, default=None, help="SLURM array task id")
    ap.add_argument("--nshards", type=int, default=None,
                    help="task --shard handles sids[shard::nshards] (strided, so the "
                         "cost of any slow tail spreads evenly across tasks)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N of this task's slice")
    ap.add_argument("--overwrite", action="store_true", default=False,
                    help="rebuild cond.pth even where one exists (default: skip, resumable)")
    ap.add_argument("--verify", action="store_true", default=False,
                    help="after writing, check every cond.pth this run produced against "
                         "the Path A contract (keys/shape/dtype/nonzero)")
    ap.add_argument("--strict", action="store_true", default=False,
                    help="exit nonzero if any shape failed preprocessing (default: report "
                         "them in COND_FAILURES + the run report and exit 0, so a multi-split "
                         "job under `set -e` is not aborted by an expected ~1%% failure rate)")
    ap.add_argument("--repair", action="store_true", default=False,
                    help="use the guarded preprocessing (preprocess_image_repair) instead of "
                         "Path A's verbatim one. ONLY for the residual shapes whose thumbnails "
                         "Path A's code cannot process at all (grayscale placeholders, "
                         "thumbnails with no BiRefNet foreground). A DOCUMENTED DEVIATION -- "
                         "see that function's docstring for exactly what changes and what it "
                         "does to the conditioning quality of those shapes.")
    ap.add_argument("--build_index", action="store_true", default=False,
                    help="(re)build the sid -> thumbnail-batch index and exit")
    ap.add_argument("--index_path", default=INDEX_PATH)
    args = ap.parse_args()

    if args.build_index:
        build_index(args.index_path)
        return
    assert args.split, "--split is required (or use --build_index)"

    idx = load_index(args.index_path)
    print(f"[index] {len(idx)} thumbnails", flush=True)

    sdir = os.path.join(args.dataset_root, args.split)
    sids = sorted(os.listdir(sdir))
    n_total = len(sids)
    if args.nshards:
        assert args.shard is not None, "--nshards requires --shard"
        sids = sids[args.shard::args.nshards]
    if args.limit:
        sids = sids[:args.limit]
    tag = f"{args.split}" + (f"_shard{args.shard}" if args.shard is not None else "")
    print(f"[data] split={args.split} total={n_total} this_task={len(sids)}", flush=True)

    # heavy imports / model load only once we know there is work
    from img_to_cond import img_to_cond
    from trellis2.pipelines.rembg import BiRefNet
    from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
    t_load = time.perf_counter()
    rembg = BiRefNet(model_name="briaai/RMBG-2.0")
    rembg.cuda()
    dino = DinoV3FeatureExtractor(model_name="facebook/dinov3-vitl16-pretrain-lvd1689m")
    dino.cuda()
    print(f"[models] loaded in {time.perf_counter() - t_load:.1f}s", flush=True)

    written, skipped, no_thumb, failed = [], 0, [], []
    repaired = {}   # sid -> which fallback branch preprocess_image_repair took
    t0 = time.perf_counter()
    t_build = 0.0
    for i, sid in enumerate(sids):
        out = os.path.join(sdir, sid, "cond.pth")
        if os.path.exists(out) and not args.overwrite:
            skipped += 1
            continue
        tp = thumb_path(sid, idx)
        if tp is None:
            # NEVER write zeros here: a zero cond is indistinguishable from
            # --cond zero and would silently poison the comparison.
            no_thumb.append(sid)
            print(f"[no_thumb] {sid}", flush=True)
            continue
        try:
            ts = time.perf_counter()
            tmp = out + f".tmp{os.getpid()}"
            if args.repair:
                used = img_to_cond_repair(rembg, dino, tp, tmp)
                repaired[sid] = used
                print(f"[repair] {sid} bbox={used}", flush=True)
            else:
                img_to_cond(rembg, dino, tp, tmp)
            os.replace(tmp, out)
            t_build += time.perf_counter() - ts
            written.append(out)
            if len(written) % 200 == 0 or len(written) <= 3:
                rate = t_build / len(written)
                print(f"[ok] {sid} ({i + 1}/{len(sids)}) written={len(written)} "
                      f"{rate:.3f}s/shape", flush=True)
        except Exception as e:
            failed.append({"sid": sid, "error": repr(e)[:500]})
            print(f"[fail] {sid}: {repr(e)[:300]}", flush=True)
            traceback.print_exc()

    dt = time.perf_counter() - t0
    per = (t_build / len(written)) if written else float("nan")
    print(f"COND_DONE split={tag} written={len(written)} skipped={skipped} "
          f"no_thumb={len(no_thumb)} failed={len(failed)} "
          f"wall={dt:.1f}s per_shape={per:.3f}s", flush=True)

    os.makedirs(REPORT_DIR, exist_ok=True)
    if args.repair:
        tag = "repair_" + tag
    json.dump({"split": args.split, "shard": args.shard, "nshards": args.nshards,
               "repair": args.repair, "repaired": repaired,
               "written": len(written), "skipped": skipped,
               "no_thumb": no_thumb, "failed": failed,
               "wall_s": dt, "per_shape_s": per},
              open(os.path.join(REPORT_DIR, f"report_{tag}.json"), "w"), indent=2)

    if args.verify and written:
        verify_against_patha(written[:8])
    if failed:
        # NOT a process failure by default. Per-shape preprocessing failures are an
        # expected, quantified outcome here (~1% of TexVerse thumbnails, see the
        # module docstring) and the --repair pass exists to absorb them. Exiting
        # nonzero once cost a whole 24-task array its train_72k half, because
        # `set -e` in the sbatch aborted the split loop after val_72k. Completeness
        # is asserted explicitly by the caller counting cond.pth, not by this
        # exit code. --strict restores the hard exit for callers that want it.
        print(f"COND_FAILURES split={tag} n={len(failed)} "
              f"sids={[f['sid'] for f in failed][:20]}", flush=True)
        if args.strict:
            raise SystemExit(f"{len(failed)} shapes failed (--strict)")


if __name__ == "__main__":
    main()
