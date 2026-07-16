"""
Agent-facing inference CLI: raw GLB in -> binary emissive-mask prediction out.

Not a training/eval script — no ground truth is required or used. Input can be ANY .glb
(any source, not just our "somage" assets); output is a predicted per-voxel emissive
mask + a decoded mesh, for a downstream agent (or a person) to consume without touching
any of the eval_emissive.py / build_dataset.py internals directly.

Pipeline (every stage below REUSES existing code; see the attribution comment at each
import/lift — nothing here is a silent fork of eval_emissive.py's or make_pred_glb.py's
logic):
  1. glb_to_vxz            (data_toolkit/glb_to_vxz.py, upstream, unmodified)
  2. vxz_to_latent_slat x2 (data_toolkit/vxz_to_slat.py, upstream, unmodified) with
     shape_encoder + tex_encoder, then intersected onto a shared coords set — see
     `_common_coords_2` below, a 2-slat analogue of vxz_to_slat.get_common_coords (that
     helper hardcodes exactly 4 slats for the train-time shape/input_tex/output_tex/
     foreground case; inference has no GT output slat, so we only ever intersect 2).
  3. cond:
       --zero_cond   -> zeros, matching eval_emissive.py's cond_mode="zero"
       --image PATH  -> skip rendering, run img_to_cond.img_to_cond() (data_toolkit,
                        unmodified) directly on the supplied image
       (default)     -> render the glb via data_toolkit/bpy_render.render_from_transforms
                        using the SAME transforms.json build_dataset.py's --real_cond
                        path uses, then img_to_cond.img_to_cond() on that render
  4. load ckpt: eval_emissive.load_eval_models() for the tex/shape decoders + norm
     stats + sampler (exact import, same object every eval/train_emissive quick-val call
     reuses), plus the identical state_dict "gen3dseg."-prefix-strip loading snippet used
     verbatim in eval_emissive.py main() and make_pred_glb.py main().
  5. sample --draws independent draws and average: this loop is lifted from
     eval_emissive.eval_sample()'s draw loop (same noise/sampler/decode calls) with the
     GT-scoring half removed, since there is no GT here.
  6. threshold @ --thr, then decode to mesh via inference_full.slat_to_glb — the same
     function make_pred_glb.py imports and calls, with the identical hard-threshold-to-
     white/black glue (lifted from make_pred_glb.py's main(), see comment at the call).

VALIDATION — untested on GPU as of this commit (see emissive/docs/EXPERIMENTS.md). What
IS verified locally (no GPU/trellis2 env available on this workstation): `python -m
py_compile` is clean, and `--help` runs (all heavy imports — torch/trellis2/inference_full/
data_toolkit/o_voxel — are deferred until after argparse, specifically so `--help` never
needs them). Smoke-test on the cluster once a GPU node is available:

    salloc -p 3dlg-hcvc-lab-debug --gres=gpu:l40s:1 --time=0:30:00 --cpus-per-task=8 --mem=64G
    source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh && conda activate trellis2
    cd <repo root, after rsync>   # emissive/slurm/*.sbatch document this convention
    export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
    python emissive/infer/predict_emissive.py --glb assets/example.glb \
        --out /tmp/predict_smoke --draws 4 --thr 0.5 --zero_cond
    # then, to also exercise the real-cond render path (needs bpy in the trellis2 env):
    python emissive/infer/predict_emissive.py --glb assets/example.glb \
        --out /tmp/predict_smoke_realcond --draws 4 --thr 0.5

Usage:
  python predict_emissive.py --glb mesh.glb --out out_dir/
  python predict_emissive.py --glb mesh.glb --out out_dir/ --draws 8 --thr 0.4
  python predict_emissive.py --glb mesh.glb --out out_dir/ --image render.png
  python predict_emissive.py --glb mesh.glb --out out_dir/ --zero_cond --ckpt /path/to/other.ckpt
"""
import os
import sys
import json
import time
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(ROOT, "inference_full.py")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError(f"could not locate SegviGen repo root (inference_full.py) above {__file__}")
    ROOT = parent   # walk up: this script lives nested under emissive/infer/, not repo root
SEGVIGEN = ROOT

# Recommended checkpoint per emissive/docs/EXPERIMENTS.md: no fine-tune beats the 0.219
# zero-shot oracle (which is oracle-assisted / not deployable), so this is the best
# *deployable* checkpoint — emis_1k_w5's EMA weights at its best epoch (0.117 nonzero
# IoU, full 111-shape val, 4-draw averaged, @0.5). See EXPERIMENTS.md for the full
# registry and the tie with emis_2k_bal (0.114) that this default breaks.
DEFAULT_CKPT = "/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_1k_w5/epoch_0016_ema.ckpt"
DEFAULT_TRANSFORMS = os.path.join(SEGVIGEN, "data_toolkit", "transforms.json")


def _common_coords_2(slat_a, slat_b):
    """2-slat analogue of data_toolkit/vxz_to_slat.py's get_common_coords(). That helper
    hardcodes 4 slats (shape/input_tex/output_tex/foreground) for the train-time case
    where a GT output slat exists to intersect against. Inference only ever has
    shape_slat + input_tex_slat (no GT), but the intersection is still required: the
    shape_encoder and tex_encoder can each downsample glb_to_vxz's input voxel grid to a
    different coords set, so shape_slat.coords and input_tex_slat.coords are not
    guaranteed to match without this step."""
    import torch
    xs = [torch.unique(x, dim=0) for x in (slat_a.coords, slat_b.coords)]
    all_coords = torch.cat(xs, dim=0)
    uniq, counts = torch.unique(all_coords, dim=0, return_counts=True)
    return uniq[counts == 2].cuda()


def _load_ckpt(gen, ckpt_path, device):
    """Identical to the state_dict loading in eval_emissive.py main() / make_pred_glb.py
    main() (the trained module is wrapped as `gen3dseg.<name>` by the standalone training
    loop's Lightning-style checkpointing; strip that prefix before load_state_dict)."""
    from collections import OrderedDict
    import torch
    sd = torch.load(ckpt_path, map_location=device)["state_dict"]
    sd = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in sd.items()])
    gen.load_state_dict(sd)
    gen.eval()


def _render_cond_image(glb_path, out_img_path, transforms_json):
    """Default (no --zero_cond, no --image) conditioning path: render the glb the same
    way build_dataset.py's --real_cond does (data_toolkit/bpy_render.render_from_transforms
    + transforms.json), for use as the DINOv3 image condition. Imports bpy lazily — only
    needed on this path, not for --zero_cond / --image."""
    sys.path.insert(0, os.path.join(SEGVIGEN, "data_toolkit"))
    from bpy_render import render_from_transforms
    render_from_transforms(glb_path, transforms_json, out_img_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glb", required=True, help="input mesh, any source (not necessarily a somage asset)")
    ap.add_argument("--out", required=True, help="output dir: mask.npz, pred_mesh.glb, meta.json")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                     help=f"fine-tuned checkpoint (default: recommended in EXPERIMENTS.md, "
                          f"emis_1k_w5 best-EMA) [{DEFAULT_CKPT}]")
    ap.add_argument("--draws", type=int, default=4,
                     help="independent flow-matching samples to average (K). K=1 is noisy "
                          "(draw-std ~0.09 IoU observed in eval); K=4 matches the headline "
                          "eval protocol in EXPERIMENTS.md.")
    ap.add_argument("--thr", type=float, default=0.5, help="probability threshold for the binary mask")
    ap.add_argument("--steps", type=int, default=12, help="flow sampler steps (matches eval_emissive.py default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zero_cond", action="store_true", default=False,
                     help="skip DINOv3 conditioning entirely (cond=zeros), matching "
                          "eval_emissive.py's --cond zero. Fastest; use for a first smoke test.")
    ap.add_argument("--image", default=None,
                     help="use this image as the DINOv3 condition instead of rendering --glb "
                          "(skips the bpy render step; still runs rembg+DINOv3 via img_to_cond.py)")
    ap.add_argument("--transforms", default=DEFAULT_TRANSFORMS,
                     help="camera transforms.json for the render-from-glb cond path (default: "
                          "the same file build_dataset.py's --real_cond uses)")
    args = ap.parse_args()

    if args.zero_cond and args.image:
        ap.error("--zero_cond and --image are mutually exclusive")

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    # --- heavy imports deferred to here so `--help` above never needs torch/trellis2/o_voxel ---
    sys.path.insert(0, SEGVIGEN)
    sys.path.insert(0, os.path.join(SEGVIGEN, "data_toolkit"))
    sys.path.insert(0, os.path.join(ROOT, "emissive", "eval"))  # sibling dir holding eval_emissive.py
    os.environ.setdefault("HF_HOME", "/3dlg-jupiter-project/lightgen/hf_cache")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    import numpy as np
    import trellis2.modules.sparse as sp
    from trellis2 import models
    from inference_full import Gen3DSeg, Sampler, slat_to_glb   # reuse: same import make_pred_glb.py does
    from eval_emissive import load_eval_models, COND_T, COND_D  # reuse: shared decoder/norm-stat loader
    from glb_to_vxz import glb_to_vxz
    from vxz_to_slat import vxz_to_latent_slat, get_slat_by_common_coords

    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- 1. glb -> vxz -> shape_slat / input_tex_slat (steps 1-2 in the module docstring) ---
    vxz_path = os.path.join(args.out, "input.vxz")
    print(f"[glb_to_vxz] {args.glb} -> {vxz_path}", flush=True)
    glb_to_vxz(args.glb, vxz_path)

    shape_encoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16").to(device).eval()
    tex_encoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16").to(device).eval()
    shape_slat_o, tex_slat_o = vxz_to_latent_slat(shape_encoder, tex_encoder, vxz_path)
    common_coords = _common_coords_2(shape_slat_o, tex_slat_o)
    shape_slat = get_slat_by_common_coords(shape_slat_o, common_coords)
    input_tex_slat = get_slat_by_common_coords(tex_slat_o, common_coords)
    coords = shape_slat.coords
    print(f"[slat] {coords.shape[0]} common voxels", flush=True)

    # --- 2. cond (step 3) ---
    if args.zero_cond:
        cond = torch.zeros(1, COND_T, COND_D, device=device)
        print("[cond] zero", flush=True)
    else:
        from img_to_cond import img_to_cond
        from trellis2.pipelines.rembg import BiRefNet
        from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
        img_path = args.image
        if img_path is None:
            img_path = os.path.join(args.out, "cond_render.png")
            print(f"[cond] rendering {args.glb} -> {img_path}", flush=True)
            _render_cond_image(args.glb, img_path, args.transforms)
        else:
            print(f"[cond] using supplied image {img_path}", flush=True)
        rembg = BiRefNet(model_name="briaai/RMBG-2.0").to(device)
        dino = DinoV3FeatureExtractor(model_name="facebook/dinov3-vitl16-pretrain-lvd1689m").to(device)
        cond_path = os.path.join(args.out, "cond.pth")
        img_to_cond(rembg, dino, img_path, cond_path)   # reuse: identical to build_dataset.py's real_cond call
        cond = torch.load(cond_path, map_location=device)["cond"]
    cond_dict = {"cond": cond, "neg_cond": torch.zeros_like(cond)}

    # --- 3. load fine-tuned model (step 4) ---
    print(f"[ckpt] {args.ckpt}", flush=True)
    flow = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")
    gen = Gen3DSeg(flow).to(device)
    _load_ckpt(gen, args.ckpt, device)

    md = load_eval_models(device)   # reuse: eval_emissive.load_eval_models — decoders + norm stats + sampler
    tex_decoder, shape_decoder, sampler = md["tex_decoder"], md["shape_decoder"], md["sampler"]
    sm, ss, tm, ts = md["sm"], md["ss"], md["tm"], md["ts"]
    sp_params = dict(md["pipeline_args"]["tex_slat_sampler"]["params"]); sp_params["steps"] = args.steps

    shp_n = sp.SparseTensor((shape_slat.feats.to(device) - sm) / ss, coords.to(device))
    itx_n = sp.SparseTensor((input_tex_slat.feats.to(device) - tm) / ts, coords.to(device))

    # --- 4. sample K draws + decode (steps 5-6), loop lifted from eval_emissive.eval_sample() ---
    with torch.no_grad():
        shape_decoder.set_resolution(512)
        meshes, subs = shape_decoder(sp.SparseTensor(shape_slat.feats.to(device), coords.to(device)), return_subs=True)

    draw_probs = []
    for k in range(args.draws):
        with torch.no_grad():
            noise = sp.SparseTensor(torch.randn_like(itx_n.feats), coords.to(device))
            out = sampler.sample(gen, noise, itx_n, shp_n, [coords.shape[0]], cond_dict, sp_params)
            out = out * ts + tm
            pred_vox = tex_decoder(out, guide_subs=subs) * 0.5 + 0.5   # base color in [0,1]-ish
        draw_probs.append(pred_vox.feats[:, :3].mean(-1).cpu())
        print(f"[draw {k + 1}/{args.draws}] mean prob={draw_probs[-1].mean().item():.3f}", flush=True)
    prob = torch.stack(draw_probs, dim=0).mean(0)   # average across draws
    mask = prob > args.thr

    out_coords = pred_vox.coords[:, 1:].cpu().numpy().astype(np.int32)
    np.savez_compressed(
        os.path.join(args.out, "mask.npz"),
        coords=out_coords,
        prob=prob.numpy().astype(np.float32),
        mask=mask.numpy().astype(bool))
    print(f"[mask.npz] {out_coords.shape[0]} voxels, {mask.float().mean().item():.3f} frac emissive", flush=True)

    # Decode to mesh: identical hard-threshold-then-bake path as make_pred_glb.py main()
    # (lifted with attribution; slat_to_glb itself is imported, not reimplemented).
    new_feats = pred_vox.feats.clone()
    white = mask.to(new_feats.device).float()[:, None]
    new_feats[:, 0:3] = white
    pred_vox_thr = pred_vox.replace(new_feats)
    print("[glb] slat_to_glb (remesh 512 + bake 4096) ...", flush=True)
    glb = slat_to_glb(meshes, pred_vox_thr)
    glb_path = os.path.join(args.out, "pred_mesh.glb")
    glb.export(glb_path)

    meta = {
        "glb_in": os.path.abspath(args.glb),
        "ckpt": args.ckpt,
        "draws": args.draws,
        "thr": args.thr,
        "steps": args.steps,
        "seed": args.seed,
        "cond_mode": "zero" if args.zero_cond else ("image" if args.image else "render"),
        "n_voxels": int(out_coords.shape[0]),
        "frac_emissive": float(mask.float().mean().item()),
        "elapsed_sec": round(time.time() - t0, 1),
        "outputs": {"mask": "mask.npz", "mesh": "pred_mesh.glb"},
    }
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    print(f"[done] {args.out} ({meta['elapsed_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
