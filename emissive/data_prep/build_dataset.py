"""
Build the emissive fine-tune dataset: per asset, produce the training tuple
  (shape_slat, input_tex_slat, output_tex_slat, cond, emis_mask)
mirroring SegviGen's non-interactive (full-seg) path, but with a fixed binary
emissive coloring (white=emissive / black=non) as the target.

Per sid:
  1. somage → input.glb (textured) + emissive.glb (binary target)   [somage_to_glb]
  2. glb_to_vxz(input.glb)  → input.vxz
  3. glb_to_vxz(emissive.glb) → output.vxz
  4. vxz_to_slat(..., interactive=False) → shape_slat / input_tex_slat / output_tex_slat .pth
  5. compute_emis_mask(output.vxz, output_tex_slat coords) → emis_mask.pth   [make_emis_mask]
  6. render input.glb (albedo appearance, NO emission) → DINOv3 cond.pth

NOTE: input appearance is albedo/PBR only (we never feed emission) → genuine inference.
Requires the TRELLIS.2 env (o_voxel, models) + weights. Run on a GPU node.

Usage:
  python build_dataset.py --split train --n 64 --out_root .../dataset
"""
import os, sys, json, argparse, traceback
ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(ROOT, "inference_full.py")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError(f"could not locate SegviGen repo root (inference_full.py) above {__file__}")
    ROOT = parent   # walk up: this script now lives nested under emissive/data_prep/, not repo root
SEGVIGEN = ROOT
sys.path.insert(0, SEGVIGEN)
sys.path.insert(0, os.path.join(SEGVIGEN, "data_toolkit"))

os.environ.setdefault("HF_HOME", "/3dlg-jupiter-project/lightgen/hf_cache")

import torch
import pandas as pd
from somage_to_glb import convert_one
from make_emis_mask import compute_emis_mask

DATA_ROOT  = "/3dlg-falas/project/omages/datasets/TexVerse/lightgen/somages_corresp_dc80k"
PARQUET    = "/3dlg-falas/project/omages/datasets/TexVerse/lightgen/emissive_thumbnails_obj_ids_df.parquet"
SPLIT_JSON = "/3dlg-jupiter-project/lightgen/diffusionnet_xg/data/data_splits_74k.json"
LABELS_DIR = "/3dlg-jupiter-project/lightgen/diffusionnet_xg/labels_uv_74k"
TRANSFORMS = os.path.join(SEGVIGEN, "data_toolkit", "transforms.json")
TRELLIS    = "microsoft/TRELLIS.2-4B/ckpts"


def _load_sids_from_file(sid_file):
    """Explicit sid list (one 32-hex id per line). Maps each to its ditem_dir via the
    build parquet. Used to build a SPECIFIC determined set (e.g. the canonical
    overfit_split_10) rather than slicing a split by index."""
    df = pd.read_parquet(PARQUET)
    if "success" in df.columns:
        df = df[df["success"] == True]
    want = [l.strip() for l in open(sid_file) if l.strip()]
    sids, ditem, missing = [], {}, []
    for s in want:
        if s in df.index:
            sids.append(s); ditem[s] = df.loc[s, "ditem_dir"]
        else:
            missing.append(s)
    if missing:
        print(f"[sid_file] WARNING {len(missing)} sids not in parquet: {missing[:3]}...", flush=True)
    print(f"[sid_file] resolved {len(sids)}/{len(want)} sids", flush=True)
    return sids, ditem


def _load_split_sids(split, pbr_only=False):
    df = pd.read_parquet(PARQUET)
    if "success" in df.columns:
        df = df[df["success"] == True]
    with open(SPLIT_JSON) as f:
        sp = json.load(f)
    idx = sp[split]["indices"]
    sids_all = [df.iloc[i].name for i in idx]
    ditem_all = {df.iloc[i].name: df.iloc[i]["ditem_dir"] for i in idx}
    if pbr_only:
        # Filter out fully-lit shapes (pbrType is <NA>). These bake lighting into the
        # albedo texture, so the model has no clean PBR signal to learn emissive vs
        # albedo from — confirmed by Xingguang to be confusing the network.
        pbrtype_all = {df.iloc[i].name: df.iloc[i]["pbrType"] for i in idx}
        # pbrType is a pandas StringDtype column → missing values are pd.NA, and
        # `v not in (pd.NA, "")` triggers "boolean value of NA is ambiguous". pd.notna
        # handles pd.NA/None/float NaN uniformly.
        sids = [s for s in sids_all if pd.notna(pbrtype_all[s]) and pbrtype_all[s] != ""]
        ditem = {s: ditem_all[s] for s in sids}
        print(f"[pbr_only] kept {len(sids)}/{len(sids_all)} ({100*len(sids)/max(1,len(sids_all)):.1f}%)", flush=True)
        return sids, ditem
    return sids_all, ditem_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--zero_cond", action="store_true", default=True,
                    help="skip DINOv3/rembg/render; image cond handled as zeros at train time (DINOv3 gated)")
    ap.add_argument("--real_cond", dest="zero_cond", action="store_false")
    ap.add_argument("--pbr_only", action="store_true", default=False,
                    help="filter to shapes with a PBR material (pbrType in {metalness,specular}); "
                         "drops ~57%% fully-lit shapes that confuse the network")
    ap.add_argument("--out_split_name", default=None,
                    help="dir name under out_root (defaults to --split); use to keep filtered "
                         "datasets separate, e.g. --split train --pbr_only --out_split_name train_pbr")
    ap.add_argument("--sid_file", default=None,
                    help="build an explicit list of sids (one per line) instead of a split slice; "
                         "requires --out_split_name. For the canonical overfit_split_10 etc.")
    args = ap.parse_args()

    # heavy imports (need the env)
    from trellis2 import models
    from glb_to_vxz import glb_to_vxz
    from vxz_to_slat import vxz_to_slat

    shape_encoder = models.from_pretrained(f"{TRELLIS}/shape_enc_next_dc_f16c32_fp16").cuda().eval()
    tex_encoder   = models.from_pretrained(f"{TRELLIS}/tex_enc_next_dc_f16c32_fp16").cuda().eval()
    rembg = dino = None
    if not args.zero_cond:
        from img_to_cond import img_to_cond
        from bpy_render import render_from_transforms
        from trellis2.pipelines.rembg import BiRefNet
        from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
        rembg = BiRefNet(model_name="briaai/RMBG-2.0"); rembg.cuda()
        dino  = DinoV3FeatureExtractor(model_name="facebook/dinov3-vitl16-pretrain-lvd1689m"); dino.cuda()

    if args.sid_file:
        assert args.out_split_name, "--sid_file requires --out_split_name"
        sids, ditem = _load_sids_from_file(args.sid_file)
    else:
        sids, ditem = _load_split_sids(args.split, pbr_only=args.pbr_only)
        sids = sids[args.start:args.start + args.n]
    os.makedirs(args.out_root, exist_ok=True)
    out_split = args.out_split_name or args.split

    ok, fail = 0, 0
    manifest = []
    for sid in sids:
        save_dir = os.path.join(args.out_root, out_split, sid)
        need = ["output_tex_slat.pth", "emis_mask.pth"] + ([] if args.zero_cond else ["cond.pth"])
        if all(os.path.exists(os.path.join(save_dir, f)) for f in need):
            ok += 1; manifest.append(sid); continue
        # FAST PATH: the heavy artifacts (slats/vxz, built earlier) already exist, only
        # emis_mask.pth and/or cond.pth are missing → backfill just those, skip the
        # full (slow) rebuild.
        inp_glb_existing = os.path.join(save_dir, "glb", f"{sid}_input.glb")
        out_vxz_existing = os.path.join(save_dir, "output.vxz")
        otx_existing = os.path.join(save_dir, "output_tex_slat.pth")
        have_heavy = os.path.exists(otx_existing) and os.path.exists(out_vxz_existing)
        need_mask = not os.path.exists(os.path.join(save_dir, "emis_mask.pth"))
        need_cond = (not args.zero_cond) and not os.path.exists(os.path.join(save_dir, "cond.pth"))
        if have_heavy and (need_mask or need_cond):
            try:
                if need_mask:
                    otx = torch.load(otx_existing, map_location="cpu")
                    mask = compute_emis_mask(out_vxz_existing, otx["coords"])
                    torch.save(mask, os.path.join(save_dir, "emis_mask.pth"))
                if need_cond and os.path.exists(inp_glb_existing):
                    img = os.path.join(save_dir, "img.png")
                    render_from_transforms(inp_glb_existing, TRANSFORMS, img)
                    img_to_cond(rembg, dino, img, os.path.join(save_dir, "cond.pth"))
                if all(os.path.exists(os.path.join(save_dir, f)) for f in need):
                    ok += 1; manifest.append(sid)
                    print(f"[backfill] {sid} ({ok}/{len(sids)})", flush=True)
                    continue
                # still incomplete (e.g. cond needed but input glb absent) → fall
                # through to the full rebuild below.
            except Exception as e:
                fail += 1
                print(f"[fail backfill] {sid}: {repr(e)[:200]}", flush=True)
                continue
        try:
            dd = ditem[sid]
            mesh_npz   = os.path.join(DATA_ROOT, dd, "somage_original_mesh.npz")
            somage_npz = os.path.join(DATA_ROOT, dd, "somage.npz")
            if not (os.path.exists(mesh_npz) and os.path.exists(somage_npz)):
                print(f"[skip] {sid}: missing somage", flush=True); fail += 1; continue
            os.makedirs(save_dir, exist_ok=True)
            glb_dir = os.path.join(save_dir, "glb")
            inp_glb, tgt_glb, frac = convert_one(sid, mesh_npz, somage_npz, glb_dir, labels_dir=LABELS_DIR)

            input_vxz  = os.path.join(save_dir, "input.vxz")
            output_vxz = os.path.join(save_dir, "output.vxz")
            glb_to_vxz(inp_glb, input_vxz)
            glb_to_vxz(tgt_glb, output_vxz)
            vxz_to_slat(shape_encoder, tex_encoder, input_vxz, output_vxz, save_dir, interactive=False)

            otx = torch.load(os.path.join(save_dir, "output_tex_slat.pth"), map_location="cpu")
            mask = compute_emis_mask(output_vxz, otx["coords"])
            torch.save(mask, os.path.join(save_dir, "emis_mask.pth"))

            if not args.zero_cond:
                img = os.path.join(save_dir, "img.png")
                render_from_transforms(inp_glb, TRANSFORMS, img)
                img_to_cond(rembg, dino, img, os.path.join(save_dir, "cond.pth"))

            json.dump({"sid": sid, "emissive_frac": frac}, open(os.path.join(save_dir, "meta.json"), "w"))
            ok += 1; manifest.append(sid)
            print(f"[ok] {sid} frac={frac:.3f} ({ok}/{len(sids)})", flush=True)
        except Exception as e:
            fail += 1
            print(f"[fail] {sid}: {repr(e)[:200]}", flush=True)
            traceback.print_exc()

    json.dump(manifest, open(os.path.join(args.out_root, f"{out_split}_manifest.json"), "w"))
    print(f"DONE split={out_split} ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
