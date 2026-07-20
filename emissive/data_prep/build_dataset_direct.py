"""
DIRECT GLB -> o-voxel emissive-GT pipeline (stage 1 of the new data pipeline).

Motivation (see the explainer page pipeline_glb_direct): the CURRENT pipeline bakes
the original TexVerse material down to a 'somage' (a repacked UV atlas + a handful of
512x512 PBR maps), rebuilds a GLB from that, then voxelizes it — so the emissive
ground truth is a binary per-face label derived by thresholding the somage's
`emission_color` MAP (somage_to_glb.emissive_face_mask, thr 0.04). That is a lossy,
two-hop derivation of "what glows". This script instead voxelizes the ORIGINAL glb
ONCE and reads the per-voxel `emissive` attribute that o_voxel's
`textured_mesh_to_volumetric_attr` computes directly from the glb's real
emissiveFactor * emissiveTexture — no somage bake in the loop.

KEY FACT this pipeline rests on: our `data_toolkit/glb_to_vxz.py` writes ALL attrs
returned by `textured_mesh_to_volumetric_attr`, including `emissive`. Upstream
TRELLIS.2 prep deletes it (`del attr['emissive']`); we do not. So a vxz written from
an original glb already carries per-voxel emissive. (Verified before the pilot — see
verify_emissive.py / the pilot's verify/ dir.)

------------------------------------------------------------------------------------
STAGE 1 (this file, for the A/B pilot) — per sid:
  original .glb  --glb_to_vxz-->  direct.vxz            (coords + ALL attrs incl emissive)
                 --read back---->  emis_gt_direct.npz    (coords int32 @512,
                                                          emissive_lum float32 [0,1],
                                                          mask bool @ EMIS_THRESH)
No somage, no slat encoding here — this is the GT-comparison artifact only.

STAGE 2 (TODO, full rebuild — stubs at the bottom): the training tuple also needs an
INPUT-side latent with emissive information REMOVED, plus the shape latent. See
`stage2_todo_*` below and the leakage note there — the released tex VAE does NOT read
emissive, so a single voxelization already gives a leak-free input, which is the whole
point of collapsing the two-hop pipeline into one.
------------------------------------------------------------------------------------

Usage (one shape; GPU node, trellis2 env):
  python build_dataset_direct.py --sid <sid> --glb /path/to/original.glb \
      --out_dir /3dlg-jupiter-project/lightgen/segvigen_emissive/direct_pilot/out/<sid>

  # or resolve the glb from the TexVerse manifest parquet by sid:
  python build_dataset_direct.py --sid <sid> --out_dir .../out/<sid>
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

# --- repo path shim: this file lives at emissive/data_prep/, data_toolkit/ is at root ---
ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(ROOT, "data_toolkit")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError(f"could not locate repo root (data_toolkit/) above {__file__}")
    ROOT = parent
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data_toolkit"))

os.environ.setdefault("HF_HOME", "/3dlg-jupiter-project/lightgen/hf_cache")

# TexVerse manifest (sid -> glb path), only needed when --glb is not passed.
PARQUET = "/3dlg-falas/project/omages/datasets/TexVerse/lightgen/emissive_thumbnails_obj_ids_df.parquet"
GLB_ROOT = "/3dlg-falas/project/omages/datasets/TexVerse/TexVerse-1K"

# ---------------------------------------------------------------------------------
# Binary emissive threshold.
#
# The somage GT this is A/B'd against binarizes with somage_to_glb.emissive_face_mask:
# it normalizes the emission_color map to [0,1], takes bright = val.max(over channels),
# and labels a face emissive iff bright > 0.04. We mirror that decision rule exactly on
# the per-voxel emissive attr: normalize to [0,1], luminance = max over the 3 channels,
# emissive iff > EMIS_THRESH. Using the SAME 0.04 cut and the SAME max-channel reduction
# keeps the two GTs comparable (any difference is then provenance/geometry, not a
# threshold artifact). The pilot also stores the raw luminance so the cut can be swept
# post-hoc without re-voxelizing.
# ---------------------------------------------------------------------------------
EMIS_THRESH = 0.04


def resolve_glb(sid):
    import pandas as pd
    df = pd.read_parquet(PARQUET)
    if "success" in df.columns:
        df = df[df["success"] == True]  # noqa: E712
    if sid not in df.index:
        raise KeyError(f"sid {sid} not in manifest parquet")
    p = os.path.join(GLB_ROOT, df.loc[sid, "glb_1k_path"])
    if not os.path.exists(p):
        raise FileNotFoundError(f"glb missing on disk: {p}")
    return p


def _normalize_emissive(emissive):
    """o_voxel stores color-like attrs as uint8 [0,255] (cf. base_color/255 in
    vxz_to_slat.py). Normalize to [0,1] regardless of whether it comes back uint8 or
    already-float, so EMIS_THRESH is a fixed [0,1] cut like somage's."""
    em = emissive.float()
    if emissive.dtype == torch.uint8 or em.max() > 1.5:
        em = em / 255.0
    return em


def build_one(sid, glb_path, out_dir, thresh=EMIS_THRESH):
    import o_voxel
    from glb_to_vxz import glb_to_vxz

    os.makedirs(out_dir, exist_ok=True)
    direct_vxz = os.path.join(out_dir, "direct.vxz")
    npz_path = os.path.join(out_dir, "emis_gt_direct.npz")

    # ONE voxelization of the ORIGINAL glb. glb_to_vxz writes every attr from
    # textured_mesh_to_volumetric_attr (incl. emissive) + dual_vertices/intersected.
    glb_to_vxz(glb_path, direct_vxz)

    coords, data = o_voxel.io.read(direct_vxz)
    if "emissive" not in data:
        raise RuntimeError(f"{sid}: direct.vxz has NO emissive attr — bake broken "
                           f"(finding (a)). attrs={sorted(data.keys())}")

    coords_np = coords.cpu().numpy().astype(np.int32)          # (M,3) @512-res
    em = _normalize_emissive(data["emissive"])                 # (M,3) in [0,1]
    lum = em.max(dim=-1).values if em.ndim == 2 else em.reshape(em.shape[0], -1).max(dim=-1).values
    lum_np = lum.cpu().numpy().astype(np.float32)              # (M,) max-channel luminance
    mask_np = (lum_np > thresh)                                # (M,) bool

    meta = dict(sid=sid, glb=glb_path, n_voxels=int(coords_np.shape[0]),
                emis_thresh=float(thresh),
                emissive_frac=float(mask_np.mean()) if mask_np.size else 0.0,
                lum_max=float(lum_np.max()) if lum_np.size else 0.0,
                lum_mean=float(lum_np.mean()) if lum_np.size else 0.0)
    np.savez_compressed(npz_path, coords=coords_np, emissive_lum=lum_np,
                        mask=mask_np, thresh=np.float32(thresh))
    json.dump(meta, open(os.path.join(out_dir, "meta_direct.json"), "w"))
    print(f"[ok] {sid} n_vox={meta['n_voxels']} emis_frac={meta['emissive_frac']:.4f} "
          f"lum_max={meta['lum_max']:.4f}", flush=True)
    return meta


# =================================================================================
# STAGE 2 (full-rebuild TODO — not exercised by the pilot) ==========================
# =================================================================================
#
# The training tuple mirrors vxz_to_slat.py's (shape_slat, input_tex_slat,
# output_tex_slat, ...). For the DIRECT pipeline the plan is:
#
#   stage2_todo_input_vxz(): from the SAME direct.vxz, produce the INPUT-side vxz by
#     ZEROING the emissive attr (and/or dropping it), so the input latent carries the
#     shape's albedo/PBR appearance with no emissive signal. Then run vxz_to_slat's
#     encoders on it.
#
#   stage2_todo_output_target(): the emissive GT (emis_gt_direct.npz here) becomes the
#     supervision target, pooled to the 32^3 latent grid exactly as make_emis_mask.py
#     pools the somage output.vxz (floor(coord/16) block-fraction). No separate
#     somage/emissive.glb voxelization is needed.
#
# LEAKAGE — answered in code, not by assertion:
#   vxz_to_slat.vxz_to_latent_slat (data_toolkit/vxz_to_slat.py:24-30) builds the tex
#   encoder input as
#       attr = cat([base_color, metallic, roughness, alpha]) / 255 * 2 - 1
#   i.e. it reads ONLY base_color(3) + metallic(1) + roughness(1) + alpha(1) = 6 ch.
#   The `emissive` attr is NEVER passed to the released tex encoder
#   (tex_enc_next_dc_f16c32_fp16). The shape encoder (lines 21-22) only sees
#   dual_vertices + intersected (geometry). So a SINGLE voxelization of the original
#   glb already yields a leak-free input latent: the emissive channel that defines the
#   TARGET is not among the channels the input encoders consume. This is the structural
#   reason the two-hop somage bake is unnecessary for leakage safety — one voxelization
#   suffices. (If a future re-trained tex VAE is given emissive as an input channel,
#   this note must be revisited and the input-side zeroing above becomes load-bearing.)
#
# def stage2_todo_input_vxz(direct_vxz, out_vxz): ...   # zero emissive, rewrite vxz
# def stage2_todo_output_target(npz, slat_coords): ...  # pool to 32^3 like make_emis_mask
# =================================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", required=True)
    ap.add_argument("--glb", default=None, help="original glb path; if omitted, resolve from manifest by sid")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--thresh", type=float, default=EMIS_THRESH)
    args = ap.parse_args()

    glb = args.glb or resolve_glb(args.sid)
    build_one(args.sid, glb, args.out_dir, thresh=args.thresh)


if __name__ == "__main__":
    main()
