#!/usr/bin/env python3
"""Ask, per panel: if we draw this mask as "mask x albedo", is there any light in it?

Both render pipelines in this project build emission the same way: take the
per-voxel emissive mask, multiply by the surface's base colour, and use the
product as the emitted colour. That convention has a failure case neither
pipeline can fix, and which looks identical to a broken pipeline: if the asset's
BASE COLOUR under the mask is black, the product is black and the panel renders
as an empty room even though the mask is perfectly correct.

A phone screen is the canonical case. In many assets the screen's base colour is
black (the screen is off in the material) and the picture lives entirely in the
emissive texture, so a mask that correctly selects the screen still multiplies to
nothing.

This probe separates the two causes. For each panel it decodes the shape's own
per-voxel base colour (the same decode the voxel-native render path uses) and
reports the mean and maximum base-colour brightness over exactly the voxels the
mask selects. A panel with a near-zero maximum would render dark through ANY
pipeline; a panel with a healthy maximum that still renders dark is a pipeline
failure.

Usage (trellis2 env; on this workstation the preload is needed because the system
C++ runtime is older than the one o_voxel's extension was built against):

  LD_PRELOAD=/cs/3dlg-jupiter-project/lightgen/miniforge3/lib/libstdc++.so.6 \\
  HF_HOME=/cs/3dlg-jupiter-project/lightgen/hf_cache \\
  /cs/3dlg-jupiter-project/lightgen/miniforge3/envs/trellis2/bin/python \\
      emissive/eval/mask_albedo_probe.py \\
      --dataset /cs/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct \\
      --npz_dir <converted_npz> --out probe.json
"""
import argparse
import glob
import json
import os
import sys
import traceback

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(ROOT, "inference_full.py")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError("could not locate the SegviGen repo root above this file")
    ROOT = parent
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/cs/3dlg-jupiter-project/lightgen/hf_cache")

import trellis2.modules.sparse as sp     # noqa: E402
from trellis2 import models              # noqa: E402

RESOLUTION = 512
SPLITS = ("val_72k", "test_72k", "train_72k")


def find_sample_dir(dataset_root, sid):
    for s in SPLITS:
        d = os.path.join(dataset_root, s, sid)
        if os.path.isdir(d):
            return d, s
    raise FileNotFoundError(f"{sid} not under {dataset_root}")


def transfer_mask(npz_coords, vals, dec_coords):
    """Match by voxel coordinate, not row order: the decoders are not bit-reproducible
    across runs, so two decodes of the same shape can differ by a handful of voxels."""
    R = RESOLUTION
    kn = (npz_coords[:, 0] * R + npz_coords[:, 1]) * R + npz_coords[:, 2]
    kd = (dec_coords[:, 0] * R + dec_coords[:, 1]) * R + dec_coords[:, 2]
    order = np.argsort(kn)
    ks = kn[order]
    pos = np.clip(np.searchsorted(ks, kd), 0, len(ks) - 1)
    hit = ks[pos] == kd
    out = np.zeros(len(kd), dtype=bool)
    out[hit] = vals[order[pos[hit]]]
    return out, float((~hit).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default="gt,draw2")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    tex_decoder = models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16").cuda().eval()
    shape_decoder = models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16").cuda().eval()

    sids = sorted(os.path.basename(p)[:-7]
                  for p in glob.glob(os.path.join(args.npz_dir, "*_gt.npz")))
    kinds = args.kinds.split(",")
    rows, n_fail = [], 0
    for sid in sids:
        try:
            d, split = find_sample_dir(args.dataset, sid)
            shp = torch.load(os.path.join(d, "shape_slat.pth"), map_location="cuda")
            itx = torch.load(os.path.join(d, "input_tex_slat.pth"), map_location="cuda")
            coords = shp["coords"].cuda()
            with torch.no_grad():
                shape_decoder.set_resolution(RESOLUTION)
                _, subs = shape_decoder(
                    sp.SparseTensor(shp["feats"].cuda(), coords), return_subs=True)
                tex = tex_decoder(sp.SparseTensor(itx["feats"].cuda(), coords),
                                  guide_subs=subs) * 0.5 + 0.5
            bc = tex.feats[:, :3].float().clamp(0, 1)
            lum = bc.mean(-1).cpu().numpy()
            dec_coords = tex.coords[:, 1:].cpu().numpy().astype(np.int64)
        except Exception:
            n_fail += 1
            print(f"[FAIL decode] {sid}", flush=True)
            traceback.print_exc()
            continue

        for kind in kinds:
            p = os.path.join(args.npz_dir, f"{sid}_{kind}.npz")
            if not os.path.isfile(p):
                continue
            z = np.load(p)
            vals = (z["gt_e"].astype(bool) if kind == "gt"
                    else z["pred_bc"].astype(np.float32) > args.thr)
            m, miss = transfer_mask(z["coords"].astype(np.int64), vals, dec_coords)
            sel = lum[m]
            rows.append({
                "sid": sid, "kind": kind, "split": split,
                "n_voxels": int(len(lum)), "mask_frac": float(m.mean()),
                "coord_miss_frac": miss,
                "albedo_mean_under_mask": float(sel.mean()) if sel.size else 0.0,
                "albedo_max_under_mask": float(sel.max()) if sel.size else 0.0,
                "albedo_p99_under_mask": float(np.percentile(sel, 99)) if sel.size else 0.0,
                "frac_masked_above_0p05": float((sel > 0.05).mean()) if sel.size else 0.0,
            })
            r = rows[-1]
            print(f"{sid} {kind:6s} mask={r['mask_frac']:.4f} "
                  f"albedo mean={r['albedo_mean_under_mask']:.4f} "
                  f"p99={r['albedo_p99_under_mask']:.4f} "
                  f"max={r['albedo_max_under_mask']:.4f} "
                  f"lit_frac={r['frac_masked_above_0p05']:.3f}", flush=True)
        torch.cuda.empty_cache()

    json.dump(rows, open(args.out, "w"), indent=1)
    print(f"DONE {len(rows)} panels, {n_fail} shapes failed to decode -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
