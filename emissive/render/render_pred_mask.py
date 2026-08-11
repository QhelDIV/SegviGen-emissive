#!/usr/bin/env python3
"""Voxel-mask visualization, the representation finetune_binary_v1's pred_w1ema.png /
pred_w5ema.png panels used: predicted (or ground-truth) voxels drawn directly as white/black
cubes, not multiplied by albedo.

This is render_predictions.py's method (coarsen_binary / voxel_mesh / the aligned-frame
formula, all copied verbatim below), generalized two ways for the fbv1_repro comparison:

  --field {pred,gt}   pred thresholds pred_bc per-voxel at --thr; gt uses the npz's own
                       gt_e boolean directly (no threshold applies)
  --thr FLOAT          only meaningful for --field pred

Input: the {coords, pred_bc, gt_e} npz written by dump_pred_voxels.py or
dump_pred_voxels_repro.py -- same schema, so this script does not care which one produced
a given file, or how many draws it averaged; it only looks at the draw-0 arrays.

Camera and frame are fixed (0,-2.6,1.4), not per-shape solved: this deliberately matches
the OLD page's rendering rather than render_emissive.py's solved camera, since the point
is a like-for-like comparison against pred_w1ema.png / pred_w5ema.png.

Run on a CPU node, shared venv, PYTHONPATH=<xgutils>/src.
"""
import os
import sys
import argparse

import numpy as np
import bpy  # noqa: E402
from xgutils import bpyutil  # noqa: E402

WHITE = np.array([0.96, 0.97, 0.98], np.float32)
BLACK = np.array([0.10, 0.11, 0.13], np.float32)

_CUBE_V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.float32) - 0.5
_CUBE_F = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                    [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]], np.int32)


def coarsen_binary(coords, is_white, factor=4, max_cells=15000):
    """Downsample to a coarse grid; per cell, majority vote of the per-voxel binary field."""
    c64 = coords.astype(np.int64)
    while True:
        cc = c64 // factor
        uniq, inv = np.unique(cc, axis=0, return_inverse=True)
        if len(uniq) <= max_cells or factor >= 64:
            break
        factor *= 2
    n = len(uniq)
    cnt = np.bincount(inv, minlength=n)
    white_cnt = np.bincount(inv, weights=is_white.astype(np.float64), minlength=n)
    maj_white = (white_cnt / cnt) > 0.5
    return uniq, maj_white, factor


def voxel_mesh(uniq, factor, color, pad=0.92):
    """Same aligned frame render_predictions.py derives: a raw 512-res voxel index maps to
    bpyutil.normalize_mesh's frame via (index+0.5)/256 - 1, so this panel lines up with the
    appearance/GT mesh panels rendered through that normalization."""
    uniq = uniq.astype(np.float32)
    centers = (uniq * factor + factor / 2.0) / 256.0 - 1.0
    cell = (factor / 256.0) * pad
    cube = _CUBE_V * cell
    V = (centers[:, None, :] + cube[None, :, :]).reshape(-1, 3)
    F = (_CUBE_F[None] + (np.arange(len(uniq)) * 8)[:, None, None]).reshape(-1, 3)
    C = np.repeat(color, 8, axis=0)
    return V.astype(np.float32), F.astype(np.int32), C.astype(np.float32)


def _save(img, path):
    from PIL import Image
    img = np.asarray(img, np.float32)
    if img.ndim == 3 and img.shape[2] == 4:
        rgb, a = img[..., :3], img[..., 3:4]
        img = rgb * a + 0.07 * (1 - a)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def render_one(npz_path, out_path, field, thr, res=512):
    d = np.load(npz_path)
    coords = d["coords"]
    if field == "gt":
        is_white = d["gt_e"].astype(bool)
    else:
        is_white = d["pred_bc"].astype(np.float32) > thr
    cc, maj_white, factor = coarsen_binary(coords, is_white)
    color = np.where(maj_white[:, None], WHITE[None], BLACK[None]).astype(np.float32)
    V, F, C = voxel_mesh(cc, factor, color)
    bpyutil.load_blend(bpyutil.preset_glb)
    bpyutil.clear_collection("workbench")
    img = bpyutil.render_mesh(V, F, vert_color=C, resolution=(res, res), samples=24,
                              shadow_catcher=False, camera_position=(0, -2.6, 1.4),
                              camera_up=(0, 0, 1))
    _save(img, out_path)
    return {"n_coarse_cells": int(len(cc)), "coarsen_factor": int(factor),
            "white_frac": float(maj_white.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True, help="dir with <sid>.npz dumps")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sids", required=True, help="comma-separated")
    ap.add_argument("--field", required=True, choices=["pred", "gt"])
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tag", required=True, help="output files: <sid>_vox_<tag>.png")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--overwrite", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sids = args.sids.split(",")
    stats = {}
    for i, sid in enumerate(sids):
        npz = os.path.join(args.npz_dir, f"{sid}.npz")
        out = os.path.join(args.out, f"{sid}_vox_{args.tag}.png")
        if not os.path.exists(npz):
            print(f"  [{i + 1}/{len(sids)}] {sid} MISSING npz {npz}", flush=True)
            continue
        if os.path.exists(out) and not args.overwrite:
            print(f"SKIP {sid}", flush=True)
            continue
        try:
            s = render_one(npz, out, args.field, args.thr, args.res)
            stats[sid] = s
            print(f"OK [{i + 1}/{len(sids)}] {sid} field={args.field} thr={args.thr} "
                  f"white_frac={s['white_frac']:.4f} cells={s['n_coarse_cells']}", flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"FAIL {sid}", flush=True)
    import json
    json.dump(stats, open(os.path.join(args.out, f"stats_{args.tag}.json"), "w"), indent=1)
    print("ALL_DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
