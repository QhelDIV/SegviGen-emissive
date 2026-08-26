#!/usr/bin/env python3
"""Recompute the mesh-vs-volume frame check for a whole directory of exports, with one
metric, after the fact.

Why this exists. render_voxel_native.py checks every panel as it exports it, but the
check's METRIC was corrected partway through a 330-panel run (symmetric occupancy IoU
replaced by coverage of the volume's cells, because the exporter's remesh legitimately
adds surface and a superset was being scored as a misalignment). Panels exported before
and after that correction carry numbers that mean different things, and a page must not
quote a worst-case over a mixed population.

This re-derives the number for every panel from the artifacts on disk, so the reported
distribution is one metric computed one way. It also re-verifies each panel
independently of the run that produced it, which is stronger than trusting a value the
producing process wrote about itself.

Writes `frame_coverage` into each panel's __stats.json (leaving the original
`frame_iou_vs_volume` untouched as a record of what the run itself saw) and prints the
distribution plus anything below --min.

Usage:
  python emissive/render/verify_frames.py \\
      --vn_dir <dir of exported glbs and __stats.json> \\
      --gen_root <generation tree>  [--min 0.9]
"""
import argparse
import glob
import json
import os
import statistics as st

import numpy as np
import trimesh

RES = 32
N_SAMPLES = 200000


def surface_points(mesh, n=N_SAMPLES, seed=0):
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces)
    tris = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0],
                                         tris[:, 2] - tris[:, 0]), axis=1)
    if area.sum() <= 0:
        return v
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tris), size=n, p=area / area.sum())
    t = tris[pick]
    u = rng.random((n, 1)); w = rng.random((n, 1))
    over = (u + w) > 1
    u[over] = 1 - u[over]; w[over] = 1 - w[over]
    return t[:, 0] + u * (t[:, 1] - t[:, 0]) + w * (t[:, 2] - t[:, 0])


def occupancy(pts, res=RES):
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    s = (hi - lo).max()
    i = np.clip((((pts - c) / s) + 0.5) * res, 0, res - 1).astype(int)
    g = np.zeros((res, res, res), dtype=bool)
    g[i[:, 0], i[:, 1], i[:, 2]] = True
    return g


def npz_for(panel_id, gen_root):
    sid, _, kind = panel_id.rpartition("_")
    sub = "draw0" if kind == "gt" else kind
    return os.path.join(gen_root, sub, f"{sid}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vn_dir", required=True)
    ap.add_argument("--gen_root", required=True)
    ap.add_argument("--min", type=float, default=0.9)
    args = ap.parse_args()

    vals, low, missing = [], [], []
    stats_files = sorted(glob.glob(os.path.join(args.vn_dir, "*__stats.json")))
    for sp in stats_files:
        pid = os.path.basename(sp)[: -len("__stats.json")]
        glb = os.path.join(args.vn_dir, f"{pid}.glb")
        npz = npz_for(pid, args.gen_root)
        if not (os.path.isfile(glb) and os.path.isfile(npz)):
            missing.append(pid)
            continue
        mesh = trimesh.load(glb, force="mesh")
        g_mesh = occupancy(surface_points(mesh))
        g_vox = occupancy(np.load(npz)["coords"].astype(np.float64))
        cov = float((g_mesh & g_vox).sum()) / max(int(g_vox.sum()), 1)
        vals.append(cov)
        d = json.load(open(sp))
        d["frame_coverage"] = round(cov, 4)
        json.dump(d, open(sp, "w"), indent=1)
        if cov < args.min:
            low.append((pid, cov))

    if vals:
        print(f"checked {len(vals)} panels: coverage min {min(vals):.4f} "
              f"median {st.median(vals):.4f} mean {st.mean(vals):.4f}")
    print(f"below {args.min}: {len(low)}")
    for pid, c in sorted(low, key=lambda x: x[1])[:20]:
        print(f"   {pid}  {c:.4f}")
    if missing:
        print(f"could not check {len(missing)} panels (glb or npz absent): {missing[:5]}")


if __name__ == "__main__":
    main()
