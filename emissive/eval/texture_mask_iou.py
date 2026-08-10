"""
Score a TEXTURE-space mask (a baseline, or a model's mask) with the same per-voxel IoU the
model numbers use, by lifting the mask into the 512-res voxel space.

Why lift rather than compare in texture space: every model number on this project is a
per-voxel IoU against the decoded GT. A baseline scored in texture space would not be
comparable to it, and the whole point of this measurement is that both come from one
evaluator in one space.

The lift is the inverse of pred_mask_to_asset.py's projection and reuses its rasteriser: a
material's UV atlas gives a surface position per texel, the lit texels become a point set in
voxel coordinates, and a GT voxel counts as lit if a lit surface point falls within `tol` of
it. Materials with no usable parameterisation carry a scalar instead, and their surface is
sampled area-weighted over their triangles -- skipping them would score a shape as predicting
nothing exactly where its only emitter lives (48af42db's `Flame_0`).

BECAUSE the lift is not free, it is CONTROLLED: the same lift is applied to the model's own
texture masks, whose direct voxel-space IoU is already known. If lifted and direct agree for
the model, the lift is neutral and the baseline's lifted number is comparable. If they do
not agree, the comparison is reported as invalid rather than quietly published.

  python emissive/eval/texture_mask_iou.py --masks .../pred_masks/albedo_matched \
      --control .../pred_masks/emis_72k_unfilt --npz_root .../pred_voxels/emis_72k_unfilt \
      --glb_dir .../glb_src --survey .../material_survey.json --out iou.json
"""
import os
import json
import argparse

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pred_mask_to_asset import read_glb, primitives, rasterise_into, GRID

RNG = np.random.default_rng(0)


def sample_triangles(positions, faces, n):
    """Area-weighted uniform samples over a material's triangles, for materials whose mask
    is a single scalar and therefore has no texel positions to use."""
    tri = positions[faces]
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    tot = area.sum()
    if tot <= 0:
        return positions
    idx = RNG.choice(len(area), size=n, p=area / tot)
    u = RNG.random((n, 1))
    v = RNG.random((n, 1))
    over = (u + v) > 1
    u[over] = 1 - u[over]
    v[over] = 1 - v[over]
    return v0[idx] + u * (v1[idx] - v0[idx]) + v * (v2[idx] - v0[idx])


def lit_points_for_shape(sid, mask_dir, glb_path, slot_order, tex=1024, n_uniform=200000):
    """Every surface point the texture mask calls emissive, in the asset's own frame."""
    gltf, bins = read_glb(glb_path)
    prims = primitives(gltf, bins)
    allpos = np.concatenate([p["positions"] for p in prims], axis=0)
    lo, hi = allpos.min(0), allpos.max(0)
    centre = (lo + hi) / 2.0
    scale = 0.99999 / (hi - lo).max()

    st_path = os.path.join(mask_dir, f"{sid}__stats.json")
    uniform = {}
    if os.path.exists(st_path):
        uniform = {int(k): float(v) for k, v in (json.load(open(st_path)).get("uniform") or {}).items()}

    name_to_slot = {n: i for i, n in enumerate(slot_order)}
    mat_names = {i: m.get("name", f"material_{i}") for i, m in enumerate(gltf.get("materials", []))}
    by_mat = {}
    for p in prims:
        by_mat.setdefault(p["material"], []).append(p)

    pts, per_mat = [], []
    for mat, plist in by_mat.items():
        name = mat_names.get(mat, f"material_{mat}")
        slot = name_to_slot.get(name)
        if slot is None:
            per_mat.append({"material": name, "slot": None, "skipped": "not in survey"})
            continue
        png = os.path.join(mask_dir, f"{sid}__mat{slot}__emis.png")
        rec = {"material": name, "slot": slot}

        if os.path.exists(png):
            m = np.asarray(Image.open(png).convert("L")).astype(np.float32) / 255.0
            pos_buf = np.zeros((tex, tex, 3), dtype=np.float64)
            valid = np.zeros((tex, tex), dtype=bool)
            for p in plist:
                if p["uv"] is None or len(p["uv"]) != len(p["positions"]):
                    continue
                rasterise_into(p["uv"], p["faces"], p["positions"], tex, pos_buf, valid)
            if m.shape != (tex, tex):
                m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(
                    (tex, tex), Image.NEAREST)).astype(np.float32) / 255.0
            sel = valid & (m > 0.5)
            rec.update(carrier="texture", texel_coverage=float(valid.mean()),
                       lit_texels=int(sel.sum()))
            if sel.any():
                pts.append(pos_buf[sel])
        elif uniform.get(slot, 0.0) >= 0.5:
            # no parameterisation: the whole material is lit, so sample its surface
            pos = np.concatenate([p["positions"] for p in plist], axis=0)
            off, faces = 0, []
            for p in plist:
                faces.append(p["faces"] + off); off += len(p["positions"])
            s = sample_triangles(pos, np.concatenate(faces, axis=0), n_uniform)
            pts.append(s)
            rec.update(carrier="uniform", uniform_value=uniform[slot], sampled=int(len(s)))
        else:
            rec.update(carrier="none")
        per_mat.append(rec)

    if not pts:
        return np.zeros((0, 3)), centre, scale, per_mat
    return np.concatenate(pts, axis=0), centre, scale, per_mat


def iou_from_points(pts, centre, scale, coords, gt_e, tol):
    if len(pts) == 0:
        pred = np.zeros(len(coords), dtype=bool)
    else:
        q = ((((pts - centre) * scale) + 0.5) * GRID) - 0.5
        tree = cKDTree(q)
        d, _ = tree.query(coords.astype(np.float64), k=1)
        pred = d <= tol
    inter = int((pred & gt_e).sum()); union = int((pred | gt_e).sum())
    return (inter / union if union > 0 else 1.0), float(pred.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True, help="the mask dir to score")
    ap.add_argument("--control", default=None,
                    help="a mask dir whose DIRECT voxel-space IoU is already known, lifted "
                         "the same way to show the lift is neutral")
    ap.add_argument("--control_summary", default=None,
                    help="summary.json holding the control's direct voxel-space IoU")
    ap.add_argument("--npz_root", required=True)
    ap.add_argument("--glb_dir", required=True)
    ap.add_argument("--survey", required=True)
    ap.add_argument("--tol", type=float, default=0.87)
    ap.add_argument("--tex", type=int, default=1024)
    ap.add_argument("--only", default=None,
                    help="comma-separated sids; score just these (the headphone stand costs ~7 min\n"
                         "of rasterisation on its own, so rescoring the whole set to add one\n"
                         "shape is not worth it)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    survey = json.load(open(args.survey))
    direct = json.load(open(args.control_summary)) if args.control_summary else {}
    sids = sorted(s for s in survey
                  if os.path.exists(os.path.join(args.masks, f"{s}__stats.json")))
    if args.only:
        keep = set(args.only.split(","))
        sids = [s for s in sids if s in keep]
    print(f"scoring {len(sids)} shapes from {args.masks}\n")

    rows = []
    for sid in sids:
        z = np.load(os.path.join(args.npz_root, f"{sid}.npz"))
        coords = z["coords"].astype(np.float64)
        gt_e = z["gt_e"].astype(bool)
        slot_order = [m["material"] for m in sorted(survey[sid]["materials"], key=lambda m: m["slot"])]
        glb = os.path.join(args.glb_dir, f"{sid}.glb")

        pts, c, s, per_mat = lit_points_for_shape(sid, args.masks, glb, slot_order, args.tex)
        iou, cov = iou_from_points(pts, c, s, coords, gt_e, args.tol)
        row = {"sid": sid, "iou": iou, "lifted_coverage": cov, "gt_coverage": float(gt_e.mean()),
               "n_lit_points": int(len(pts)), "materials": per_mat}

        if args.control:
            cpts, cc, cs, _ = lit_points_for_shape(sid, args.control, glb, slot_order, args.tex)
            ciou, ccov = iou_from_points(cpts, cc, cs, coords, gt_e, args.tol)
            row["control_lifted_iou"] = ciou
            row["control_lifted_coverage"] = ccov
            if sid in direct:
                row["control_direct_iou"] = direct[sid]["iou_by_thr"]["0.5"]
                row["control_direct_coverage"] = direct[sid]["pred_frac_by_thr"]["0.5"]
        rows.append(row)
        # persist after EVERY shape. This job's slowest shape (the 317k-vertex headphone
        # stand, 13 atlases rasterised twice) can eat a third of the walltime on its own, and
        # writing only at the end means a timeout loses every shape rather than the last one.
        json.dump({"masks": args.masks, "tol": args.tol, "n": len(rows), "partial": True,
                   "per_shape": rows}, open(args.out, "w"), indent=1)
        print(f"{sid[:8]}  IoU {iou:.4f}  lifted_cov {cov:.4f}  gt_cov {gt_e.mean():.4f}"
              + (f"   | control lifted {row.get('control_lifted_iou', float('nan')):.4f} "
                 f"direct {row.get('control_direct_iou', float('nan')):.4f}" if args.control else ""),
              flush=True)

    ious = np.array([r["iou"] for r in rows])
    out = {"masks": args.masks, "tol": args.tol, "n": len(rows),
           "mean_iou": float(ious.mean()), "median_iou": float(np.median(ious)),
           "per_shape": rows}
    if args.control:
        cl = np.array([r["control_lifted_iou"] for r in rows])
        cd = np.array([r["control_direct_iou"] for r in rows if "control_direct_iou" in r])
        out["control_lifted_mean"] = float(cl.mean())
        if len(cd) == len(cl):
            out["control_direct_mean"] = float(cd.mean())
            out["control_lift_bias_mean_abs"] = float(np.mean(np.abs(cl - cd)))
            print(f"\nCONTROL  lifted mean {cl.mean():.4f} vs direct mean {cd.mean():.4f}   "
                  f"mean |per-shape difference| {np.mean(np.abs(cl - cd)):.4f}")
    print(f"\n{os.path.basename(args.masks)}  mean IoU {ious.mean():.4f}  "
          f"median {np.median(ious):.4f}  n={len(rows)}")
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
