"""
Round-trip validation of the voxel->UV resampler, and calibration of its one free parameter.

Push the DECODED GT emissive mask through exactly the path a prediction takes, then compare
the resulting area-weighted lit fraction against the published oracle column's own
`area_lit_frac` (gallery.json, produced by render_emissive's `rebuild_emission`). If the GT
mask does not come back looking like the GT column, the resampler is wrong, and a predicted
column built on it would be wrong the same way without ever announcing it.

The resampler has one free parameter: `--tol`, how close a texel's surface point must be to
a lit voxel to count as lit. It cannot be zero (texels sample the surface continuously,
voxels sit on a 512 lattice) and too large dilates thin emissive regions. Calibrating it
against GT is legitimate and is the point of the round-trip; it is fixed there and then
applied unchanged to the predictions, never re-tuned per model or per shape.

  python emissive/probes/roundtrip_check.py --pv3 .../paper_v3 --gallery .../gallery.json \
      --dirs gt_roundtrip_tol0.87 gt_roundtrip_tol1.0 gt_roundtrip_tol1.5 gt_roundtrip
"""
import os
import json
import argparse

import numpy as np


def area_weighted_lit(stats, survey_entry):
    """Sum area_frac * lit over materials, using the uniform scalar where the material has
    no usable parameterisation -- the same quantity render_emissive's box summary computes
    as `area_lit_frac` (it sums area * mask_frac over all materials)."""
    area = {m["material"]: m["area_frac"] for m in survey_entry["materials"]}
    tot, matched = 0.0, 0
    # `materials` is the renderer's flat slot-ordered name list; the per-material records
    # live under `materials_detail`. Older stats files predate the split.
    for m in stats.get("materials_detail") or stats["materials"]:
        a = area.get(m["material_name"])
        if a is None:
            continue
        matched += 1
        lit = m.get("uniform_value") if m.get("uv_degenerate") else m["lit_texel_frac_of_covered"]
        tot += a * (lit or 0.0)
    return tot, matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv3", required=True)
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    survey = json.load(open(os.path.join(args.pv3, "material_survey.json")))
    gal = {e["sid"]: e for e in json.load(open(args.gallery))}

    out = {}
    print(f"{'tol dir':26}{'n':>4}{'r':>8}{'median ratio':>14}{'max ratio':>11}"
          f"{'mean |log2 ratio|':>19}")
    for d in args.dirs:
        root = os.path.join(args.pv3, "pred_masks", d)
        pub, rt, sids = [], [], []
        for sid in sorted(gal):
            sp = os.path.join(root, f"{sid}__stats.json")
            if not os.path.exists(sp) or sid not in survey:
                continue
            tot, _ = area_weighted_lit(json.load(open(sp)), survey[sid])
            p = gal[sid]["area_lit_frac"]
            if p <= 0:
                continue
            pub.append(p); rt.append(tot); sids.append(sid)
        if len(pub) < 3:
            print(f"{d:26}{len(pub):>4}   (too few shapes)")
            continue
        pub, rt = np.array(pub), np.array(rt)
        ratio = rt / pub
        # symmetric error: over- and under-shooting by the same factor score the same
        score = float(np.mean(np.abs(np.log2(np.maximum(ratio, 1e-9)))))
        print(f"{d:26}{len(pub):>4}{np.corrcoef(pub, rt)[0, 1]:>8.3f}"
              f"{np.median(ratio):>14.3f}{ratio.max():>11.2f}{score:>19.3f}")
        out[d] = {"n": len(pub), "r": float(np.corrcoef(pub, rt)[0, 1]),
                  "median_ratio": float(np.median(ratio)), "max_ratio": float(ratio.max()),
                  "mean_abs_log2_ratio": score,
                  "per_shape": [{"sid": s, "what": gal[s]["what"], "published": float(a),
                                 "roundtrip": float(b), "ratio": float(b / a)}
                                for s, a, b in zip(sids, pub, rt)]}

    if out:
        best = min(out, key=lambda k: out[k]["mean_abs_log2_ratio"])
        print(f"\nBEST {best}  (mean |log2 ratio| = {out[best]['mean_abs_log2_ratio']:.3f}, "
              f"r = {out[best]['r']:.3f})")
        print(f"\nper-shape for {best}:")
        print(f"  {'sid':10}{'what':24}{'published':>11}{'roundtrip':>11}{'ratio':>8}")
        for e in sorted(out[best]["per_shape"], key=lambda e: e["published"]):
            print(f"  {e['sid'][:8]:10}{(e['what'] or '')[:23]:24}"
                  f"{e['published']:>11.4f}{e['roundtrip']:>11.4f}{e['ratio']:>8.2f}")
    if args.json_out:
        json.dump(out, open(args.json_out, "w"), indent=1)
        print(f"\nWROTE {args.json_out}")


if __name__ == "__main__":
    main()
