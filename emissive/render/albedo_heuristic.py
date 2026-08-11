#!/usr/bin/env python3
"""The coverage-matched albedo-brightness baseline, emitted as a "prediction".

Rule: light the brightest texels of the asset, choosing how many so that the LIT
SURFACE AREA equals that shape's own ground-truth emissive coverage.

Two decisions in that sentence carry the argument:

- Matched PER SHAPE, not at a fixed global percentile. A fixed percentile scores
  well for the same degenerate reason "light everything" does, and picking the
  percentile by sweeping on the eval set is oracle selection. Matching each
  shape's own coverage removes "it guessed the right amount" and leaves "it
  guessed the right place", which is the only question a picture can answer.
- Matched by AREA, not by texel count. Texel density per unit surface varies
  between materials, so a texel-count match would silently over-light whichever
  material happens to be densely parameterised. One global luminance threshold
  is solved against the area-weighted distribution instead.

Output is written in the SAME format as a model prediction, so the heuristic
goes through the same renderer and the same evaluator as the model rather than
through a parallel path that could differ. That is also what makes its IoU
comparable with the model's: one evaluator, one space.

  <out>/<sid>__mat<N>__emis.png   per material slot, 8-bit, white = lit
  <out>/<sid>__stats.json         target vs achieved coverage, threshold,
                                  "uniform" scalars for UV-less materials

Run on a CPU node with the shared venv plus PYTHONPATH=<xgutils>/src.
"""
import argparse
import json
import os
import traceback

import numpy as np
from PIL import Image

import bpy  # noqa: E402
from xgutils import bpyutil  # noqa: E402

import render_emissive as re_  # noqa: E402

# Rec.709 luma on LINEAR values. Blender's importer has already converted sRGB
# textures to linear, so this is applied to light, not to encoded bytes.
LUMA = np.array([0.2126, 0.7152, 0.0722])

# Cap for the threshold search only; the mask itself is applied at full texture
# resolution. Keeps the weighted sort bounded on assets with many 4K maps.
SEARCH_MAX = 512


def material_albedo(f):
    """(array_or_None, constant) for one material's base colour."""
    bsdf = f.get("bsdf")
    if bsdf is None:
        return None, np.zeros(3, dtype=np.float32)
    b_sock = bsdf.inputs["Base Color"]
    node = re_.upstream_image(b_sock)
    const = re_.socket_rgb(b_sock)
    if node is None:
        return None, const
    return re_.img_array(node.image)[..., :3] * const, const


def solve_threshold(entries, target):
    """The luminance cut whose AREA-weighted lit fraction equals `target`.

    entries: [(luminance array or scalar, area)] per material.
    """
    lums, weights = [], []
    for lum, area in entries:
        if area <= 0:
            continue
        if np.isscalar(lum):
            lums.append(np.array([float(lum)]))
            weights.append(np.array([area]))
        else:
            small = re_.resize_to(lum[..., None],
                                 (min(lum.shape[0], SEARCH_MAX),
                                  min(lum.shape[1], SEARCH_MAX)))[..., 0]
            v = small.ravel()
            lums.append(v)
            weights.append(np.full(v.shape, area / v.size))
    if not lums:
        return float("inf")
    v = np.concatenate(lums)
    w = np.concatenate(weights)
    if target <= 0:
        return float(v.max()) + 1.0        # nothing lit
    if target >= 1:
        return float(v.min()) - 1.0        # everything lit
    order = np.argsort(-v)                 # brightest first
    v, w = v[order], w[order]
    frac = np.cumsum(w) / w.sum()
    i = int(np.searchsorted(frac, target))
    return float(v[min(i, len(v) - 1)])


def refine_threshold(facts, areas, albedo, uv_ok, entries, target, seed):
    """Bisect the threshold against full-resolution coverage."""
    vals = [float(l.min()) if not np.isscalar(l) else float(l) for l, a in entries if a > 0]
    his = [float(l.max()) if not np.isscalar(l) else float(l) for l, a in entries if a > 0]
    if not vals:
        return seed
    lo, hi = min(vals), max(his)
    if target <= 0:
        return hi + 1.0
    if target >= 1:
        return lo - 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if coverage_at(facts, areas, albedo, uv_ok, mid, True) > target:
            lo = mid                      # too much lit, raise the cut
        else:
            hi = mid
    best = min((lo, hi),
               key=lambda t: abs(coverage_at(facts, areas, albedo, uv_ok, t, True)
                                 - target))
    return float(best)


def apply_cut(lum, thr, inclusive):
    return (lum >= thr) if inclusive else (lum > thr)


def coverage_at(facts, areas, albedo, uv_ok, thr, inclusive):
    """Area-weighted lit fraction for one cut, at full texture resolution."""
    lit = total = 0.0
    for f in facts:
        slot = f["slot"]
        area = float(areas[slot]) if slot < len(areas) else 0.0
        total += area
        arr, const = albedo[slot]
        if arr is None or not uv_ok.get(slot, False):
            lum = float((arr @ LUMA).mean()) if arr is not None else float(const @ LUMA)
            frac = 1.0 if apply_cut(np.array([lum]), thr, inclusive)[0] else 0.0
        else:
            frac = float(apply_cut(arr @ LUMA, thr, inclusive).mean())
        lit += area * frac
    return (lit / total) if total else 0.0


def one(sid, glb, out, args):
    bpyutil.load_blend(bpyutil.preset_glb)
    bpyutil.clear_collection("workbench")
    obj = bpyutil.load_glb(glb, import_shading=None)
    facts = re_.analyze(obj)
    areas = re_.material_areas(obj)
    target = re_.measure_lit_fraction(facts, areas)

    # pass 1: the luminance distribution, weighted by surface area
    albedo, entries = {}, []
    for f in facts:
        slot = f["slot"]
        area = float(areas[slot]) if slot < len(areas) else 0.0
        arr, const = material_albedo(f)
        albedo[slot] = (arr, const)
        entries.append(((arr @ LUMA) if arr is not None
                        else float(const @ LUMA), area))

    thr = solve_threshold(entries, target)
    # The subsample above only seeds the search. Solve the threshold against the
    # FULL-resolution coverage, because a nearest-neighbour subsample of a
    # structured texture is a biased estimate of its luminance distribution: on
    # the vending machine the seeded cut landed at 7.7 percent coverage against
    # a 13.9 percent target, a 45 percent miss that had nothing to do with ties.
    # Coverage is monotone in the threshold, so bisection is exact to tolerance.
    thr = refine_threshold(facts, areas, albedo, uv_usable(obj), entries, target,
                           thr)

    # Ties decide this baseline. Large flat regions of identical albedo are
    # common (a white panel, an untextured material), so a strict cut can
    # exclude a whole tied block and an inclusive one can admit it: on the
    # vending machine that was the difference between 7.7 and 13.9 percent
    # coverage, and on the untextured headphone stand a strict cut lit nothing
    # at all, making the baseline vacuous. Evaluate both at full resolution and
    # keep whichever lands closer to the shape's own coverage.
    uv_ok = uv_usable(obj)
    cands = [(abs(coverage_at(facts, areas, albedo, uv_ok, thr, inc) - target), inc)
             for inc in (False, True)]
    inclusive = min(cands)[1]

    # pass 2: apply that one cut everywhere, at full texture resolution
    os.makedirs(out, exist_ok=True)
    uniform, per_mat, lit_area, total_area = {}, [], 0.0, 0.0
    for f in facts:
        slot = f["slot"]
        area = float(areas[slot]) if slot < len(areas) else 0.0
        total_area += area
        arr, const = albedo[slot]
        if arr is None or not uv_ok.get(slot, False):
            # no texture, or no usable UVs to hold one: the material can only
            # answer whether it fires, exactly as the ground-truth column
            # renders such a material
            lum = float((arr @ LUMA).mean()) if arr is not None else float(const @ LUMA)
            on = 1.0 if apply_cut(np.array([lum]), thr, inclusive)[0] else 0.0
            uniform[str(slot)] = on
            frac = on
        else:
            mask = apply_cut(arr @ LUMA, thr, inclusive)
            frac = float(mask.mean())
            if frac > 0:
                Image.fromarray((mask * 255).astype(np.uint8), mode="L").convert(
                    "RGB").save(os.path.join(out, f"{sid}__mat{slot}__emis.png"))
        lit_area += area * frac
        per_mat.append({"slot": slot,
                        "material": f["mat"].name if f["mat"] else None,
                        "area": area, "lit_frac": frac,
                        "carrier": "uniform" if (arr is None or not uv_ok.get(slot, False))
                                   else "texture",
                        "gt_emits": bool(f.get("emits"))})

    achieved = lit_area / total_area if total_area else 0.0
    stats = {"sid": sid, "baseline": "albedo_brightness_coverage_matched",
             # one name per slot, in the asset's order: the renderer refuses a
             # prediction whose slot keying it cannot verify against these
             "materials": [m["material"] for m in per_mat],
             "target_coverage": target, "achieved_coverage": achieved,
             "luminance_threshold": thr, "cut_inclusive": bool(inclusive),
             "coverage_abs_error": abs(achieved - target),
             "uniform": uniform,
             "materials": per_mat}
    json.dump(stats, open(os.path.join(out, f"{sid}__stats.json"), "w"), indent=1)
    return stats


def uv_usable(obj):
    """Per material slot: does it have a non-degenerate UV parameterisation?

    A slot whose loops all collapse to one UV point cannot carry a texture at
    all; the jack-o'-lantern's flame is such a slot and is that shape's ONLY
    emitter, so getting this wrong silently discards the whole answer there.
    """
    me = obj.data
    uv = me.uv_layers.active
    out = {i: False for i in range(len(obj.material_slots))}
    if uv is None:
        return out
    n = len(me.loops)
    arr = np.empty(n * 2, dtype=np.float32)
    uv.data.foreach_get("uv", arr)
    arr = arr.reshape(n, 2)
    loop_mat = np.empty(n, dtype=np.int32)
    for p in me.polygons:
        for li in p.loop_indices:
            loop_mat[li] = p.material_index
    for i in out:
        sel = arr[loop_mat == i]
        if len(sel) == 0:
            continue
        span = sel.max(0) - sel.min(0)
        out[i] = bool(span[0] > 1e-6 and span[1] > 1e-6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--glb_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    rows = json.load(open(args.manifest))
    if args.only:
        keep = set(args.only.split(","))
        rows = [r for r in rows if r["sid"] in keep]
    for r in rows:
        sid = r["sid"]
        try:
            s = one(sid, os.path.join(args.glb_dir, f"{sid}.glb"), args.out, args)
            print(f"OK {sid} target={s['target_coverage']:.4f} "
                  f"achieved={s['achieved_coverage']:.4f} thr={s['luminance_threshold']:.4f}",
                  flush=True)
        except Exception:
            traceback.print_exc()
            print(f"FAIL {sid}", flush=True)
    print("ALL_DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
