"""Rebake the emissive mask against Blender's OWN imported UVs (the same
interpretation render_emissive_closest.py reads at render time) instead of the
raw glTF UVs pred_mask_to_asset.py used, so write side and read side share one
UV source by construction -- no divergence possible, by design rather than by
hope.

Also runs a diagnostic UV-diff first: for every face, compares Blender's
imported per-corner UV against the raw glTF per-vertex UV (matched via the
SAME per-material-cursor face correspondence validated earlier in this
investigation), under the best-fitting corner permutation, and reports what
changed -- not just that something changed.

Flat-material branch: for a material with no baseColorTexture actually wired
to Base Color in Blender's OWN imported node graph (checked directly, not
inferred from the glTF JSON), the per-texel mask carries no useful albedo
regardless of how well it is rasterized. Instead, compute that material's lit
FACE fraction straight from the voxel field (bypassing the texture atlas
entirely); if it clears --flat_thr, write a literal RGB uniform emission color
= baseColorFactor * lit_fraction into the stats.json 'uniform' entry (a list,
not the old binary scalar); otherwise no emission. render_emissive_closest.py
is patched (this same local copy) to treat a list-valued uniform entry as a
literal emission colour rather than the old on/off flag.

Usage (bpy job, solar only):
  PYTHONPATH=<xgutils>/src <venv>/bin/python bpy_rebake.py \
      --glb <path> --npz <coords+pred_bc+gt_e> --value pred|gt \
      --out_dir <dir> --sid <sid> --thr 0.5 --tex 1024 --tol 2.0 \
      --flat_thr 0.5
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from PIL import Image

sys.path.insert(0, "/project/3dlg-hcvc/omages/xgutils/src")
from xgutils import bpyutil
import bpy

sys.path.insert(0, "/3dlg-jupiter-project/lightgen/segvigen_emissive/code")
from pred_mask_to_asset import read_glb, primitives, rasterise_into  # noqa: E402

GRID = 512


def compute_frame(gltf, bins):
    prims = primitives(gltf, bins)
    allpos = np.concatenate([p["positions"] for p in prims], axis=0)
    lo, hi = allpos.min(0), allpos.max(0)
    centre = (lo + hi) / 2.0
    scale = 0.99999 / (hi - lo).max()
    return centre, scale, prims


def to_voxel(pts, centre, scale):
    return ((((pts - centre) * scale) + 0.5) * GRID) - 0.5


def material_albedo_const(gltf, mat_idx):
    mat = gltf["materials"][mat_idx]
    pbr = mat.get("pbrMetallicRoughness", {})
    if "baseColorFactor" in pbr:
        return np.array(pbr["baseColorFactor"][:3], dtype=np.float64)
    ext = mat.get("extensions", {}).get("KHR_materials_pbrSpecularGlossiness", {})
    if "diffuseFactor" in ext:
        return np.array(ext["diffuseFactor"][:3], dtype=np.float64)
    return np.array([1.0, 1.0, 1.0])


# --------------------------------------------------------------- correspondence
def match_faces_to_polygons(me, prims, mw, centre, scale, slot_to_gltf):
    """[(polygon, raw_prim, raw_face_local_idx)] in Blender polygon order.

    BUG FIX (caught live, 2026-08-11): the original version assumed Blender's
    polygon order, within one material, exactly replays the raw glTF face
    order (validated on the hammer/saber/robot, all single clean primitives).
    The desk lamp broke that assumption outright: material 0 alone has 3516
    Blender polygons against 2000 raw glTF faces, a real topology difference
    from import (not just a reorder), so no index-based walk can be correct.
    Matches instead by NEAREST 3D POSITION, per material, in the SAME
    normalized frame used everywhere else in this script (raw faces via
    centre/scale, Blender polygons via the axis-swap + /2.0 correction) --
    robust to a different triangle count, not just a different order: a
    Blender polygon with no exact raw counterpart still gets the geometrically
    closest one, which is the right answer for "what should this triangle's
    lit status be" even when it did not exist as a discrete face on the raw
    side."""
    raw_centroids_norm_by_mat = {}
    raw_lookup_by_mat = {}
    for p in prims:
        c = (p["positions"][p["faces"]].mean(axis=1) - centre) * scale
        raw_centroids_norm_by_mat.setdefault(p["material"], []).append(c)
        for i in range(len(p["faces"])):
            raw_lookup_by_mat.setdefault(p["material"], []).append((p, i))

    trees = {}
    for mat, clist in raw_centroids_norm_by_mat.items():
        trees[mat] = cKDTree(np.concatenate(clist, axis=0))

    # BUG FIX (caught live, 2026-08-11), pred_mask_to_asset.py's own
    # documented Trap 5, reproduced here: Blender's material SLOT index is
    # not the glTF material index (the desk lamp's slot 0 is glTF material 7,
    # a full rotation of all 8). Group Blender polygons by the TRANSLATED
    # glTF index, not the raw slot index, or every material gets another
    # material's albedo/texture-presence/UV data.
    uv_layer = me.uv_layers.active.data
    by_mat_polys = {}
    for poly in me.polygons:
        gltf_idx = slot_to_gltf.get(poly.material_index)
        if gltf_idx is None:
            continue
        by_mat_polys.setdefault(gltf_idx, []).append(poly)

    out = []
    for mat, polys in by_mat_polys.items():
        if mat not in trees:
            continue
        cents_blender = []
        for poly in polys:
            vert_idx = [me.loops[li].vertex_index for li in poly.loop_indices]
            wpos = np.array([list(mw @ me.vertices[vi].co) for vi in vert_idx], dtype=np.float64)
            cents_blender.append(wpos.mean(axis=0))
        cents_blender = np.array(cents_blender)
        cents_norm = (cents_blender[:, [0, 2, 1]] * np.array([1.0, 1.0, -1.0])) / 2.0
        _, nn_idx = trees[mat].query(cents_norm, k=1)
        for poly, ni in zip(polys, nn_idx):
            prim, local_idx = raw_lookup_by_mat[mat][ni]
            out.append((poly, prim, local_idx))
    # me.polygons order is not preserved above (grouped by material instead);
    # that is fine, every consumer of `matched` only needs the pairing, not
    # polygon enumeration order.
    return out


def best_perm_diff(raw_uv3, blend_uv3):
    """Min L1 diff over the 6 corner correspondences (3 rotations x 2 windings)
    between two UV triangles, plus which permutation won and whether a V-flip
    (u unchanged, v -> 1-v) fits the winning alignment better than identity."""
    idxs = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2)]
    names = ["id", "rot1", "rot2", "rev", "rev_rot1", "rev_rot2"]
    best = (1e9, None, None)
    for name, idx in zip(names, idxs):
        cand = blend_uv3[list(idx)]
        d = float(np.abs(raw_uv3 - cand).sum())
        if d < best[0]:
            best = (d, name, cand)
    d, name, cand = best
    vflip_cand = cand.copy()
    vflip_cand[:, 1] = 1.0 - vflip_cand[:, 1]
    d_vflip = float(np.abs(raw_uv3 - vflip_cand).sum())
    return d, name, d_vflip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--value", choices=["pred", "gt"], required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tex", type=int, default=1024)
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--flat_thr", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gltf, bins = read_glb(args.glb)
    centre, scale, prims = compute_frame(gltf, bins)
    mat_names = {i: m.get("name", f"material_{i}") for i, m in enumerate(gltf.get("materials", []))}

    z = np.load(args.npz)
    coords_vox = z["coords"].astype(np.float64)
    lit = (z["pred_bc"] > args.thr) if args.value == "pred" else z["gt_e"].astype(bool)
    print(f"LIT_VOXEL_FRAC {lit.mean():.6f} ({int(lit.sum())}/{len(lit)})", flush=True)
    tree_lit = cKDTree(coords_vox[lit]) if lit.sum() else None

    bpyutil.load_blend(bpyutil.preset_glb)
    bpyutil.clear_collection("workbench")
    obj = bpyutil.load_glb(args.glb, import_shading=None)
    me = obj.data
    mw = obj.matrix_world

    # Blender's material SLOT index is not the glTF material index
    # (pred_mask_to_asset.py's own documented Trap 5). Match by NAME, the
    # only thing import preserves, and fail loudly on any slot that cannot
    # be matched rather than silently mis-keying that material's albedo/
    # texture-presence data.
    name_to_gltf = {name: i for i, name in mat_names.items()}
    slot_to_gltf = {}
    for slot_i, slot in enumerate(obj.material_slots):
        gltf_idx = name_to_gltf.get(slot.material.name) if slot.material else None
        assert gltf_idx is not None, (
            f"Blender slot {slot_i} ({slot.material.name if slot.material else None}) "
            f"does not match any glTF material name in {list(mat_names.values())}")
        slot_to_gltf[slot_i] = gltf_idx
    gltf_to_slot = {v: k for k, v in slot_to_gltf.items()}
    print(f"SLOT_MAP slot_to_gltf={slot_to_gltf}", flush=True)

    assert all(len(p.loop_indices) == 3 for p in me.polygons), (
        "non-triangle polygon found; this script assumes a triangulated import")
    uv_layer = me.uv_layers.active.data
    n_poly = len(me.polygons)
    n_raw = sum(len(p["faces"]) for p in prims)
    # Informational only, not an assertion: the desk lamp proved import can
    # change triangle count (3516 Blender polygons vs 2000 raw glTF faces on
    # one material alone), so a mismatch here is no longer evidence of a
    # broken correspondence, match_faces_to_polygons() is position-based now.
    print(f"CORRESPONDENCE_CHECK n_blender_polygons={n_poly} n_raw_gltf_faces={n_raw} "
          f"{'(match)' if n_poly == n_raw else '(differs -- expected on some imports, matching is position-based)'}",
          flush=True)

    matched = match_faces_to_polygons(me, prims, mw, centre, scale, slot_to_gltf)

    # ---------------------------------------------------------- UV-diff report
    per_mat_diff = {}
    for poly, prim, local_idx in matched:
        mat = prim["material"]  # the correct glTF index; poly.material_index is the Blender SLOT, a different number
        f = prim["faces"][local_idx]
        raw_uv3 = prim["uv"][f] if prim["uv"] is not None else None
        loop_idx = list(poly.loop_indices)
        blend_uv3 = np.array([uv_layer[li].uv for li in loop_idx], dtype=np.float64)
        d = per_mat_diff.setdefault(mat, {"n": 0, "n_no_raw_uv": 0, "diffs": [],
                                          "perms": {}, "vflip_better": 0})
        d["n"] += 1
        if raw_uv3 is None:
            d["n_no_raw_uv"] += 1
            continue
        best_d, perm_name, d_vflip = best_perm_diff(raw_uv3, blend_uv3)
        d["diffs"].append(best_d)
        d["perms"][perm_name] = d["perms"].get(perm_name, 0) + 1
        if d_vflip < best_d - 1e-6:
            d["vflip_better"] += 1

    print("=== UV-DIFF REPORT (raw glTF vs Blender-imported, best-permutation L1 per face) ===", flush=True)
    for mat, d in sorted(per_mat_diff.items()):
        diffs = np.array(d["diffs"]) if d["diffs"] else np.zeros(0)
        n_diverging = int((diffs > 0.02).sum())
        print(f"MAT {mat} ({mat_names.get(mat)}): n_faces={d['n']} "
              f"no_raw_uv={d['n_no_raw_uv']} "
              f"mean_diff={diffs.mean() if len(diffs) else float('nan'):.5f} "
              f"max_diff={diffs.max() if len(diffs) else float('nan'):.5f} "
              f"n_diverging(>0.02)={n_diverging}/{len(diffs)} "
              f"perm_hist={d['perms']} "
              f"vflip_fits_better_count={d['vflip_better']}/{len(diffs)}", flush=True)

    # ------------------------------------------------------- rebake by Blender UV
    tex_bufs = {}   # gltf material index -> (pos_buf, valid_buf)
    for poly, prim, local_idx in matched:
        mat = prim["material"]  # gltf index, not poly.material_index (Blender slot)
        if mat not in tex_bufs:
            tex_bufs[mat] = (np.zeros((args.tex, args.tex, 3), dtype=np.float64),
                             np.zeros((args.tex, args.tex), dtype=bool))
        pos_buf, valid_buf = tex_bufs[mat]
        loop_idx = list(poly.loop_indices)
        blend_uv3 = np.array([uv_layer[li].uv for li in loop_idx], dtype=np.float64)
        vert_idx = [me.loops[li].vertex_index for li in loop_idx]
        pos3_blender = np.array([list(mw @ me.vertices[vi].co) for vi in vert_idx], dtype=np.float64)
        # BUG FIX (caught live, 2026-08-11), TWO stacked effects, not one:
        # (1) Blender's glTF importer always converts glTF's Y-up convention
        # to Blender's Z-up on load (blender_xyz = (x,-z,y), 8e-6 mean fit
        # error). (2) bpyutil.load_glb() ALSO normalizes every mesh so its
        # longest bbox extent is exactly 2.0, regardless of the asset's own
        # authored scale (confirmed: for the hammer, whose raw extent is
        # ~367 units, blend_pos / raw_frame_normalized_pos = 0.500020 on
        # every axis, i.e. exactly 2x). compute_frame()'s own `scale` below
        # normalizes to extent ~1.0 (0.99999), not 2.0, so feeding Blender
        # positions through to_voxel(pts, centre, scale) mixed a Blender-
        # normalized frame with a raw-normalized one on top of the axis
        # mismatch: every query collapsed to a tiny wrong corner of voxel
        # space. Invisible on the saber (its raw asset already happens to be
        # authored near unit scale, so Blender's normalization was near
        # identity); total on the hammer and robot, whose raw scale is
        # nowhere near 1. pos_buf keeps Blender's own axis-corrected,
        # 2.0-normalized positions; the later to_voxel() call uses
        # centre=(0,0,0), scale=0.5 to undo exactly that normalization
        # instead of the raw-frame centre/scale, which does not apply here.
        pos3 = pos3_blender[:, [0, 2, 1]] * np.array([1.0, 1.0, -1.0])
        rasterise_into(blend_uv3, np.array([[0, 1, 2]]), pos3, args.tex, pos_buf, valid_buf)

    stats = {"sid": args.sid, "npz": args.npz, "glb": args.glb, "thr": args.thr,
             "tex": args.tex, "tol_voxels": args.tol, "source": args.value,
             "flat_thr": args.flat_thr, "rebake": "blender_imported_uv",
             "pred_voxel_frac": float(lit.mean()), "materials": [], "uniform": {}}

    n_mats = len(gltf.get("materials", []))
    for mat in range(n_mats):
        name = mat_names.get(mat, f"material_{mat}")
        has_tex = bool(gltf["materials"][mat].get("pbrMetallicRoughness", {}).get("baseColorTexture"))
        # Output files/keys must use the BLENDER SLOT index (what load_pred()/
        # rebuild_emission_predicted() key by at render time), not this loop's
        # gltf index -- the whole point of slot_to_gltf/gltf_to_slot.
        slot = gltf_to_slot.get(mat)
        rec = {"material_index": mat, "blender_slot": slot, "material_name": name,
               "has_base_color_texture": has_tex}

        if mat in tex_bufs and has_tex:
            pos_buf, valid_buf = tex_bufs[mat]
            texel_coverage = float(valid_buf.mean())
            rec["texel_coverage"] = texel_coverage
            mask = np.zeros((args.tex, args.tex), dtype=np.float32)
            if valid_buf.any() and tree_lit is not None:
                # centre=(0,0,0), scale=0.5: undoes bpyutil's own
                # extent-to-2.0 normalization, NOT the raw-frame centre/scale
                # (see the note above where pos_buf is filled).
                q = to_voxel(pos_buf[valid_buf], np.zeros(3), 0.5)
                d_lit, _ = tree_lit.query(q, k=1)
                hit = d_lit <= args.tol
                mask[valid_buf] = hit.astype(np.float32)
                rec["lit_texel_frac_of_covered"] = float(hit.mean())
            else:
                rec["lit_texel_frac_of_covered"] = 0.0
            img = np.zeros((args.tex, args.tex, 4), dtype=np.uint8)
            img[..., 0] = img[..., 1] = img[..., 2] = (mask * 255).astype(np.uint8)
            img[..., 3] = 255
            png = os.path.join(args.out_dir, f"{args.sid}__mat{slot}__emis.png")
            Image.fromarray(img).save(png)
            rec["mask_png"] = png
            rec["carrier"] = "texture"
            # team-lead request (2026-08-11): print per-material lit-texel
            # count so a zero transfer is visible in the log, not just in a
            # stats.json field nobody is looking at.
            print(f"MAT {mat} ({name}) carrier=texture texel_coverage={texel_coverage:.4f} "
                  f"n_lit_texels={int(mask.sum())} lit_texel_frac_of_covered={rec['lit_texel_frac_of_covered']:.4f}",
                  flush=True)
        else:
            # FLAT-MATERIAL BRANCH: no baseColorTexture actually usable, so a
            # per-texel mask carries no albedo regardless of rasterization
            # quality. Decide from FACE centroids directly, bypassing the
            # texture atlas.
            faces_this_mat = [(prim, local_idx) for poly, prim, local_idx in matched
                              if prim["material"] == mat]
            if faces_this_mat and tree_lit is not None:
                centroids = np.array([prim["positions"][prim["faces"][li]].mean(axis=0)
                                      for prim, li in faces_this_mat])
                q = to_voxel(centroids, centre, scale)
                d_lit, _ = tree_lit.query(q, k=1)
                lit_frac = float((d_lit <= args.tol).mean())
            else:
                lit_frac = 0.0
            rec["lit_face_frac"] = lit_frac
            rec["carrier"] = "flat" if lit_frac >= args.flat_thr else "none"
            if lit_frac >= args.flat_thr:
                albedo = material_albedo_const(gltf, mat)
                rgb = (albedo * lit_frac).clip(0, 1).tolist()
                stats["uniform"][str(slot)] = rgb  # keyed by Blender slot, not gltf index
                rec["uniform_rgb"] = rgb
            print(f"FLAT_MATERIAL mat={mat} ({name}) lit_face_frac={lit_frac:.4f} "
                  f"threshold={args.flat_thr} -> "
                  f"{'EMIT ' + str(rec.get('uniform_rgb')) if lit_frac >= args.flat_thr else 'off'}",
                  flush=True)
        stats["materials"].append(rec)

    # GUARD (added live, 2026-08-11, after it was caught missing by the
    # gallery's own pixel-stat verification, not by this script): a nonzero
    # source voxel mask that transfers to literally nothing anywhere -- no
    # texel, no flat material -- must not exit clean. Two real gallery cases
    # hit this: the source lit voxels can legitimately sit outside every
    # face's --tol distance (a genuine near-miss, not a bug) or land on
    # faces whose UV footprint never rasterizes a texel centre at this
    # --tex resolution. Either way the caller needs to know before treating
    # the render as a valid empty panel. Not a hard SystemExit (unlike
    # render_uvfree.py's equivalent guard): a rebake runs unattended across
    # a whole gallery batch, so a loud, greppable warning that still lets
    # the other shapes in the same fan-out finish is worth more here than a
    # crash that would have to be re-run shape by shape.
    any_lit = int(lit.sum()) > 0
    any_transferred = any(r.get("carrier") in ("texture", "flat") and
                          (r.get("lit_texel_frac_of_covered", 0) > 0 or "uniform_rgb" in r)
                          for r in stats["materials"])
    if any_lit and not any_transferred:
        print(f"ZERO_TRANSFER_WARNING {args.sid}: source has {int(lit.sum())} lit "
              f"voxels but nothing transferred to any material -- this render "
              f"will be visually black despite a nonzero prediction. Check "
              f"before using this panel.", flush=True)

    # BUG FIX (caught live, 2026-08-11): this used to overwrite stats["materials"]
    # (the real per-material diagnostic recs: texel_coverage, lit_texel_frac_of_
    # covered, carrier) with the bare slot-list load_pred() needs, silently
    # discarding the exact numbers that would have caught the zero-transfer bug
    # from the stats.json alone. The loader's required list now lives under its
    # own key; "materials" keeps the diagnostics. Built straight from Blender's
    # own obj.material_slots (not from gltf_to_slot) so it is correct even if
    # the remapping above has a bug of its own.
    stats["materials_list_for_loader"] = [
        {"slot": i, "material": slot.material.name if slot.material else f"slot_{i}"}
        for i, slot in enumerate(obj.material_slots)]
    stats_out = dict(stats)
    stats_out["materials"] = stats["materials_list_for_loader"]
    stats_out["materials_detail"] = stats["materials"]
    with open(os.path.join(args.out_dir, f"{args.sid}__stats.json"), "w") as f:
        json.dump(stats_out, f, indent=1)
    print(f"REBAKE_DONE {args.sid} n_materials={n_mats}", flush=True)


if __name__ == "__main__":
    main()
