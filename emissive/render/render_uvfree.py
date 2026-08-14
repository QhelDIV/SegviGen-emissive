"""UV-free emission box render: assigns emission per FACE, from a nearest-voxel
lookup against the shape's own predicted (or GT) voxel field, with no texture,
no UV atlas, and no dependence on Blender's import UV-layer handling at all.
By construction this cannot produce chart-boundary artifacts, since there is no
chart: every face gets a value from its own 3D centroid, independent of every
other face's texture-space location.

Frame math (centre/scale/to_voxel) is copied verbatim from pred_mask_to_asset.py
(the same proven-correct convention used throughout this investigation), reading
face centroids directly from the raw glTF geometry -- NOT from Blender's
post-import mesh -- so there is no dependency on whatever Blender's importer
does with UV layers, vertex welding, or per-primitive joins. Face-to-polygon
correspondence with the Blender mesh is order-preserving (glTF primitive order
== Blender polygon order for a single load_glb call) and is validated by the
GT run: a correct correspondence reproduces the known carved-face glow shape,
a wrong one would scramble it into visible noise.

Usage:
  PYTHONPATH=<xgutils>/src <venv>/bin/python render_uvfree.py \
      --glb <path> --npz <pred_voxels npz, coords+pred_bc+gt_e> \
      --value pred|gt --out <path> --sid <sid> \
      --azimuth 38 --elevation 17 --lens 52 --margin 1.06 \
      --res 768 --samples 1024 --strength 4.0 [--white_floor 0.3]
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/project/3dlg-hcvc/omages/xgutils/src")
from xgutils import bpyutil
import bpy

sys.path.insert(0, "/project/3dlg-hcvc/omages/yanxg_scratch/ckpt8_eval/render")
import render_emissive as re_  # noqa: E402  (emission_only_box / place_camera / render / etc.)

sys.path.insert(0, "/3dlg-jupiter-project/lightgen/segvigen_emissive/code")
from pred_mask_to_asset import read_glb, primitives  # noqa: E402

GRID = 512

# BUG, caught live and fixed: an earlier local reimplementation of primitives()
# read raw local-space POSITION accessors with no node-transform applied.
# pred_mask_to_asset.py's real primitives() applies each node's world matrix
# (node_transforms()) before returning positions; skipping that step is
# invisible for a single-node mesh with an identity/near-identity node
# transform (the pumpkin validation happened to pass for exactly this reason)
# but produces grossly wrong world-space positions for any asset whose node
# carries a real transform -- confirmed on the warhammer, whose face-centroid
# bounding box came out with Y and Z swapped and offset against the actual
# voxel grid until this fix. Now imports the real, validated primitives()
# directly instead of a hand-copied reimplementation.


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
    """Flat RGB albedo for a material, read directly from the glTF JSON:
    pbrMetallicRoughness.baseColorFactor, or the spec-gloss extension's
    diffuseFactor, or white if neither is present. Deliberately ignores any
    albedo TEXTURE (this path is UV-free end to end)."""
    mat = gltf["materials"][mat_idx]
    pbr = mat.get("pbrMetallicRoughness", {})
    if "baseColorFactor" in pbr:
        return np.array(pbr["baseColorFactor"][:3], dtype=np.float64)
    ext = mat.get("extensions", {}).get("KHR_materials_pbrSpecularGlossiness", {})
    if "diffuseFactor" in ext:
        return np.array(ext["diffuseFactor"][:3], dtype=np.float64)
    return np.array([1.0, 1.0, 1.0])


def face_colors_for_material(gltf, prims, mat_idx, centre, scale, tree_lit, tol, white_floor):
    """Per-face linear RGB = lit * albedo (+ optional white floor), in the SAME
    order the raw glTF primitives list presents this material's faces.

    Matches pred_mask_to_asset.py's OWN convention exactly (not "nearest voxel
    overall, is it lit" -- a much stricter test that misses faces near but not
    literally touching a lit region): tree_lit is a KDTree over LIT voxels
    ONLY, and a face is lit if the nearest LIT voxel is within --tol voxel
    units, regardless of what the single nearest occupied voxel happens to be.
    Getting this backwards was a real bug caught live: it silently rendered
    the warhammer and lightsaber fully black despite both having nonzero
    predicted voxel fraction, because their few lit voxels never happened to
    be literally the closest occupied voxel to any face centroid."""
    albedo = material_albedo_const(gltf, mat_idx)
    colors = []
    for p in prims:
        if p["material"] != mat_idx:
            continue
        pos, faces = p["positions"], p["faces"]
        centroids = pos[faces].mean(axis=1)
        vox = to_voxel(centroids, centre, scale)
        if tree_lit is None:
            v = np.zeros(len(vox))
        else:
            d_lit, _ = tree_lit.query(vox, k=1)
            v = (d_lit <= tol).astype(np.float64)
        if white_floor:
            v = v * (1 - white_floor) + white_floor * (v > 0.5)
        rgb = v[:, None] * albedo[None, :]
        colors.append(rgb)
    return np.concatenate(colors, axis=0) if colors else np.zeros((0, 3))


def build_uvfree_material(name, strength):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    attr = nt.nodes.new("ShaderNodeVertexColor")
    attr.layer_name = "emis_uvfree"
    nt.links.new(attr.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = strength
    bsdf.inputs["Base Color"].default_value = (0, 0, 0, 1)
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.9
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--value", choices=["pred", "gt"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--azimuth", type=float, default=38.0)
    ap.add_argument("--elevation", type=float, default=17.0)
    ap.add_argument("--lens", type=float, default=52.0)
    ap.add_argument("--margin", type=float, default=1.06)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--samples", type=int, default=1024)
    ap.add_argument("--strength", type=float, default=4.0)
    ap.add_argument("--white_floor", type=float, default=0.0,
                    help="0 = pure mask x albedo (current formulation). >0 blends "
                         "in that fraction of pure white wherever the mask fires, "
                         "a brightness/whiteness-floor VARIANT, not a silent change.")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=2.0,
                    help="max distance, in voxel units, from a face centroid to "
                         "the nearest LIT voxel for that face to count as lit -- "
                         "matches pred_mask_to_asset.py's --tol default exactly")
    args = ap.parse_args()

    gltf, bins = read_glb(args.glb)
    centre, scale, prims = compute_frame(gltf, bins)

    z = np.load(args.npz)
    coords_vox = z["coords"].astype(np.float64)
    if args.value == "pred":
        lit = (z["pred_bc"] > args.thr)
    else:
        lit = z["gt_e"].astype(bool)
    print(f"LIT_VOXEL_FRAC {lit.mean():.6f} ({int(lit.sum())}/{len(lit)})", flush=True)
    tree_lit = cKDTree(coords_vox[lit]) if lit.sum() else None

    n_mats = len(gltf.get("materials", []))
    per_mat_colors = {}
    for mi in range(n_mats):
        c = face_colors_for_material(gltf, prims, mi, centre, scale, tree_lit, args.tol, args.white_floor)
        if len(c):
            per_mat_colors[mi] = c
            print(f"MAT {mi} n_faces={len(c)} n_lit_faces={(c.max(axis=1) > 0).sum()}", flush=True)

    total_lit_faces = sum(int((c.max(axis=1) > 0).sum()) for c in per_mat_colors.values())
    print(f"TRANSFER_CHECK source_voxel_lit={int(lit.sum())} total_lit_faces={total_lit_faces}", flush=True)
    if lit.sum() > 0 and total_lit_faces == 0:
        raise SystemExit(
            f"REFUSING TO RENDER: source voxel mask has {int(lit.sum())} lit "
            f"voxels but zero faces transferred as lit -- a silent zero "
            f"transfer, not a genuinely empty prediction. Check frame/scale "
            f"and --tol before re-running.")

    bpyutil.load_blend(bpyutil.preset_glb)
    bpyutil.clear_collection("workbench")
    obj = bpyutil.load_glb(args.glb, import_shading=None)
    me = obj.data

    n_poly = len(me.polygons)
    n_raw_faces = sum(len(c) for c in per_mat_colors.values())
    print(f"CORRESPONDENCE_CHECK n_blender_polygons={n_poly} n_raw_gltf_faces={n_raw_faces}", flush=True)
    assert n_poly == n_raw_faces, (
        f"face count mismatch: Blender has {n_poly} polygons, raw glTF has "
        f"{n_raw_faces} faces for this object's materials; order-preserving "
        f"per-face assignment is unsafe without matching counts")

    # per-material colors are ordered by (material index, then raw-glTF-primitive
    # order); Blender's polygons are grouped the same way for a single load_glb
    # import (verified by the GT sanity render, not merely assumed)
    ordered = []
    for mi in sorted(per_mat_colors):
        ordered.append(per_mat_colors[mi])
    face_rgb = np.concatenate(ordered, axis=0)

    # Blender's own polygon order may not match glTF's raw material-then-face
    # order 1:1 (materials can interleave across primitives at import); assign
    # by matching each Blender polygon's OWN material_index to the right slice.
    col = me.color_attributes.new(name="emis_uvfree", type="FLOAT_COLOR", domain="CORNER")
    per_mat_cursor = {mi: 0 for mi in per_mat_colors}
    corner_rgba = np.zeros((len(me.loops), 4), dtype=np.float32)
    corner_rgba[:, 3] = 1.0
    for p in me.polygons:
        mi = p.material_index
        if mi not in per_mat_colors:
            continue
        rgb = per_mat_colors[mi][per_mat_cursor[mi]]
        per_mat_cursor[mi] += 1
        for li in p.loop_indices:
            corner_rgba[li, :3] = rgb
    col.data.foreach_set("color", corner_rgba.ravel())

    for slot in obj.material_slots:
        if slot.material is None:
            continue
        mat = build_uvfree_material(f"uvfree_{slot.material.name}", args.strength)
        slot.material = mat

    box = re_.emission_only_box(obj, args.azimuth)
    re_.drop_to_floor(obj)
    re_.place_camera(obj, args.azimuth, args.elevation, args.lens, args.margin)
    re_.assert_emission_is_only_light()
    re_.clear_compositor()
    os.makedirs(args.out, exist_ok=True)
    re_.render(os.path.join(args.out, f"{args.sid}_{args.value}_uvfree.png"),
              (args.res, args.res), args.samples, False, "Filmic", 1.5)
    print(f"UVFREE_DONE {args.sid} {args.value}", flush=True)


if __name__ == "__main__":
    main()
