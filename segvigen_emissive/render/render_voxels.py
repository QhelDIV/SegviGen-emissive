#!/usr/bin/env python3
"""Render the staged voxel grids as cubes, on the same camera as the mesh panels.

Two panels per shape:
  <sid>_vox_pbr.png    every occupied cell, coloured by its base colour
  <sid>_vox_mask.png   the same occupancy in neutral grey, with the ground-truth
                       emissive cells marked in the page accent

The mask panel deliberately shows the FULL occupancy, not the emissive subset
alone: the emissive set on its own is a handful of disconnected cells floating in
space with no object around them, and a reader cannot tell what part of what
shape they belong to.

Camera: the voxel object is normalised exactly the way bpyutil.normalize_mesh
normalises a loaded GLB (bounding box centred, longest side scaled to 1.9999),
then dropped to the floor, then framed by render_emissive.place_camera with the
same arguments as the mesh renders. Both objects therefore present the same
bounding box to the same solver, so the viewpoint matches panel for panel.

Run on a CPU node with the shared venv plus PYTHONPATH=<xgutils>/src.
"""
import argparse
import json
import os
import traceback

import numpy as np

import bpy  # noqa: E402
from xgutils import bpyutil  # noqa: E402

import render_emissive as re_  # noqa: E402  (place_camera / render / drop_to_floor)

# The page's accent, in linear RGB: the design system uses it to mark data and
# nothing else, and "these are the voxels the model must select" is data.
ACCENT = (0.663, 0.140, 0.043)
BODY = (0.045, 0.045, 0.050)


def grid_to_blender(cells, res):
    """Cell indices to Blender coordinates inside the [-1, 1] cube.

    The bake is in the GLB's own frame, which is glTF's Y-up; Blender's importer
    maps (x, y, z) -> (x, -z, y), and the mesh panels are rendered from that
    import, so the same permutation is applied here or the voxels would face a
    different way from the mesh.
    """
    p = (cells.astype(np.float64) + 0.5) / res * 2.0 - 1.0
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)


def cube_mesh(centres, size, colors):
    """One mesh of axis-aligned cubes, with a per-corner colour attribute."""
    unit = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
                    dtype=np.float64) * (size / 2.0)
    faces_unit = np.array([[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
                           [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]])
    n = centres.shape[0]
    verts = (centres[:, None, :] + unit[None, :, :]).reshape(-1, 3)
    offs = (np.arange(n) * 8)[:, None, None]
    faces = (faces_unit[None, :, :] + offs).reshape(-1, 4)
    vcols = np.repeat(colors, 8, axis=0)
    return verts, faces, vcols


def build_object(name, centres, size, colors):
    me = bpy.data.meshes.new(name)
    verts, faces, vcols = cube_mesh(centres, size, colors)
    me.from_pydata(verts.tolist(), [], faces.tolist())
    me.update()
    col = me.color_attributes.new(name="vcol", type="FLOAT_COLOR", domain="POINT")
    rgba = np.concatenate([vcols, np.ones((vcols.shape[0], 1))], axis=1)
    col.data.foreach_set("color", rgba.astype(np.float32).ravel())

    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def vertex_color_material(name, emission_strength=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    attr = nt.nodes.new("ShaderNodeVertexColor")
    attr.layer_name = "vcol"
    nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.62
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.0
    if emission_strength > 0:
        nt.links.new(attr.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    return mat


def normalize_like_mesh(obj):
    """bpyutil.normalize_mesh's transform, applied to an object we built."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN", center="BOUNDS")
    extent = max(obj.dimensions)
    obj.scale = (1.9999 / extent,) * 3 if extent else (1.0,) * 3
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def one(sid, npz_path, out, args):
    data = np.load(npz_path)
    meta = json.loads(str(data["meta"]))
    res = meta["res_display"]
    cells, color, lit = data["cells"], data["color"], data["lit"].astype(bool)
    centres = grid_to_blender(cells, res)
    size = (2.0 / res) * args.fill

    # sRGB texture bytes to the linear values Blender's colour attribute wants
    srgb = color.astype(np.float64) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92,
                      ((srgb + 0.055) / 1.055) ** 2.4)

    for kind in ("pbr", "mask"):
        bpyutil.load_blend(bpyutil.preset_glb)
        bpyutil.clear_collection("workbench")
        if kind == "pbr":
            cols = linear
            mat = vertex_color_material("vox_pbr")
        else:
            cols = np.tile(np.array(BODY), (cells.shape[0], 1))
            cols[lit] = ACCENT
            mat = vertex_color_material("vox_mask", emission_strength=1.4)
        obj = build_object(f"vox_{kind}", centres, size, cols)
        obj.data.materials.append(mat)
        normalize_like_mesh(obj)
        re_.drop_to_floor(obj)
        re_.place_camera(obj, args.azimuth, args.elevation, args.lens, args.margin)
        floor = bpy.data.objects.get("Floor")
        if floor is not None:
            floor.location.z = -0.004
            floor.cycles.is_shadow_catcher = True
        re_.clear_compositor()
        re_.render(os.path.join(out, f"{sid}_vox_{kind}.png"),
                   (args.res, args.res), args.samples, True,
                   "Khronos PBR Neutral")
        bpyutil.purge_obj(obj)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxel_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sids", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--fill", type=float, default=0.92,
                    help="cube side as a fraction of the cell, so cubes read as cubes")
    # identical to the mesh renders, so the viewpoint is the same
    ap.add_argument("--azimuth", type=float, default=38.0)
    ap.add_argument("--elevation", type=float, default=17.0)
    ap.add_argument("--lens", type=float, default=52.0)
    ap.add_argument("--margin", type=float, default=1.06)
    ap.add_argument("--overwrite", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sids = args.sids.split(",")[args.shard::args.nshards]
    for sid in sids:
        done = os.path.join(args.out, f"{sid}_vox_mask.png")
        if os.path.exists(done) and not args.overwrite:
            print(f"SKIP {sid}", flush=True)
            continue
        print(f"=== {sid}", flush=True)
        try:
            m = one(sid, os.path.join(args.voxel_dir, f"{sid}.npz"), args.out, args)
            print(f"OK {sid} cells={m['n_cells_display']} "
                  f"emissive={m['emissive_frac_display']:.4f}", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"FAIL {sid}", flush=True)
    print("ALL_DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
