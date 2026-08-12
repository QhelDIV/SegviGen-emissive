"""Red-padding probe texture, rasterized against Blender's OWN imported UVs
(the fixed path) instead of raw glTF UVs (the old, broken write side). White
inside the write side's own coverage, red in padding, exactly as before, but
now writer and reader share one UV interpretation by construction: if the
V-flip mechanism (found and confirmed 2026-08-11: Blender's UV V = 1 - raw
glTF V, U unchanged) was the whole story, red should collapse to near zero.

Usage (bpy job, solar only):
  PYTHONPATH=<xgutils>/src <venv>/bin/python bpy_redpad_texture.py \
      --glb <path> --sid <sid> --out_dir <dir> --tex 1024
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/project/3dlg-hcvc/omages/xgutils/src")
from xgutils import bpyutil
import bpy

sys.path.insert(0, "/3dlg-jupiter-project/lightgen/segvigen_emissive/code")
from pred_mask_to_asset import read_glb, primitives, rasterise_into  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tex", type=int, default=1024)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gltf, bins = read_glb(args.glb)
    prims = primitives(gltf, bins)
    mat_names = {i: m.get("name", f"material_{i}") for i, m in enumerate(gltf.get("materials", []))}

    bpyutil.load_blend(bpyutil.preset_glb)
    bpyutil.clear_collection("workbench")
    obj = bpyutil.load_glb(args.glb, import_shading=None)
    me = obj.data
    mw = obj.matrix_world
    assert all(len(p.loop_indices) == 3 for p in me.polygons)
    uv_layer = me.uv_layers.active.data

    n_mats = len(gltf.get("materials", []))
    names_in_slot_order = []
    for mat in range(n_mats):
        name = mat_names.get(mat, f"material_{mat}")
        names_in_slot_order.append(name)
        pos_buf = np.zeros((args.tex, args.tex, 3), dtype=np.float64)
        valid = np.zeros((args.tex, args.tex), dtype=bool)
        for poly in me.polygons:
            if poly.material_index != mat:
                continue
            loop_idx = list(poly.loop_indices)
            blend_uv3 = np.array([uv_layer[li].uv for li in loop_idx], dtype=np.float64)
            vert_idx = [me.loops[li].vertex_index for li in loop_idx]
            pos3 = np.array([list(mw @ me.vertices[vi].co) for vi in vert_idx], dtype=np.float64)
            rasterise_into(blend_uv3, np.array([[0, 1, 2]]), pos3, args.tex, pos_buf, valid)
        img = np.zeros((args.tex, args.tex, 3), dtype=np.uint8)
        img[valid] = [250, 250, 250]
        img[~valid] = [255, 0, 0]
        out_png = os.path.join(args.out_dir, f"{args.sid}__mat{mat}__emis.png")
        Image.fromarray(img, mode="RGB").save(out_png)
        print(f"MAT {mat} ({name}): blender-uv coverage={valid.mean():.4f} -> {out_png}", flush=True)

    stats = {"materials": names_in_slot_order, "uniform": {}}
    with open(os.path.join(args.out_dir, f"{args.sid}__stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(f"BPY_REDPAD_TEXTURE_DONE {args.sid} n_materials={n_mats}", flush=True)


if __name__ == "__main__":
    main()
