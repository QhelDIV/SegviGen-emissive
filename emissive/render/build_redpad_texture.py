"""Red-padding probe texture: white inside the write side's own rasterized UV
coverage, saturated red in what it considers padding. Feeds --pred_emission
so the render shows exactly what the write side wrote, with no mask x albedo
multiply and no Blender UV re-read involved -- the same construction already
used and decided on for the pumpkin/lantern probes earlier in this
investigation, now generalized to any single-material shape.

Write side ONLY: reuses primitives()/rasterise_into() from
pred_mask_to_asset.py verbatim, no reimplementation.

Usage (no bpy needed, pure Python + PIL):
  <venv>/bin/python build_redpad_texture.py --glb <path> --sid <sid> \
      --out_dir <dir> --tex 1024
Writes <out_dir>/<sid>__mat<N>__emis.png (one per material) and
<out_dir>/<sid>__stats.json in the --pred_emission file convention.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

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
    by_mat = {}
    for p in prims:
        by_mat.setdefault(p["material"], []).append(p)

    names_in_slot_order = []
    for mat, plist in sorted(by_mat.items(), key=lambda kv: (kv[0] is None, kv[0])):
        pos_buf = np.zeros((args.tex, args.tex, 3), dtype=np.float64)
        valid = np.zeros((args.tex, args.tex), dtype=bool)
        for p in plist:
            if p["uv"] is None or len(p["uv"]) != len(p["positions"]):
                continue
            rasterise_into(p["uv"], p["faces"], p["positions"], args.tex, pos_buf, valid)
        img = np.zeros((args.tex, args.tex, 3), dtype=np.uint8)
        # 250, not 255: deliberately off pure white so this texel can never be
        # mistaken for a coincidental pure-white value in a real asset texture
        # (the exact confound that made the earlier, broken box-mode probe look
        # like a real result). Red at pure [255,0,0] is already unmistakable.
        img[valid] = [250, 250, 250]
        img[~valid] = [255, 0, 0]
        name = mat_names.get(mat, f"material_{mat}")
        names_in_slot_order.append(name)
        out_png = os.path.join(args.out_dir, f"{args.sid}__mat{mat}__emis.png")
        Image.fromarray(img, mode="RGB").save(out_png)
        print(f"MAT {mat} ({name}): coverage={valid.mean():.4f} -> {out_png}", flush=True)

    stats = {"materials": names_in_slot_order, "uniform": {}}
    with open(os.path.join(args.out_dir, f"{args.sid}__stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(f"REDPAD_TEXTURE_DONE {args.sid} n_materials={len(names_in_slot_order)}", flush=True)


if __name__ == "__main__":
    main()
