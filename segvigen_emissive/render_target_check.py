#!/usr/bin/env python3
"""target_check visualization: render the DECODED output_tex_slat.pth colors
(the actual training/eval target) side by side with the raw ALBEDO (input.vxz)
and the raw binary EMISSIVE mask (output.vxz), to show visually whether the
target that's actually on disk is the binary mask (correct) or the shape's own
albedo (the suspected bug).

Input: <name>_decode.npz written by target_check/decode_check.py on solar
  (coords, dec_out_bc, dec_in_bc, raw_coords, raw_out_bc, raw_in_bc)

Run: lightgen_repo venv (bpy pip package, no GPU needed)
  /localhome/xya120/studio/misc/lightgen/lightgen_repo/.venv/bin/python \
      render_target_check.py <npz_path> <out_prefix>
"""
import sys
import numpy as np

sys.path.insert(0, '/localhome/xya120/studio/misc/lightgen/lightgen_repo')
import bpy  # noqa: E402
from xgutils import bpyutil  # noqa: E402
from xgutils.miscutil import preset_glb  # noqa: E402

_CUBE_V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.float32) - 0.5
_CUBE_F = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                    [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]], np.int32)


def coarsen_color(coords, colors, factor=4, max_cells=15000):
    """Downsample to a coarse grid; per cell, mean color + occupancy."""
    c64 = coords.astype(np.int64)
    while True:
        cc = c64 // factor
        uniq, inv = np.unique(cc, axis=0, return_inverse=True)
        if len(uniq) <= max_cells or factor >= 64:
            break
        factor *= 2
    n = len(uniq)
    cnt = np.bincount(inv, minlength=n)
    mean_color = np.zeros((n, 3), np.float64)
    for c in range(3):
        mean_color[:, c] = np.bincount(inv, weights=colors[:, c].astype(np.float64), minlength=n) / cnt
    return uniq, mean_color.astype(np.float32), factor


def voxel_mesh(uniq, factor, colors, pad=0.92):
    uniq = uniq.astype(np.float32)
    centers = (uniq * factor + factor / 2.0) / 256.0 - 1.0
    cell = (factor / 256.0) * pad
    cube = _CUBE_V * cell
    V = (centers[:, None, :] + cube[None, :, :]).reshape(-1, 3)
    F = (_CUBE_F[None] + (np.arange(len(uniq)) * 8)[:, None, None]).reshape(-1, 3)
    C = np.repeat(colors, 8, axis=0)
    return V.astype(np.float32), F.astype(np.int32), C.astype(np.float32)


def render_panel(coords, colors, out_path):
    uniq, mean_color, factor = coarsen_color(coords, colors)
    V, F, C = voxel_mesh(uniq, factor, mean_color)
    bpyutil.load_blend(preset_glb)
    bpyutil.clear_collection('workbench')
    img = bpyutil.render_mesh(
        V, F, vert_color=C, resolution=(512, 512), samples=32,
        shadow_catcher=False, camera_position=(0, -2.6, 1.4), camera_up=(0, 0, 1),
    )
    img = np.asarray(img, np.float32)
    if img.ndim == 3 and img.shape[2] == 4:
        rgb, a = img[..., :3], img[..., 3:4]
        img = rgb * a + 0.93 * (1 - a)
    img = np.clip(img, 0, 1)
    from PIL import Image
    Image.fromarray((img * 255).astype(np.uint8)).save(out_path)
    print(f"[saved] {out_path}")


def main():
    npz_path, out_prefix = sys.argv[1], sys.argv[2]
    d = np.load(npz_path)

    # panel 1: decoded output_tex_slat.pth (the ACTUAL training/eval target)
    render_panel(d["coords"], d["dec_out_bc"].astype(np.float32), f"{out_prefix}_decoded_target.png")
    # panel 2: raw albedo from input.vxz (for comparison -- is the target actually this?)
    render_panel(d["raw_coords"], d["raw_in_bc"].astype(np.float32) / 255.0, f"{out_prefix}_raw_albedo.png")
    # panel 3: raw binary emissive mask from output.vxz (what the target SHOULD look like)
    render_panel(d["raw_coords"], d["raw_out_bc"].astype(np.float32) / 255.0, f"{out_prefix}_raw_emissive_mask.png")


if __name__ == "__main__":
    main()
