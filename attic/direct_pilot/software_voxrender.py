"""
Fast software voxel rasterizer (numpy + PIL, no bpy/GPU) for gallery thumbnails.
Renders a voxel set as depth-shaded squares from a fixed 3/4 view: surface grey,
emissive orange. Tight crop, dark backdrop, square output.

~ms per shape even at 300k voxels (painter's order + vectorized square scatter),
so it runs in a SLURM array on solar CPU nodes.
"""
import numpy as np
from PIL import Image

# 3/4 view: camera from front-right-above looking at origin. Voxel frame is z-up.
_CAMDIR = np.array([1.9, -2.3, 1.5], float); _CAMDIR /= np.linalg.norm(_CAMDIR)
_WORLD_UP = np.array([0, 0, 1.0])
_RIGHT = np.cross(_WORLD_UP, _CAMDIR); _RIGHT /= np.linalg.norm(_RIGHT)
_UP = np.cross(_CAMDIR, _RIGHT); _UP /= np.linalg.norm(_UP)
# light from above-front-left for a little modelling on the depth shade
_LIGHT = np.array([-0.4, -0.5, 0.8]); _LIGHT /= np.linalg.norm(_LIGHT)

GREY = np.array([190, 192, 198], float)
ORANGE = np.array([214, 106, 54], float)     # terracotta accent (matches pilot)
BG = np.array([22, 22, 26], float)           # dark card backdrop


def render(coords, mask, out_png, px=300, fill=0.86, seed=0):
    """coords (N,3) int @grid, mask (N,) bool emissive. Writes px-square PNG."""
    if coords.shape[0] == 0:
        Image.fromarray(np.tile(BG.astype(np.uint8), (px, px, 1))).save(out_png); return 0, 0
    # UPRIGHT FIX: Dongchen's emission_voxels_256 frame has the visual up along its
    # Y (2nd) axis, not Z. Rotate +90 deg about X -> (x,y,z)->(x,-z,y), bringing that
    # up onto the renderer's vertical. Proper rotation (not an axis-swap reflection),
    # so shapes aren't mirrored. Determined empirically (standing figures/furniture);
    # note this is NOT the voxrecon som transform's -Z, which left them lying down.
    coords = np.stack([coords[:, 0], -coords[:, 2], coords[:, 1]], axis=1)
    c = coords.astype(np.float64)
    c = c - (c.min(0) + c.max(0)) / 2.0           # center by bbox
    scale = (c.max(0) - c.min(0)).max()
    c /= max(scale, 1e-6)                          # -> ~[-0.5,0.5]

    # project
    sx = c @ _RIGHT; sy = c @ _UP; depth = c @ _CAMDIR
    order = np.argsort(depth)                      # far -> near (near drawn last, wins)
    sx, sy, depth = sx[order], sy[order], depth[order]
    m = mask[order].astype(bool)

    # screen mapping: tight-fit projected bbox into px*fill
    ex = max(sx.max() - sx.min(), sy.max() - sy.min(), 1e-6)
    s = px * fill / ex
    cx = (sx - (sx.min() + sx.max()) / 2) * s + px / 2
    cy = px / 2 - (sy - (sy.min() + sy.max()) / 2) * s     # flip y for image
    pxi = np.clip(cx.astype(np.int64), 0, px - 1)
    pyi = np.clip(cy.astype(np.int64), 0, px - 1)

    # voxel screen pitch -> square size
    span_vox = (c.max(0) - c.min(0)).max() * max(scale, 1e-6)  # in voxel units (~grid extent frac)
    # approximate #voxels across the largest projected axis:
    nvox_across = max((coords.max(0) - coords.min(0)).max(), 1)
    pitch = px * fill / nvox_across
    half = max(int(round(pitch / 2)), 1)

    # depth shade (near brighter) in [0.55,1.05]
    dn = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-6)
    shade = 0.55 + 0.5 * dn
    base = np.where(m[:, None], ORANGE[None], GREY[None]) * shade[:, None]
    base = np.clip(base, 0, 255)

    img = np.tile(BG, (px, px, 1))
    # painter's order already sorted; vectorized square scatter, last write (nearest) wins
    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            xx = np.clip(pxi + dx, 0, px - 1); yy = np.clip(pyi + dy, 0, px - 1)
            img[yy, xx] = base
    Image.fromarray(img.astype(np.uint8)).save(out_png)
    return int(m.sum()), int(coords.shape[0])


if __name__ == "__main__":
    import sys
    out_png, npz = sys.argv[1], sys.argv[2]
    d = np.load(npz)
    ne, nt = render(d["coords"], d["mask"], out_png)
    print(f"SWRENDER {out_png} n_emis={ne} n_tot={nt} frac={ne/max(1,nt):.4f}")
