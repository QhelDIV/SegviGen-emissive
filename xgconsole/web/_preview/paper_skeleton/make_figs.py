#!/usr/bin/env python3
"""Prepare the real-image assets for the LightGen paper-skeleton page.

CURRENTLY UNUSED. The page is a claim chain (see build.py) and carries three
figures, all of them computed schematics or a matrix, so it references no
raster images and the generated img/ directory was retired 2026-08-06. This
script is kept because it records where each image came from and regenerates
the whole set in one run if a photographic figure is wanted back.


Every image this script writes is derived from an existing verified asset, no
new rendering and no cluster work:

  src_<sid>.png   TexVerse source thumbnail (square centre crop of the 1920x1080
                  render at /cs/3dlg-falas/datasets/TexVerse/thumbnails/).
  alb_<sid>.png   the shape's 512^3 albedo (base_color) voxel render, taken from
                  web/_preview/data_compare/img/<sid>_geom512.png and cropped to
                  the object's bounding box (design law D9).
  msk_<sid>.png   the matching emissive-mask render (<sid>_emis512.png), same
                  crop, so it stays pixel-registered with the albedo panel.
  prd_<sid>.png   mask x albedo, composed IN IMAGE SPACE from the two renders
                  above: they come from one camera and are pixel-aligned, so
                  taking the albedo pixel wherever the mask render marks the
                  surface emissive, and black elsewhere on the object, is the
                  formula applied to what the reader can already see. This is an
                  illustration of the formula, not a model output.
                  ONLY produced for the shapes in PRODUCT_SIDS. The mask render
                  is diffuse-shaded, so a brightness threshold separates
                  emissive from non-emissive surface reliably only where the
                  foreground histogram is cleanly bimodal. It is, for the
                  lightsaber (0.35): of 1,161 object pixels, 675 fall below
                  brightness 128 and 428 at or above 208, leaving 58 (5%) in the
                  valley between, and the resulting emissive pixel fraction
                  0.371 lands next to the shape's recorded voxel coverage 0.35.
                  It is not for shapes whose two populations overlap, so no
                  product panel is produced for them.
  vae_*.png       three panels copied from the emission-VAE page's own figure
                  set, so this page stays self-contained (SKILL.md rule 11).

Run: .venv2/bin/python web/_preview/paper_skeleton/make_figs.py
  (.venv2 = /cs/3dlg-project/3dlg-hcvc/omages/omages_internal/.venv2/bin/python)
"""
import glob
import json
import os
import shutil

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(os.path.dirname(HERE))
IMG = os.path.join(HERE, "img")
DC = os.path.join(WEB, "_preview", "data_compare", "img")
VAE = os.path.join(WEB, "_preview", "emission_vae", "img")
GALLERY = os.path.join(WEB, "_preview", "experiment_overview", "gallery_sids.json")
THUMBS = "/cs/3dlg-falas/datasets/TexVerse/thumbnails/thumbnails_batch"

OUT_PX = 460          # matches the native size of the data_compare renders
BG = 239              # the uniform background value of those renders
EMIS_LEVEL = 205      # emissive voxels render near-white (254); surfaces are darker

# Shapes whose mask render is cleanly bimodal enough to composite (see module
# docstring); every other shape gets source / albedo / mask panels only.
PRODUCT_SIDS = {"900e70f33acf409799eb19d11f78d60c"}


def thumb_path(sid):
    hits = glob.glob(f"{THUMBS}/batch_*/{sid}.png")
    if not hits:
        raise FileNotFoundError(f"no TexVerse thumbnail for {sid}")
    return hits[0]


def square_centre(im):
    w, h = im.size
    side = min(w, h)
    return im.crop(((w - side) // 2, (h - side) // 2,
                    (w - side) // 2 + side, (h - side) // 2 + side))


def load_pair(sid):
    alb = np.array(Image.open(f"{DC}/{sid}_geom512.png").convert("RGB")).astype(np.int16)
    msk = np.array(Image.open(f"{DC}/{sid}_emis512.png").convert("RGB")).astype(np.int16)
    fg = (np.abs(alb - BG).max(2) > 6) | (np.abs(msk - BG).max(2) > 6)
    emis = fg & (msk.mean(2) > EMIS_LEVEL)
    return alb, msk, fg, emis


def product(alb, fg, emis):
    """mask x albedo: the albedo where the mask is on, zero on the rest of the
    object, background untouched."""
    out = np.full_like(alb, BG)
    out[fg] = 0
    out[emis] = alb[emis]
    return out


def content_box(fg, pad_frac=0.08):
    """Square bounding box of the object, with a small margin (D9)."""
    ys, xs = np.where(fg)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    half = max(y1 - y0, x1 - x0) / 2.0 * (1 + pad_frac) + 4
    h, w = fg.shape
    half = min(half, min(cx, w - cx, cy, h - cy))
    return (int(cx - half), int(cy - half), int(cx + half), int(cy + half))


def save(arr_or_im, name, box=None):
    im = arr_or_im if isinstance(arr_or_im, Image.Image) else Image.fromarray(
        arr_or_im.astype(np.uint8))
    if box is not None:
        im = im.crop(box)
    im.resize((OUT_PX, OUT_PX), Image.LANCZOS).save(os.path.join(IMG, name))


def main():
    os.makedirs(IMG, exist_ok=True)
    gallery = json.load(open(GALLERY))

    for g in gallery:
        sid, ef = g["sid"], g["emissive_frac_512"]
        alb, msk, fg, emis = load_pair(sid)
        box = content_box(fg)
        save(square_centre(Image.open(thumb_path(sid)).convert("RGB")), f"src_{sid}.png")
        save(alb, f"alb_{sid}.png", box)
        save(msk, f"msk_{sid}.png", box)
        if sid in PRODUCT_SIDS:
            save(product(alb, fg, emis), f"prd_{sid}.png", box)
        print(f"{sid}  coverage {ef:.2f}  "
              f"object px {int(fg.sum()):>7}  emissive px {int(emis.sum()):>7}")

    for src, dst in [("ovf_xy_gt_boost.png", "vae_gt.png"),
                     ("ovf_xy_rec_boost.png", "vae_rec.png"),
                     ("ovf_xy_leak.png", "vae_leak.png")]:
        shutil.copyfile(os.path.join(VAE, src), os.path.join(IMG, dst))
        print(f"copied {src} -> {dst}")


if __name__ == "__main__":
    main()
