#!/usr/bin/env python3
"""Stage the bloom-sweep assets into the page's own img/ directory.

Read-only against the sweep on scratch. Three jobs:
  1. copy every bloom cell, re-encoded RGB + optimize (the sources are RGBA with
     alpha 255 everywhere, so dropping alpha is lossless and saves ~35 percent);
  2. cut the carved-eye crop pair that carries the section;
  3. render the bloom-extent maps (accent over a dimmed grey base), the same
     device the clipped-pixel maps use in the exposure half of the page.

The shared cells (none, t1_m-0.15_s7, t1_m-0.15_s5) appear in BOTH sweep runs and
are pixel-identical (verified); bloom_refine is the single source for those.

Run once: .venv2/bin/python web/_preview/render_sweep/prep_bloom.py
"""
import os
import shutil
import numpy as np
from PIL import Image

SRC = "/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "img")
ACC = np.array([201, 100, 66], dtype=np.float64)   # theme2 --accent #C96442

PUMPKIN = "48af42db48c44cd9bfab32bbb057a39c"
CANDLES = "658ecf9f837246509b0b1c4aa81e9e5b"
VENDING = "9418a924a50d44c186dd499006b62424"
SIDS = [PUMPKIN, CANDLES, VENDING]

SWEEP_ONLY = ["t1_m-0.15_s9", "t1.5_m-0.15_s9", "t2.5_m-0.15_s9",
              "t1_m-0.45_s9", "t1_m-0.7_s9"]
REFINE = ["none", "t1_m-0.15_s7", "t1_m-0.45_s7", "t1_m-0.15_s6",
          "t1_m-0.45_s6", "t1_m-0.15_s5", "t1_m-0.45_s5"]

# The chosen setting applied to all ELEVEN shapes (box3_raw), not just the three
# that were swept. The three swept shapes' cells there are pixel-identical to
# the refinement run's (verified), so only the other eight are staged and the
# page reads the same bytes for a shape wherever it appears.
SHIPPED_ONLY = ["1e9c6545b4da42e0ba4e5dbcd2e0e8ff", "4e105e043a6447439e98e9831aed122e",
                "51a60b164e874bf891597d9c6c1941af", "8f4c281aef1b4563b6103efbcd77fac1",
                "b74fc2533d5345629f2c3ce2c8ab340a", "b7709a651d144134a5babce33223380a",
                "c1e3035d1ccb49df9c09aa86681faf30", "e5eecab2bc8649548b48b79e705d768e"]

OLD, NEW = "t1_m-0.15_s9", "t1_m-0.45_s7"
# The carved eye plus the top of the mouth: the region where the old setting's
# halo swallows the crescent. Box read off the 768px render, not guessed.
CROP = (200, 285, 560, 645)

# The refinement grid's cells are shown as CROPS, not full frames. A halo two
# glare-size steps wider is plainly visible in a 300 pixel window and invisible
# in a 768 pixel frame shown at 266: the first cut of that matrix used full
# frames and six cells of it looked identical (checked on the rendered page, at
# reading size, not in the source files). One window per shape, the same window
# in every cell of that shape's matrix, so only the setting varies.
#   pumpkin: the carved eye, where the halo swallows the crescent
#   vending: the lit product column against the dark cabinet, where the halo's
#            reach onto unlit surface is legible
GRID_CROPS = {
    "48af42db48c44cd9bfab32bbb057a39c": (230, 300, 530, 600),
    "9418a924a50d44c186dd499006b62424": (355, 305, 655, 605),
}
GRID_TAGS = [f"t1_{m}_{s}" for m in ("m-0.15", "m-0.45") for s in ("s5", "s6", "s7")]


def copy_cell(src_dir, sid, tag):
    dst = os.path.join(OUT, f"bloom_{sid}_{tag}.png")
    im = Image.open(os.path.join(SRC, src_dir, f"{sid}_{tag}.png"))
    assert im.size == (768, 768), f"{sid}_{tag}: {im.size}"
    im.convert("RGB").save(dst, optimize=True)
    return dst


def extent_map(sid, tag):
    """Pixels the glare lifts by 3 or more 8-bit steps, in the accent colour
    over a dimmed grey base. Same construction as the clipped-pixel maps.

    The base is the GLARE-FREE render, identical in every panel, so two panels
    of this figure differ only in the accent region. Basing it on each panel's
    own graded image instead would have made the halo brighten its own backdrop,
    so the panel with more bloom would read lighter overall and the comparison
    would carry two changes at once."""
    a = np.asarray(Image.open(os.path.join(OUT, f"bloom_{sid}_{tag}.png"))).astype(np.int16)
    b = np.asarray(Image.open(os.path.join(OUT, f"bloom_{sid}_none.png"))).astype(np.int16)
    lifted = (a.max(-1) - b.max(-1)) >= 3
    base = np.repeat(b.mean(-1, keepdims=True).astype(np.float64), 3, axis=2) * 0.55 + 36.0
    out = np.where(lifted[..., None], ACC[None, None, :], base)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(
        os.path.join(OUT, f"bloomext_{sid}_{tag}.png"), optimize=True)
    return float(lifted.mean())


def main():
    n = 0
    for sid in SIDS:
        for tag in SWEEP_ONLY:
            copy_cell("bloom_sweep", sid, tag); n += 1
        for tag in REFINE:
            copy_cell("bloom_refine", sid, tag); n += 1
    for sid in SHIPPED_ONLY:
        for tag in ("none", NEW):
            copy_cell("box3_raw", sid, tag); n += 1
        shutil.copy2(os.path.join(SRC, "box_exr", f"{sid}_box.json"),
                     os.path.join(OUT, f"bloomshape_{sid}.json"))
    print(f"copied {n} bloom cells -> {OUT}")

    for tag, name in [(OLD, "old"), (NEW, "new")]:
        im = Image.open(os.path.join(OUT, f"bloom_{PUMPKIN}_{tag}.png")).crop(CROP)
        im.save(os.path.join(OUT, f"eyecrop_{name}.png"), optimize=True)
        print(f"crop {name}: {im.size} from {CROP}")

    for sid in (PUMPKIN, VENDING):
        for tag in (OLD, NEW):
            print(f"extent {sid[:8]} {tag}: {100 * extent_map(sid, tag):.2f}%")

    for sid, box in GRID_CROPS.items():
        for tag in GRID_TAGS:
            im = Image.open(os.path.join(OUT, f"bloom_{sid}_{tag}.png")).crop(box)
            im.save(os.path.join(OUT, f"gridcrop_{sid}_{tag}.png"), optimize=True)
        print(f"grid crops {sid[:8]}: {len(GRID_TAGS)} at {im.size} from {box}")


if __name__ == "__main__":
    main()
