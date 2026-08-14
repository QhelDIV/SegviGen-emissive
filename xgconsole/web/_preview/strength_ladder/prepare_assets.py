#!/usr/bin/env python3
"""Stage the strength-ladder renders into this page's own img/, and measure
claim 2 (a surface the model did not select stays at 0 across every rung).

Reads from /project/3dlg-hcvc/omages/yanxg_scratch/paper_v3/ladder/out/, the
solar jobs' raw output (see ../../../segvigen_emissive/render/README.md and
the sbatch scripts this page's provenance section quotes). Writes:
  img/<sid>_<band>_s<strength>.png   one PNG per rung. band in:
                                      {box_true, box_mask} (primary, Cornell
                                      box, no key light: the asset's own
                                      emission vs GT mask x albedo) and
                                      {gt, pred} (secondary, key-lit, key=8:
                                      ground truth vs the model's prediction)
  img/<sid>_lit.png                  the neutral-studio reference (band-
                                      independent; emission strength cannot
                                      change a render with no emission)
  img/<sid>_contact.png              a composite contact sheet, gt over pred
  measurements.json                  the claim-2 pixel sample, per shape,
                                      one measurement per band measured

Claim 2 is measured on the box_mask band: no key light at all (box mode
removes every light but the object's own emission), so a genuinely unselected
surface is lit only by bounced light from whatever DOES emit, exactly the
box treatment's own subject. The key-lit "pred" band is kept too (secondary
section), reported honestly with its own confound named.

Run: /cs/3dlg-project/3dlg-hcvc/omages/omages_internal/.venv2/bin/python \
        web/_preview/strength_ladder/prepare_assets.py
"""
import json
import os
import shutil

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
SRC = "/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3/ladder/out"

STRENGTHS = [0, 1, 4, 8, 16]
# band -> (source subdir under SRC/<sid>/, output filename stem under that dir)
BAND_SUBDIR = {"box_true": "box_true", "box_mask": "box_mask",
              "gt": "gt", "pred": "pred", "pred_k0": "pred_k0"}
BAND_FILESTEM = {"box_true": "box_true", "box_mask": "box_mask",
                 "gt": "glow", "pred": "glow", "pred_k0": "glow"}
# claim-2 is measured on these bands
MEASURE_BANDS = ["box_mask", "pred"]

HEADPHONE = {"sid": "8f4c281aef1b4563b6103efbcd77fac1", "tier": "sparse",
            "what": "headphone stand", "author": "Serega_SHTOPOR",
            "license": "CC Attribution",
            "url": "https://sketchfab.com/models/8f4c281aef1b4563b6103efbcd77fac1"}
WEAPON = {"sid": "51a60b164e874bf891597d9c6c1941af", "tier": "mid",
         "what": "sci-fi weapon", "author": "George B",
         "license": "CC Attribution",
         "url": "https://sketchfab.com/models/51a60b164e874bf891597d9c6c1941af"}
CANDLES = {"sid": "658ecf9f837246509b0b1c4aa81e9e5b", "tier": "large",
          "what": "three lit candles", "author": "Rexotec",
          "license": "CC Attribution",
          "url": "https://sketchfab.com/models/658ecf9f837246509b0b1c4aa81e9e5b"}
WALL_LAMP = {"sid": "e5eecab2bc8649548b48b79e705d768e", "tier": "large-qualifying",
            "what": "wall light fixtures", "author": "Archistoric",
            "license": "CC Attribution",
            "url": "https://sketchfab.com/models/e5eecab2bc8649548b48b79e705d768e"}

# primary: box treatment, chosen freely (no prediction constraint)
SHAPES_BOX = [HEADPHONE, WEAPON, CANDLES]
# secondary: key-lit treatment, the earlier prediction-constrained shape set
SHAPES_KEYLIT = [HEADPHONE, WEAPON, WALL_LAMP]
# sid -> which bands that shape actually has renders for
SHAPE_BANDS = {}
for _g in SHAPES_BOX:
    SHAPE_BANDS.setdefault(_g["sid"], set()).update({"box_true", "box_mask"})
for _g in SHAPES_KEYLIT:
    SHAPE_BANDS.setdefault(_g["sid"], set()).update({"gt", "pred"})
ALL_SHAPES = {g["sid"]: g for g in SHAPES_BOX + SHAPES_KEYLIT}.values()


def find_rung_dir(sid, band, st):
    """strength_ladder.py's own rung dirs (gt/pred/pred_k0) label rungs as
    str(float(strength)).replace('.','p'), e.g. strength 0 -> '0.0' -> 's0p0'.
    This page's own box_ladder.sh (box_true/box_mask) uses plain shell
    integers instead ("s%s" % $ST, no decimal point) -> 's0'. Both are
    confirmed against the actual output tree rather than re-derived blind;
    try both rather than assuming one."""
    base = os.path.join(SRC, sid, BAND_SUBDIR[band])
    for candidate in (f"s{float(st)}".replace(".", "p"), f"s{int(st)}"):
        d = os.path.join(base, candidate)
        if os.path.isdir(d):
            return d
    raise RuntimeError(f"no rung dir for {sid}/{band}/s={st}: tried under {base}")


def band_path(sid, band, st):
    return os.path.join(find_rung_dir(sid, band, st),
                        f"{sid}_{BAND_FILESTEM[band]}.png")


def lit_path(sid):
    return os.path.join(find_rung_dir(sid, "gt", 0), f"{sid}_lit.png")


def stage_images():
    n = 0
    for g in ALL_SHAPES:
        sid = g["sid"]
        shutil.copy2(lit_path(sid), os.path.join(IMG, f"{sid}_lit.png"))
        n += 1
        for band in SHAPE_BANDS[sid]:
            for st in STRENGTHS:
                src = band_path(sid, band, st)
                dst = os.path.join(IMG, f"{sid}_{band}_s{st}.png")
                shutil.copy2(src, dst)
                n += 1
    print(f"staged {n} images -> {IMG}")


def contact_sheet(sid, band_a, band_b, suffix):
    """band_a row over band_b row, 5 columns, labelled. Generated from the
    SAME per-panel files the page's CSS grid uses (not a separate render)."""
    W, PAD, LABEL_H = 220, 4, 20
    rows = [band_a, band_b]
    sheet = Image.new("RGB", (len(STRENGTHS) * (W + PAD),
                              LABEL_H + len(rows) * (W + PAD)), (18, 18, 18))
    d = ImageDraw.Draw(sheet)
    for j, st in enumerate(STRENGTHS):
        d.text((j * (W + PAD) + 4, 2), f"strength {st}", fill=(230, 230, 230))
    for i, band in enumerate(rows):
        for j, st in enumerate(STRENGTHS):
            p = os.path.join(IMG, f"{sid}_{band}_s{st}.png")
            im = Image.open(p).convert("RGB").resize((W, W))
            sheet.paste(im, (j * (W + PAD), LABEL_H + i * (W + PAD)))
    out = os.path.join(IMG, f"{sid}_contact_{suffix}.png")
    sheet.save(out)
    return out


def sample_window(arr, cx, cy, half=6):
    y0, y1 = max(0, cy - half), min(arr.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(arr.shape[1], cx + half)
    return arr[y0:y1, x0:x1].reshape(-1, arr.shape[2]).mean(axis=0)


def series(sid, band, cx, cy):
    out = {}
    for st in STRENGTHS:
        arr = np.asarray(Image.open(
            os.path.join(IMG, f"{sid}_{band}_s{st}.png")).convert("RGB"),
            dtype=np.float32)
        out[str(st)] = [round(float(v), 3) for v in
                        sample_window(arr, cx, cy)]
    return out


SELECTED_COLOR = (255, 130, 40)   # orange: the selected-point sample box
UNSELECTED_COLOR = (70, 175, 255)  # blue: the unselected-point sample box


def mark_points(sid, band, st, points, out_suffix):
    """Box one or more sample windows on one rung's panel, so the reader can
    see exactly where a reported number came from, not just trust it.
    points: [(xy, color), ...]; a None xy is skipped."""
    p = os.path.join(IMG, f"{sid}_{band}_s{st}.png")
    im = Image.open(p).convert("RGB")
    d = ImageDraw.Draw(im)
    r = 14
    for xy, color in points:
        if xy is None:
            continue
        x, y = xy
        d.rectangle([x - r, y - r, x + r, y + r], outline=color, width=3)
    out = os.path.join(IMG, f"{sid}_{band}_s{st}_{out_suffix}.png")
    im.save(out)
    return out


def measure_claim2(g, band):
    """Find a selected point and an unselected point in the given band, then
    sample the SAME two 12x12 pixel windows across every rung. Returns
    per-rung RGB means at both, so the reader sees numbers, not a
    description.

    Points are chosen by DELTA from the band's own s=0 (no-emission) panel,
    not by absolute brightness: selected = the on-object pixel whose value
    grows the most from s=0 to s=16; unselected = the on-object pixel whose
    value grows the LEAST. A small box blur picks the location robustly
    (avoids a single noisy pixel); the sample itself is read unblurred.

    On "box_mask" (no key light, GT mask x albedo, the box treatment's own
    claim-2 band) a genuinely unselected surface can still gain a little
    light from BOUNCED illumination off whatever does emit, since box mode's
    room is lit only by the object and multi-bounce GI is real physics, not
    a mask leak; that is reported as what it is, not hidden as "flat".

    On "pred" (key-lit, key=8) an unselected point still reflects the key
    light and can pick up bloom bleed from a bright, large selected region:
    real effects of THAT RENDER, not of the mask.

    Also writes a marked copy of the band's s16 panel with both sample
    windows boxed, so the reported numbers are traceable to actual pixels.
    """
    sid = g["sid"]
    lit = np.asarray(Image.open(os.path.join(IMG, f"{sid}_lit.png"))
                      .convert("RGB"), dtype=np.float32)
    s0 = np.asarray(Image.open(os.path.join(IMG, f"{sid}_{band}_s0.png"))
                     .convert("RGB"), dtype=np.float32).sum(axis=2)
    s16 = np.asarray(Image.open(os.path.join(IMG, f"{sid}_{band}_s16.png"))
                      .convert("RGB"), dtype=np.float32).sum(axis=2)
    on_object = lit.sum(axis=2) > 8  # bg is near-black in the studio-lit pass
    delta = s16 - s0
    delta_smooth = ndimage.uniform_filter(delta, size=13)

    core = ndimage.binary_erosion(on_object, iterations=4)
    core = core if core.any() else on_object

    masked_for_max = np.where(core, delta_smooth, -np.inf)
    sy, sx = np.unravel_index(np.argmax(masked_for_max), delta.shape)
    sel_xy = [int(sx), int(sy)]

    masked_for_min = np.where(core, delta_smooth, np.inf)
    uy, ux = np.unravel_index(np.argmin(masked_for_min), delta.shape)
    unsel_xy = [int(ux), int(uy)]

    result = {
        "band": band,
        "unselected_xy": unsel_xy,
        "unselected_rgb_by_strength": series(sid, band, *unsel_xy),
        "unselected_delta_s0_to_s16": round(float(delta[uy, ux]), 3),
        "selected_xy": sel_xy,
        "selected_rgb_by_strength": series(sid, band, *sel_xy),
        "selected_delta_s0_to_s16": round(float(delta[sy, sx]), 3),
    }

    mark_points(sid, band, 16,
                [(sel_xy, SELECTED_COLOR), (unsel_xy, UNSELECTED_COLOR)],
                "marked")
    return result


def main():
    os.makedirs(IMG, exist_ok=True)
    stage_images()
    measurements = []
    for g in ALL_SHAPES:
        sid = g["sid"]
        bands = SHAPE_BANDS[sid]
        if {"box_true", "box_mask"} <= bands:
            contact_sheet(sid, "box_true", "box_mask", "box")
        if {"gt", "pred"} <= bands:
            contact_sheet(sid, "gt", "pred", "keylit")
        entry = {"sid": sid}
        for band in MEASURE_BANDS:
            if band in bands:
                entry[band] = measure_claim2(g, band)
        measurements.append(entry)
    with open(os.path.join(HERE, "measurements.json"), "w") as f:
        json.dump(measurements, f, indent=2)
    print(f"wrote measurements.json ({len(measurements)} shapes, "
          f"bands {MEASURE_BANDS})")
    for e in measurements:
        for band in MEASURE_BANDS:
            m = e.get(band)
            if m is None:
                continue
            print(e["sid"], band, "selected@", m["selected_xy"],
                  "unselected@", m["unselected_xy"])
            print("  selected  ", m["selected_rgb_by_strength"])
            print("  unselected", m["unselected_rgb_by_strength"])


if __name__ == "__main__":
    main()
