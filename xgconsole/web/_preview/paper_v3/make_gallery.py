#!/usr/bin/env python3
"""Copy the chosen renders and preview GLBs into the page, and write gallery.json.

Source: the solar render job's output directory (read-only here). The gallery is
ordered by measured emissive coverage, smallest first, so the figure itself shows
the range rather than asserting it.

ATTRIBUTION IS PART OF THE BUILD, not a caption someone remembers to write. Each
shape is a Sketchfab model by a named author under a Creative Commons licence
that obliges us to credit them wherever we show the work, and this page is
public. The author, the licence string and the model's own URL are read here
from TexVerse's metadata, per shape, and the build REFUSES any shape whose
licence is not on ALLOWED_LICENSES rather than guessing what the obligation is.
An excluded shape is printed, not silently dropped.

Run: /cs/3dlg-project/3dlg-hcvc/omages/omages_internal/.venv2/bin/python \
        web/_preview/paper_v3/make_gallery.py
"""
import json
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3/final3"
# The viewer GLBs are exported separately, with the same emission rewrite but
# a texture cap and a face budget, so a 46 MB source asset does not become a
# 46 MB download when a tile is clicked.
GLB_SRC = "/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3/preview_glb2"
VOX_SRC = "/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3/voxpanels_final"
BASE_SRC = "/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3/baselines_k8"
VOXELS = "/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3/voxels"

# Every panel the page shows, as (published name, source dir, source name).
# The gallery needs the first four for every shape; the comparison figure needs
# the last three as well, for its own subset.
PANELS = [
    ("{sid}_input.png", SRC, "{sid}_lit.png"),
    ("{sid}_vox_pbr.png", VOX_SRC, "{sid}_vox_pbr.png"),
    ("{sid}_vox_mask.png", VOX_SRC, "{sid}_vox_mask.png"),
    ("{sid}_glow.png", SRC, "{sid}_glow.png"),
]
HEUR_SRC = "/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3/pred/albedo_matched_k8"
# The predicted column is produced by another workstream. While PRED_PENDING is
# set the figure carries a labelled placeholder instead; clearing it makes every
# predicted panel required, so a missing one fails the build rather than being
# silently replaced. Defined HERE and imported by build.py, so the copier and the
# page cannot disagree about whether the column has landed.
PRED_PENDING = False
PRED_SRC = ("/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3/pred/"
            "emis_72k_unfilt_method_k8")
COMPARE_PANELS = [
    ("{sid}_random.png", BASE_SRC, "{sid}_random.png"),
    ("{sid}_albedo.png", HEUR_SRC, "{sid}_glow.png"),
    ("{sid}_pred_72k.png", PRED_SRC, "{sid}_glow.png"),
    ("{sid}_true.png", SRC, "{sid}_true.png"),
]

# THE TREATMENT GUARD.
#
# Every dark panel in the comparison figure must have been rendered with the
# same lighting and tone; a column that differs in anything but its mask makes
# any visual difference unattributable, which is the one thing that figure
# cannot afford. This is checked, not documented: a sidecar that disagrees fails
# the build.
#
# The trap this exists for is real and already happened once. After the bloom
# sweep the renderer's DEFAULTS moved from size 9 / mix -0.15 to 7 / -0.45, so a
# column rendered afterwards without explicit flags would have differed silently
# from the four rendered before it.
#
# THE KEY IS 8, AND IT USED TO BE 20. That change is the reason this figure can
# be believed. At key 20 the Glare node fired on 108,865 pixels of a panel whose
# prediction was EMPTY, against 123,948 on the ground truth, so the glow was
# reporting the key light rather than the object, in a figure whose whole subject
# is where light comes from. At key 8 nothing non-emissive reaches the node's
# linear threshold of 1.0: it fires on 0 pixels of the empty panel and 50,458 of
# the ground truth. Bloom became a property of emission instead of a property of
# brightness. Raising the key back would quietly undo that.
REFERENCE_TREATMENT = {
    "view_transform": "AgX", "exposure": 0.0, "key": 8.0, "bg": 0.012,
    "samples": 256, "bloom_size": 9, "bloom_threshold": 1.0, "bloom_mix": -0.15,
}

# NO EXEMPTIONS. There used to be four, for panels rendered before the sidecar
# carried a `treatment` block, whose settings had to be recovered from their job
# scripts. The key-8 re-render replaced every one of those panels with a run that
# wrote its own treatment out, so the recovery is no longer load-bearing and the
# guard now applies to every comparison panel. Keep this empty: an exemption is a
# panel nobody is checking.
#
# The `input` column is the one panel outside the guard, and deliberately: it is
# the studio preset under Khronos PBR Neutral with no bloom, because what it has
# to show is the object. That exception is stated in the caption rather than
# hidden, and it is a measurement rather than a belief: its background sits at
# 0.5/255 on every shape against 22.0/255 for the five emission columns. It is
# not in COMPARE_PANELS, so check_treatment never sees it.
TREATMENT_LEGACY = set()


def check_treatment(dest_tpl, src_dir, sid, src_name):
    """Fail the build if a comparison panel was rendered on other settings.

    The sidecar's name follows the render, not a single convention: the method
    path writes `<sid>_stats.json` for all three of its panels, while a baseline
    mode writes `<sid>_<mode>.json`. Derive it from the source file and fall
    back, rather than assuming, since assuming reported a missing sidecar for a
    panel whose sidecar was sitting beside it under another name.
    """
    if dest_tpl in TREATMENT_LEGACY:
        return None
    side = os.path.join(src_dir, src_name.format(sid=sid).replace(".png", ".json"))
    if not os.path.exists(side):
        side = os.path.join(src_dir, f"{sid}_stats.json")
    if not os.path.exists(side):
        raise SystemExit(
            f"{dest_tpl.format(sid=sid)}: no sidecar at {side}; a comparison "
            f"panel must record the treatment it was rendered with")
    t = (json.load(open(side)) or {}).get("treatment")
    if not t:
        raise SystemExit(
            f"{dest_tpl.format(sid=sid)}: sidecar has no 'treatment' block; "
            f"re-render with the current renderer so the settings are recorded")
    bad = {k: (t.get(k), v) for k, v in REFERENCE_TREATMENT.items()
           if t.get(k) != v}
    if bad:
        raise SystemExit(
            f"{dest_tpl.format(sid=sid)}: treatment differs from the other "
            f"comparison columns (got, expected): {bad}")
    return t
# box3: the same linear renders as box2, re-graded through the Glare node at
# the swept bloom (radius 7, mix -0.45). Bloom is post-process, so the change
# needed a compositor pass over the saved EXRs, not a re-render.
BOX_SRC = "/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3/box3"
BOX_PANEL = ("{sid}_box.png", BOX_SRC, "{sid}_box.png")

# Shapes whose emission-only box render is too dark to be worth a tile. Judged
# by looking at every render, not by a threshold on coverage: what matters is
# whether the object visibly lights anything, and that depends on where the
# emitter sits as much as on how much of it there is.
BOX_SKIP = set()
# Empty again. The wall light fixtures were dropped when the box renders capped
# diffuse light at Cycles' default 4 bounces: their cups face upward on thin
# arms, so the object is lit almost entirely by light that has already bounced
# off the walls, which is exactly the light that truncation was discarding. With
# 16 diffuse bounces the arms and the fixture body read normally. The lesson is
# that the shape which looked like a bad subject was diagnosing a bug.

# The comparison figure's rows. Six of the eleven, deliberately including the
# three where mask x albedo departs most from the authored emission (vending
# machine, headphone stand, candles): a comparison figure that only showed the
# agreeing cases would be arguing the opposite of what it measures.
COMPARE = [
    "48af42db48c44cd9bfab32bbb057a39c",
    "1e9c6545b4da42e0ba4e5dbcd2e0e8ff",
    "9418a924a50d44c186dd499006b62424",
    "8f4c281aef1b4563b6103efbcd77fac1",
    "b7709a651d144134a5babce33223380a",
    "658ecf9f837246509b0b1c4aa81e9e5b",
    # The seventh row, added last and the most informative of the set. On every
    # other shape the model either predicts nothing or floods the object at 16x
    # to 144x the true area, which reads as a model that simply does not work.
    # Here it predicts 0.103 of the area against a true 0.069, a plausible
    # amount of light, and still scores 0.0002: right quantity, wrong place.
    # That is the failure an aggregate hides and the one a reader would
    # otherwise assume we had left out.
    "51a60b164e874bf891597d9c6c1941af",
]

# Keyed by our shape id, which IS the Sketchfab model uid: every record's
# thumbnail_url is https://media.sketchfab.com/models/<sid>/thumbnails/...
METADATA = "/cs/3dlg-falas/datasets/TexVerse/metadata.json"
MODEL_URL = "https://sketchfab.com/models/{sid}"

# Plain attribution, nothing else. ShareAlike is deliberately NOT here: it puts
# conditions on derivative works that are a decision for the owner, not for this
# build, and one of the twelve candidates carries it.
ALLOWED_LICENSES = {"CC Attribution"}

# Plain names, read off each shape's TexVerse caption and confirmed against its
# own render. No adjectives the picture does not support.
WHAT = {
    "8f4c281aef1b4563b6103efbcd77fac1": "headphone stand",
    "b74fc2533d5345629f2c3ce2c8ab340a": "ghost character",
    "c1e3035d1ccb49df9c09aa86681faf30": "humanoid robot",
    "1e9c6545b4da42e0ba4e5dbcd2e0e8ff": "street lamp",
    "48af42db48c44cd9bfab32bbb057a39c": "jack-o'-lantern",
    "51a60b164e874bf891597d9c6c1941af": "sci-fi weapon",
    "e5eecab2bc8649548b48b79e705d768e": "wall light fixtures",
    "9418a924a50d44c186dd499006b62424": "vending machine",
    "619f0732286f4a4683412d7f1cae983b": "handheld game console",
    "4e105e043a6447439e98e9831aed122e": "war hammer",
    "b7709a651d144134a5babce33223380a": "animatronic character",
    "658ecf9f837246509b0b1c4aa81e9e5b": "three lit candles",
}


def main():
    img_dir = os.path.join(HERE, "img")
    glb_dir = os.path.join(HERE, "glb")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(glb_dir, exist_ok=True)
    meta = json.load(open(METADATA))

    rows, excluded = [], []
    for sid, what in WHAT.items():
        record = meta.get(sid)
        if record is None:
            excluded.append((sid, what, "no metadata record"))
            continue
        license_ = record.get("license")
        author = record.get("user_display") or record.get("user")
        if license_ not in ALLOWED_LICENSES or not author:
            excluded.append((sid, what, f"licence {license_!r}, author {author!r}"))
            continue

        stats = json.load(open(os.path.join(SRC, f"{sid}_stats.json")))
        with np.load(os.path.join(VOXELS, f"{sid}.npz")) as z:
            vox = json.loads(str(z["meta"]))
        wanted = list(PANELS)
        if sid in COMPARE:
            wanted += COMPARE_PANELS
        has_box = sid not in BOX_SKIP and os.path.exists(
            os.path.join(BOX_SRC, f"{sid}_box.png"))
        if has_box:
            wanted.append(BOX_PANEL)
        glb = os.path.join(GLB_SRC, f"{sid}_mod.glb")
        if not os.path.exists(glb):
            raise SystemExit(f"missing {glb}")
        for dest, src_dir, src_name in wanted:
            src = os.path.join(src_dir, src_name.format(sid=sid))
            if not os.path.exists(src):
                # the predicted column may be absent, and ONLY while
                # PRED_PENDING is set; build.py draws a labelled placeholder in
                # its place. Every other missing panel fails the build, because
                # a figure with a silent gap is worse than no figure.
                if src_dir == PRED_SRC and PRED_PENDING:
                    continue
                raise SystemExit(f"missing {src}")
            if sid in COMPARE and (dest, src_dir, src_name) in COMPARE_PANELS:
                check_treatment(dest, src_dir, sid, src_name)
            shutil.copyfile(src, os.path.join(img_dir, dest.format(sid=sid)))
        shutil.copyfile(glb, os.path.join(glb_dir, f"{sid}.glb"))
        rows.append({
            "sid": sid,
            "what": what,
            "author": author,
            "license": license_,
            "url": MODEL_URL.format(sid=sid),
            "area_lit_frac": stats["area_lit_frac"],
            "n_emissive_materials": stats["n_emissive_materials"],
            "in_compare": sid in COMPARE,
            "has_box": has_box,
            "voxels": vox,
            "glb_bytes": os.path.getsize(glb),
        })

    # An excluded shape leaves its copied render behind on a rebuild; clear the
    # page's own directories of anything the manifest no longer lists, so a
    # withdrawn model cannot stay published by accident.
    # Prune against the EXACT set of files this run wrote, not against a sid
    # prefix: a prefix test keeps every file that merely starts with a live
    # shape id, which quietly left eleven renders from an earlier naming scheme
    # on the page and in the published directory.
    expected = {os.path.join(glb_dir, f"{r['sid']}.glb") for r in rows}
    for r in rows:
        wanted = list(PANELS) + (COMPARE_PANELS if r["sid"] in COMPARE else [])
        if r["has_box"]:
            wanted.append(BOX_PANEL)
        expected |= {os.path.join(img_dir, dest.format(sid=r["sid"]))
                     for dest, _, _ in wanted}
    for d in (img_dir, glb_dir):
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if p not in expected:
                os.remove(p)
                print(f"removed stale {p}")

    rows.sort(key=lambda r: r["area_lit_frac"])
    with open(os.path.join(HERE, "gallery.json"), "w") as f:
        json.dump(rows, f, indent=1)
    total = sum(r["glb_bytes"] for r in rows)
    print(f"{len(rows)} examples, coverage "
          f"{rows[0]['area_lit_frac']:.4f} .. {rows[-1]['area_lit_frac']:.4f}, "
          f"preview GLBs {total / 1e6:.1f} MB total")
    for r in rows:
        print(f"  {r['sid'][:8]} {r['what']:24s} lit={r['area_lit_frac']:.4f} "
              f"vox={r['voxels'].get('emissive_frac_raw', 0):.4f} "
              f"glb={r['glb_bytes'] / 1e6:.2f}MB  {'cmp' if r['in_compare'] else '   '} "
              f"{'box' if r['has_box'] else '   '}  "
              f"{r['license']}  {r['author']}")
    for sid, what, why in excluded:
        print(f"EXCLUDED {sid[:8]} {what}: {why}")


if __name__ == "__main__":
    main()
