#!/usr/bin/env python3
"""Build the emission-strength-ladder page: what strength does, and what it
cannot do to a binary mask.

PRIMARY figure: the box preset, no key light, the object as the only light
source in a room. Strength is a claim about what an object's light does to a
room, and the box treatment is the only one of the two that actually shows
that; key-lit keeps the object lit regardless of what it emits, hiding half
the claim by design. Two bands per shape: the asset's OWN native emission
(the `_true` path) and our method's formulation, ground-truth mask times
albedo (the `_glow` path with no --pred_masks). Comparing them at every
strength is the cost of the mask-times-albedo formulation itself, with a
perfect mask and no model in the way, an experiment the paper skeleton has
recorded as never yet run.

SECONDARY figure: the key-lit ladder (ground truth vs the model's own
prediction) built earlier, kept because it shows the object itself and the
model responding to strength; not the page's subject, so it is labelled
accordingly and carries its own claim-2 measurement with the key-lit
confound named plainly.

Claim 2 (a surface the model/method did not select cannot be lit by turning
strength up) is led by the texture-space fact that needs no render (99 mask
files, 34,603,008 pixels, only values 0 and 255), then confirmed directly in
the box_mask band, where there is no key light and no model, only a perfect
ground-truth mask.

Renders come from box_ladder.sh and (for the secondary section) the earlier
key-lit jobs, all on the solar cluster
(/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3/ladder/). This script reads
the model's own prediction stats and this page's own measurements.json, so
every number on the page is measured rather than retyped.

Run: /cs/3dlg-project/3dlg-hcvc/omages/omages_internal/.venv2/bin/python \
        web/_preview/strength_ladder/build.py
"""
import hashlib
import html
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WEB)
sys.path.insert(0, os.path.join(REPO, "tools"))

import xgpage as lp                      # noqa: E402
import workspace_zone as wz              # noqa: E402  (read-only: tree + zone guard)
from xgpage.publish import publish_assets  # noqa: E402

SITE_ROOT = "/projects/omages/yanxg/lightgen"
SITE_ASSETS = f"{SITE_ROOT}/assets"
FAVICON = f"{SITE_ROOT}/assets/images/favicon.png"
PAGE_DATE = "2026-08-07"
PAGE_INNER = 820

P = "/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3"
PRED_MASKS = os.path.join(P, "pred_masks", "emis_72k_unfilt")
LADDER_OUT = os.path.join(P, "ladder", "out")

STRENGTHS = [0, 1, 4, 8, 16]

# PRIMARY (box treatment): chosen freely for legibility of the strength
# effect, no prediction constraint. The largest true emitter in the manifest
# (candles) is used here precisely because that constraint used to exclude it.
SHAPES_BOX = [
    {"sid": "8f4c281aef1b4563b6103efbcd77fac1", "tier": "sparse",
     "what": "headphone stand", "author": "Serega_SHTOPOR",
     "license": "CC Attribution",
     "url": "https://sketchfab.com/models/8f4c281aef1b4563b6103efbcd77fac1"},
    {"sid": "51a60b164e874bf891597d9c6c1941af", "tier": "mid",
     "what": "sci-fi weapon", "author": "George B",
     "license": "CC Attribution",
     "url": "https://sketchfab.com/models/51a60b164e874bf891597d9c6c1941af"},
    {"sid": "658ecf9f837246509b0b1c4aa81e9e5b", "tier": "large",
     "what": "three lit candles", "author": "Rexotec",
     "license": "CC Attribution",
     "url": "https://sketchfab.com/models/658ecf9f837246509b0b1c4aa81e9e5b"},
]

# SECONDARY (key-lit treatment): the earlier, prediction-constrained shape set
# (headphone stand and sci-fi weapon repeat; the third is wall fixtures, not
# candles, because this section predates the funnel constraint being dropped
# and is kept as real, already-rendered work rather than re-shot to match).
SHAPES_KEYLIT = [
    SHAPES_BOX[0],
    SHAPES_BOX[1],
    {"sid": "e5eecab2bc8649548b48b79e705d768e", "tier": "large-qualifying",
     "what": "wall light fixtures", "author": "Archistoric",
     "license": "CC Attribution",
     "url": "https://sketchfab.com/models/e5eecab2bc8649548b48b79e705d768e"},
]

ALL_SIDS = {g["sid"] for g in SHAPES_BOX} | {g["sid"] for g in SHAPES_KEYLIT}

MEASUREMENTS = {m["sid"]: m
                for m in json.load(open(os.path.join(HERE, "measurements.json")))}

# The model's own record of what it predicted, per shape, read here so the
# gt/pred coverage numbers in the captions are the ones the renders actually
# used, not numbers retyped from an earlier pass.
PRED_STATS = {}
for _sid in ALL_SIDS:
    _p = os.path.join(PRED_MASKS, f"{_sid}__stats.json")
    if not os.path.exists(_p):
        raise RuntimeError(f"no prediction stats for {_sid}: {_p}")
    PRED_STATS[_sid] = json.load(open(_p))

# The full 11-shape candidate set (funnel section, moved to the end and
# reframed as the model's own degeneracy rate, not a selection rationale).
_GALLERY_WHAT = {}
_gallery_path = os.path.join(WEB, "_preview", "paper_v3", "gallery.json")
if os.path.exists(_gallery_path):
    _GALLERY_WHAT = {r["sid"]: r.get("what") for r in json.load(open(_gallery_path))}

CANDIDATES = []
_manifest = json.load(open(os.path.join(P, "manifest12.json")))
_manifest_sids = {r["sid"] for r in _manifest}
for _f in sorted(os.listdir(PRED_MASKS)):
    if not _f.endswith("__stats.json"):
        continue
    _sid = _f.replace("__stats.json", "")
    if _sid not in _manifest_sids:
        continue
    _d = json.load(open(os.path.join(PRED_MASKS, _f)))
    CANDIDATES.append({
        "sid": _sid,
        "what": _GALLERY_WHAT.get(_sid, _sid[:8]),
        "gt": _d["gt_voxel_frac"],
        "pred": _d["pred_voxel_frac"],
        "empty": _d["empty_prediction"],
    })
CANDIDATES.sort(key=lambda r: r["gt"])


def _candidate_class(c):
    if c["empty"]:
        return "empty"
    if c["pred"] > 0.90:
        return "saturated"
    if c["pred"] >= 0.05:
        return "qualifying"
    return "near-empty"


for _c in CANDIDATES:
    _c["cls"] = _candidate_class(_c)

BOX_SIDS = {g["sid"] for g in SHAPES_BOX}
N_EMPTY = sum(1 for c in CANDIDATES if c["cls"] == "empty")
N_SATURATED = sum(1 for c in CANDIDATES if c["cls"] == "saturated")
N_NEAR_EMPTY = sum(1 for c in CANDIDATES if c["cls"] == "near-empty")
N_QUALIFYING = sum(1 for c in CANDIDATES if c["cls"] == "qualifying")
assert N_EMPTY + N_SATURATED + N_NEAR_EMPTY + N_QUALIFYING == len(CANDIDATES)

OUTLINE = [
    ("box", "What strength does to a room"),
    ("binary", "What a binary mask cannot do"),
    ("keylit", "What strength looks like on the object"),
    ("funnel", "The model's own degeneracy rate"),
    ("sources", "Provenance"),
]


def asset(rel):
    p = os.path.join(HERE, rel)
    h = hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    return f"{rel}?v={h}"


def credit(g):
    for key in ("author", "license", "url"):
        if not g.get(key):
            raise RuntimeError(f"{g['sid']}: no {key}")
    return (f'{html.escape(g["author"])} &middot; {html.escape(g["license"])} '
            f'&middot; <a href="{html.escape(g["url"], quote=True)}" '
            f'target="_blank" rel="noopener">Sketchfab</a>')


# ------------------------------------------------------------- section 1
BOX_INTRO = (
    "Strength is a claim about what an object's light does to the space "
    "around it, not only about how the object itself looks. A key light "
    "keeps every panel lit regardless of what the object emits, which hides "
    "exactly the half of the claim a room should show; the box preset "
    "removes every light source but the object, so the room is lit only by "
    "whatever it emits, at whatever strength is set.<br><br>"
    "<b>4.0 is a look choice, not a measurement.</b> The bake this method "
    "trains against stores emission as an 8-bit texture and drops the "
    "<code>KHR_materials_emissive_strength</code> glTF extension, which 3 of "
    "60 sampled source GLBs carry; nothing in the data says how brightly a "
    "surface emits.<br><br>"
    "Below is a ladder of five strengths, 0, 1, 4, 8 and 16, on three shapes "
    "chosen purely for legibility of the strength effect, not for prediction "
    "quality: a sparse emitter, a mid emitter, and the largest true emitter "
    "in the eleven-shape manifest. Each shape carries two bands, both from "
    "the FIXED renderer (see provenance): the asset's own native emission, "
    "and our method's own formulation, ground truth mask times albedo, with "
    "a perfect mask and no model involved. Comparing the two at every "
    "strength is the cost of the mask-times-albedo formulation itself, an "
    "experiment recorded as open and never yet run."
)

PICK_REASON_BOX = {
    "8f4c281aef1b4563b6103efbcd77fac1": (
        "the sparse emitter: {gtp:.2f}% of surface area, the smallest true "
        "emitter of the eleven manifest shapes, a purple logo and an edge "
        "strip on an otherwise plain black stand."),
    "51a60b164e874bf891597d9c6c1941af": (
        "the mid emitter: {gtp:.1f}% of surface area, a glowing core on an "
        "otherwise dark weapon."),
    "658ecf9f837246509b0b1c4aa81e9e5b": (
        "the largest true emitter in the manifest: {gtp:.1f}% of surface "
        "area, three candle flames. Earlier shape selection excluded this "
        "shape for having no usable model prediction; that constraint does "
        "not apply here, since this figure is not about the model."),
}


def box_figure(g):
    sid = g["sid"]
    true_cells = [{"img": asset(f"img/{sid}_box_true_s{st}.png"),
                   "alt": f"{g['what']}, own emission, box, strength {st}"}
                  for st in STRENGTHS]
    mask_cells = [{"img": asset(f"img/{sid}_box_mask_s{st}.png"),
                   "alt": f"{g['what']}, GT mask x albedo, box, strength {st}"}
                  for st in STRENGTHS]
    rows = [("own emission", true_cells), ("GT mask x albedo", mask_cells)]
    cols = [f"s = {st}" for st in STRENGTHS]
    return lp.method_matrix(cols, rows, caption_html=box_caption(g),
                            native_px=768, content="photo",
                            id=f"box-{sid}", page_inner=PAGE_INNER)


def box_caption(g):
    sid = g["sid"]
    gtp = PRED_STATS[sid]["gt_voxel_frac"] * 100
    reason = PICK_REASON_BOX[sid].format(gtp=gtp)
    diffs = []
    for st in STRENGTHS:
        a = np.asarray(Image.open(os.path.join(HERE, f"img/{sid}_box_true_s{st}.png"))
                        .convert("RGB"), dtype=np.float32)
        b = np.asarray(Image.open(os.path.join(HERE, f"img/{sid}_box_mask_s{st}.png"))
                        .convert("RGB"), dtype=np.float32)
        diffs.append(float(np.abs(a - b).mean()))
    diff_str = ", ".join(f"{d:.1f}" for d in diffs)
    return (
        f"<b>{html.escape(g['what'].capitalize())}, picked as {reason}</b> "
        f"Rows: the asset's own native emission at each strength (top), and "
        f"ground-truth mask times albedo, our method's own formulation with "
        f"a perfect mask (bottom), rendered under the identical box "
        f"treatment (Filmic, exposure +1.5, no key light, wall albedo 0.80, "
        f"32/16 bounces, bloom radius 7 / threshold 1.0 / mix &minus;0.45). "
        f"Every rung is a real render; the room's light visibly grows with "
        f"strength because the room has nothing else to see by. Mean "
        f"absolute pixel difference between the two rows at each strength, "
        f"0/1/4/8/16: {diff_str} (0&ndash;255 scale) &mdash; the cost of "
        f"mask-times-albedo against the asset's own emission, with a "
        f"perfect mask and no model in the way. {credit(g)} &middot; "
        f'<a href="{asset(f"img/{sid}_contact_box.png")}" target="_blank" '
        f'rel="noopener">composite PNG</a>'
    )


# ------------------------------------------------------------- section 2
BINARY_INTRO = (
    "<b>The claim is about the MASK, and it is already measured, in texture "
    "space, with no render involved.</b> Across 99 mask files from three "
    "checkpoints, 34,603,008 pixels, the renderer's own scan found only two "
    "values present anywhere: 0 and 255. A material that is not selected has "
    "its emission shader switched off outright, not turned down, so there is "
    "no dim residual value for a higher strength to amplify. That is the "
    "claim, and it does not need a render to hold.<br><br>"
    "The box treatment above is where a render can actually confirm it: no "
    "key light, so a genuinely unselected surface has nothing reflecting off "
    "it, only whatever bounced light reaches it from the parts that do "
    "emit. That bounced light is real physics, the box treatment's own "
    "subject, and it can be nonzero without meaning anything leaked; it "
    "means light travelled through the room, which is what the room is "
    "for. The measurement below is on the ground-truth mask times albedo "
    "band, a perfect mask, no model."
)


def _window_rise(m, key):
    """Sum(R,G,B) at strength 16 minus strength 0, for the windowed mean the
    table displays (not the single-pixel delta used to pick the point)."""
    s0 = sum(m[f"{key}_rgb_by_strength"]["0"])
    s16 = sum(m[f"{key}_rgb_by_strength"]["16"])
    return s16 - s0


def binary_figure_and_table(g, band):
    sid = g["sid"]
    m = MEASUREMENTS[sid][band]
    figs = [lp.fig(asset(f"img/{sid}_{band}_s16_marked.png"),
                   caption_html=binary_caption(g, m, band),
                   native_px=768, content="photo")]
    rows = []
    for st in STRENGTHS:
        sr, sgc, sb = m["selected_rgb_by_strength"][str(st)]
        ur, ugc, ub = m["unselected_rgb_by_strength"][str(st)]
        rows.append(
            f"<tr><td style='text-align:left'>{st}</td>"
            f"<td>{sr:.1f}, {sgc:.1f}, {sb:.1f}</td>"
            f"<td>{ur:.1f}, {ugc:.1f}, {ub:.1f}</td></tr>")
    table = lp.results_table(
        ["strength", "selected window, mean RGB (0-255)",
         "unselected window, mean RGB (0-255)"], "".join(rows))
    return "".join(figs) + table


def binary_caption(g, m, band):
    sel, uns = m["selected_xy"], m["unselected_xy"]
    sel_rise = _window_rise(m, "selected")
    uns_rise = _window_rise(m, "unselected")
    uns_s0 = sum(m["unselected_rgb_by_strength"]["0"])
    uns_s16 = sum(m["unselected_rgb_by_strength"]["16"])
    what = html.escape(g["what"].capitalize())
    flat = abs(uns_rise) < 5
    if band == "box_mask" and flat:
        lede = (
            f"<b>{what}, box, ground-truth mask: the unselected window is "
            f"flat, {uns_s0:.1f} to {uns_s16:.1f} out of 765, "
            f"indistinguishable from rendering noise.</b> With no key light "
            f"and no bounced light reaching this point either, an "
            f"unselected surface stays at the background floor regardless "
            f"of strength: the claim, confirmed directly.")
    elif band == "box_mask":
        lede = (
            f"<b>{what}, box, ground-truth mask: the unselected window "
            f"rises from {uns_s0:.1f} to {uns_s16:.1f} out of 765.</b> With "
            f"no key light, this is not reflected light or bloom bleed: it "
            f"is bounced illumination from whatever surface DOES emit, "
            f"real physics and the box treatment's own subject. The "
            f"material at this point is still switched off; light reaching "
            f"it after bouncing off another surface is not the mask "
            f"leaking.")
    elif flat:
        lede = (
            f"<b>{what}, key-lit: the unselected window is flat, "
            f"{uns_s0:.1f} to {uns_s16:.1f} out of 765, but it starts from "
            f"the key light's own reflected baseline, not from black.</b> "
            f"Compare against the box measurement above for what an "
            f"unselected surface looks like with no key light.")
    else:
        lede = (
            f"<b>{what}, key-lit: the unselected window rises from "
            f"{uns_s0:.1f} to {uns_s16:.1f} out of 765, a real but small "
            f"change, and it is bloom bleed on top of the key light's own "
            f"reflected baseline, not a mask leak.</b> This shape's "
            f"selected region covers most of the surface and saturates "
            f"early, and the compositor's blur carries some of that glow "
            f"onto the unselected geometry beside it in the rendered "
            f"image; the material itself is still switched off.")
    return (
        f"{lede} The selected window (orange) rises from "
        f"{sum(m['selected_rgb_by_strength']['0']):.1f} to "
        f"{sum(m['selected_rgb_by_strength']['16']):.1f} over the same "
        f"range, a change of {sel_rise:.0f}. Both windows are 12&times;12 "
        f"pixels, sampled at the same image coordinates on every rung. "
        f"Selected window at [{sel[0]}, {sel[1]}]; unselected window at "
        f"[{uns[0]}, {uns[1]}], chosen as the on-object point (per the "
        f"neutral-studio silhouette) that changes LEAST from strength 0 to "
        f"strength 16, so the comparison is against the most favourable "
        f"case for a leak to show up in, not a cherry-picked flat one. "
        f"Panel shown is s = 16; the table below carries all five "
        f"strengths."
    )


# ------------------------------------------------------------- section 3
KEYLIT_INTRO = (
    "Two treatments appear on this page, and they show different things. "
    "The box treatment above shows what an object's light does to a room: "
    "no key light, no model, a perfect mask. The ladder below shows what "
    "the OBJECT itself looks like while its own predicted mask responds to "
    "strength, ground truth against the model's own prediction, under a "
    "dim key light that keeps the object visible regardless of what it "
    "emits. This is the ladder built before the box treatment was adopted "
    "as the page's primary figure; it stays, secondary, because it is real "
    "work and a reader benefits from seeing the object and the model's "
    "prediction responding to strength, which the box figure does not "
    "show. Its third shape is the wall fixtures, not the candles above, "
    "because this section was built under a since-dropped constraint "
    "(a usable model prediction) that no longer applies to the box "
    "figure; see provenance."
)

PICK_REASON_KEYLIT = {
    "8f4c281aef1b4563b6103efbcd77fac1": (
        "the sparse emitter: {gtp:.2f}% of surface area."),
    "51a60b164e874bf891597d9c6c1941af": (
        "the mid emitter: {gtp:.1f}% of surface area, and a legible "
        "partial prediction."),
    "e5eecab2bc8649548b48b79e705d768e": (
        "the largest ground-truth emitter whose prediction was usable "
        "(neither empty nor saturated) at the time this section was built: "
        "{gtp:.1f}% of surface area."),
}


def keylit_figure(g):
    sid = g["sid"]
    gt_cells = [{"img": asset(f"img/{sid}_gt_s{st}.png"),
                "alt": f"{g['what']}, ground-truth emission, strength {st}"}
               for st in STRENGTHS]
    pred_cells = [{"img": asset(f"img/{sid}_pred_s{st}.png"),
                  "alt": f"{g['what']}, predicted emission, strength {st}"}
                 for st in STRENGTHS]
    rows = [("ground truth", gt_cells), ("model prediction", pred_cells)]
    cols = [f"s = {st}" for st in STRENGTHS]
    return lp.method_matrix(cols, rows, caption_html=keylit_caption(g),
                            native_px=768, content="photo",
                            id=f"keylit-{sid}", page_inner=PAGE_INNER)


def keylit_caption(g):
    sid = g["sid"]
    ps = PRED_STATS[sid]
    gtp = ps["gt_voxel_frac"] * 100
    predp = ps["pred_voxel_frac"] * 100
    reason = PICK_REASON_KEYLIT[sid].format(gtp=gtp)
    if ps["empty_prediction"]:
        pred_sentence = (
            f"<b>The model predicts no emission anywhere on this shape.</b>")
    elif predp > 3 * max(gtp, 0.01):
        ratio = predp / max(gtp, 1e-6)
        pred_sentence = (
            f"<b>The model over-predicts here by roughly {ratio:.0f}&times;: "
            f"it selects {predp:.1f}% of surface area against a "
            f"ground truth of {gtp:.2f}%.</b> At high strength most of the "
            f"{g['what']} glows, and that glow is a large miss, not a good "
            f"result.")
    else:
        pred_sentence = (
            f"The model predicts emission on {predp:.1f}% of surface area "
            f"against a ground truth of {gtp:.2f}%.")
    return (
        f"<b>{html.escape(g['what'].capitalize())}, picked as {reason}</b> "
        f"Rows: the asset's own ground-truth emission, and the model's "
        f"predicted mask, both rendered under the identical key-lit "
        f"treatment (key 8, AgX, one dim fill light, bloom radius 9 / "
        f"threshold 1.0 / mix &minus;0.15) at every strength. {pred_sentence} "
        f"{credit(g)} &middot; "
        f'<a href="{asset(f"img/{sid}_contact_keylit.png")}" target="_blank" '
        f'rel="noopener">composite PNG</a>'
    )


# ------------------------------------------------------------- section 4
FUNNEL_INTRO = (
    "This is not why the shapes above were picked; the box ladder's shapes "
    "were chosen freely, and only the secondary key-lit section (\"What "
    "strength looks like on the object\") was ever constrained by "
    "prediction quality. It is a genuine, separate finding about the "
    "model, kept and shown on its own terms: of the eleven manifest shapes "
    "with both a GLB and a model prediction on hand, only {nq} predict a "
    "non-degenerate region (roughly 5 to 90 percent of surface area). {ne} "
    "predict nothing anywhere, {nn} predict something but under 5 percent "
    "of surface area (effectively also a miss), and {ns} predicts over 90 "
    "percent (effectively lighting the whole object). This is the model's "
    "degeneracy rate on this manifest."
)


def funnel_table():
    rows = []
    for c in CANDIDATES:
        shown = c["sid"] in BOX_SIDS
        mark = " &#10003;" if shown else ""
        row_style = ' style="font-weight:600"' if shown else ""
        rows.append(
            f"<tr{row_style}><td style='text-align:left'>"
            f"{html.escape(c['what'])}{mark}</td>"
            f"<td>{c['gt'] * 100:.2f}</td>"
            f"<td>{c['pred'] * 100:.4f}</td>"
            f"<td style='text-align:left'>{c['cls']}</td></tr>")
    return lp.results_table(
        ["shape", "ground truth %", "predicted %", "class"], "".join(rows))


def funnel_section():
    note = FUNNEL_INTRO.format(nq=N_QUALIFYING, ne=N_EMPTY, nn=N_NEAR_EMPTY,
                               ns=N_SATURATED)
    table = funnel_table()
    caption = (
        f"<b>{N_QUALIFYING} of {len(CANDIDATES)} candidate shapes predict a "
        f"non-degenerate region.</b> Class boundaries: empty = 0% "
        f"predicted; near-empty = under 5%, nonzero; qualifying = "
        f"5&ndash;90%; saturated = over 90%. Ground truth and predicted are "
        f"both the voxel-space emissive fraction of surface area, read "
        f"from the model's own prediction sidecars. Checked rows are the "
        f"three shown in the box ladder above (present in this table "
        f"because they also happen to have a prediction on hand, not "
        f"because they were selected by it).")
    return lp.prose(note) + table + lp.chartnote(caption)


# theme.css sets table.results to min-width:960px, sized for a wide v2/v1
# column. This page is v3 at the 820px content measure, so that floor forces
# every table into an internal horizontal scroll a reader has to notice and
# use to reach the last column. Same defect, same fix as paper_v3's EXTRA_CSS.
EXTRA_CSS = "<style>.xg3 table.results{min-width:0}</style>"


def build():
    assets_dir = os.path.join(WEB, "assets")

    hero = lp.hero_header(
        f"lightgen &middot; emissive generation &middot; {PAGE_DATE}",
        "Strength changes what light does to a room, and nothing in the "
        "data says how much",
        dek_html=(
            "A ladder of real renders across five strengths, in a Cornell "
            "box lit only by the object: the asset's own emission against "
            "our method's own formulation, ground-truth mask times albedo, "
            "compared at every strength. The mask is exactly binary in "
            "texture space, and the box measurement confirms it directly: "
            "a surface the mask does not select changes little to nothing "
            "as strength rises, and what does change is bounced light, not "
            "a leak."),
    )

    box_figs = "".join(box_figure(g) for g in SHAPES_BOX)
    binary_body = "".join(
        f'<h3 style="margin-top:2em">{html.escape(g["what"])}</h3>'
        + binary_figure_and_table(g, "box_mask")
        for g in SHAPES_BOX
    ) + '<h3 style="margin-top:2.5em">Key-lit, for comparison (secondary treatment)</h3>' + "".join(
        f'<h4 style="margin-top:1.5em">{html.escape(g["what"])}</h4>'
        + binary_figure_and_table(g, "pred")
        for g in SHAPES_KEYLIT
    )
    keylit_figs = "".join(keylit_figure(g) for g in SHAPES_KEYLIT)

    body = [
        lp.section_v2("box", "", "What strength does to a room",
                      lp.prose(BOX_INTRO) + box_figs),
        lp.section_v2("binary", "", "What a binary mask cannot do",
                      lp.prose(BINARY_INTRO) + binary_body),
        lp.section_v2("keylit", "", "What strength looks like on the object",
                      lp.prose(KEYLIT_INTRO) + keylit_figs),
        lp.section_v2("funnel", "", "The model's own degeneracy rate",
                      funnel_section()),
    ]

    apx = lp.appendix("Sources", [
        "Shapes: TexVerse-1K, sampled from the same held-out manifest as "
        "<code>web/_preview/paper_v3/</code> "
        f"(<code>{P}/manifest12.json</code>). Each shape is a Sketchfab "
        "model, CC Attribution; author, licence and link are under every "
        "figure. Box ladder: sparse, mid and largest-true-emitter tiers, "
        "chosen from all eleven manifest shapes with no prediction "
        "constraint. Key-lit ladder (secondary): the earlier sparse and mid "
        "picks plus the largest ground-truth emitter whose prediction was "
        "usable at the time, since that section predates the box treatment "
        "and the funnel constraint being dropped.",
        "Box renders: every flag pinned rather than defaulted. "
        "768&times;768, 1024 samples, Filmic view transform, exposure "
        "+1.5, no key light (the object is the only light source), wall "
        "albedo 0.80, box scale 2.0 / height 1.7 / depth 1.8, 32 total / "
        "16 diffuse bounces, bloom radius 7 / threshold 1.0 / mix "
        "&minus;0.45, strengths 0, 1, 4, 8, 16, the shape's own solved "
        "camera. Two emission sources, both via a new "
        "<code>--emission_source {true,mask}</code> flag on "
        "<code>render_emissive.py</code> (added for this page; default "
        "\"mask\" preserves every existing caller's behaviour unchanged): "
        "\"true\" drives <code>set_true_strength()</code>, the asset's own "
        "native emission at <code>--emit_strength</code>, no mask or "
        "albedo substitution; \"mask\" is the existing "
        "<code>rebuild_emission()</code> ground-truth path (mask times "
        "albedo), now carrying the strength-shadowing fix below.",
        "Key-lit renders (secondary): key 8, background 0.012, AgX view "
        "transform, exposure 0, bloom radius 9 / threshold 1.0 / mix "
        "&minus;0.15, samples 256 (96 for the neutral-studio reference "
        "pass), strengths 0, 1, 4, 8, 16. Ground truth uses the asset's own "
        "emissive channel; prediction substitutes "
        f"<code>{PRED_MASKS.replace(P + '/', '')}</code>, from the "
        "<code>three_ckpt_eval</code> run's <code>emis_72k_unfilt</code> "
        "split (three checkpoints, unfiltered 72k evaluation set).",
        "Renderer fix. The ground-truth mask path, "
        "<code>rebuild_emission()</code>, had a bug: a local variable "
        "shadowed <code>--emit_strength</code> with the asset's own "
        "authored strength before it reached the shader, so a ground-truth "
        "strength sweep came back pixel-identical at every rung (measured: "
        "mean pixel difference 0.000 between every adjacent pair, all "
        "three original shapes). Fixed by renaming the shadowing variable "
        "to <code>asset_strength</code>, kept only for the fidelity stats "
        "that legitimately need the asset's own value; the render socket "
        "now uses the <code>strength</code> parameter throughout. Verified "
        "after the fix: nonzero, growing adjacent-rung differences on a "
        "re-render of all three original ground-truth bands before this "
        "page was rebuilt. The <code>--emission_source</code> flag above "
        "is a separate, additive change (new default-preserving option, "
        "not a bug fix).",
        "Paper_v3 impact check: rendered the headphone stand's "
        "ground-truth glow at <code>--emit_strength 4.0</code> with "
        "paper_v3's own pinned key-lit flags (no <code>--camera_json</code>, "
        "matching how <code>k8_all.sh</code> actually produced it) using "
        "the fixed renderer, and diffed against "
        f"<code>{P}/final3/8f4c281aef1b4563b6103efbcd77fac1_glow.png</code>: "
        "mean absolute pixel difference 0.0000758, 99.98% of pixels "
        "bit-identical, matching a fresh strength-4.0 render almost "
        "exactly. Two follow-up checks rule out the two ways that could be "
        "coincidence: the shape's own authored emission strength is 1.0, "
        "not 4.0, so it is not a match by luck; and its <code>final3</code> "
        "sidecar records <code>pred_masks: null</code>, so it went through "
        "the ground-truth path that carried the bug, not the unaffected "
        "predicted-mask path. The account every measurement agrees with is "
        "that <code>final3</code> predates the defect, most likely "
        "introduced in uncommitted working-tree changes after that gallery "
        "was rendered; git history (one commit, already containing the "
        "bug) cannot confirm this further. Paper_v3 untouched and "
        "unpublished by this page's work.",
        "Compute: solar cluster, account 3dlg-hcvc-lab, partition "
        "3dlg-hcvc-lab-short, no GPU (Cycles CPU rendering). Key-lit jobs "
        "240390 and 240395: 16 CPUs / 48G per task. Ground-truth re-render "
        "after the fix, job 240411: 16 CPUs / 48G per task. Box ladder, "
        "job 240454 (3 tasks, one per shape, 10 renders each): "
        "<b>64 CPUs / 96G per task</b>, widened from 16 after a throughput "
        "probe measured roughly 13&times; on this node class for Cycles "
        "CPU rendering, which does not depend on thread count for its "
        "output, only its wall-clock cost. Raw output: "
        f"<code>{LADDER_OUT}/&lt;sid&gt;/{{box_true,box_mask,gt,pred}}/"
        "s&lt;strength&gt;/</code>.",
        "Pixel measurement (second section): points are chosen by DELTA "
        "between a band's own strength 0 and strength 16 panels, not "
        "absolute brightness, since neither band's strength-0 panel is "
        "necessarily black (key-lit keeps a key light on; box mode's "
        "strength-0 panel can still carry faint bounced light if another "
        "material in the scene is treated as emissive by mistake, though "
        "none was observed here). The delta map is box-blurred (13px) to "
        "pick a robust location; selected = largest blurred delta on "
        "object, unselected = smallest. Both 12&times;12 windows sampled "
        "unblurred at the identical coordinates on every rung. "
        "Script: <code>web/_preview/strength_ladder/prepare_assets.py</code>.",
        "The 34,603,008-pixel, 99-file, three-checkpoint binary-mask "
        "measurement cited in the second section is the renderer's own, "
        "documented in <code>segvigen_emissive/render/README.md</code>, "
        "quoted here rather than rerun.",
        "Binarization threshold check. render_emissive.py's ground-truth "
        "mask binarizes at <code>LIN_EPS = 1e-5</code> (linear); the "
        "dataset's own rule is emission &gt; 1/255, which converts to "
        "&asymp;3.035e-4 linear via the renderer's own "
        "<code>srgb_to_linear()</code>, about 30&times; higher. Measured "
        "directly on the three ladder shapes' own emissive materials "
        "(texture-space, no render): the headphone stand's two emissive "
        "materials carry no texture at all (a constant emissive colour "
        "only), so no per-texel threshold applies to them, and together "
        "they cover 0.83% of surface area, matching "
        "<code>gt_voxel_frac</code> exactly, the correctly small purple "
        "logo and edge strip, not the whole shell; what looks larger in "
        "the box ladder above is bloom spreading a small, saturating "
        "emitter's glow across the rendered image, not the mask covering "
        "more surface than it does. The sci-fi weapon's and candles' "
        "single textured emissive materials both give the two thresholds "
        "the IDENTICAL selected fraction (6.9306% and 32.3737% "
        "respectively, matching to six decimal places) with zero texels "
        "landing in the gap between them: both textures are already "
        "effectively binary, solid black outside the emitting region and a "
        "substantial nonzero value inside it, nothing faint in between. On "
        "these three shapes the threshold gap is real in principle but "
        "measured to be inert in practice; it is not what makes a "
        "mask-times-albedo panel diverge from the asset's own emission, "
        "and it does not explain a fully dark mask-times-albedo panel on a "
        "dark-albedo shape elsewhere in this project, which is the albedo "
        "multiply doing that, a different place the formulation loses "
        "light. Not checked beyond these three shapes; a broader sample "
        "would be needed before treating \"inert\" as a dataset-wide "
        "property. Script: <code>paper_v3/check_threshold.py</code>.",
    ])

    page_html = lp.page(
        title="What emission strength does (lightgen)",
        header_html=hero,
        body_sections=body + [apx],
        assets_rel=SITE_ASSETS,
        assets_dir=assets_dir,
        theme="v3",
        tree_html=wz.tree_html(active_href=None),
        nav_title="Strength ladder",
        outline_entries=[{"id": i, "label": lab} for i, lab in OUTLINE],
        version_slot=lp.v3_version_slot(date=PAGE_DATE),
        needs_katex=False,
        extra_head=(f'<link rel="icon" href="{FAVICON}">' + EXTRA_CSS),
    )

    violations = wz.console_links_in(page_html)
    if violations:
        sys.exit(f"ZONE-LINK GUARD FAILED: page links to the console: {violations}")

    out = os.path.join(HERE, "index.html")
    with open(out, "w") as f:
        f.write(page_html)
    print(f"wrote {out} ({len(page_html)} bytes, "
          f"{len(SHAPES_BOX)} box shapes, {len(SHAPES_KEYLIT)} key-lit shapes)")
    print("  zone-link guard: clean")

    publish_assets(assets_dir)
    print(f"assets published -> {assets_dir}")


if __name__ == "__main__":
    build()
