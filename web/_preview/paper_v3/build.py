#!/usr/bin/env python3
"""Build the paper_v3 page: four claims, then a gallery of glowing objects.

The page is deliberately short. Four claims carried by the `.skel` claim-chain
component (the same component as web/_preview/paper_skeleton/), then one figure:
twelve TexVerse shapes rendered in a dark room with emission = mask x albedo.

The renders come from render_emissive.py, run on the solar cluster
(/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3/). Its per-shape stats.json is
read here, so every number on the page is measured rather than retyped.

Run: /cs/3dlg-project/3dlg-hcvc/omages/omages_internal/.venv2/bin/python \
        web/_preview/paper_v3/build.py
"""
import hashlib
import html
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WEB)
sys.path.insert(0, os.path.join(REPO, "tools"))

import xgpage as lp                      # noqa: E402
import workspace_zone as wz              # noqa: E402  (read-only: tree + zone guard)
# one source for the comparison treatment: the guard that enforces it and the
# appendix that states it must not be able to disagree
sys.path.insert(0, HERE)
from make_gallery import PRED_PENDING, PRED_SRC, REFERENCE_TREATMENT  # noqa: E402
from xgpage.publish import publish_assets  # noqa: E402

SITE_ROOT = "/projects/omages/yanxg/lightgen"
SITE_ASSETS = f"{SITE_ROOT}/assets"
FAVICON = f"{SITE_ROOT}/assets/images/favicon.png"
PAGE_DATE = "2026-08-06"

GALLERY = json.load(open(os.path.join(HERE, "gallery.json")))
COMPARE = [g for g in GALLERY if g.get("in_compare")]

# The albedo baseline's own record of what it did, per shape. Read here so the
# caption's coverage numbers are the ones the mask stage actually reached, not
# numbers retyped from a run that may since have been superseded.
HEUR_MASKS = ("/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3/"
              "pred_masks/albedo_matched")
HEUR = {}
for _g in COMPARE:
    _p = os.path.join(HEUR_MASKS, f"{_g['sid']}__stats.json")
    if not os.path.exists(_p):
        raise RuntimeError(f"albedo baseline has no stats for {_g['sid']}: {_p}")
    HEUR[_g["sid"]] = json.load(open(_p))

# The voxel gallery's own measured resolution and occupied-cell count, per
# shape (owner ask 2026-08-11: the earlier renders drew a 64-cell display grid
# for legibility, which does not reflect the real output). Read from the
# converter's own npz meta (voxel_true_res/voxel_dir/<sid>.npz), never
# retyped: geometry is native 512^3 (o_voxel.convert.mesh_to_flexible_dual_grid,
# the same extraction build_dataset_direct.py runs for training), base-colour
# and the emissive label are Dongchen's 256^3 bake nearest-neighbor upsampled
# onto that grid (build_dataset_direct.py's Upsampler256to512) -- read straight
# from dataset_direct/train_72k's own input.vxz/output.vxz, the exact tensors
# this checkpoint trains on.
VOXEL_DIR = "/project/3dlg-hcvc/omages/yanxg_scratch/voxel_true_res/voxel_dir"
VOXEL_META = {}
for _g in GALLERY:
    _vp = os.path.join(VOXEL_DIR, f"{_g['sid']}.npz")
    if not os.path.exists(_vp):
        raise RuntimeError(f"no true-resolution voxel data for {_g['sid']}: {_vp}")
    VOXEL_META[_g["sid"]] = json.loads(str(np.load(_vp)["meta"]))

# This page is the workspace zone's living "paper_skeleton" page; the preview
# directory is only where it is built. The tree's active leaf and the version
# manifest are therefore the LIVING page's, not this preview's, so the rail
# highlights correctly and the picker resolves from either location.
LIVING_HREF = f"{wz.WORKSPACE_URL}/paper_skeleton/index.html"
VERSIONS_URL = f"{wz.WORKSPACE_URL}/paper_skeleton/versions.json"
# v3's content measure. method_matrix does its tile arithmetic against this.
PAGE_INNER = 820

# The outline rail. Shorter than the headings on purpose: a rail entry is a
# label, not a restatement of the claim the heading makes.
OUTLINE = [
    ("claims", "The claims"),
    ("gallery", "What the output looks like"),
    ("box", "What the light does to a room"),
    ("compare", "Ours versus ground truth"),
]


def latest_version():
    """The newest label in the living page's manifest, so the preview's slot is
    honest today. tools/publish_version.py strips and re-injects this slot at
    mint time, so the minted copies carry the tool's value, not this one."""
    p = os.path.join(str(wz.WORKSPACE_DIR), "paper_skeleton", "versions.json")
    try:
        entries = json.load(open(p))
        return str(entries[-1]["v"]) if entries else None
    except (OSError, ValueError, KeyError, IndexError):
        return None


def asset(rel):
    """Page-relative src with a content hash, so a republished same-name file is
    never served from a stale cache."""
    p = os.path.join(HERE, rel)
    h = hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    return f"{rel}?v={h}"


# ------------------------------------------------------------- claim chain
# The four claims, verbatim. Claim 3's three reasons are sub-items of claim 3.
CLAIMS = [
    "3D generation methods like TRELLIS.2 can generate geometry and texture "
    "very well, but do not produce emissive objects.",
    "Generating emission is important.",
    ("It is hard for three reasons:", [
        "the data is scarce and noisy",
        "strength has to be determined, not only texture",
        "evaluation is hard, because an object cannot be judged alone and has "
        "to be placed in an environment",
    ]),
    "We fine-tune SegviGen for emissive mask detection and use the masked "
    "albedo as the emissive texture.",
]


def claim_chain():
    items = []
    for c in CLAIMS:
        if isinstance(c, tuple):
            lead, reasons = c
            subs = "".join(f"<li>{html.escape(r)}</li>" for r in reasons)
            items.append(f"<li>{html.escape(lead)}"
                         f'<ul class="skel-sub">{subs}</ul></li>')
        else:
            items.append(f"<li>{html.escape(c)}</li>")
    return ('<div class="skelwrap" id="claims"><div class="prose">'
            f'<ol class="skel">{"".join(items)}</ol></div></div>')


# ----------------------------------------------------------------- gallery
# Left to right: what goes in, the structure the model reads, the structure it
# has to produce, and what comes out.
STAGES = [
    ("input", "{sid}_input.png", "input: {what}"),
    ("pbr voxels", "{sid}_vox_pbr.png", "{what} as base-colour voxels"),
    ("emissive voxels", "{sid}_vox_mask.png",
     "{what}: the emissive voxels marked on the full occupancy"),
    ("emission", "{sid}_glow.png", "{what}, lit by its own emission"),
]


def gallery():
    """One four-panel row per shape.

    Each shape is its own four-column `fig_grid`, rather than all shapes in one
    grid, because the credit has to sit with ITS OWN object: a licence line
    under a shared figure would not say which model it credits. Panel labels are
    emitted on the FIRST row only, so the column headers appear once at the top
    the way method_matrix would do it, without giving every row a repeated
    header strip. Every grid is the same width with the same column count, so
    the columns line up down the page.
    """
    out = []
    for i, g in enumerate(GALLERY):
        panels = []
        for label, pat, alt in STAGES:
            src = asset("img/" + pat.format(sid=g["sid"]))
            panels.append((label if i == 0 else "", src,
                           alt.format(what=g["what"], sid=g["sid"])))
        grid = lp.fig_grid(panels, cols=4, caption_html=credit(g),
                           native_px=768, content="photo")
        out.append(with_viewer(grid, g))
    return '<div class="gal">' + "".join(out) + "</div>"


def with_viewer(grid, g):
    """Make the row's INPUT tile open the shape's preview GLB in the lightbox.

    Only the first tile is a click target: it is the one panel that shows the
    object as it actually is, so it is the one a 3D view of the object belongs
    on. The voxel panels are a different representation and the emission panel
    is a lighting result; hanging the same GLB off all four would say they are
    four views of one thing.
    """
    # alt must be the one gallery() already emitted for this tile, or the
    # replacement silently matches nothing
    label, pat, alt_tpl = STAGES[0]
    src = asset("img/" + pat.format(sid=g["sid"]))
    alt = html.escape(alt_tpl.format(what=g["what"], sid=g["sid"]), quote=True)
    old_img = f'<img loading="lazy" src="{src}" alt="{alt}">'
    if old_img not in grid:
        raise RuntimeError(f"input tile markup for {g['sid']} not found")
    glb_rel = f"glb/{g['sid']}.glb"
    if not os.path.exists(os.path.join(HERE, glb_rel)):
        raise RuntimeError(f"missing preview GLB for {g['sid']}")
    title = f"{g['what']} · {g['sid'][:8]}"
    new_img = (f'<img loading="lazy" class="v3d" src="{src}" alt="{alt}" '
               f'data-glb="{asset(glb_rel)}" '
               f'data-title="{html.escape(title, quote=True)}">')
    return grid.replace(old_img, new_img, 1)


# ------------------------------------------------------------- box figure
def box_figure():
    """Each object as the only light in a closed room.

    A SEPARATE figure rather than a fifth panel in the gallery row, for three
    reasons. A fifth column takes the gallery's tiles from 236px to 187px, and
    the voxel panels stop being readable as voxels at that size. The lighting
    setup here is not the next stage of the pipeline the gallery walks through,
    it is a different measurement of the same output, so putting it in that row
    would claim a sequence that does not exist. And the thing to look at is a
    soft gradient across a wall, which needs the pixels.
    """
    shapes = [g for g in GALLERY if g.get("has_box")]
    panels = [("", asset(f"img/{g['sid']}_box.png"),
               f"{g['what']}, lit only by its own emission") for g in shapes]
    # sentence-initial, so it keeps its capital; and it says "all" only when it
    # really is all of them, so a future dropped shape cannot go unmentioned
    word = NUMWORD.get(len(shapes), str(len(shapes)))
    subject = (f"All {word.lower()} objects above" if len(shapes) == len(GALLERY)
               else f"{word} of the objects above")
    caption = BOX_CAPTION.format(N=subject)
    return lp.fig_grid(panels, cols=3, caption_html=caption,
                       native_px=768, content="photo")


# ------------------------------------------------------- comparison figure
# Six columns. The middle three share one predicted density and differ only in
# WHERE they put it: uninformed, brightness-informed, and the model. That triple
# is the argument; all-emissive was dropped because "the task is not solved by
# lighting everything" is a fact a table states better than a panel.
COMPARE_COLS = ["input", "random", "albedo", "predicted", "ours (gt mask)",
                "ground truth"]
COMPARE_FILES = ["{sid}_input.png", "{sid}_random.png", "{sid}_albedo.png",
                 "{sid}_pred_72k.png", "{sid}_glow.png", "{sid}_true.png"]
SHORT_LABEL = {"b7709a651d144134a5babce33223380a": "animatronic",
               "658ecf9f837246509b0b1c4aa81e9e5b": "lit candles"}
IOU = json.load(open(os.path.join(HERE, "baseline_iou.json")))
IOU_BY_SID = {r["sid"]: r for r in IOU["shapes"]}


def iou_table():
    """The baseline against the model, per shape, with the lift's trust attached.

    A table rather than a sentence, because no aggregate carries this. The
    baseline's masks are authored in texture space and the model is scored in
    voxel space, so comparing them means lifting one into the other, and the
    lift turned out to inflate some shapes and deflate others. A single mean
    would hide exactly the thing a reader needs in order to weigh the number.
    """
    rows = []
    for r in IOU["shapes"]:
        # a shape whose lift moved the mask is dimmed rather than annotated: the
        # reader needs to know which rows to weigh, and a seventh column to say
        # so cost more width than the 820px measure has
        trust = r["trust"]
        cls = "" if trust == "ok" else ' style="opacity:.6"'
        name = SHORT_LABEL.get(r["sid"], r["what"])
        rows.append(
            f'<tr{cls}><td style="text-align:left">{html.escape(name)}</td>'
            f'<td>{r["baseline_iou"]:.4f}</td>'
            f'<td>{r["model_iou"]:.4f}</td>'
            f'<td>{r["model_coverage"]:.3f}</td>'
            f'<td>{r["lift_ratio"]:.2f}</td></tr>')
    # short headers on purpose: at 960px the table scrolled sideways inside its
    # own box even on a desktop, and the lift column is the caveat. A caveat you
    # have to scroll to reach is a caveat nobody reads.
    return lp.results_table(
        ["shape", "baseline", "model", "model cov.", "lift"], "".join(rows))


VENDING = "9418a924a50d44c186dd499006b62424"
HEADPHONE = "8f4c281aef1b4563b6103efbcd77fac1"
# Measured from the vending machine's own base-colour texture (scratchpad
# plateau.py, reproduced against the mask stage's achieved coverage): 16.07% of
# its texels hold luminance 0.737255 exactly, so the coverage a cut can reach
# jumps from 7.67% to 23.73% across that one value.
VEND_PLATEAU_SHARE = 0.1607
VEND_PLATEAU_ABOVE = 0.2373


# The predicted column's renders are produced by another workstream, so until
# they land the figure carries a labelled placeholder rather than a gap that
# reads as a broken image. Flipping this to False makes every predicted panel
# REQUIRED: the build fails on a missing one instead of quietly substituting the
# placeholder, which is the state the figure has to be in before it is minted.
# A placeholder that can outlive its reason is a defect, not a convenience.
PRED_FILE = "{sid}_pred_72k.png"

# Shapes reviewed by eye after the near-empty guard stopped the build on them.
# The band is NOT widened: each decision is recorded here with the measurement
# that triggered it and the badge the panel carries, so the guard stays tight for
# every other shape and a future near-empty case still has to be looked at.
# Coverages are three-ckpt's, from the voxel field the model predicts in.
PRED_NEAR_EMPTY = {
    # 6 voxels of 2.4 million. Not empty, and not visible at any size, so the
    # panel says what it is rather than rounding to "no emission", which would
    # be false, or to nothing at all, which would read as a failed render.
    "c1e3035d1ccb49df9c09aa86681faf30": (2.511e-06, "6 voxels"),
}


def pred_empty(sid):
    """True when the model predicted no emission anywhere on this shape.

    Read from the render's own sidecar rather than decided by looking at the
    panel. A dark panel and an empty prediction are different things, and only
    one of them is the model's result; the badge has to be carried by the
    number, or a dim-but-nonzero prediction would get mislabelled.

    An exact float test would also leave a gap: a prediction covering a
    thousandth of the surface renders as a black tile and would ship with no
    badge, reading as a failed render. So anything under a tenth of a percent
    stops the build instead, and gets looked at rather than guessed about. The
    outcome of that look is recorded in PRED_NEAR_EMPTY, per shape, rather than
    the band being widened until nothing trips it.

    Returns the badge text, or None when the panel needs no badge.
    """
    if sid in PRED_NEAR_EMPTY:
        return PRED_NEAR_EMPTY[sid][1]
    p = os.path.join(PRED_SRC, f"{sid}_stats.json")
    if not os.path.exists(p):
        return None
    frac = json.load(open(p)).get("area_lit_frac")
    if frac is None:
        raise SystemExit(f"{sid}: predicted sidecar carries no area_lit_frac")
    if 0.0 < frac < 1e-3:
        raise SystemExit(
            f"{sid}: predicted area_lit_frac is {frac:g}, too small to read as "
            f"emission and not zero either; look at the panel, then add it to "
            f"PRED_NEAR_EMPTY with the badge it should carry. Do not widen the "
            f"band")
    return "no emission" if frac == 0.0 else None


def comparison():
    rows = []
    for g in COMPARE:
        cells = []
        for p in COMPARE_FILES:
            rel = "img/" + p.format(sid=g["sid"])
            if not os.path.exists(os.path.join(HERE, rel)):
                if p != PRED_FILE or not PRED_PENDING:
                    raise SystemExit(
                        f"comparison figure is missing {rel}; only the "
                        f"predicted column may be absent, and only while "
                        f"PRED_PENDING is set")
                cells.append({"placeholder": "72k model", "sub": "rendering"})
                continue
            cell = {"img": asset(rel), "alt": g["what"]}
            # an empty prediction is labelled on the panel, so a reader cannot
            # mistake the model predicting nothing for a render that failed.
            # Kept short: the badge sits inside a 131px tile, and at .7rem mono
            # a longer string becomes a bar across the panel.
            if p == PRED_FILE:
                badge = pred_empty(g["sid"])
                if badge:
                    cell["badge"] = badge
            cells.append(cell)
        # the row-label gutter is 26px wide and the type is set for a 131px
        # cell, so long names are truncated here rather than in the gallery,
        # where the full name is the one the credit line has to match
        rows.append((SHORT_LABEL.get(g["sid"], g["what"]), cells))
    # not "compare": that id belongs to the section this figure sits in, and two
    # elements sharing one id is invalid and breaks anchor links and querying
    return lp.method_matrix(COMPARE_COLS, rows, caption_html=compare_caption(),
                            native_px=768, content="photo", id="compare-fig",
                            page_inner=PAGE_INNER)


def credit(g):
    """The licence credit for one shape: author, licence, link to the source.

    Required, not decorative. Every model here is somebody's Creative Commons
    work, and the licence obliges attribution wherever the work is shown, so it
    sits on the item rather than in a collective line at the page foot.
    """
    for key in ("author", "license", "url"):
        if not g.get(key):
            raise RuntimeError(f"{g['sid']}: no {key}; a shape cannot ship "
                               f"without its attribution")
    vm = VOXEL_META[g["sid"]]
    return (f'{html.escape(g["author"])} &middot; {html.escape(g["license"])} '
            f'&middot; <a href="{html.escape(g["url"], quote=True)}" '
            f'target="_blank" rel="noopener">Sketchfab</a>'
            f' &middot; voxel panels: {vm["res_display"]}&sup3; grid, '
            f'{vm["n_cells_display"]:,} cells occupied')


CAPTION = (
    "<b>{N} objects, one per row: the object, the structure the model reads, the "
    "structure it has to produce, and the result.</b> Columns, left to right: the "
    "object under studio light; its surface as base-colour voxels; the same "
    "occupancy in grey with the emissive voxels marked in orange; and the object "
    "in a dark room. That last panel keeps one dim key light so the silhouette "
    "stays readable, which means the shell is not lit by the emission alone; the "
    "figure below removes every other source. The emissive region is the "
    "ground-truth mask that ships with the shape, not our model's prediction, so "
    "these show what the method's output looks like rather than what the model "
    "currently produces. Inside that mask the emission is the object's own "
    "albedo, at one emission strength shared by every shape; outside it the "
    "emission is exactly zero. Voxels are drawn at their true 512&sup3; grid, "
    "the same resolution this checkpoint trains on; the geometry is native at "
    "that resolution, and the base colour and the emissive label are Dongchen's "
    "256&sup3; scan, nearest-neighbor upsampled onto it, so the surface shape is "
    "the real fine grid but a colour or emissive value can still repeat across "
    "several neighbouring cells that shared one coarser answer underneath. "
    "Each row's credit line states its own measured cell count. Rows are "
    "ordered by emissive coverage, {lo} to {hi} percent of surface area. Click "
    "an object to open it in 3D."
)

AGG = IOU["aggregates"]
IOU_CAPTION = (
    f"<b>A brightness threshold beats the model on {AGG['baseline_wins']} of "
    f"these {AGG['n']} shapes.</b> Mask IoU, the same measure the model is "
    "trained against, computed by one evaluator so the two columns are the same "
    f"quantity. Across the seven the baseline averages "
    f"{AGG['baseline_mean_all7']:.2f} against the model's "
    f"{AGG['model_mean_all7']:.3f}, and {AGG['baseline_mean_excl_headphone']:.2f} "
    f"against {AGG['model_mean_excl_headphone']:.3f} with the headphone stand "
    "set aside. "
    "<b>Read the rows, not the average.</b> Two things make any single figure "
    "misleading here. The headphone stand carries no albedo texture, so a "
    "luminance cut selects whole materials there and coverage matching hands the "
    "baseline the two the asset really emits from; it nearly doubles the mean by "
    "construction. And the baseline's masks are authored in texture space while "
    "the model is scored in voxels, so comparing them at all means lifting one "
    "into the other, and that lift inflates some shapes and deflates others "
    "rather than biasing everything one way. The lift column says which rows to "
    f"trust: only {AGG['n_lift_ok']} of {AGG['n']} sit close enough to 1 to take "
    "at face value, and the dimmed rows are the ones where the lift moved the "
    "mask enough to matter. What survives all of it is the ranking, because "
    "deflating the baseline by the largest distortion measured anywhere still "
    "leaves it ahead. "
    "<b>Two rows are worth more than the average.</b> On the jack-o'-lantern the "
    "model lights 99.3 percent of the object and the baseline lights the wrong "
    "third, so both are wrong and only one of them looks it. On the sci-fi "
    "weapon the two scores are 0.026 and 0.0002, small enough that the honest "
    "reading is that neither method puts the light in the right place, and the "
    "model's predicted region barely intersects the true one at all rather than "
    "being approximately right and blurry. "
    "<b>The model's column is bimodal.</b> It predicts 99.3 percent of one shape "
    "and nothing whatever on four others. A mean near 0.1 is what "
    "everything-or-nothing looks like after averaging, which is why the rows are "
    "here and the mean is not the headline."
)

BOX_CAPTION = (
    "<b>{N}, now as the only light in a closed room.</b> "
    "Every lamp is deleted and the world is set to zero, so the only photons in "
    "the scene leave the object: the pool on the floor, the wash up the walls "
    "and the colour bleeding onto neutral grey are all light this object cast. "
    "The room is a five-sided box of matte 0.80-albedo grey, open toward the "
    "camera, on the same camera as every other panel of that shape, and the "
    "light is followed through sixteen diffuse bounces, because in a room lit "
    "only by the object the ambient fill IS the later bounces. The view is "
    "exposed a stop and a half up, the way an eye adapts on walking into a "
    "dark room, so the emitters clip to white and the room stays readable. "
    "<b>Brightness here is ours, not the asset's.</b> The gallery's renders keep "
    "a key light, so the emission strength was a stylistic choice; with the "
    "object as the sole source, how far its light reaches is the subject, and "
    "every shape is emitting at one strength we picked rather than at anything "
    "recovered from the data. That is the third difficulty on this page, in a "
    "picture: strength has to be determined, and nothing here determines it. "
    "Compare the candles, which fill the room, against the jack-o'-lantern, "
    "whose emitter sits inside a closed shell: its own surface is readable only "
    "by light that has bounced off the walls and come back, and it stays the "
    "dimmest room here by an order of magnitude."
)

def compare_caption():
    """The comparison caption, with the baseline's coverage numbers measured.

    Written as a function because three of its sentences quote figures the
    albedo baseline produced, and a caption that quotes a number it does not
    read is a caption that goes stale without anything failing.
    """
    err = {sid: abs(s["achieved_coverage"] - s["target_coverage"])
           for sid, s in HEUR.items()}
    # counted from the renders' sidecars, so the sentence cannot drift from the
    # panels; the badge on those cells comes from the same test
    n_empty = sum(pred_empty(g["sid"]) is not None for g in COMPARE)
    if PRED_PENDING:
        empty_note = ("its panels are still rendering, and the count of shapes "
                      "it predicts nothing on is stated once they are in.")
    elif n_empty:
        empty_note = (f"on {NUMWORD.get(n_empty, n_empty).lower()} of the "
                      f"{NUMWORD.get(len(COMPARE), len(COMPARE)).lower()} "
                      "shapes it predicts no emission anywhere, and those "
                      "panels are labelled. That is its output, not a render "
                      "that failed.")
    else:
        empty_note = "it predicts some emission on every shape here."
    vend = HEUR[VENDING]
    close = sorted(v for sid, v in err.items() if sid != VENDING)
    return (
        "<b>The gap between the last two columns is the question this method "
        "turns on.</b> The five emission panels are the same shape on the same "
        "camera in the same dark room under the same key light, differing only "
        "in which surface emits and in what colour. The <i>input</i> panel is "
        "the one exception, and deliberately: it is the same camera under a "
        "neutral studio light with no emission at all, because what it has to "
        "show is the object. "
        "<i>Random</i> and <i>albedo</i> are each handed the shape's own "
        "ground-truth emissive coverage and differ only in where they put it: "
        "random places it in blocky patches, albedo puts it on the brightest "
        "texels. <i>Predicted</i> is the emission-mask model's own output at "
        f"72k shapes, rendered under the same treatment; {empty_note} "
        "<i>Ours</i> is the "
        "albedo restricted to the ground-truth mask, exactly as in the gallery "
        "above. <i>Ground truth</i> is the asset's own emissive texture. "
        "<br><br>"
        "Ours and ground truth agree wherever the emissive texture is the base "
        "colour and part company wherever it is not: the vending machine's "
        "authored emission is a plain white panel while its albedo carries the "
        "artwork, and the headphone stand's light strip is authored violet over "
        "white plastic. "
        "<br><br>"
        "<b>Read the predicted column against the last one, not on its own.</b> "
        "The model fails in two opposite directions and both are visible here. "
        "It predicts nothing at all on several shapes. On the headphone stand "
        "and the jack-o'-lantern it does the reverse, lighting 144 and 16 times "
        "the area that really emits, and because both of those assets are "
        "near-white where they emit, <b>the two most wrong panels are also the "
        "two brightest</b>. Scanning the column for whichever panel looks most "
        "lit gives the answer backwards. The sci-fi weapon is the row worth "
        "spending time on: the model puts a plausible quantity of light on it, "
        "a tenth of the surface against a true fifteenth, and still scores "
        "0.0002, because the light is in the wrong place. That failure is "
        "invisible in any average and it is the one this method has to solve. "
        "<br><br>"
        "Two things about the albedo column, both of which say more about the "
        "assets than about the baseline. Its luminance cut is solved per shape "
        "by bisection against the shape's own coverage, and it lands within "
        f"{max(close) * 100:.1f} percentage points on "
        f"{NUMWORD[len(close)].lower()} of the "
        f"{NUMWORD[len(COMPARE)].lower()}; on the "
        f"vending machine it stops at {vend['achieved_coverage'] * 100:.1f} "
        f"percent against a {vend['target_coverage'] * 100:.1f} percent target, "
        f"because {VEND_PLATEAU_SHARE * 100:.1f} percent of that texture sits at "
        f"one single luminance value and the reachable coverage steps from "
        f"{vend['achieved_coverage'] * 100:.1f} straight to "
        f"{VEND_PLATEAU_ABOVE * 100:.1f} percent across it, with no cut in "
        "between. And the headphone stand carries no albedo texture at all: its "
        f"{len(HEUR[HEADPHONE]['materials'])} materials each hold one constant "
        "colour, so a luminance cut selects whole materials rather than regions, "
        "and matching the coverage exactly picks out the two materials the asset "
        "really does emit from. On a shape built that way the baseline is a "
        "material classifier, and coverage matching makes it a well-informed one."
    )

NUMWORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six",
           7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven",
           12: "Twelve"}


def build():
    assets_dir = os.path.join(WEB, "assets")

    fracs = [g["area_lit_frac"] for g in GALLERY]
    caption = CAPTION.format(N=NUMWORD.get(len(GALLERY), str(len(GALLERY))),
                             lo=f"{min(fracs) * 100:.1f}",
                             hi=f"{max(fracs) * 100:.1f}")

    hero = lp.hero_header(
        f"lightgen &middot; emissive generation &middot; {PAGE_DATE}",
        "Generating emissive objects",
    )

    body = [
        claim_chain(),
        lp.section_v2("gallery", "", "What the output looks like",
                      gallery() + lp.chartnote(caption)),
        lp.section_v2("box", "", "What the light actually does to a room",
                      box_figure()),
        lp.section_v2("compare", "", "Where mask times albedo stops being the "
                                     "real emission",
                      comparison() + iou_table() + lp.chartnote(IOU_CAPTION)),
    ]

    apx = lp.appendix("Sources", [
        "Shapes: TexVerse-1K, "
        "<code>/cs/3dlg-falas/datasets/TexVerse-1K/glbs/glbs_1k/</code>, sampled "
        "across object category and emissive coverage from the held-out "
        "<code>val_72k</code> and <code>test_72k</code> splits of "
        "<code>segvigen_emissive/dataset_direct</code>. Each shape is a Sketchfab "
        "model; the author, licence and link under every panel are read per shape "
        "from <code>TexVerse/metadata.json</code> at build time, and a shape whose "
        "licence is not plain CC Attribution is not published here.",
        "Renders: Cycles at 768&times;768. The dark-room panels use 256 samples, "
        "one dim key light and a near-black world; the box panels use 1,024 "
        "samples with every lamp deleted and the world at zero, so the object is "
        "the only emitter and the build asserts it before rendering, 32 total and "
        "16 diffuse bounces against Cycles' defaults of 12 and 4, and the Filmic "
        "view transform at +1.5 stops. Bloom is a compositor fog glow, and the "
        "two sets of panels carry different ones because they were tuned "
        "separately: radius 7, threshold 1.0, mix &minus;0.45 on the box panels, "
        f"and radius {REFERENCE_TREATMENT['bloom_size']}, threshold "
        f"{REFERENCE_TREATMENT['bloom_threshold']}, mix "
        f"&minus;{abs(REFERENCE_TREATMENT['bloom_mix'])} on the key-lit ones. "
        "The threshold is deliberately left low in both, because raising it "
        "silences the bloom on a dim shape long before it tames a bright one. "
        "AgX view transform on the key-lit panels, "
        "one three-quarter camera rule solved per shape from its bounding box, "
        "shared by every panel of that shape. The mask is each asset's own "
        "emissive channel, binarized at any nonzero value; the emission is that "
        "mask times the albedo feeding the same material. Generator: "
        "<code>yanxg_scratch/paper_v3/render_emissive.py</code>.",
        "Voxels: the 256&sup3; surface bake at "
        "<code>uv_voxel_pipeline/out_uv_voxel_74k/&lt;sid&gt;/{pbr_voxels_256, "
        "emission_voxels_256}</code>, read with <code>o_voxel.io.read_vxz</code> "
        "and pooled to 64&sup3; for display. The model's own input is the "
        "512&sup3; upsample of the same bake. Generators: "
        "<code>extract_voxels.py</code> and <code>render_voxels.py</code>.",
        "Lighting. Every key-lit panel uses a key of 8, and it used to use 20. "
        "That is not a preference about how dark the figure should look. The "
        "compositor's fog glow has a fixed threshold of 1.0 in linear space, and "
        "at a key of 20 a pale non-emissive surface clears it: on a shape whose "
        "prediction was empty the glow fired on 108,865 pixels, against 123,948 "
        "on the same shape's ground truth. The glow was reporting the lamp "
        "rather than the object, in a figure whose subject is where light comes "
        "from. At a key of 8 nothing non-emissive reaches the threshold, so the "
        "glow fires on no pixels of the empty panel and on 50,458 of the ground "
        "truth, which makes it a property of emission rather than of brightness. "
        "The measurement that forced this: an empty prediction sat 10.1 levels "
        "from the ground truth and 22.1 from its own unlit input, so a panel "
        "containing nothing was closer to the answer than to the object. "
        "Measured at the emissive texels rather than averaged over the material, "
        "ten of the eleven shapes carry emission on light-coloured albedo, so "
        "this was the normal case and not a few rows to caption around.",
        "Baselines. The random baseline draws a blocky mask over every material "
        "at the shape's own measured emissive coverage. The albedo baseline "
        "thresholds base-colour luminance and solves the threshold per shape by "
        "bisection so its coverage matches that same figure, which removes "
        "coverage as a source of difference and leaves only placement; the cut "
        "is evaluated in both its inclusive and its exclusive form and the "
        "closer one is kept, because on a shape whose materials are constant "
        "colours a strict cut can land on a tie and light nothing. Neither is a "
        "trained model. An all-emissive baseline was rendered and dropped from "
        "the figure: that lighting everything is not a solution is a claim a "
        "number states better than a panel.",
        "Substituting a predicted mask. The renderer takes a directory of "
        "per-material masks and replaces the asset's emissive channel with "
        "them, and two of its behaviours are deliberate. Every material is "
        "considered, not only the ones the asset emits from, so a prediction "
        "can light a surface that is dark in the ground truth. And a material "
        "the prediction does not select is switched off even where the asset "
        "does emit, so a miss reads as a miss rather than as the asset showing "
        "through. Masks are keyed by material slot index and the loader "
        "requires the producer to ship the slot's material name alongside it, "
        "which it checks against the loaded object before applying anything: a "
        "bare index range check passes when a producer keys by glTF material "
        "index instead, and the masks then land on the wrong materials and "
        "render as a model error.",
    ])

    page_html = lp.page(
        title="Generating emissive objects (lightgen)",
        header_html=hero,
        body_sections=body + [apx],
        assets_rel=SITE_ASSETS,
        assets_dir=assets_dir,
        theme="v3",
        tree_html=wz.tree_html(active_href=LIVING_HREF),
        nav_title="Emissive objects",
        outline_entries=[{"id": i, "label": lab} for i, lab in OUTLINE],
        version_slot=lp.v3_version_slot(version=latest_version(), living=True,
                                        manifest=VERSIONS_URL),
        needs_katex=False,
        extra_head=(f'<link rel="icon" href="{FAVICON}">'
                    + lightbox_head() + EXTRA_CSS),
        extra_body_end=lightbox_modal(),
    )

    # ZONE BOUNDARY LAW: nothing in the workspace zone may link to the operator
    # console. The console lives at the site ROOT, so this checks its real
    # hrefs rather than a "/console/" substring. Fail the build, never publish.
    violations = wz.console_links_in(page_html)
    if violations:
        sys.exit(f"ZONE-LINK GUARD FAILED: page links to the console: {violations}")

    out = os.path.join(HERE, "index.html")
    with open(out, "w") as f:
        f.write(page_html)
    print(f"wrote {out} ({len(page_html)} bytes, {len(GALLERY)} examples)")
    print("  zone-link guard: clean")

    publish_assets(assets_dir)
    print(f"assets published -> {assets_dir}")


# --------------------------------------------------------- the 3D lightbox
# Page-local, following web/_preview/emission_vae/. The shared v1 stylesheet and
# ui.js no longer carry the .v3d / #mv3d rules that tools/xgpage_ext.py's helpers
# assume: web/assets is a published copy of the xgpage package's assets, and the
# package has never had them, so a page that only emits the markup gets a
# thumbnail that does nothing when clicked (measured on the first publish of this
# page: every one of the twelve tiles was inert). Shipping the behaviour with the
# page is what makes it work.
LIGHTBOX_CSS = """
img.v3d { cursor: zoom-in; }
.mv3d-modal { position: fixed; inset: 0; z-index: 90; display: none;
  background: color-mix(in srgb, var(--bg2) 90%, transparent);
  backdrop-filter: blur(4px);
  flex-direction: column; align-items: center; justify-content: center;
  padding: 2vh 2vw; }
.mv3d-modal.open { display: flex; }
.mv3d-modal model-viewer { width: min(92vw, 860px); height: min(80vh, 860px);
  background: #0b0b0d; border: 1px solid var(--line); border-radius: 12px; }
.mv3d-bar { display: flex; gap: 1rem; align-items: center; color: var(--ink);
  font-size: .84rem; margin-bottom: .6rem; width: min(92vw, 860px); }
.mv3d-bar #mv3d-title { font-family: ui-monospace, Menlo, monospace;
  font-size: .76rem; color: var(--ink-2); overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.mv3d-dl { color: var(--accent-ink); font-size: .76rem; white-space: nowrap; }
.mv3d-close { background: var(--accent); color: #fff; border: 0; border-radius: 6px;
  padding: .35rem .8rem; cursor: pointer; font-size: .78rem; white-space: nowrap; }
@media (max-width: 760px) {
  .mv3d-modal model-viewer { width: 96vw; height: 74vh; }
}
"""

LIGHTBOX_JS = """
/* Click a .v3d thumbnail -> open its data-glb in the #mv3d model-viewer modal.
   The src is set on open and removed on close, so nothing downloads until a
   thumbnail is clicked and only one WebGL context is ever live. */
(function () {
  function init() {
    var modal = document.getElementById('mv3d');
    if (!modal) return;
    var mv = document.getElementById('mv3d-viewer');
    var titleEl = document.getElementById('mv3d-title');
    var dl = document.getElementById('mv3d-dl');
    function open(glb, ttl) {
      if (!glb) return;
      mv.setAttribute('src', glb);
      titleEl.textContent = ttl || '';
      dl.setAttribute('href', glb);
      modal.classList.add('open');
    }
    function close() {
      modal.classList.remove('open');
      mv.removeAttribute('src');
    }
    document.querySelectorAll('img.v3d').forEach(function (im) {
      im.addEventListener('click', function () {
        open(im.dataset.glb, im.dataset.title);
      });
    });
    modal.addEventListener('click', function (e) { if (e.target === modal) close(); });
    document.getElementById('mv3d-close').addEventListener('click', close);
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && modal.classList.contains('open')) close();
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else { init(); }
})();
"""


def lightbox_head():
    return (f'<script type="module" src="{SITE_ASSETS}/model-viewer.min.js">'
            f'</script><style>{LIGHTBOX_CSS}</style>')


def lightbox_modal():
    """The modal shell plus its behaviour. Exposure is set well below the
    default: these assets carry an emission strength of four, so a neutral
    exposure washes the lit surfaces up to meet the emissive ones and the glow
    stops reading as a light source."""
    return (
        '<div id="mv3d" class="mv3d-modal">'
        '<div class="mv3d-bar"><span id="mv3d-title"></span>'
        '<a id="mv3d-dl" class="mv3d-dl" href="#" download>download GLB</a>'
        '<button id="mv3d-close" class="mv3d-close" type="button">'
        '&#10005; close</button></div>'
        '<model-viewer id="mv3d-viewer" camera-controls auto-rotate '
        'rotation-per-second="16deg" interaction-prompt="none" exposure="0.45" '
        # the render's own three-quarter view: azimuth 38 degrees, and a polar
        # angle of 90 minus the render's 17 degree elevation
        'camera-orbit="38deg 73deg auto" '
        'tone-mapping="neutral" shadow-intensity="0.25" shadow-softness="1">'
        '</model-viewer></div>'
        f'<script>{LIGHTBOX_JS}</script>')


# ------------------------------------------------------------ page-local CSS
EXTRA_CSS = """
<style>
/* theme.css sets table.results to min-width:960px, which suits a wide results
   table in v2's 972px column. This page is v3 at 820px with a five-column
   table, so that floor forced 140px of sideways scrolling on a desktop and put
   the lift column, which is the caveat, off the right edge. A caveat you have to
   scroll to reach is a caveat nobody reads. Overridden here rather than in the
   shared theme, since other pages rely on the floor. */
.xg3 table.results{min-width:0}

/* The .skel claim chain, as used by web/_preview/paper_skeleton/. Four claims,
   one counter, sub-items for claim 3's three reasons. */
.skelwrap{counter-reset:skl}
.skel{counter-reset:none;list-style:none;padding:0;margin:0}
.skel > li{counter-increment:skl;position:relative;padding:.62em 0 .62em 2.1em;
  border-top:1px solid var(--line);font-size:1.05em}
.skel > li:first-child{border-top:0}
.skel > li::before{content:counter(skl);position:absolute;left:0;top:.78em;
  font:11px ui-monospace,Menlo,monospace;color:var(--ink-2)}
.skel-sub{list-style:none;padding:0;margin:.45em 0 0}
.skel-sub li{position:relative;padding:.16em 0 .16em 1.1em;color:var(--ink-2)}
/* a middle dot, not a dash: the project register bans em dashes in rendered
   text, and a CSS-escaped one is still an em dash on the page. */
.skel-sub li::before{content:"\\00B7";position:absolute;left:.2em;top:.16em}

/* The gallery is the page. One four-column grid per shape, stacked; the tile
   background matches the dark-room renders' own background (measured: rgb
   22,22,22) so the three transparent panels and the opaque emission panel read
   as one continuous strip rather than four cards on two different greys. */
.gal{display:flex;flex-direction:column;gap:18px;margin:8px 0 6px}
.xg2 .gal .fig-grid[data-cols="4"]{gap:8px}
.xg2 .gal .fig-grid figure img{background:#161616;border-radius:3px;
  display:block;width:100%}
/* Four panels is the figure's whole point, so they stay four across as long as
   a tile clears ~86px; below that the row becomes two columns of two, which
   keeps input next to its voxels and the mask next to the result. */
@media (max-width:560px){
  .xg2 .gal .fig-grid[data-cols="4"]{grid-template-columns:repeat(2,1fr)}
}
/* Column headers on the first row only: keep the label row's height on every
   other row so the tiles stay on one baseline down the page. */
.xg2 .gal .fig-grid .panel-label:empty::after{content:"\\00a0"}

/* Per-item credit. Quiet enough not to compete with the renders, large enough
   to read: .76rem sits above the skill's 12.5px secondary-text floor. */
/* the engine puts 14px above a fig_grid caption; a credit line is not a
   figure caption and should sit close to the row it credits */
.xg2 .gal figure[style*="margin-top"]{margin-top:6px !important}
.xg2 .gal figcaption{margin:0;font-size:.76rem;line-height:1.4;
  color:var(--ink-2);max-width:none}
.xg2 .gal figcaption a{color:var(--accent-ink);white-space:nowrap}

/* The comparison matrix's tiles are renders on their own dark background. */
.xg2 .mm .mm-cell{background:#161616}

/* The box figure: three across, so a wall gradient gets enough pixels to read
   as a gradient. These renders are opaque and fill their tile, so they need no
   tile background of their own. */
.xg2 #box .fig-grid[data-cols="3"]{gap:10px}
.xg2 #box .fig-grid figure img{border-radius:3px;display:block;width:100%}
@media (max-width:640px){
  .xg2 #box .fig-grid[data-cols="3"]{grid-template-columns:repeat(2,1fr)}
}
</style>
"""


if __name__ == "__main__":
    build()
