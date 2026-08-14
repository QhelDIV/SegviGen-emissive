#!/usr/bin/env python3
"""Build the "Rendering setups" page on the xgpage v3 workspace shell.

The project makes figures under five different lighting setups. This page names
each one, shows what it looks like on ONE asset, and says when to reach for it.
The repo-side companion, with every flag and a file:line reference for every
parameter, is RENDERING.md at the repo root.

The opening figure is the whole vocabulary on ONE asset (TEASER_SID below):
four stills plus the emission sweep as a short looping video. The box, key-lit
and studio stills are copies of already-published renders; the condition image
and the segmentation panel were produced for this page, and so were the video
frames. make_video.py in this directory encodes the frames; it is a separate
step because encoding is local and cheap while the frames are a cluster job.

The first teaser candidate was the sci-fi weapon 51a60b16. Its zero-shot
segmentation came back as one flat colour over almost the whole object, so it
could not carry the segmentation cell; the asset was switched and the weapon's
panels were kept as the honest illustration of that failure in section 05.

Run: .venv_console/bin/python web/_preview/rendering/build.py [--publish]
"""
import hashlib
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WEB)
sys.path.insert(0, os.path.join(REPO, "tools"))
import workspace_zone as wz  # noqa: E402

import xgpage as lp  # noqa: E402
from xgpage.publish import publish_assets  # noqa: E402

SITE_ROOT = "/projects/omages/yanxg/lightgen"
SITE_ASSETS = f"{SITE_ROOT}/assets"
FAVICON = f"{SITE_ROOT}/assets/images/favicon.png"

PAGE_SLUG = "rendering"
PAGE_HREF = f"{wz.WORKSPACE_URL}/{PAGE_SLUG}/index.html"
PUBLISH_DIR = os.path.join(str(wz.WORKSPACE_DIR), PAGE_SLUG)
PAGE_DATE = "2026-08-09"

IMG = os.path.join(HERE, "img")
SWEEP_URL = f"{wz.WORKSPACE_URL}/render_sweep/index.html"

CLUSTER = "/project/3dlg-hcvc/omages/yanxg_scratch"

# The one asset the opening figure shows under all five setups. It has to carry
# box and key-lit stills already (only the strength-ladder shapes do), and its
# zero-shot segmentation has to be legible as a part decomposition rather than
# one flat colour, which is what disqualified the first choice. Everything the
# teaser pulls is keyed off these three constants, so changing the asset is a
# one-line edit plus a re-render of the sweep frames.
TEASER_SID = "51a60b164e874bf891597d9c6c1941af"
TEASER_WHAT = "sci-fi weapon"
LADDER = "web/_preview/strength_ladder/img"

# The four other emissive assets whose zero-shot segmentation was checked while
# choosing the teaser. They are on the page as evidence, in section 05: the
# pretrained model does not reconstruct this asset family, and one failure could
# be an unlucky shape while five is a pattern.
FAMILY = [
    ("48af42db48c44cd9bfab32bbb057a39c", "pumpkin", "jack-o'-lantern"),
    ("8f4c281aef1b4563b6103efbcd77fac1", "headphone", "headphone stand"),
    ("658ecf9f837246509b0b1c4aa81e9e5b", "candles", "three lit candles"),
    ("9418a924a50d44c186dd499006b62424", "vending", "vending machine"),
]

# (destination under img/, source path, one-line provenance). A source starting
# with "/" is absolute; anything else is relative to the repo root. stage_images()
# copies these and check_assets() proves each destination exists and is non-empty,
# so a renamed upstream file fails the build instead of shipping a broken tag.
STAGED = [
    ("cmp_box.png", f"{LADDER}/{TEASER_SID}_box_mask_s4.png",
     "box render, ground-truth mask x albedo, emission strength 4"),
    ("cmp_keylit.png", f"{LADDER}/{TEASER_SID}_gt_s4.png",
     "key-lit emission render, same asset, same mask, same strength"),
    ("cmp_input.png", f"{LADDER}/{TEASER_SID}_lit.png",
     "input render, studio reference panel written by the same run"),
    ("teaser_cond.png", f"{CLUSTER}/render_doc/cond_img/{TEASER_SID}.png",
     "input render, condition image, rendered for this page by job 242221"),
    ("teaser_seg.png", f"{CLUSTER}/render_doc/seg_img/{TEASER_SID}_va.png",
     "segmentation render, camera A, prediction from job 242225, rendered by job 242226"),
    ("seg_input.png",
     "web/_preview/fullseg_19/img/f6314284e0e84a14ba466613ae776110_input.png",
     "input render, the condition image fed to the model"),
    ("seg_parts.png",
     "web/_preview/fullseg_19/img/f6314284e0e84a14ba466613ae776110_seg_a.png",
     "segmentation render, camera A, the model's own part colours"),
] + [
    (f"fam_{tag}_{kind}.png",
     f"{CLUSTER}/render_doc/{'cond_img/' + sid + '.png' if kind == 'in' else 'seg_img/' + sid + '_va.png'}",
     f"{'input render' if kind == 'in' else 'segmentation render'} of the {what}, job 242228")
    for sid, tag, what in FAMILY
    for kind in ("in", "seg")
] + [
    (f"sweep_s{k}.png", f"{LADDER}/{TEASER_SID}_box_mask_s{k}.png",
     f"emission sweep rung, strength {k}, box render")
    for k in (0, 1, 4, 8, 16)
]

# Produced locally by make_video.py from the cluster frames of job 242222, so
# they are not staged from anywhere; they only have to be present.
VIDEO_ASSETS = ["sweep.webm", "sweep.mp4", "sweep_poster.png"]

OUTLINE = [
    {"id": "default", "label": "The default rule"},
    {"id": "box", "label": "Box render"},
    {"id": "keylit", "label": "Key-lit emission render"},
    {"id": "input", "label": "Input render"},
    {"id": "seg", "label": "Segmentation render"},
    {"id": "sweep", "label": "Emission sweep"},
    {"id": "provenance", "label": "Where this came from"},
]

EXTRA_CSS = """
<style>
/* theme.css carries v1-era floors (table.results{min-width:960px},
   td.rowhead{min-width:210px}) that every v3 page inherits, so a short
   parameter table would scroll internally for no reason. Lifted here, as the
   render_sweep and paper_skeleton pages already do. */
.xg2 table.results { min-width: 0; }
.xg2 table.results td.rowhead { min-width: 0; }
.xg2 table.results td, .xg2 table.results th { padding-left: 8px; padding-right: 8px; }
/* The settings column holds short prose, not numbers, so the engine's centered
   numeric default reads as a ragged middle column. Left-aligned here only. */
.xg2 table.results td + td, .xg2 table.results th + th { text-align: left; }

/* fig_row's panel labels butt straight against the paragraph above them.
   Give the row the breathing space a figure gets elsewhere. */
.xg3 .fig-pair, .xg3 .fig-triple { margin-top: 20px; }

/* theme3.css releases .prose's max-width so the 820px column IS the measure,
   but leaves .chartnote on v2's narrower cap, which reads as a stray indent
   next to the prose above it. Released to one edge, same fix render_sweep
   applies. */
.xg3 .chartnote { max-width: none; }

/* The dark renders are shown well below their 768px native size; the tile
   carries the comparison, the link carries the detail. */
.mm-cell a { display: contents; }
.mm-cell a:hover img { outline: 2px solid var(--accent); outline-offset: -2px; }

/* A video standing in for an image inside a matrix cell has to sit in the
   cell exactly as the engine's own <img> does, or the tile geometry shifts
   under it and the strip stops reading as one row. */
.mm-cell video { width: 100%; height: 100%; object-fit: contain; display: block; }

/* The standalone sweep video in its own section: sized like a figure, on the
   same constant light figure background every other figure uses, so a dark
   render does not float on the page background. */
.xg3 figure.xg-video { margin: 20px auto 0; }
.xg3 figure.xg-video video { display: block; width: 100%; height: auto;
  background: var(--fig-bg, #fff); border: 1px solid var(--line);
  border-radius: 8px; }

/* Scroll room past the last section: xg3.js's scrollspy marks the nearest
   [id] above a 110px line, and the final section is too short to reach it. */
.xg3 .v3-main .page::after { content: ""; display: block; height: 40vh; }
</style>
"""


# ------------------------------------------------------------------- staging
def _src_path(src):
    return src if src.startswith("/") else os.path.join(REPO, src)


def stage_images():
    os.makedirs(IMG, exist_ok=True)
    for dest, src_rel, _why in STAGED:
        src = _src_path(src_rel)
        if not os.path.exists(src):
            sys.exit(f"STAGING FAILED: source missing for {dest}: {src}")
        shutil.copy2(src, os.path.join(IMG, dest))
    print(f"staged {len(STAGED)} images -> {IMG}")


def check_assets():
    """Every asset the page references exists in img/ and is non-empty."""
    problems = []
    for name in [d for d, _s, _w in STAGED] + VIDEO_ASSETS:
        p = os.path.join(IMG, name)
        if not os.path.exists(p):
            problems.append(f"{name}: missing"
                            + (" (run make_video.py first)" if name in VIDEO_ASSETS else ""))
        elif os.path.getsize(p) == 0:
            problems.append(f"{name}: zero bytes")
    return problems


def src(name):
    """Content-hashed src for a file under img/."""
    h = hashlib.md5(open(os.path.join(IMG, name), "rb").read()).hexdigest()[:8]
    return f"img/{name}?v={h}"


# ------------------------------------------------------------------ page bits
def ptable(rows):
    body = "".join(
        f'<tr><td class="rowhead">{k}</td><td>{v}</td></tr>' for k, v in rows)
    return lp.results_table(["parameter", "setting"], body)


def linkify_cells(html):
    """Make every matrix tile click through to its full-resolution render.
    Videos are skipped: the tile already plays, and a click-through to a raw
    video file is worse than the tile."""
    return re.sub(
        r'(<div class="mm-cell">)(<img [^>]*?src="([^"]+)"[^>]*>)',
        lambda m: f'{m.group(1)}<a href="{m.group(3)}" target="_blank" '
                  f'rel="noopener">{m.group(2)}</a>', html)


def video_tag(*, poster, cls="", extra_style=""):
    """One muted, looping, inline autoplay video with its poster as the
    fallback. The poster is a real rendered frame, so a reader whose browser
    blocks autoplay or video entirely still sees the setup rather than a gap;
    `preload="metadata"` keeps the page from pulling two video files before
    the reader has scrolled to them."""
    return (f'<video{cls} autoplay loop muted playsinline preload="metadata" '
            f'poster="{poster}"{extra_style} '
            f'aria-label="the emission sweep as a looping video">'
            f'<source src="{src("sweep.webm")}" type="video/webm">'
            f'<source src="{src("sweep.mp4")}" type="video/mp4">'
            f'<img src="{poster}" alt="the brightest frame of the emission sweep">'
            f'</video>')


def swap_cell_for_video(html, poster_src):
    """Replace the matrix cell holding the poster image with the video itself.

    Building the strip through method_matrix and then substituting one cell's
    contents keeps the engine's tile geometry, column labels, caption width and
    centering, rather than hand-rolling a parallel five-up grid that would drift
    from every other matrix on the site.
    """
    pat = re.compile(r'<img loading="lazy"[^>]*?src="' + re.escape(poster_src) + r'"[^>]*?>')
    out, n = pat.subn(lambda _m: video_tag(poster=poster_src), html)
    if n != 1:
        sys.exit(f"VIDEO CELL SWAP FAILED: matched {n} cells, expected exactly 1")
    return out


def tree_html():
    entries = wz.tree_entries()
    for group in entries:
        for leaf in group.get("children", []):
            leaf["active"] = leaf["href"] == PAGE_HREF
    return lp.v3_tree(entries, title="Lightgen", subtitle="research workspace",
                      tree_src=wz.TREE_JSON_URL)


# ------------------------------------------------------------------ sections
def sec_default():
    poster = src("sweep_poster.png")
    strip = lp.method_matrix(
        columns=["box", "key-lit", "input", "segmentation", "sweep"],
        rows=[("one asset", [src("cmp_box.png"), src("cmp_keylit.png"),
                             src("teaser_cond.png"), src("teaser_seg.png"),
                             poster])],
        caption_html=(
            "<b>The whole vocabulary on one asset.</b> A sci-fi weapon with emissive "
            "strips along its limbs and a lamp at the front, under all five setups. "
            "Box: the asset alone in a closed neutral room, so the green wash on the "
            "floor is its own light. Key-lit: one dim lamp recovers the shell, and the "
            "room goes quiet. Input: bright neutral studio light, the image the model "
            "is actually given. Segmentation: the model's own part colours on its "
            "predicted mesh. Sweep: the same box render as emission strength ramps up "
            "and back down, playing on loop. The first two share a camera with each "
            "other and with the sweep; the input and segmentation panels use the "
            "model's own canonical camera, which is part of what distinguishes them. "
            "The segmentation panel is the weakest of the five: the model returns "
            "almost one flat colour on this asset, which section 05 shows is true "
            "across the whole emissive asset family rather than bad luck here. Each "
            "still opens at full resolution."),
        native_px=768, content="photo", page_inner=820)
    strip = swap_cell_for_video(linkify_cells(strip), poster)

    row = lp.fig_row(
        [("box render", src("cmp_box.png"),
          "the asset inside a closed neutral box, lit only by its own emission"),
         ("key-lit emission render", src("cmp_keylit.png"),
          "the asset in a dark room with one dim key light, plus its emission"),
         ("input render", src("cmp_input.png"),
          "the asset under bright neutral studio lighting on white")],
        caption_html=(
            "<b>With the camera held fixed, the lighting is the only variable.</b> "
            "The three setups that share one solved viewpoint, shown larger. Same "
            "asset, same emission (ground-truth mask times albedo, strength 4), same "
            "camera. Left, the box render: the pool on the floor and the falloff up "
            "the back wall are the weapon's own light, because there is no other "
            "light in the scene. Middle, the key-lit render: a single dim lamp "
            "recovers the shell, so a reader can see which parts of the body the "
            "strips sit on, but the room now says nothing about the emission. Right, "
            "the studio reference panel: no emission contribution visible at all, "
            "which is what makes it the panel for showing what the object is."))

    body = lp.prose(
        "Five setups produce every figure in this project. Naming them means a "
        "figure can be specified by one word instead of eight flags, and it means "
        "two figures are comparable exactly when they name the same setup."
    ) + lp.callout(
        "Any figure that shows emission output, ground truth or predicted, on a "
        "page or in the paper, is a <b>box render</b>. The other four are used only "
        "for their own purposes: the key-lit emission render when the shell's "
        "geometry has to stay legible alongside the emission, the input render for "
        "the panel showing what the model was given, the segmentation render for "
        "part decomposition, the emission sweep for the strength ladder.",
        title="The default") + strip + row
    return lp.section_v2("default", "01",
                         "Emission figures get the box render", body)


def sec_box():
    body = lp.prose(
        "The asset sits in a closed neutral box with no environment light and no "
        "lamps. Every lamp is deleted rather than dimmed and the world is set to "
        "zero strength, so the only photons in the scene come off the asset itself. "
        "That is what makes the pool on the floor and the wash up the walls "
        "evidence: they are the asset's own light and nothing else. The script "
        "asserts the condition and fails the render rather than shipping a panel "
        "whose claim is false."
    ) + lp.fig(
        src("cmp_box.png"),
        caption_html=(
            "<b>The box render answers what the emission does, not merely where it "
            "is.</b> Neutral matte walls at albedo 0.80, rotated so the open face "
            "squares up with the camera. The dark body of the weapon is legible only "
            "through bounced light, which is the point: a surface that receives "
            "nothing stays black."),
        alt="a sci-fi weapon in a neutral box, lit only by its own green emission",
        native_px=768, content="photo",
    ) + lp.prose("<b>When to use it.</b> By default, for anything showing emission "
                 "output. Prefer something else only when a specific reason applies, "
                 "and say which setup you used."
    ) + ptable([
        ("lighting", "none; every lamp deleted, world strength 0, world black"),
        ("box", "five quads, open toward the camera, rotated to the camera azimuth"),
        ("walls", "albedo 0.80, roughness 0.9, metallic 0"),
        ("view transform", "Filmic, exposure +1.5 stops"),
        ("bounces", "32 total, 16 diffuse"),
        ("samples", "1024, denoising on"),
        ("bloom", "Fog Glow, size 7, threshold 1.0, mix -0.45"),
        ("background", "none; the box fills the frame"),
        ("resolution", "768 x 768"),
        ("camera", "three-quarter view, azimuth 38, elevation 17, 52 mm, distance solved"),
        ("emission strength", "4.0, the same for every shape"),
    ]) + lp.chartnote(
        "The 32/16 bounce count is not a quality setting. In a box lit only by the "
        "asset, the ambient fill <i>is</i> multi-bounce diffuse light, and Cycles' "
        "default of 4 diffuse bounces truncates it, costing 39 percent of the image "
        "mean on the dimmest shape. Filmic and +1.5 were chosen by measurement over "
        "26 exposure renders and 33 bloom grades: "
        f'<a href="{SWEEP_URL}">the render sweep page</a> carries the evidence, '
        "including what +1.5 costs."
    )
    return lp.section_v2("box", "02",
                         "Box render: the asset is the only light", body)


def sec_keylit():
    body = lp.prose(
        "A near-black room, one dim key light at energy 8 on a matte dark floor, and "
        "the asset's emission on top. The lamp recovers the silhouette and enough "
        "surface detail to locate the emission on the body, which a box render can "
        "leave ambiguous on a dark asset. The cost is that a lamp is now in the "
        "scene, so nothing in the frame can be read as a consequence of the emission."
    ) + lp.fig(
        src("cmp_keylit.png"),
        caption_html=(
            "<b>The key light buys geometry and spends the room.</b> The same weapon "
            "and the same emission as the box render above. The body, the arm and the "
            "curved boom now read as shapes, so a reader can say which parts carry "
            "emissive strips. The floor carries nothing."),
        alt="a sci-fi weapon in a dark room under one dim key light, its emissive strips glowing",
        native_px=768, content="photo",
    ) + lp.prose(
        "<b>When to use it.</b> When the claim is about <i>where on the object</i> "
        "the emission sits. The overfit and fbv1 galleries use it, and it is the "
        "secondary band of the strength ladder."
    ) + ptable([
        ("world", "near-black, colour 0.012, strength 1.0"),
        ("key light", "energy 8 on the first lamp, 1.6 on the rest, soft size 0.6+"),
        ("floor", "matte dark plate, base colour 0.10, roughness 0.5"),
        ("view transform", "AgX, exposure 0"),
        ("samples", "256"),
        ("bloom", "Fog Glow, size 9, threshold 1.0, mix -0.15"),
        ("background", "the dark room; the film is opaque"),
        ("resolution", "768 x 768"),
        ("camera", "identical to the box render"),
        ("emission strength", "4.0"),
    ]) + lp.chartnote(
        "The key was 20 until it was measured. The Glare node's threshold is 1.0 in "
        "linear space, and at key 20 the bloom fired on 108,865 pixels of a panel "
        "whose prediction was empty, against 123,948 on real ground truth, so the "
        "glow was reporting the lamp rather than the object. At key 8 nothing "
        "non-emissive reaches the threshold: 0 pixels against 50,458."
    )
    return lp.section_v2("keylit", "03",
                         "Key-lit emission render: geometry stays legible", body)


def sec_input():
    row = lp.fig_row(
        [("condition image", src("teaser_cond.png"),
          "the weapon under SegviGen's own three-lamp rig on a white background"),
         ("studio reference panel", src("cmp_input.png"),
          "the same weapon under the preset studio scene with a contact shadow")],
        caption_html=(
            "<b>Two implementations, two different renders, one name.</b> The same "
            "asset both times. Left, the condition image: SegviGen's own renderer, "
            "the same call the model's inference path makes to build its DINOv3 "
            "conditioning, so a panel made this way is literally the model's input. "
            "Its scene is normalized to a unit bounding box before the camera is "
            "placed, which is why the asset sits smaller and at a different angle. "
            "Right, the studio reference panel that accompanies every key-lit run: a "
            "different renderer, a different lighting rig, and the same solved camera "
            "as the emission panels beside it, so those three line up. Neither "
            "carries a background of its own: both render on a transparent film and "
            "take whatever they are placed on, which is why both read as white here."))
    body = lp.prose(
        "Bright neutral studio lighting, white background, no emission story. This "
        "setup has two jobs: the panel that tells a reader what the shape looks "
        "like, and the image actually fed to the model. Those are served by two "
        "different implementations, and they are not the same render."
    ) + row + lp.prose(
        "<b>Which to use.</b> If the panel's claim is \"this is what the model saw\", "
        "use the condition image. If the claim is \"this is what the object looks "
        "like, from the same viewpoint as the emission panels beside it\", use the "
        "studio reference panel."
    ) + ptable([
        ("condition image: lighting",
         "three lamps: point 1000 at (4, 1, 6), area 10000 overhead, area 1000 below"),
        ("condition image: framing",
         "scene scaled to unit longest edge and centred, then one canonical camera"),
        ("condition image: camera", "32 mm square sensor, 43.95 mm lens (40 degree FOV)"),
        ("condition image: samples", "128, denoising on, 1 diffuse bounce"),
        ("condition image: resolution", "512 x 512"),
        ("studio panel: lighting", "the xgutils preset studio scene, unchanged"),
        ("studio panel: view transform", "Khronos PBR Neutral"),
        ("studio panel: background", "transparent, floor as a shadow catcher"),
        ("studio panel: samples", "96"),
        ("studio panel: resolution", "768 x 768"),
    ]) + lp.chartnote(
        "Normalizing the scene before placing the camera is what keeps a set of "
        "assets on one framing, and it is also the failure mode: a stray unrelated "
        "mesh in a source GLB inflates the bounding box and shrinks the asset to a "
        "speck. That happened once on the 19-shape gallery and was traced to an "
        "unused sphere in the source file."
    )
    return lp.section_v2("input", "04",
                         "Input render: what the object is, and what the model saw",
                         body)


def sec_seg():
    row_teaser = lp.fig_row(
        [("condition image", src("teaser_cond.png"),
          "the weapon under the canonical camera"),
         ("segmentation render", src("teaser_seg.png"),
          "the same weapon, almost entirely one pale colour")],
        caption_html=(
            "<b>The setup works; the model's answer on this asset does not.</b> The "
            "pretrained full-segmentation model bakes its part colouring into the mesh "
            "it exports, and that mesh is rendered by the same renderer under the same "
            "camera as the condition image, run zero-shot with no fine-tuning. The "
            "silhouette is roughly the right object, which is more than the four "
            "assets below manage, but the model reports four parts and returns three "
            "of them in nearly the same pale colour, so the panel reads as a "
            "monochrome object rather than a decomposition."))
    row_tumbled = lp.fig_row(
        [("input render", src("seg_input.png"),
          "a grey and orange bench under studio lighting"),
         ("segmentation render", src("seg_parts.png"),
          "the same bench with each predicted part in its own colour")],
        caption_html=(
            "<b>The orientation is not guaranteed, and that is a pipeline issue rather "
            "than a property of the setup.</b> A bench from the 19-shape gallery under "
            "the same nominal camera in both panels. The predicted mesh's up-axis "
            "often differs from the source asset's, so the object arrives tumbled; on "
            "that gallery only a minority of shapes came out aligned. Check the "
            "particular shape before building a figure whose argument depends on the "
            "two panels registering."))
    fam_matrix = linkify_cells(lp.method_matrix(
        columns=[w for _s, _t, w in FAMILY],
        rows=[("input", [src(f"fam_{t}_in.png") for _s, t, _w in FAMILY]),
              ("prediction", [src(f"fam_{t}_seg.png") for _s, t, _w in FAMILY])],
        caption_html=(
            "<b>On the emissive asset family the pretrained model does not "
            "reconstruct the object, so there is nothing left to colour.</b> Four more "
            "assets from the same set as the teaser, each run zero-shot through the "
            "same pipeline. Top row, what went in; bottom row, what came back. The "
            "jack-o'-lantern becomes a smooth green ball with its carved face gone, "
            "the headphone stand becomes a barrel and a plate, the candles become a "
            "disc, and the vending machine keeps only its box. A part count is "
            "reported for every one of them, so a number on its own would not have "
            "caught this. It is a property of the model on these assets rather than "
            "of the rendering setup, and it is why the opening figure's segmentation "
            "panel is the weakest of its five."),
        native_px=512, content="photo", page_inner=820))

    body = lp.prose(
        "Per-part colours on a white background, for showing how a model decomposes "
        "a shape. As with the input render there are two implementations, and the "
        "difference is more than plumbing."
    ) + row_teaser + row_tumbled + fam_matrix + lp.prose(
        "The panels above use the model's own colours under ordinary three-lamp "
        "shading, which is what the 19-shape gallery ships. An earlier variant, on "
        "the canonical mesh and voxel pages, recolours part labels through a fixed "
        "20-entry high-contrast palette and paints them as vertex colours, which "
        "reads as flat per-part colour. Use the model's own colours when the point "
        "is what the model produced; use the palette when parts have to be told "
        "apart at a glance and the model's colours are too close together."
    ) + ptable([
        ("model colours: source", "the part colouring baked into the exported mesh"),
        ("model colours: lighting", "the same three-lamp rig as the condition image"),
        ("model colours: cameras", "camera A (shared with the input panel) and camera B, about 140 degrees around"),
        ("model colours: resolution", "512 x 512"),
        ("palette: colours", "20 fixed high-contrast triples, indexed by part label"),
        ("palette: assignment", "each mesh vertex takes its nearest segmentation voxel's label"),
        ("palette: camera", "fixed at (0, -2.6, 1.4), up +Z"),
        ("palette: resolution", "460 x 460 on the mesh view, 440 x 440 on the voxel view"),
    ])
    return lp.section_v2("seg", "05",
                         "Segmentation render: one colour per part", body)


def sec_sweep():
    poster = src("sweep_poster.png")
    video = (f'<figure class="xg-video" style="max-width:520px">'
             f'{video_tag(poster=poster)}'
             f'<figcaption style="max-width:520px">'
             "<b>Emission strength is a look choice, and the sweep is how that is "
             "shown rather than asserted.</b> The weapon under the box render as "
             "strength ramps from 0 to 16 and back, eased so the turnarounds are "
             "smooth, on a loop. Watch the floor rather than the emitters: the pool "
             "of light grows faster than the strips themselves, because what the "
             "sweep changes is how much light the asset puts into the room. At 0 the "
             "room is black, which is the honest baseline."
             f'</figcaption></figure>')
    cells = [src(f"sweep_s{k}.png") for k in (0, 1, 4, 8, 16)]
    matrix = linkify_cells(lp.method_matrix(
        columns=["0", "1", "4", "8", "16"],
        rows=[("box render", cells)],
        caption_html=(
            "<b>The same sweep as stills, for reading exact rungs.</b> Five of the "
            "video's strengths, side by side, so a reader can compare two values "
            "without scrubbing. Every rung is a real render because strength changes "
            "light transport. The halo grows with strength, and that is correct: the "
            "Glare threshold is 1.0 in linear space, so raising strength moves "
            "emitters further above it. Any tile opens at full resolution."),
        native_px=768, content="photo", page_inner=820))
    body = lp.prose(
        "The same asset and the same view at a ladder of emission strengths. The "
        "bake stores emission as uint8 and drops the glTF emissive-strength "
        "extension, so nothing in the data says how brightly a surface emits; 4.0 is "
        "the value every published panel uses, and the sweep is what makes that "
        "choice inspectable. It is also the basis of the user study design."
    ) + video + matrix + lp.prose(
        "<b>When to use it.</b> Only for the strength question. A sweep is not a "
        "results figure, and a single frame pulled out of one is a box render or a "
        "key-lit render, so cite it as that."
    ) + ptable([
        ("rungs", "0, 1, 4, 8, 16 by default; one real render each"),
        ("what varies", "emission strength only; every other flag is passed through"),
        ("base setup", "the box render for the primary band, the key-lit render for the secondary one"),
        ("video", "33 rendered strengths eased with a smoothstep, ping-ponged to 64 frames at 24 fps"),
        ("per-rung check", "a rung that produced no file fails the run"),
        ("output", "one image per rung, plus a labelled montage and a JSON index"),
    ]) + lp.chartnote(
        "What the sweep over our own masks cannot show: the model's masks are "
        "exactly binary. Across 99 mask files from three checkpoints, 34,603,008 "
        "pixels, the only values present are 0 and 255, so a surface the model did "
        "not select stays exactly black at every rung. Amplification cannot make "
        "this formulation leak, which is the opposite of what a continuous emission "
        "generator would show."
    )
    return lp.section_v2("sweep", "06",
                         "Emission sweep: the strength ladder", body)


def sec_provenance():
    rows = "".join(
        f'<tr><td class="rowhead">{dest}</td><td>{why}</td>'
        f'<td><code>{srcp}</code></td></tr>'
        for dest, srcp, why in STAGED)
    table = lp.results_table(["image", "what it is", "copied from"], rows)
    body = lp.prose(
        "Every parameter on this page was read off the scripts rather than recalled. "
        "The repo-side companion, <code>RENDERING.md</code> at the lightgen root, "
        "carries the full flag lists and a file and line reference for each value, "
        "including the three cases where the shipped setting is not the script's own "
        "default."
    ) + lp.prose(
        "<b>Scripts.</b> The box render, key-lit render, studio reference panel and "
        "emission sweep all come from "
        "<code>segvigen_emissive/render/render_emissive.py</code> and its wrapper "
        "<code>segvigen_emissive/render/strength_ladder.py</code>. The condition "
        "image and the segmentation panels come from SegviGen's own "
        "<code>data_toolkit/bpy_render.py</code>. Parameters were read at repo commit "
        "<code>cc218ed</code>."
    ) + lp.prose(
        "<b>What was rendered for this page.</b> Five cluster jobs, all on CPU nodes "
        "except the two inference ones. Job <code>242221</code> rendered the teaser "
        "asset's condition image. Job <code>242225</code> ran the pretrained "
        "full-segmentation model on it, zero-shot, 25 sampling steps, 48 seconds on one "
        "GPU. Job <code>242226</code> rendered that predicted mesh under cameras A and "
        "B. Job <code>242228</code> repeated all three stages for the four other "
        "emissive assets in section 05. Job <code>242222</code> rendered the 33 sweep "
        "frames as an 11-task array at 64 cores each, one box render per emission "
        "strength, with the camera pinned from the shape's own camera file so the "
        "viewpoint is identical across the sequence. Encoding to WebM and MP4 ran on "
        "the workstation, from <code>make_video.py</code> in this page's directory; it "
        "ping-pongs the 33 rendered strengths into 64 frames and fails if the encoded "
        "frame count does not match."
    ) + lp.prose(
        "<b>Everything else is a copy.</b> The remaining panels are already-published "
        "renders, restaged here so the page is self-contained."
    ) + table + lp.prose(
        "<b>Reading the page cold.</b> The asset in the opening figure, the sweep and "
        "sections 02 to 04 is a sci-fi weapon from the strength-ladder shape set; the "
        "bench in section 05 is from the 19-shape pretrained-segmentation gallery. "
        "None of it is a fine-tuned model result: the emission shown is ground truth "
        "read off the source asset, and the part colours come from the pretrained "
        "segmentation model run with no fine-tuning."
    )
    return lp.section_v2("provenance", "07", "Where this came from", body)


# ----------------------------------------------------------------- the build
def build(publish=False):
    assets_dir = os.path.join(WEB, "assets")
    stage_images()
    problems = check_assets()
    if problems:
        sys.exit("ASSET CHECK FAILED:\n  " + "\n  ".join(problems))

    hero = lp.hero_header(
        f"lightgen &middot; figure rendering &middot; {PAGE_DATE}",
        "Five Rendering Setups, and Which One a Figure Gets",
        dek_html=(
            "Every figure in this project is made under one of five lighting setups. "
            "Emission results get the box render, where the asset is the only light "
            "in a closed room; the other four exist for questions the box render "
            "cannot answer. The figure below shows all five on one asset. Each "
            "section then says when to reach for that setup and lists the settings "
            "that make two figures comparable. The flags themselves live in "
            "RENDERING.md in the repository."),
        toc=[("default", "The rule"), ("box", "Box"), ("keylit", "Key-lit"),
             ("input", "Input"), ("seg", "Segmentation"), ("sweep", "Sweep"),
             ("provenance", "Provenance")],
    )

    page_html = lp.page(
        title="Five Rendering Setups, and Which One a Figure Gets",
        header_html=hero,
        body_sections=[sec_default(), sec_box(), sec_keylit(), sec_input(),
                       sec_seg(), sec_sweep(), sec_provenance()],
        assets_rel=SITE_ASSETS,
        assets_dir=assets_dir,
        theme="v3",
        tree_html=tree_html(),
        nav_title="Rendering setups",
        version_slot=lp.v3_version_slot(date=PAGE_DATE),
        needs_katex=False,
        extra_head=f'<link rel="icon" href="{FAVICON}">' + EXTRA_CSS,
        outline_entries=OUTLINE,
    )

    # ZONE-BOUNDARY LAW: nothing in the workspace zone may link to the console.
    violations = wz.console_links_in(page_html)
    if violations:
        sys.exit(f"ZONE-LINK GUARD FAILED: page links to the console: {violations}")

    out_path = os.path.join(HERE, "index.html")
    with open(out_path, "w") as f:
        f.write(page_html)
    print(f"wrote {out_path}")
    print(f"  {len(STAGED)} images staged, {len(VIDEO_ASSETS)} video assets checked")
    print("  zone-link guard: clean")

    publish_assets(assets_dir)
    print(f"assets published -> {assets_dir}")

    if publish:
        os.makedirs(PUBLISH_DIR, exist_ok=True)
        shutil.copytree(IMG, os.path.join(PUBLISH_DIR, "img"), dirs_exist_ok=True)
        shutil.copy2(out_path, os.path.join(PUBLISH_DIR, "index.html"))
        wz.write_tree_json()
        for p in [PUBLISH_DIR, *[os.path.join(dp, f)
                                 for dp, _dn, fn in os.walk(PUBLISH_DIR)
                                 for f in fn]]:
            try:
                os.chmod(p, os.stat(p).st_mode | (0o005 if os.path.isdir(p) else 0o004))
            except OSError:
                pass
        print(f"published -> {PUBLISH_DIR}")
        print(f"tree.json refreshed -> {wz.TREE_JSON}")


if __name__ == "__main__":
    build(publish="--publish" in sys.argv)
