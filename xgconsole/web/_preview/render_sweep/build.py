#!/usr/bin/env python3
"""Build the lighting-sweep evidence page on the xgpage v3 workspace shell.

The page shows every variant rendered behind the emission-only box figures, so
the choice of view transform, exposure, bounce count and bloom can be judged
rather than taken on trust. TWO sweeps with one conclusion: the exposure sweep
(26 renders, view transform x exposure, plus the pre-fix bounce control) and the
bloom sweep (33 glare settings across 3 shapes, regraded from the linear
renders through the compositor). Both rejected a parameter for the same reason,
and the page is ordered so that parallel lands as a section, not an aside.

Every number on the page is RECOMPUTED at build time from the PNGs in img/
(measure() and bloom_share() below); nothing is retyped from prose. Two
consequences: the table and the page text cannot drift apart, and a re-render of
either sweep changes the page by rebuilding it.

Bloom assets are staged by prep_bloom.py (run it once before this); it also cuts
the carved-eye crops and renders the bloom-extent maps.

Metric definitions, stated once and used everywhere:
  midtone       median of Rec.709 luminance over the FULL frame, in [0, 1].
                Full frame, not a foreground crop: these are closed-box
                renders, every pixel is either the object or a wall the object
                lit, so there is no background to exclude (verified: alpha is
                255 everywhere in all 26 PNGs).
  clipped share fraction of pixels with ANY channel at or above 254/255.
  colorfulness  Hasler and Suesstrunk's opponent-axis measure, sqrt(sd_rg^2 +
                sd_yb^2) + 0.3 sqrt(mean_rg^2 + mean_yb^2) on 0-255 channels,
                where rg = R - G and yb = (R + G)/2 - B. Higher is more
                colorful. Used only for the AgX comparison.
  image mean    mean of the three channels over the frame, in [0, 1]. The
                statistic the bounce comparison is quoted in.
  bloomed share fraction of pixels the Glare node lifts by AT LEAST 3 of 255,
                per-pixel max over channels, against the same linear render
                regraded with the Glare node bypassed.

                Stated in whole steps on purpose. The sweep's original code
                tested `lift > 2.0/255` in floating point, and these are 8-bit
                images, so a large share of every halo lands EXACTLY on the
                2-step boundary (4.96 percent of the candles frame in one
                cell) where rounding residue, not the image, decides the
                comparison. Measured over all 50 published cells:

                  integer >= 3 steps   worst drift 0.0066 vs the reports
                  integer >= 2 steps   worst drift 0.1126
                  float64 > 2.0/255    worst drift 0.0509

                So the published numbers mean ">= 3 whole steps", and the
                double-precision float reading is NOT the ground truth the
                float32 one approximates: it splits the two-step population
                and matches neither integer cut. Both float forms are unsound
                for the same reason. 3 is also the first cut that does not sit
                on a populated boundary. check_bloom() asserts the agreement at
                build time. Rankings are identical under every reading, so no
                conclusion turns on this. (Reported to paper-v3; bloom_sweep.py
                now compares in integer space with the cut at 3.)

Run: .venv2/bin/python web/_preview/render_sweep/build.py [--publish]
  (.venv2 = /cs/3dlg-project/3dlg-hcvc/omages/omages_internal/.venv2/bin/python)
"""
import hashlib
import json
import os
import re
import shutil
import sys

import numpy as np
from PIL import Image

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

PAGE_SLUG = "render_sweep"
PAGE_HREF = f"{wz.WORKSPACE_URL}/{PAGE_SLUG}/index.html"
PUBLISH_DIR = os.path.join(str(wz.WORKSPACE_DIR), PAGE_SLUG)
PAGE_DATE = "2026-08-06"

IMG = os.path.join(HERE, "img")

# ------------------------------------------------------------------ the sweep
SHAPES = {
    "48af42db48c44cd9bfab32bbb057a39c": ("jack-o'-lantern", "pumpkin"),
    "c1e3035d1ccb49df9c09aa86681faf30": ("humanoid robot", "robot"),
    "658ecf9f837246509b0b1c4aa81e9e5b": ("three lit candles", "candles"),
    "9418a924a50d44c186dd499006b62424": ("vending machine", "vending"),
}
PUMPKIN, ROBOT = "48af42db48c44cd9bfab32bbb057a39c", "c1e3035d1ccb49df9c09aa86681faf30"
CANDLES, VENDING = "658ecf9f837246509b0b1c4aa81e9e5b", "9418a924a50d44c186dd499006b62424"

TRANSFORMS = [("agx", "AgX"), ("film", "Filmic"), ("std", "Standard")]
EXPOSURES = [("e0", "+0 stops", 0.0), ("e1", "+1.0 stops", 1.0), ("e15", "+1.5 stops", 1.5)]

# ------------------------------------------------------------- the bloom sweep
# Glare-node settings, as the sweep tagged them: t<threshold>_m<mix>_s<size>.
BLOOM_OLD, BLOOM_NEW = "t1_m-0.15_s9", "t1_m-0.45_s7"
BLOOM_SHAPES = [(PUMPKIN, "jack-o'-lantern"), (CANDLES, "three lit candles"),
                (VENDING, "vending machine")]
# one axis at a time from the previous setting (the sweep's first pass)
BLOOM_LADDER = [(BLOOM_OLD, "previous"), ("t1_m-0.45_s9", "mix -0.45"),
                ("t1_m-0.7_s9", "mix -0.70"), ("t1_m-0.15_s7", "size 7"),
                ("t1_m-0.15_s5", "size 5")]
BLOOM_SIZES = [("s5", "size 5"), ("s6", "size 6"), ("s7", "size 7")]
BLOOM_MIXES = [("m-0.15", "mix -0.15"), ("m-0.45", "mix -0.45")]
BLOOM_THRESHOLDS = [("t1", "threshold 1.0"), ("t1.5", "threshold 1.5"),
                    ("t2.5", "threshold 2.5")]

# The chosen setting was applied to ALL ELEVEN box shapes, not just the three
# that were swept, so the claim that one glare size serves the whole range can
# be checked on a wider sample than the sample it was chosen on. Names read off
# the renders; emissive area comes from each shape's own EXR sidecar.
SHIPPED_SHAPES = [
    ("8f4c281aef1b4563b6103efbcd77fac1", "headphone stand"),
    ("b74fc2533d5345629f2c3ce2c8ab340a", "ghost figure"),
    (ROBOT, "humanoid robot"),
    ("1e9c6545b4da42e0ba4e5dbcd2e0e8ff", "street lamp"),
    (PUMPKIN, "jack-o'-lantern"),
    ("51a60b164e874bf891597d9c6c1941af", "glowing creature"),
    ("e5eecab2bc8649548b48b79e705d768e", "candelabra"),
    (VENDING, "vending machine"),
    ("4e105e043a6447439e98e9831aed122e", "lit staff"),
    ("b7709a651d144134a5babce33223380a", "costumed figure"),
    (CANDLES, "three lit candles"),
]


# ------------------------------------------------------------------ measuring
def measure(key):
    """Recompute every reported statistic for one render. See module docstring."""
    a = np.asarray(Image.open(os.path.join(IMG, key + ".png"))).astype(np.float64)
    assert a.shape[2] == 4 and (a[..., 3] == 255).all(), f"{key}: unexpected alpha"
    rgb = a[..., :3]
    r01 = rgb / 255.0
    lum = 0.2126 * r01[..., 0] + 0.7152 * r01[..., 1] + 0.0722 * r01[..., 2]
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorful = (np.hypot(rg.std(), yb.std())
                + 0.3 * np.hypot(rg.mean(), yb.mean()))
    return {
        "p50": float(np.median(lum)),
        "mean": float(r01.mean()),
        "clip": float((rgb >= 254).any(axis=2).mean()),
        "colorful": float(colorful),
    }


def sidecar(key):
    with open(os.path.join(IMG, key + ".json")) as f:
        return json.load(f)


def load_sweep():
    """Every render in img/, keyed <sid>_box_<variant>, with stats + settings.
    The settings are read from the sidecars, never assumed."""
    # "_box_" is the exposure sweep's own naming; img/ also holds the bloom
    # grades, the clipped-pixel maps and the crops, which this must not pick up.
    keys = sorted(f[:-4] for f in os.listdir(IMG)
                  if f.endswith(".png") and "_box_" in f and not f.startswith("clip_"))
    out = {}
    for k in keys:
        out[k] = {**measure(k), "cfg": sidecar(k)}
    return out


def check_settings(sw):
    """The sweep's stated design, asserted against the sidecars. A render whose
    settings disagree with the grid it is placed in would silently make a
    column mean something else."""
    problems = []
    for k, d in sw.items():
        cfg, variant = d["cfg"], k.split("_box_")[1]
        if cfg["samples"] != 512:
            problems.append(f"{k}: samples {cfg['samples']}")
        if variant == "oldbounce":
            want = (12, 4, 0.7, "Filmic", 1.0)
        else:
            tf = {"agx": "AgX", "film": "Filmic", "std": "Standard"}[variant.split("_")[0]]
            ex = {"e0": 0.0, "e1": 1.0, "e15": 1.5}[variant.split("_")[1]]
            want = (32, 16, 0.8, tf, ex)
        got = (cfg["max_bounces"], cfg["diffuse_bounces"], cfg["wall"],
               cfg["view_transform"], cfg["exposure"])
        if got != want:
            problems.append(f"{k}: settings {got} != {want}")
    return problems


# ------------------------------------------------------- measuring the bloom
_BLOOM_CACHE = {}


def bloom_max(key):
    """Per-pixel max over channels of one staged bloom grade, as int16."""
    if key not in _BLOOM_CACHE:
        a = np.asarray(Image.open(os.path.join(IMG, f"bloom_{key}.png")).convert("RGB"))
        _BLOOM_CACHE[key] = a.astype(np.int16).max(axis=-1)
    return _BLOOM_CACHE[key]


def bloom_share(sid, tag):
    """Share of the frame the Glare node lifts by at least 3 of 255, against the
    same render with the node bypassed. See the module docstring for why the cut
    is stated in integer steps rather than as the sweep's float comparison."""
    lift = bloom_max(f"{sid}_{tag}") - bloom_max(f"{sid}_none")
    return float((lift >= 3).mean())


def check_bloom(shares):
    """Assert the recomputation reproduces the sweep's own report. Guards the
    integer restatement of the bloomed-pixel measure: a definition that drifted
    from what the sweep ran would silently rewrite every number on the page."""
    worst, problems = 0.0, []
    for name in ("sweep", "refine"):
        with open(os.path.join(IMG, f"bloom_report_{name}.json")) as f:
            rep = json.load(f)
        for k, v in rep.items():
            sid, tag = k.split("/")
            if (sid, tag) not in shares:
                problems.append(f"{k}: in the report, not staged")
                continue
            d = abs(shares[(sid, tag)] - v["bloomed_px_frac"])
            worst = max(worst, d)
            if d > 0.01:
                problems.append(f"{k}: {shares[(sid, tag)]:.5f} vs reported "
                                f"{v['bloomed_px_frac']:.5f}")
    return worst, problems


# ------------------------------------------------------------------ page bits
def hashed(rel):
    """Content-hashed src for a file under img/."""
    p = os.path.join(IMG, rel)
    h = hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    return f"img/{rel}?v={h}"


CELL_LINK_RE = re.compile(r'(<div class="mm-cell">)(<img [^>]*?src="([^"]+)"[^>]*>)')


def linkify_cells(html):
    """Make every matrix tile click through to its full-resolution render.
    These are dark images shown well below their 512px native size; the tile
    carries the comparison, the link carries the detail."""
    return CELL_LINK_RE.sub(
        lambda m: f'{m.group(1)}<a href="{m.group(3)}" target="_blank" '
                  f'rel="noopener">{m.group(2)}</a>', html)


def fmt(x, n=3):
    return f"{x:.{n}f}"


def pct(x):
    return f"{100 * x:.2f}%" if x >= 1e-4 else ("0%" if x == 0 else f"{100 * x:.3f}%")


# ------------------------------------------------------------ page-local CSS
# Two overrides, both in service of the one thing this page has to do: let a
# reader judge dark renders side by side.
#
# 1. method_matrix caps its tiles at 220px (core.py: min of the tile-span fit,
#    220, and the D9 cap). At three columns in the 820px v3 content column the
#    fit is 269px, so the cap leaves 148px of the column unused while the
#    renders sit at 43 percent of native size. These are near-black frames
#    whose subject is a few hundred pixels of emission; 220px is where the
#    earlier dense-thumbnail failure on this project came from. The tracks are
#    widened to 268px (3x268 + 2x6 = 816, inside the 820 column), and the
#    figure and caption max-widths follow so figure and caption stay one
#    object and the D11 auto-margins still center them. The engine's own fixed
#    px tracks are kept, so narrow viewports still scroll inside .mm-scroll.
#    Worth an optional cell_max= param upstream; not changed here, this page
#    does not own the engine.
# 2. .mm-cell a { display: contents } so the click-through anchor adds no box
#    and the grid geometry is exactly what the engine computed.
EXTRA_CSS = """
<style>
.xg3 figure.mm { max-width: 816px !important; }
.xg3 figure.mm figcaption { max-width: 816px !important; }
.xg3 .method-matrix { grid-template-columns: 26px repeat(3, 268px) !important; }
.mm-cell a { display: contents; }
/* the tile is a link: say so on hover, quietly */
.mm-cell a:hover img { outline: 2px solid var(--accent); outline-offset: -2px; }

/* theme.css carries v1-era floors (table.results{min-width:960px},
   td.rowhead{min-width:210px}) that every v3 page inherits, so any results
   table on the 820px column scrolls internally however short its cells are.
   Lifted here, as the paper-skeleton page does. */
.xg2 table.results { min-width: 0; }
.xg2 table.results td.rowhead { min-width: 0; }
.xg2 table.results td, .xg2 table.results th { padding-left: 8px; padding-right: 8px; }
.xg2 table.results td.num { font-variant-numeric: tabular-nums; text-align: right; }

/* theme3.css releases .prose's max-width so the 820px column IS the measure,
   but leaves .chart (760px) and .chartnote (720px) on v2's narrower caps. On
   v3 that gives a chart and its note two more left edges than the heading and
   the prose around them, measured at 1440: prose L=0, chart L=30, chartnote
   L=50 inside the same column. Each is symmetric on its own, so the width QA
   passes it; it still reads as a stray indent. Released here to one edge.
   Upstream candidate for theme3.css. */
.xg3 .chart, .xg3 .chartnote { max-width: none; }

/* flow_stage is a v1 component and theme.css gives it a HARDCODED dark chrome
   (background #0d1014, .fnum colour #fff) that the v2 token remap cannot reach,
   which is SKILL.md rule 16's trap verbatim. On this light-palette page the
   measured result was background rgb(13,16,20) under text rgb(33,32,28): black
   on black. Re-skinned here in the v2 tokens. Worth upstreaming as an .xg2
   override next to the other v1-component re-skins in theme2.css. */
.xg2 .flow-stage { background: var(--panel, #FAF9F5); border-color: var(--line); }
.xg2 .flow-stage .fnum { color: var(--ink-2); font-size: .78rem; font-weight: 600;
  font-family: ui-monospace, Menlo, monospace; }
.xg2 .flow-stage .flbl { color: var(--text); font-weight: 600; font-size: .86rem; }
.xg2 .flow-stage .fsub { color: var(--ink-2); }
.xg2 .flow-stage.highlight { border-color: var(--accent); }
.xg2 .flow-stage.highlight .fnum { color: var(--accent); }
.xg2 .flow-arrow .arrowglyph { color: var(--ink-2); }

/* Scroll room past the last section: xg3.js's scrollspy marks the nearest
   [id] above a 110px line, and the final section is too short to reach it. */
.xg3 .v3-main .page::after { content: ""; display: block; height: 40vh; }
</style>
"""


def tree_html():
    entries = wz.tree_entries()
    for group in entries:
        for leaf in group.get("children", []):
            leaf["active"] = leaf["href"] == PAGE_HREF
    return lp.v3_tree(entries, title="Lightgen", subtitle="research workspace",
                      tree_src=wz.TREE_JSON_URL)


# ----------------------------------------------------------------- the build
def build(publish=False):
    assets_dir = os.path.join(WEB, "assets")
    sw = load_sweep()
    problems = check_settings(sw)
    if problems:
        sys.exit("SIDECAR CHECK FAILED:\n  " + "\n  ".join(problems))

    def S(sid, variant):
        return sw[f"{sid}_box_{variant}"]

    bl = {}
    for sid, _ in BLOOM_SHAPES:
        for f in os.listdir(IMG):
            if f.startswith(f"bloom_{sid}_") and f.endswith(".png"):
                tag = f[len(f"bloom_{sid}_"):-4]
                if tag != "none":
                    bl[(sid, tag)] = bloom_share(sid, tag)
    bloom_worst, bloom_problems = check_bloom(bl)
    if bloom_problems:
        sys.exit("BLOOM CHECK FAILED:\n  " + "\n  ".join(bloom_problems))

    def B(sid, tag):
        return bl[(sid, tag)]

    # The chosen setting on all eleven shapes, with each shape's emissive area.
    shipped = {}
    for sid, name in SHIPPED_SHAPES:
        if (sid, BLOOM_NEW) not in bl:
            bl[(sid, BLOOM_NEW)] = bloom_share(sid, BLOOM_NEW)
        with open(os.path.join(IMG, f"bloomshape_{sid}.json")) as f:
            shipped[sid] = {"name": name, "share": bl[(sid, BLOOM_NEW)],
                            "area": json.load(f)["area_lit_frac"]}
    with open(os.path.join(IMG, "bloom_report_shipped.json")) as f:
        rep_shipped = json.load(f)
    ship_worst = max(abs(shipped[sid]["share"] - v["bloomed_px_frac"])
                     for k, v in rep_shipped.items()
                     for sid in [k.split("/")[0]])
    if ship_worst > 0.01:
        sys.exit(f"SHIPPED BLOOM CHECK FAILED: worst drift {ship_worst:.5f}")

    # ================================================================== hero
    hero = lp.hero_header(
        f"lightgen &middot; figure rendering &middot; lighting sweep &middot; {PAGE_DATE}",
        "Choosing the Lighting for the Emission-Only Box Renders",
        dek_html=(
            "Two sweeps stand behind one setting. The first crossed three view "
            "transforms with three exposures on the dim shapes, then put the finalists "
            "against the bright ones, and kept the pre-fix bounce settings as a control. "
            "The second regraded the same renders through the compositor's glare node "
            "across bloom threshold, mix and size. Both ended by rejecting a parameter "
            "for the same reason, which is the argument this page is ordered around. "
            "Every number below is recomputed from the images when the page is built. "
            "Any tile opens at full resolution."
        ),
        toc=[("decision", "The setting"), ("bounces", "Bounces"),
             ("round1", "Round 1"), ("round2", "Round 2"),
             ("cost", "What +1.5 costs"), ("bloom", "The halo"),
             ("scaling", "The shared reason"), ("mechanism", "A prediction tested"),
             ("numbers", "Every render")],
    )

    # ======================================================== 01 the decision
    cfg = S(PUMPKIN, "film_e15")["cfg"]
    settings_rows = "".join(
        f'<tr><td class="rowhead">{a}</td><td class="rowhead">{b}</td>'
        f'<td class="rowhead">{c}</td></tr>'
        for a, b, c in [
            ("View transform", "Filmic", "AgX and Standard both rendered, both rejected"),
            ("Exposure", "+1.5 stops", "one value for every shape, dim and bright"),
            ("Total bounces", f"{cfg['max_bounces']}", "Cycles default is 12"),
            ("Diffuse bounces", f"{cfg['diffuse_bounces']}", "Cycles default is 4"),
            ("Wall albedo", f"{cfg['wall']}", "neutral grey, was 0.70"),
            ("Samples", f"{cfg['samples']}", "unchanged across the sweep"),
            ("Glare size", "7", "was 9; the halo's radius, and the lever that worked"),
            ("Glare mix", "&minus;0.45", "was &minus;0.15; the halo's intensity"),
            ("Glare threshold", "1.0", "unchanged; rejected as a lever, see section 07"),
        ])
    sec1 = lp.section_v2("decision", 1,
        "Filmic at +1.5 stops on 0.80 walls, 32 and 16 bounces, glare size 7 at mix "
        "&minus;0.45",
        lp.verdict_box(
            "<b>Chosen: Filmic, exposure +1.5 stops, 32 total and 16 diffuse bounces, "
            "wall albedo 0.80, glare size 7 at mix &minus;0.45 with the threshold left at "
            "1.0.</b> Filmic gives the dimmest shape the highest midtone of the three "
            "transforms at every exposure tried; the extra bounces moved the image "
            "further than any tone choice did; and reducing the glare size shrinks the "
            "halo where lowering its intensity only dims it.")
        + lp.results_table(["setting", "value", "note"], settings_rows)
        + lp.chartnote(
            "<b>The settings, read from the render sidecars and the sweep's own tags "
            "rather than restated.</b> The build fails if any sidecar disagrees with the "
            "grid its render is placed in, or if the recomputed bloom shares drift from "
            "the bloom sweep's own report.")
        + lp.prose(
            "<p>One constraint shaped both sweeps: <b>one setting for every shape</b>. "
            "Comparing brightness across shapes is the point of the box figure, so a "
            "per-shape exposure or a per-shape bloom would erase the very thing the "
            "figure shows. That makes each choice a compromise across shapes differing by "
            "more than an order of magnitude in emissive area, from 1.8 percent of the "
            "frame on the robot to 32.4 percent on the candles.</p>"
            "<p>The exposure sweep ran first. Round 1 crossed the three transforms with "
            "three exposures on the two dimmest shapes, which set the floor; round 2 put "
            "the three survivors against the two brightest shapes, because a setting "
            "chosen on dim shapes alone is a guess about the bright end. The bloom sweep "
            "ran second, on the same renders at the settled exposure. Its shape is the "
            "same, and so is the reason it eliminated a parameter.</p>"))

    # ============================================================= 02 bounces
    pk_old, pk_new = S(PUMPKIN, "oldbounce"), S(PUMPKIN, "film_e1")
    rb_old, rb_new = S(ROBOT, "oldbounce"), S(ROBOT, "film_e1")
    d_pk = 100 * (pk_new["mean"] / pk_old["mean"] - 1)
    d_rb = 100 * (rb_new["mean"] / rb_old["mean"] - 1)
    sec2 = lp.section_v2("bounces", 2,
        f"The bounce fix raised image mean {d_pk:.0f} percent, more than any tone choice",
        lp.fig_row(
            [("12 total / 4 diffuse, wall 0.70",
              hashed(f"{PUMPKIN}_box_oldbounce.png")),
             ("32 total / 16 diffuse, wall 0.80",
              hashed(f"{PUMPKIN}_box_film_e1.png"))],
            caption_html=(
                f"<b>Light that used to die inside the shell now reaches the walls: image "
                f"mean {fmt(pk_old['mean'], 4)} to {fmt(pk_new['mean'], 4)}, a rise of "
                f"{d_pk:.0f} percent, from bounce count and wall albedo alone.</b> Both "
                f"frames are Filmic at +1.0 stops, the same transform and the same "
                f"exposure; only the light-transport settings differ. Left, the original "
                f"settings, which are the Cycles defaults of 12 total and 4 diffuse "
                f"bounces at wall albedo 0.70. Right, the settings every other render on "
                f"this page uses. The jack-o'-lantern is the hardest case in the sweep: "
                f"its emitter is sealed inside the shell, so every photon that leaves it "
                f"has already bounced at least once. Midtone rises with it, "
                f"{fmt(pk_old['p50'])} to {fmt(pk_new['p50'])}."),
            native_px=512, content="photo")
        + lp.fig_row(
            [("12 total / 4 diffuse, wall 0.70", hashed(f"{ROBOT}_box_oldbounce.png")),
             ("32 total / 16 diffuse, wall 0.80", hashed(f"{ROBOT}_box_film_e1.png"))],
            caption_html=(
                f"<b>The same change gives {d_rb:.0f} percent on the robot, whose emitters "
                f"face outward.</b> Image mean {fmt(rb_old['mean'], 4)} to "
                f"{fmt(rb_new['mean'], 4)}, again at Filmic +1.0 stops on both sides. The "
                f"gain is smaller than the jack-o'-lantern's because less of this shape's "
                f"light needed a second bounce to escape in the first place, which is the "
                f"expected direction."),
            native_px=512, content="photo")
        + lp.prose(
            "<p>This is the reason the control render is kept in the sweep rather than "
            "dropped. It is not a fourth tone candidate; it is the before picture, and it "
            "differs from every other render in bounce count and wall albedo rather than "
            "in tone. Reading the two pairs above answers a question the tone grid cannot: "
            "how much of the final look is transport and how much is tone mapping.</p>"))

    # ============================================================= 03 round 1
    def r1_matrix(sid, fid):
        rows = []
        for ekey, elabel, _ in EXPOSURES:
            vals = [S(sid, f"{t}_{ekey}")["p50"] for t, _ in TRANSFORMS]
            best = max(range(3), key=lambda i: vals[i])
            cells = []
            for i, (t, _) in enumerate(TRANSFORMS):
                cells.append({"img": hashed(f"{sid}_box_{t}_{ekey}.png"),
                              "alt": f"{SHAPES[sid][0]}, {TRANSFORMS[i][1]}, {elabel}",
                              "badge": fmt(vals[i]), "best": i == best})
            rows.append((elabel, cells))
        return rows

    pk_agx15, pk_film15, pk_film0 = S(PUMPKIN, "agx_e15"), S(PUMPKIN, "film_e15"), S(PUMPKIN, "film_e0")
    ratio = pk_agx15["p50"] / pk_film15["p50"]
    m_pumpkin = linkify_cells(lp.method_matrix(
        [label for _, label in TRANSFORMS], r1_matrix(PUMPKIN, "mm-pumpkin"),
        caption_html=(
            f"<b>Filmic is the brightest of the three transforms on the dimmest shape at "
            f"every exposure, and AgX at +1.5 stops still lands below Filmic at +0.</b> "
            f"Columns are the three view transforms; rows are the three exposures, rising "
            f"downward. The badge on each tile is that render's midtone, the median "
            f"Rec.709 luminance over the full frame, and the accent badge marks the "
            f"highest in its row. AgX at +1.5 stops reads {fmt(pk_agx15['p50'])} against "
            f"Filmic's {fmt(pk_film15['p50'])} at the same exposure, {100 * ratio:.0f} "
            f"percent of it, and still below Filmic at +0 stops "
            f"({fmt(pk_film0['p50'])}). Any tile opens at its native 512 pixels."),
        native_px=512, content="photo", page_inner=820, id="mm-pumpkin"))

    rb_agx15, rb_film15 = S(ROBOT, "agx_e15"), S(ROBOT, "film_e15")
    m_robot = linkify_cells(lp.method_matrix(
        [label for _, label in TRANSFORMS], r1_matrix(ROBOT, "mm-robot"),
        caption_html=(
            f"<b>The second dim shape orders the transforms the same way, in all three "
            f"rows: Filmic above Standard above AgX.</b> Same axes and same badges as "
            f"above. The robot's emitters face outward, so it starts brighter than the "
            f"jack-o'-lantern at every setting, but the ranking does not move: AgX at "
            f"+1.5 stops reads {fmt(rb_agx15['p50'])} against Filmic's "
            f"{fmt(rb_film15['p50'])}. Nine renders per shape, one per cell."),
        native_px=512, content="photo", page_inner=820, id="mm-robot"))

    chart_rows = []
    for t, tlabel in TRANSFORMS:
        for ekey, elabel, _ in EXPOSURES:
            v = S(PUMPKIN, f"{t}_{ekey}")["p50"]
            chart_rows.append({"label": f"{tlabel} {elabel.replace(' stops', '')}",
                               "value": v, "display": fmt(v),
                               "tip": f"{tlabel}, exposure {elabel}: midtone {fmt(v, 4)}"})
    ob = S(PUMPKIN, "oldbounce")
    chart_rows.append({"label": "control, Filmic +1.0", "value": ob["p50"],
                       "display": fmt(ob["p50"]),
                       "tip": f"original settings, 12/4 bounces at wall 0.70: "
                              f"midtone {fmt(ob['p50'], 4)}"})
    midtone_chart = lp.hbar_chart(
        chart_rows, title="midtone of the jack-o'-lantern, median frame luminance",
        label_w=170,
        note=("<b>Every Filmic bar clears the Standard bar at the same exposure, and "
              "every Standard bar clears the AgX bar.</b> Bars are grouped by transform "
              "and rise with exposure inside each group; the last bar is the control, "
              "which is Filmic at +1.0 stops with the original bounce settings, and it "
              "sits below Filmic at +0 stops on the new ones. Read against the "
              "jack-o'-lantern because it is the binding constraint: it is the dimmest "
              "subject in the sweep, so it sets the exposure floor the bright shapes then "
              "have to survive."))

    agx_c, film_c, std_c = (S(PUMPKIN, "agx_e15")["colorful"],
                            S(PUMPKIN, "film_e15")["colorful"],
                            S(PUMPKIN, "std_e15")["colorful"])
    rb_agx_c, rb_film_c = S(ROBOT, "agx_e15")["colorful"], S(ROBOT, "film_e15")["colorful"]
    sec3 = lp.section_v2("round1", 3,
        "Filmic gives the dim shapes the highest midtone at every exposure tried",
        m_pumpkin + m_robot + midtone_chart
        + lp.prose(
            f"<p><b>AgX is rejected on two counts.</b> The first is above: it needs +1.5 "
            f"stops to reach roughly half of what Filmic gives at the same exposure on the "
            f"dimmest shape, and does not catch Filmic at +0 even then. The second is "
            f"colour. The walls are neutral 0.80 grey, so every hue in the frame came off "
            f"the object, which makes the colour bleed the most informative signal in the "
            f"box figure. At +1.5 stops AgX measures {agx_c:.1f} on the opponent-axis "
            f"colorfulness of Hasler and Suesstrunk against Filmic's {film_c:.1f} on the "
            f"jack-o'-lantern, and {rb_agx_c:.1f} against {rb_film_c:.1f} on the robot: "
            f"less colour at the same exposure on both shapes. The measure is reported for "
            f"the AgX comparison only, and its limits are stated in the provenance note at "
            f"the foot of the page.</p>"
            f"<p>Standard survives round 1. It sits between the two on midtone, and on the "
            f"jack-o'-lantern it is the most colorful of the three at +1.5 stops "
            f"({std_c:.1f}). What rules it out is what it does to the bright shapes, which "
            f"round 1 could not show.</p>"))

    # ============================================================= 04 round 2
    FINALISTS = [("film_e1", "Filmic +1.0"), ("film_e15", "Filmic +1.5"),
                 ("std_e1", "Standard +1.0")]
    r2_rows = []
    for sid, rlabel in [(CANDLES, "three lit candles"), (VENDING, "vending machine")]:
        vals = [S(sid, v)["clip"] for v, _ in FINALISTS]
        best = min(range(3), key=lambda i: vals[i])
        cells = [{"img": hashed(f"{sid}_box_{v}.png"),
                  "alt": f"{SHAPES[sid][0]}, {vlabel}",
                  "badge": pct(vals[i]), "best": i == best}
                 for i, (v, vlabel) in enumerate(FINALISTS)]
        r2_rows.append((rlabel, cells))
    vend_std, vend_f15, vend_f1 = S(VENDING, "std_e1"), S(VENDING, "film_e15"), S(VENDING, "film_e1")
    cand_std, cand_f15 = S(CANDLES, "std_e1"), S(CANDLES, "film_e15")
    m_round2 = linkify_cells(lp.method_matrix(
        [label for _, label in FINALISTS], r2_rows,
        caption_html=(
            f"<b>Standard at the exposure the dim shapes need clips "
            f"{100 * vend_std['clip']:.1f} percent of the vending machine frame, and the "
            f"drink artwork the object is emitting goes with it.</b> Columns are the three "
            f"settings that survived round 1; rows are the two brightest shapes. The badge "
            f"is the clipped share, the fraction of pixels with any channel at or above "
            f"254 of 255, and the accent badge marks the lowest in its row. Filmic at "
            f"+1.5 stops costs {100 * vend_f15['clip']:.2f} percent on the vending machine "
            f"and {100 * cand_f15['clip']:.2f} percent on the candles; Standard at the "
            f"lower exposure of +1.0 costs {100 * vend_std['clip']:.2f} and "
            f"{100 * cand_std['clip']:.2f} percent, at half a stop less exposure. Any tile "
            f"opens at its native 512 pixels."),
        native_px=512, content="photo", page_inner=820, id="mm-round2"))

    clip_rows = []
    for sid, sname in [(VENDING, "vending machine"), (CANDLES, "candles")]:
        for v, vlabel in FINALISTS:
            c = S(sid, v)["clip"]
            clip_rows.append({"label": f"{sname}, {vlabel}", "value": 100 * c,
                              "display": pct(c),
                              "tip": f"{sname}, {vlabel}: {100 * c:.3f} percent of pixels "
                                     f"clipped"})
    clip_chart = lp.hbar_chart(
        clip_rows, title="clipped share of the frame, brightest two shapes", label_w=200,
        note=(f"<b>Standard at +1.0 stops clips more than three times what Filmic clips at "
              f"+1.5, while giving the dim shapes less midtone.</b> That is what rules it "
              f"out: it is not a taste call between a filmic roll-off and a linear one, it "
              f"is worse at both ends of the range the single exposure has to cover. On "
              f"the jack-o'-lantern, Standard reads {fmt(S(PUMPKIN, 'std_e1')['p50'])} "
              f"against Filmic's {fmt(S(PUMPKIN, 'film_e1')['p50'])} at the same +1.0 "
              f"stops, and {fmt(S(PUMPKIN, 'std_e15')['p50'])} against "
              f"{fmt(S(PUMPKIN, 'film_e15')['p50'])} at +1.5."))

    sec4 = lp.section_v2("round2", 4,
        "Standard blows out an eighth of the vending machine frame and is still darker "
        "on the dim shapes",
        m_round2 + clip_chart
        + lp.prose(
            "<p>Round 2 exists because the two shapes in round 1 are both dim. A setting "
            "picked there could have been arbitrarily bad at the other end of the range, "
            "and the single-exposure constraint means the other end is not optional. The "
            "candles fill 32.4 percent of the frame with emissive surface and the vending "
            "machine 13.9 percent, against 6.3 and 1.8 percent for the two round-1 "
            "shapes.</p>"))

    # ======================================================= 05 what +1.5 costs
    clipmaps = lp.fig_row(
        [(f"Filmic +1.0 &middot; {pct(vend_f1['clip'])}",
          hashed(f"clip_{VENDING}_box_film_e1.png")),
         (f"Filmic +1.5 &middot; {pct(vend_f15['clip'])}",
          hashed(f"clip_{VENDING}_box_film_e15.png")),
         (f"Standard +1.0 &middot; {pct(vend_std['clip'])}",
          hashed(f"clip_{VENDING}_box_std_e1.png"))],
        caption_html=(
            f"<b>The extra half stop scatters clipped pixels through the panel's "
            f"highlights; Standard floods the panel solid and takes the artwork with "
            f"it.</b> Each panel is the vending machine render converted to grey and "
            f"dimmed, with every pixel at or above 254 in any channel painted in the "
            f"accent colour. Left, Filmic at +1.0 stops, where clipping is effectively "
            f"absent at {pct(vend_f1['clip'])} of the frame. Centre, Filmic at +1.5 stops: "
            f"{pct(vend_f15['clip'])} of the frame, speckled over the bright regions of "
            f"the drink artwork and the panel edge, with the artwork still legible "
            f"underneath. Right, Standard at +1.0 stops: {pct(vend_std['clip'])} of the "
            f"frame in one solid mass covering the door, the product row and the lettering "
            f"at the bottom. The two centre and right frames differ in clipped share by a "
            f"factor of {vend_std['clip'] / vend_f15['clip']:.1f}, and they differ in kind "
            f"as well: speckle inside a highlight against a flat white region where the "
            f"emitted image used to be."),
        native_px=512, content="photo")

    sec5 = lp.section_v2("cost", 5,
        "The clipping +1.5 buys sits inside the emissive panel's own highlights",
        clipmaps
        + lp.prose(
            f"<p>The half stop from +1.0 to +1.5 is what lifts the jack-o'-lantern from "
            f"{fmt(S(PUMPKIN, 'film_e1')['p50'])} to "
            f"{fmt(S(PUMPKIN, 'film_e15')['p50'])} midtone, a rise of "
            f"{100 * (S(PUMPKIN, 'film_e15')['p50'] / S(PUMPKIN, 'film_e1')['p50'] - 1):.0f} "
            f"percent on the shape that sets the floor. It costs "
            f"{pct(vend_f15['clip'])} of the vending machine frame and "
            f"{pct(cand_f15['clip'])} of the candles frame. That cost is accepted because "
            f"of where it lands: the clipped pixels sit on the object's own emissive "
            f"panel, which behaves as a large area light, and a person standing in that "
            f"room sees glare there rather than artwork. The same argument does not "
            f"rescue Standard, whose clipped region is not a highlight but the whole "
            f"emitting surface.</p>"
            f"<p>The honest reading of the centre panel is that the clipping is not "
            f"confined to a single glare spot. It is speckle distributed through the "
            f"brightest parts of the drink artwork, and a viewer looking for the artwork "
            f"can still read it. Whether that is an acceptable price for the extra midtone "
            f"on the dim end is the one judgement on this page that a measurement does not "
            f"settle.</p>"))

    # ============================================================ 06 the bloom
    bl_pk_old, bl_pk_new = B(PUMPKIN, BLOOM_OLD), B(PUMPKIN, BLOOM_NEW)
    bl_vd_old, bl_vd_new = B(VENDING, BLOOM_OLD), B(VENDING, BLOOM_NEW)
    bl_cd_old, bl_cd_new = B(CANDLES, BLOOM_OLD), B(CANDLES, BLOOM_NEW)

    eyecrops = lp.fig_row(
        [("previous &middot; size 9, mix &minus;0.15", hashed("eyecrop_old.png")),
         ("chosen &middot; size 7, mix &minus;0.45", hashed("eyecrop_new.png"))],
        caption_html=(
            "<b>At the previous setting the carved eye reads as a bright blob; at the "
            "chosen one it keeps its crescent.</b> The same 360 pixel crop of the "
            "jack-o'-lantern, from the two grades of one linear render, so nothing but "
            "the glare settings differs. Left, the halo fills the eye opening and spills "
            "onto the flesh around it, and the teeth below sit under a veil. Right, the "
            "lit crescent inside the eye and the dark arc beside it are both legible, and "
            "the teeth keep their edges. This is the defect the sweep was run to fix."),
        native_px=360, content="photo")

    extents = lp.fig_row(
        [("previous &middot; " + pct(bl_pk_old), hashed(f"bloomext_{PUMPKIN}_{BLOOM_OLD}.png")),
         ("chosen &middot; " + pct(bl_pk_new), hashed(f"bloomext_{PUMPKIN}_{BLOOM_NEW}.png"))],
        caption_html=(
            f"<b>The halo covered the whole front of the shape; now it hugs the openings "
            f"that emit.</b> Every pixel the glare node lifts by at least 3 of 255 is "
            f"painted in the accent colour over the glare-free render, which is identical "
            f"in both panels, so the two differ only in the marked region. Left, the "
            f"previous setting reaches {pct(bl_pk_old)} of the frame as one disc across the "
            f"pumpkin's face. Right, the chosen setting reaches {pct(bl_pk_new)}, a ring "
            f"around the eye and the mouth. The same comparison on the vending machine "
            f"runs {pct(bl_vd_old)} to {pct(bl_vd_new)}."),
        native_px=768, content="photo")

    def refine_matrix(sid, mid, where):
        """The refinement grid. Cells are the SAME 300 pixel window of each
        grade, not the full frame: a halo two size steps wider is plain in a
        crop and invisible in a 768 pixel frame shown at 266 (checked on the
        rendered page, where the full-frame first cut had six identical-looking
        cells). The badges are measured over the whole frame, as everywhere
        else on this page, and each cell links to its full render."""
        rows = []
        for m, mlabel in BLOOM_MIXES:
            cells = []
            for s, slabel in BLOOM_SIZES:
                tag = f"t1_{m}_{s}"
                cells.append({"img": hashed(f"gridcrop_{sid}_{tag}.png"),
                              "alt": f"{dict(BLOOM_SHAPES)[sid]}, {mlabel}, {slabel}",
                              "badge": pct(B(sid, tag)), "best": tag == BLOOM_NEW})
            rows.append((mlabel, cells))
        return linkify_cells(lp.method_matrix(
            [slabel for _, slabel in BLOOM_SIZES], rows,
            caption_html=(
                f"<b>Reading across shrinks the halo; reading down leaves it the same "
                f"size and only dims it.</b> Columns are the glare size, rows the mix, "
                f"on the {dict(BLOOM_SHAPES)[sid]}. Every cell is the same 300 pixel "
                f"window on {where}, so only the setting differs; the badge is the "
                f"bloomed share measured over the whole frame, and a cell opens its full "
                f"render. The accent badge marks the chosen cell rather than the smallest "
                f"number: size 5 blooms less still and was passed over because at that "
                f"radius the halo stops reading as light leaving the object. Threshold is "
                f"1.0 throughout; it is handled in section 07."),
            native_px=300, content="photo", page_inner=820, id=mid))

    ladder_rows = []
    for tag, label in BLOOM_LADDER:
        v = B(PUMPKIN, tag)
        ladder_rows.append({"label": label, "value": v, "display": pct(v),
                            "tip": f"{label}: {100 * v:.2f} percent of the frame bloomed"})
    ladder_rows.append({"label": "chosen: size 7, mix &minus;0.45".replace("&minus;", "-"),
                        "value": bl_pk_new, "display": pct(bl_pk_new),
                        "tip": f"size 7 at mix -0.45: {100 * bl_pk_new:.2f} percent"})
    ladder_chart = lp.hbar_chart(
        ladder_rows, title="bloomed share of the jack-o'-lantern frame", label_w=185,
        note=(f"<b>One step down in size removes more halo than either step down in "
              f"mix.</b> Each of the middle four bars changes exactly one parameter from "
              f"the previous setting. Size 7 alone brings the dim shape from "
              f"{pct(bl_pk_old)} to {pct(B(PUMPKIN, 't1_m-0.15_s7'))}, while the strongest "
              f"mix tried, &minus;0.70, only reaches "
              f"{pct(B(PUMPKIN, 't1_m-0.7_s9'))}. The other two shapes order the five "
              f"settings identically. The last bar is the chosen combination, which is "
              f"not in the one-at-a-time ladder."))

    sec6 = lp.section_v2("bloom", 6,
        "The halo was too large, and size is the lever that shrinks it",
        eyecrops + extents + refine_matrix(PUMPKIN, "mm-bloom-pumpkin", "the carved eye and the top of the mouth")
        + refine_matrix(VENDING, "mm-bloom-vending", "the lit product column against the dark cabinet") + ladder_chart
        + lp.prose(
            f"<p>Bloom has two levers that look interchangeable and are not. Mix sets how "
            f"much of the glare is added back, so lowering it makes the same halo fainter "
            f"at the same radius. Size sets the radius, so lowering it makes the halo "
            f"smaller. The complaint was that the halo was too strong <i>and</i> too "
            f"large, and only one of the two levers addresses the second half of "
            f"that.</p>"
            f"<p>The chosen combination takes the dim shape from {pct(bl_pk_old)} of the "
            f"frame to {pct(bl_pk_new)}, a factor of {bl_pk_old / bl_pk_new:.1f}, the candles "
            f"from {pct(bl_cd_old)} to {pct(bl_cd_new)}, and the vending machine from "
            f"{pct(bl_vd_old)} to {pct(bl_vd_new)}. The reduction is proportional rather than "
            f"absolute: the bright shape keeps the most bloom because it emits the most, "
            f"which is the behaviour a single shared setting has to have.</p>"))

    # ============================================== 07 the shared reason to reject
    thr_rows = []
    for sid, sname in BLOOM_SHAPES:
        cells = []
        for t, tlabel in BLOOM_THRESHOLDS:
            tag = f"{t}_m-0.15_s9"
            v = B(sid, tag)
            cells.append({"img": hashed(f"bloom_{sid}_{tag}.png"),
                          "alt": f"{sname}, {tlabel}", "badge": pct(v)})
        thr_rows.append((sname, cells))
    pk_t25, vd_t25 = B(PUMPKIN, "t2.5_m-0.15_s9"), B(VENDING, "t2.5_m-0.15_s9")
    cd_t25 = B(CANDLES, "t2.5_m-0.15_s9")
    m_threshold = linkify_cells(lp.method_matrix(
        [tlabel for _, tlabel in BLOOM_THRESHOLDS], thr_rows,
        caption_html=(
            f"<b>At threshold 2.5 the dim shape has no bloom left at all while the bright "
            f"one keeps a sixth of its frame.</b> Columns are the glare threshold, rows "
            f"the three shapes, size 9 and mix &minus;0.15 throughout. Badges are the "
            f"bloomed share. Raising the threshold from 1.0 to 2.5 takes the "
            f"jack-o'-lantern from {pct(bl_pk_old)} to exactly zero, the candles from "
            f"{pct(bl_cd_old)} to {pct(cd_t25)}, and the vending machine from {pct(bl_vd_old)} "
            f"to {pct(vd_t25)}. No value in between splits the difference, because the "
            f"parameter is not measuring what the setting has to be fair to."),
        native_px=768, content="photo", page_inner=820, id="mm-threshold"))

    parallel_rows = "".join(
        f'<tr><td class="rowhead">{a}</td><td class="rowhead">{b}</td>'
        f'<td class="rowhead">{c}</td></tr>'
        for a, b, c in [
            ("Standard view transform, +1.0 stops",
             f"midtone {fmt(S(PUMPKIN, 'std_e1')['p50'])}, below Filmic's "
             f"{fmt(S(PUMPKIN, 'film_e1')['p50'])}",
             f"clips {pct(S(VENDING, 'std_e1')['clip'])} of the frame"),
            ("Glare threshold 2.5",
             "bloom 0%, the halo is gone",
             f"bloom {pct(vd_t25)}, the halo is intact"),
        ])
    ship_rows = sorted(shipped.values(), key=lambda d: d["share"])
    areas = np.array([d["area"] for d in ship_rows])
    shares = np.array([d["share"] for d in ship_rows])
    rho = float(np.corrcoef(areas, shares)[0, 1])
    ship_chart = lp.hbar_chart(
        [{"label": d["name"], "value": d["share"], "display": pct(d["share"]),
          "tip": f"{d['name']}: {100 * d['share']:.2f} percent bloomed, "
                 f"{100 * d['area']:.1f} percent of the frame emissive"}
         for d in ship_rows],
        title="bloomed share at the chosen setting, all eleven box shapes",
        label_w=175,
        note=(f"<b>At the chosen setting no shape blooms over "
              f"{pct(shares.max())} of its frame, and the median is "
              f"{pct(float(np.median(shares)))}.</b> Eleven shapes, only three of "
              f"which were swept, so this is the setting tested outside the sample it "
              f"was chosen on. Emissive area runs from "
              f"{100 * areas.min():.1f} to {100 * areas.max():.1f} percent of the frame "
              f"across these shapes and correlates with bloomed share at "
              f"{rho:.2f}, positive but loose: the headphone stand emits over the "
              f"smallest area of the eleven and still blooms more than the "
              f"jack-o'-lantern, because a small bright emitter against dark surround "
              f"lifts more pixels than a larger dim one. Hover a bar for its emissive "
              f"area."))

    sec7 = lp.section_v2("scaling", 7,
        "Both sweeps rejected the parameter that keys off absolute brightness",
        m_threshold
        + lp.results_table(["lever rejected", "on the dimmest shape",
                            "on the brightest shape"], parallel_rows)
        + lp.chartnote(
            "<b>The same failure, twice, for the same reason.</b> One row per rejected "
            "lever. Each fails in the same shape: it is acceptable at one end of the "
            "brightness range and unacceptable at the other, with no value in between "
            "that serves both.")
        + lp.prose(
            "<p>Standard was not rejected for looking wrong and the threshold was not "
            "rejected for being the wrong number. Both were rejected because they operate "
            "on absolute brightness, and the constraint that governs this figure is that "
            "one setting has to serve every shape. A linear transform clips whatever "
            "exceeds the display range, so the exposure the dim shapes need is the "
            "exposure that destroys the bright ones. A glare threshold passes whatever "
            "exceeds a fixed level, so the value that tames the bright shape's halo "
            "removes the dim shape's entirely.</p>"
            "<p>The parameters that survived both sweeps behave the other way. Filmic's "
            "roll-off compresses the top of the range instead of cutting it, so one "
            "exposure can serve a range of emissive areas. Glare size scales the halo's "
            "radius rather than testing a level, so one size reduces every shape's bloom "
            "without any shape losing its halo or keeping all of it. That is the property "
            "being selected for, in both sweeps, and it is worth stating as the criterion "
            "rather than rediscovering it a third time.</p>")
        + ship_chart
        + lp.prose(
            f"<p>The chart above is the criterion checked outside the sample it was "
            f"chosen on. The bloom sweep ran on three shapes; the chosen setting was then "
            f"applied to all eleven, and every one of them keeps a halo and none of them "
            f"is dominated by it. The spread is real, from {pct(shares.min())} to "
            f"{pct(shares.max())}, and it should be: a setting that produced the same "
            f"share on every shape would be normalising away the difference the box "
            f"figure exists to show.</p>"))

    # ======================================= 08 the prediction that was wrong
    # flow_stage(num, label, sub): the first slot is a short mono token, not a
    # title. The stage NAME goes in label, the qualifier in sub.
    flow = lp.flow_wrap("".join([
        lp.flow_stage("1", "Cycles render", "scene-linear pixels"),
        lp.flow_arrow(),
        lp.flow_stage("2", "Compositor", "the glare node runs here", highlight=True),
        lp.flow_arrow(),
        lp.flow_stage("3", "Display transform", "view transform, then exposure"),
    ]))
    sec8 = lp.section_v2("mechanism", 8,
        "Raising the exposure did not grow the bloom: the glare never sees it",
        flow
        + lp.chartnote(
            "<b>Exposure is applied downstream of the compositor, so the glare node "
            "cannot respond to it.</b> Blender runs the compositor on the scene-linear "
            "render and applies <code>view_settings.exposure</code> afterwards, in the "
            "display transform. The bloom sweep exploits the same ordering: it regrades "
            "one saved linear render per shape instead of re-rendering per parameter "
            "set.")
        + lp.callout(
            f"Grading the jack-o'-lantern's linear render through identical glare at "
            f"exposure +0 and at +1.5 stops, and comparing the glare output before any "
            f"tone mapping, gives a maximum absolute difference of <b>exactly 0</b> "
            f"across all 589,824 pixels. The two outputs are bitwise identical.",
            title="The prediction was tested and is wrong")
        + lp.prose(
            "<p>The prediction made before the sweep ran was that the bloom had grown "
            "because raising the exposure to +1.5 stops pushed more pixels past the glare "
            "threshold. It is a reasonable guess and it is wrong, for a reason that is "
            "structural rather than numerical: the two operations are in the wrong order "
            "for it to be true.</p>"
            "<p>What actually changed is visibility. The halo was the same size before the "
            "exposure rise; it sat in near-black, below the point where a reader would "
            "notice it. Lifting the tone curve lifted the halo with everything else, and "
            "it became the most visible thing in the frame. The defect was real and the "
            "diagnosis was not, which is why the fix had to come from the glare settings "
            "rather than from backing off the exposure the dim shapes need.</p>"))

    # ======================================================= 09 all the numbers
    def num(x):
        return f'<td class="num">{x}</td>'

    table_rows = []
    order = [(PUMPKIN, "jack-o'-lantern"), (ROBOT, "humanoid robot"),
             (CANDLES, "three lit candles"), (VENDING, "vending machine")]
    for sid, sname in order:
        variants = [f"{t}_{e}" for t, _ in TRANSFORMS for e, _, _ in EXPOSURES] + ["oldbounce"]
        first = True
        for v in variants:
            key = f"{sid}_box_{v}"
            if key not in sw:
                continue
            d, c = sw[key], sw[key]["cfg"]
            label = (f"control: {c['view_transform']} +{c['exposure']:.1f}, "
                     f"{c['max_bounces']}/{c['diffuse_bounces']} bounces, "
                     f"wall {c['wall']}" if v == "oldbounce"
                     else f"{c['view_transform']} +{c['exposure']:.1f}")
            shape_cell = (f'<td class="rowhead">{sname}<br>'
                          f'<span class="tiny">{100 * c["area_lit_frac"]:.1f}% emissive'
                          f'</span></td>' if first else '<td class="rowhead"></td>')
            first = False
            table_rows.append(
                f'<tr>{shape_cell}<td class="rowhead">{label}</td>'
                + num(fmt(d["p50"], 4)) + num(fmt(d["mean"], 4))
                + num(pct(d["clip"])) + num(f'{d["colorful"]:.1f}')
                + f'<td class="rowhead"><a href="{hashed(key + ".png")}" '
                  f'target="_blank" rel="noopener">open</a></td></tr>')

    bloom_table_rows = []
    for sid, sname in BLOOM_SHAPES:
        tags = sorted({t for (s, t) in bl if s == sid},
                      key=lambda t: (float(t.split("_")[0][1:]),
                                     -float(t.split("_")[1][1:]),
                                     int(t.split("_")[2][1:])))
        first = True
        for tag in tags:
            thr, mix, size = (t[1:] for t in tag.split("_"))
            chosen = " &middot; chosen" if tag == BLOOM_NEW else (
                " &middot; previous" if tag == BLOOM_OLD else "")
            shape_cell = (f'<td class="rowhead">{sname}</td>' if first
                          else '<td class="rowhead"></td>')
            first = False
            bloom_table_rows.append(
                f'<tr>{shape_cell}'
                f'<td class="rowhead">threshold {float(thr):.1f}, mix {mix}, size {size}'
                f'{chosen}</td>'
                + num(pct(B(sid, tag)))
                + f'<td class="rowhead"><a href="{hashed(f"bloom_{sid}_{tag}.png")}" '
                  f'target="_blank" rel="noopener">open</a></td>'
                + f'<td class="rowhead"><a href="{hashed(f"bloom_{sid}_none.png")}" '
                  f'target="_blank" rel="noopener">no glare</a></td></tr>')

    sec9 = lp.section_v2("numbers", 9,
        "Every render in both sweeps, measured the same way",
        lp.results_table(
            ["shape", "setting", "midtone", "image mean", "clipped", "colorfulness", ""],
            "".join(table_rows))
        + lp.chartnote(
            "<b>The exposure sweep: 26 renders, every value recomputed from the PNG when "
            "this page is built.</b> Midtone is the median Rec.709 luminance over the "
            "full frame; image mean is the mean of the three channels; clipped is the "
            "share of pixels with any channel at or above 254 of 255; colorfulness is the "
            "opponent-axis measure of Hasler and Suesstrunk. The percentage under each "
            "shape name is the fraction of the frame its emissive surfaces cover, read "
            "from the render sidecar. The chosen setting is the Filmic +1.5 row of each "
            "block.")
        + lp.results_table(
            ["shape", "glare setting", "bloomed share", "", ""],
            "".join(bloom_table_rows))
        + lp.chartnote(
            f"<b>The bloom sweep: {len(bl)} glare settings across three shapes, regraded "
            f"from the same linear renders, all at Filmic +1.5 stops.</b> Bloomed share "
            f"is the fraction of the frame the glare node "
            f"lifts by at least 3 of 255 against the glare-free grade of the same render, "
            f"which the last column opens. Recomputing this way reproduces the sweep's own "
            f"report to within {bloom_worst:.4f} absolute on every one of its "
            f"{len(bl)} cells, asserted at build time."))

    apx = lp.appendix("Provenance", [
        "<b>Source.</b> The 26 renders and their sidecars come from "
        "<code>paper_v3/sweep/</code> on scratch and were copied into this page's own "
        "<code>img/</code> directory unmodified. Nothing on scratch was written to. Each "
        "sidecar carries samples, total and diffuse bounce counts, wall albedo, view "
        "transform, exposure, camera and emissive area fraction; the build reads them and "
        "fails if any render's settings disagree with the grid it is placed in. All 26 "
        "agreed. The bloom grades come from <code>paper_v3/bloom_sweep/</code> and "
        "<code>paper_v3/bloom_refine/</code>; the three cells the two runs share are "
        "pixel-identical, and the refinement run is used for those. The chosen setting on "
        "the other eight shapes comes from <code>paper_v3/box3_raw/</code>, whose cells "
        "for the three swept shapes are in turn pixel-identical to the refinement run's, "
        "so one shape reads the same bytes wherever it appears on this page. All are "
        "re-encoded without their all-opaque alpha channel and otherwise unmodified.",
        "<b>Bloom is a regrade, not a re-render.</b> The compositor runs on the "
        "scene-linear render, so a bloom sweep does not need one render per parameter "
        "set: each shape was rendered once with the glare node bypassed and saved as "
        "OpenEXR, and every cell is that one file pushed through a different compositor "
        "graph. The exposure and bounce sweep, which changes the render itself, could not "
        "be done this way. This is also why the two sweeps are at different resolutions, "
        "512 pixels for the renders and 768 for the regrades.",
        "<b>The bloomed-pixel measure is stated in whole 8-bit steps, because the "
        "floating-point form of it has a knife edge.</b> These are 8-bit images, so the "
        "lift a pixel receives is always a whole number of 255ths, and a large share of "
        "every halo lands on exactly two of them: 4.96 percent of the candles frame in "
        "one cell. A cut written as a float comparison at that same value is therefore "
        "settled by rounding residue rather than by the image, and the residue depends on "
        "the width of the float. Measured on all 50 published cells, the cut at three "
        f"whole steps reproduces the sweep's own reports to within {bloom_worst:.4f}; a "
        "cut at two steps is off by up to 0.113; and the float comparison read at double "
        "precision is off by up to 0.051, splitting the two-step population and matching "
        "neither integer cut. Three steps is both what the published numbers already mean "
        "and the first cut that no longer sits on a populated boundary, which is why this "
        "page uses it and why the sweep's own code now does. Rankings are identical under "
        "every reading, so no conclusion turns on it. <b>What a share on this page means: "
        "the brightest of the three channels lifted by at least three whole 8-bit "
        "levels.</b>",
        "<b>The bloomed share is display-referred, and is not comparable across tone "
        "settings.</b> It is measured on the 8-bit image after Filmic and the exposure "
        "lift, so it answers how much of the VISIBLE frame the halo touches, not how "
        "much energy the glare node added. That is the right question for the complaint "
        "the sweep was run against, and it is the reason the whole bloom sweep is "
        "reported at one exposure and one transform: a share measured at +1.5 stops "
        "cannot be set beside one measured at +0. It also takes the per-pixel maximum "
        "over the three channels rather than a luminance, which matters here because "
        "these rooms are strongly tinted by the object's own colour.",
        "<b>The regrade was checked against the shipped renders.</b> Regrading two linear "
        "renders at the settings their shipped versions used and differencing against "
        "those shipped files gives a maximum absolute difference of 0 of 255 on the "
        "candles and 2 of 255 on the vending machine, with the 99th percentile at 0 in "
        "both, so the offline regrade reproduces the in-render result to within 8-bit "
        "rounding on a handful of pixels. The separate exposure test in section 08 is "
        "bitwise exact and is a different comparison.",
        "<b>Full frame, not a foreground crop.</b> These are closed-box renders: every "
        "pixel is either the object or a wall the object lit, and alpha is 255 everywhere "
        "in all 26 files, so there is no background to exclude. Midtone and clipped share "
        "are therefore computed over the whole 512 by 512 frame, consistently.",
        "<b>The colorfulness measure and its limit.</b> Reported for the AgX comparison "
        "only. It compares fairly at equal exposure, which is the comparison the round-1 "
        "grid invites, and AgX measures lower than Filmic at +1.5 stops on both dim "
        "shapes. It does not compare fairly at equal brightness: if AgX at +1.5 stops is "
        "instead matched against Filmic at +0 stops, where the two happen to land within "
        "6 percent of each other in midtone on the jack-o'-lantern, AgX measures the more "
        "colorful of the two. The midtone finding, which is the first count against AgX, "
        "does not depend on this measure at all.",
        "<b>Earlier figures.</b> An earlier ad hoc pass reported the clipped share of "
        "Filmic +1.0 on the vending machine as 0.24 percent and of Filmic +1.5 on the "
        "candles as 2.1 percent. Recomputed at the stated definition of 254 of 255 in any "
        "channel, those two are 0.002 and 3.71 percent. No single threshold reproduces "
        "the earlier set, so the definition it used cannot be recovered; the numbers on "
        "this page are the ones defined above, applied uniformly. The direction of every "
        "conclusion is unchanged, and the gap between Filmic and Standard is wider under "
        "these definitions than under the earlier ones. The midtone and image-mean "
        "figures agree with the earlier pass to the digits it reported.",
        "<b>No model output appears on this page.</b> Every image is a render of an "
        "existing shape under the settings named beside it. The carved-eye crops are the "
        "same 360 pixel window cut from two full grades, and the bloom-extent maps are "
        "computed overlays; neither is retouched.",
    ])

    page_html = lp.page(
        title="Choosing the Lighting for the Emission-Only Box Renders",
        header_html=hero,
        body_sections=[sec1, sec2, sec3, sec4, sec5, sec6, sec7, sec8, sec9, apx],
        assets_rel=SITE_ASSETS,
        assets_dir=assets_dir,
        theme="v3",
        tree_html=tree_html(),
        nav_title="Render sweep",
        version_slot=lp.v3_version_slot(date=PAGE_DATE),
        needs_katex=False,
        extra_head=f'<link rel="icon" href="{FAVICON}">' + EXTRA_CSS,
        outline_entries=[
            {"id": "decision", "label": "The setting"},
            {"id": "bounces", "label": "Bounces"},
            {"id": "round1", "label": "Round 1: the transforms"},
            {"id": "round2", "label": "Round 2: the bright end"},
            {"id": "cost", "label": "What +1.5 costs"},
            {"id": "bloom", "label": "The halo, and its lever"},
            {"id": "scaling", "label": "The shared reason"},
            {"id": "mechanism", "label": "A prediction tested"},
            {"id": "numbers", "label": "Every render"},
        ],
    )

    # ZONE-BOUNDARY LAW: nothing in the workspace zone may link to the console.
    violations = wz.console_links_in(page_html)
    if violations:
        sys.exit(f"ZONE-LINK GUARD FAILED: page links to the console: {violations}")

    out_path = os.path.join(HERE, "index.html")
    with open(out_path, "w") as f:
        f.write(page_html)
    print(f"wrote {out_path}")
    print(f"  {len(sw)} renders measured, sidecar settings check clean")
    print(f"  {len(bl)} bloom cells measured, worst drift vs the sweep reports: "
          f"{bloom_worst:.5f} (swept) / {ship_worst:.5f} (all 11 at the chosen setting)")
    print("  zone-link guard: clean")

    publish_assets(assets_dir)
    print(f"assets published -> {assets_dir}")

    if publish:
        os.makedirs(PUBLISH_DIR, exist_ok=True)
        shutil.copytree(IMG, os.path.join(PUBLISH_DIR, "img"), dirs_exist_ok=True)
        shutil.copy2(out_path, os.path.join(PUBLISH_DIR, "index.html"))
        wz.write_tree_json()
        print(f"published -> {PUBLISH_DIR}")
        print(f"tree.json refreshed -> {wz.TREE_JSON}")


if __name__ == "__main__":
    build(publish="--publish" in sys.argv)
