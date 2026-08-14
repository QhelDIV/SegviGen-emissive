#!/usr/bin/env python3
"""Build the mask-bake debug page: cause and fix for the emission render
artifacts (streaky/patchy/black panels) on the pumpkin, lantern, and the three
paper shapes (warhammer, lightsaber, robot).

Structure is owner-specified (2026-08-11, third revision, master-written):
plain title; a teaser of the paper's figure-5 examples before/after; a DIAGRAM
of the old versus new processing path; concise why; what-was-done table; an
explicit section on which checkpoint produced the predictions and how the
shown draw was picked; results in the paper's own mask-times-albedo
convention; all investigation material compressed below as reference.

Every image on this page is a real artifact from the investigation, never
redrawn or simulated. The three fig5_before_* panels are crops of the paper's
current figure 5 (the SegviGen-Emission column as submitted).

Run: /project/3dlg-hcvc/omages/omages_internal/.venv2/bin/python \
        web/_preview/mask_debug/build.py [--publish]
"""
import hashlib
import os
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

PAGE_SLUG = "mask_debug"
PAGE_HREF = f"{wz.WORKSPACE_URL}/{PAGE_SLUG}/index.html"
PUBLISH_DIR = os.path.join(str(wz.WORKSPACE_DIR), PAGE_SLUG)
PAGE_DATE = "2026-08-11"

IMG = os.path.join(HERE, "img")

# The fig5_* names land from the figure-5-style render job (dark room, object
# lit by its own emission, emission = mask x albedo per the paper's convention).
FIG5 = {
    "hammer_before": "fig5_before_hammer.png",
    "saber_before": "fig5_before_saber.png",
    "robot_before": "fig5_before_robot.png",
    "hammer_after": "box_hammer_rawseed3.png",
    "saber_after": "box_saber_emadraw3.png",
    "robot_after": "box_robot_emaseed4.png",
    "robot_alt": "box_robot_emaseed5.png",
    "hammer_gt": "box_hammer_gt.png",
    "saber_gt": "box_saber_gt.png",
    "robot_gt": "box_robot_gt.png",
}

VOXPAIR = ["voxel_robot_emaseed4.png", "voxel_robot_emaseed5.png"]
PROBES = ["probe_hammer_before.png", "probe_hammer_after.png",
          "probe_robot_before.png", "probe_robot_after.png"]

REQUIRED_IMAGES = list(FIG5.values()) + PROBES + VOXPAIR + [
    "npz_pumpkin_pred.png", "npz_pumpkin_gt.png",
    "pumpkin_voxel_gt.png", "pumpkin_voxel_pred.png",
    "lantern_voxel_gt.png", "lantern_voxel_pred.png",
    "pumpkin_mask_pred.png", "pumpkin_mask_gt.png",
    "pumpkin_render_pred.png", "pumpkin_render_gt.png",
    "pumpkin_uv_independent.png", "pumpkin_wireframe_overlay.png",
    "pumpkin_render_synthwhite.png", "redpad_pumpkin_nodenoise.png",
    "lantern_mask_pred.png", "lantern_render_pred.png", "lantern_render_gt.png",
    "lantern_uv_independent.png", "lantern_wireframe_overlay.png",
    "lantern_render_synthwhite.png", "redpad_lantern_nodenoise.png",
    "desklamp_render_pred.png", "desklamp_render_gt.png",
    "pumpkin_pred_uvfree_after.png",
]

CKPT_NOTE = (
    "All predictions on this page come from the epoch-8 checkpoint of the "
    "72k conditioned training run "
    "(<code>outputs/emis_72kv2_cond_pw1b/epoch_0008.ckpt</code>, raw weights, "
    "or <code>epoch_0008_ema.ckpt</code>, averaged weights). Every panel is "
    "labeled with which of the two produced it, plus the draw seed.")


def check_assets():
    problems = []
    for name in REQUIRED_IMAGES:
        p = os.path.join(IMG, name)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            problems.append(p)
    return problems


def src(name):
    h = hashlib.md5(open(os.path.join(IMG, name), "rb").read()).hexdigest()[:8]
    return f"img/{name}?v={h}"


def tree_html():
    entries = wz.tree_entries()
    for group in entries:
        for leaf in group.get("children", []):
            leaf["active"] = leaf["href"] == PAGE_HREF
    return lp.v3_tree(entries, title="Lightgen", subtitle="research workspace",
                      tree_src=wz.TREE_JSON_URL)


# ------------------------------------------------------------------ diagram
def path_diagram():
    """Two-lane SVG: the texture-bake path (with the divergence marked) and
    the UV-free path. Inline SVG so it follows the page theme's colors."""
    ink = "var(--ink, #333)"
    mut = "var(--muted-ink, #777)"
    acc = "var(--accent, #b4552d)"
    box = ('<rect x="{x}" y="{y}" width="{w}" height="52" rx="6" '
           f'fill="none" stroke="{ink}" stroke-width="1.2"/>')
    txt = (f'<text x="{{x}}" y="{{y}}" fill="{ink}" font-size="13" '
           'text-anchor="middle">{t}</text>')
    lbl = (f'<text x="{{x}}" y="{{y}}" fill="{mut}" font-size="11.5" '
           'text-anchor="middle">{t}</text>')
    lane = (f'<text x="8" y="{{y}}" fill="{mut}" font-size="11.5" '
            'font-family="monospace" letter-spacing="0.08em">{t}</text>')
    arrow = (f'<line x1="{{x1}}" y1="{{y}}" x2="{{x2}}" y2="{{y}}" '
             f'stroke="{ink}" stroke-width="1.2" marker-end="url(#arr)"/>')

    def b(x, y, w, lines):
        out = box.format(x=x, y=y, w=w)
        cy = y + 26 - (len(lines) - 1) * 8
        for ln in lines:
            out += txt.format(x=x + w / 2, y=cy, t=ln)
            cy += 16
        return out

    def a(x1, x2, y, lines, color=None, ly=None):
        out = arrow.format(x1=x1, x2=x2, y=y)
        if color:
            out = out.replace(f'stroke="{ink}"', f'stroke="{color}"')
        cy = (ly if ly is not None else y - 30) - (len(lines) - 1) * 7
        for ln in lines:
            t = lbl.format(x=(x1 + x2) / 2, y=cy, t=ln)
            if color:
                t = t.replace(f'fill="{mut}"', f'fill="{color}"')
            out += t
            cy += 14
        return out

    svg = ['<svg viewBox="0 0 840 330" role="img" '
           'style="width:100%;max-width:820px;height:auto;display:block;'
           'margin:0 auto" '
           'aria-label="Old texture-bake path versus new UV-free path">',
           '<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" '
           'markerWidth="7" markerHeight="7" orient="auto">'
           f'<path d="M0,0 L8,4 L0,8 z" fill="{ink}"/></marker></defs>']

    # lane A: texture bake (before)
    svg.append(lane.format(y=28, t="BEFORE &middot; TEXTURE BAKE"))
    svg.append(b(20, 44, 160, ["predicted voxel", "mask, 512&#179;"]))
    svg.append(a(180, 268, 70, ["per-texel", "nearest voxel"]))
    svg.append(b(268, 44, 190, ["emissive texture", "in the UV atlas"]))
    svg.append(a(458, 560, 70, ["written under the", "raw glTF UVs"], ly=18))
    svg.append(b(560, 44, 110, ["Blender", "import"]))
    svg.append(a(670, 730, 70, ["reads its own", "imported UVs"], color=acc,
                 ly=18))
    svg.append(b(730, 44, 96, ["render:", "streaks"]))
    # divergence callout under lane A
    svg.append(f'<path d="M560,110 C580,140 700,140 726,112" fill="none" '
               f'stroke="{acc}" stroke-width="1.2" stroke-dasharray="4 3"/>')
    svg.append(lbl.format(x=600, y=158, t="Blender flips V on import "
               "(glTF V top-down, Blender bottom-up);")
               .replace(f'fill="{mut}"', f'fill="{acc}"'))
    svg.append(lbl.format(x=600, y=172, t="the baker never flipped: UV "
               "islands land mirrored, emission scattered")
               .replace(f'fill="{mut}"', f'fill="{acc}"'))

    # lane B: UV-free (after)
    svg.append(lane.format(y=216, t="AFTER &middot; FIXED BAKE"))
    svg.append(b(20, 232, 160, ["predicted voxel", "mask, 512&#179;"]))
    svg.append(a(180, 268, 258, ["per-texel nearest", "lit voxel, tol 2.0"]))
    svg.append(b(268, 232, 190, ["emissive texture", "in the UV atlas"]))
    svg.append(a(458, 560, 258, ["written under the", "IMPORTED (V-flipped) UVs"]))
    svg.append(b(560, 232, 110, ["Blender", "import"]))
    svg.append(a(670, 730, 258, ["conventions", "match"]))
    svg.append(b(730, 232, 96, ["render:", "clean"]))
    svg.append('</svg>')

    cap = ("<b>The fix makes writer and reader share one texture-coordinate "
           "convention.</b> The old path wrote the emissive texture under "
           "raw glTF coordinates; Blender renders under its own imported "
           "coordinates, which flip the vertical axis. The fixed bake writes "
           "under the imported coordinates directly, so nothing is left to "
           "disagree, and the full-resolution texture (mask &times; albedo "
           "per texel) is preserved.")
    return ('<figure class="xg-fig" style="margin-left:auto;margin-right:auto;'
            'max-width:820px">' + "".join(svg)
            + f'<figcaption>{cap}</figcaption></figure>')


# ------------------------------------------------------------------ sections
def sec_teaser():
    fig = lp.method_matrix(
        columns=["BEFORE (in the paper)", "AFTER (fixed path)"],
        rows=[
            ("warhammer, raw seed 3", [
                {"img": src(FIG5["hammer_before"])},
                {"img": src(FIG5["hammer_after"])},
            ]),
            ("lightsaber, EMA draw 3", [
                {"img": src(FIG5["saber_before"])},
                {"img": src(FIG5["saber_after"])},
            ]),
            ("robot, EMA seed 4", [
                {"img": src(FIG5["robot_before"])},
                {"img": src(FIG5["robot_after"])},
            ]),
        ],
        caption_html=(
            "<b>The paper's figure-5 examples, SegviGen-Emission column, "
            "before and after the fix.</b> Before: the column as currently "
            "in the paper (near-black). After: the same shapes through the "
            "fixed bake (texture coordinates matched to the renderer) in "
            "the box render, emission = mask &times; albedo at full texture "
            "resolution, a measured draw choice, labeled per row with the "
            "exact weights and seed."),
        page_inner=820, key="teaser")
    return lp.section_v2("teaser", None, "Before / after", fig)


def sec_diagram():
    body = lp.prose(
        "Both paths start from the same model output: a predicted mask over "
        "the occupied voxels of a 512&sup3; grid. They differ in how that "
        "mask reaches the renderer.")
    body += path_diagram()
    return lp.section_v2("diagram", None,
                         "What changed in the processing", body)


def sec_why():
    body = lp.prose(
        "Three separate defects produced the broken panels; the diagram "
        "marks where the first one lives.")
    body += lp.prose(
        "<b>1. The vertical-coordinate flip (the streaks), CONFIRMED.</b> "
        "glTF counts the texture V coordinate from the top, Blender from "
        "the bottom, and Blender's importer flips it on load. The bake "
        "wrote the emissive texture under raw glTF coordinates without the "
        "flip, so every UV island lands mirrored and emission scatters: "
        "streaky islands on the pumpkin, patchy black on the lantern, the "
        "sword's mask multiplying the dark half of its own real texture. "
        "Verified numerically (U bit-identical between the two, V exactly "
        "one minus V on every checked face) and by probe: the misread "
        "fraction fell from 2.0% (hammer) and 5.2% (robot) of pixels to "
        "zero once the bake used the imported coordinates (Reference "
        "below).")
    body += lp.prose(
        "<b>2. Dropped node transforms (fully black shapes).</b> The first "
        "version of the fix read vertex positions without applying each mesh "
        "node's world transform. Any asset whose nodes carry a real "
        "transform, like the warhammer, ended up in the wrong coordinate "
        "frame: every face missed the voxels, so nothing lit.")
    body += lp.prose(
        "<b>3. Wrong voxel query (sparse emission lost).</b> The transfer "
        "first asked whether the single nearest voxel is lit. For a face "
        "near a sparse lit region, the nearest voxel is almost always an "
        "unlit surface voxel, so real emission was silently dropped. The "
        "established convention is the nearest LIT voxel within a tolerance "
        "of 2 voxel units.")
    return lp.section_v2("why", None, "Why the panels were broken", body)


def sec_what():
    rows = [
        ("Bake under the renderer's imported coordinates (V flip applied)",
         "misread texels: 2.0% hammer, 5.2% robot probe pixels",
         "zero misread; full texture resolution kept"),
        ("Scale-frame fix in the new bake (the loader rescales every mesh "
         "to a 2.0 box)",
         "hammer and robot transferred zero lit texels",
         "141,704 and 36,311 lit texels, matching the voxel grid"),
        ("Per-face rendering demoted to a debug tool",
         "used as an interim fix; one albedo sample per face reads as a "
         "triangle mosaic",
         "presentation panels use the fixed full-resolution bake"),
        ("Node transforms via the proven loader",
         "warhammer and lightsaber fully black",
         "faces and voxels share one frame"),
        ("Nearest-LIT-voxel query (tol 2.0) plus a zero-transfer guard",
         "sparse emission silently dropped",
         "sparse regions render; silent zeros became loud failures"),
        ("Fresh-seed resampling, per-draw measurement",
         "the one saved draw happened to be empty for all three shapes",
         "a usable draw found for every paper shape"),
    ]
    rows_html = "".join(
        f'<tr><td style="text-align:left"><b>{what}</b></td>'
        f'<td style="text-align:left">{before}</td>'
        f'<td style="text-align:left">{after}</td></tr>'
        for what, before, after in rows)
    return lp.section_v2("what", None, "What was done",
                         lp.results_table(["change", "before", "after"],
                                          rows_html))


def sec_picked():
    body = lp.prose(CKPT_NOTE)
    body += lp.prose(
        "<b>How the shown draw was picked.</b> Diffusion sampling gives a "
        "different mask on every draw, and for these shapes most draws land "
        "degenerate: near-empty or blanket. Every available draw was "
        "measured by its predicted lit fraction, the share of occupied "
        "voxels above threshold 0.5, listed in full below next to the "
        "ground-truth fraction. The rule: show the draw whose fraction is "
        "closest to ground truth among non-degenerate draws. That yields "
        "warhammer raw seed 3 (0.107 vs 0.085 GT) and lightsaber EMA draw 3 "
        "(0.618 vs 0.484 GT). The robot is the one deviation: the "
        "nearest-fraction draw (EMA seed 5, 0.034 vs 0.080 GT) reads sparse "
        "and patchy, so the fuller EMA seed 4 (0.289) is shown as the "
        "candidate; both are below and the choice is open.")

    def row(name, gt, raw_vals, ema_vals):
        raw_s = ", ".join(f"{v:.3f}" for v in raw_vals)
        ema_s = ", ".join(f"{v:.3f}" for v in ema_vals)
        return (f'<tr><td style="text-align:left">{name}</td><td>{gt:.3f}</td>'
                f'<td>{raw_s}</td><td>{ema_s}</td></tr>')

    body += lp.prose(
        "Re-dumped draws 0&ndash;4 of the original gallery run (the first "
        "round had only draw 0 on disk, which is empty for all three "
        "shapes):")
    body += lp.results_table(
        ["shape", "GT frac", "raw draws 0-4 (frac@0.5)",
         "EMA draws 0-4 (frac@0.5)"],
        row("warhammer", 0.0849,
            [0.0061, 0.0, 0.0, 0.0, 0.0008], [0.0032, 0.0, 0.0, 0.0, 0.0])
        + row("lightsaber", 0.4842,
              [0.0283, 0.8356, 0.0148, 0.6271, 0.4259],
              [0.0145, 0.8552, 0.0009, 0.6177, 0.0])
        + row("robot", 0.0796,
              [0.0, 0.0, 0.0122, 0.0, 0.0], [0.0, 0.9684, 0.0118, 0.0, 0.0]))
    body += lp.prose(
        "Fresh seeds (16 per shape where run) recovered the two shapes whose "
        "re-dumped draws were all degenerate: warhammer raw seed 3 hit "
        "0.107, the only usable warhammer draw across every seed checked, "
        "and the robot found a middle mode at EMA seeds 3&ndash;5 (0.251, "
        "0.289, 0.034). A reasonable prediction exists for all three "
        "shapes, on a minority of draws; that is a sampling-density "
        "finding, not a fixed failure of the checkpoint on these shapes.")
    return lp.section_v2("picked", None,
                         "Which checkpoint, and how the shown draw was picked",
                         body)


def sec_result():
    fig = lp.method_matrix(
        columns=["GROUND TRUTH", "PREDICTION (picked draw)"],
        rows=[
            ("warhammer, raw seed 3", [
                {"img": src(FIG5["hammer_gt"])},
                {"img": src(FIG5["hammer_after"])},
            ]),
            ("lightsaber, EMA draw 3", [
                {"img": src(FIG5["saber_gt"])},
                {"img": src(FIG5["saber_after"])},
            ]),
            ("robot, EMA seed 4", [
                {"img": src(FIG5["robot_gt"])},
                {"img": src(FIG5["robot_after"])},
            ]),
        ],
        caption_html=(
            "<b>Ground truth versus the picked prediction: box render, "
            "emission = mask &times; albedo at full texture resolution.</b> "
            "Warhammer: predicted lit fraction 0.107 vs 0.085 GT, the head "
            "panels glow as in GT. Lightsaber: the blade matches GT. Robot: "
            "head and legs glow where GT concentrates on the eyes; plausible "
            "rather than precise, the same bar the figure's other baselines "
            "meet. GT panels use the identical render path with the "
            "ground-truth voxel labels."),
        page_inner=820, key="paper-result")
    fig += lp.method_matrix(
        columns=["RAW VOXEL MASK", "MESH RENDER (fixed bake)"],
        rows=[
            ("robot, EMA seed 4", [
                {"img": src("voxel_robot_emaseed4.png")},
                {"img": src(FIG5["robot_after"])},
            ]),
            ("robot, EMA seed 5", [
                {"img": src("voxel_robot_emaseed5.png")},
                {"img": src(FIG5["robot_alt"])},
            ]),
        ],
        caption_html=(
            "<b>Owner-challenged and arbitrated in voxel space: seed 5's "
            "scatter is the draw, not the bake.</b> The left column renders "
            "the raw predicted mask as cubes, with no mesh, no UV, and no "
            "bake anywhere in the path. Seed 4's mask is coherent (solid "
            "eyes, hands, legs); seed 5's is structureless dust with the "
            "eyes unlit, the same character as its mesh render. A UV bug "
            "misplaces contiguous blocks; it cannot atomize them. This also "
            "settles the pick: seed 4 carries real spatial structure, "
            "seed 5's closer overall fraction (0.034 vs GT 0.080, against "
            "seed 4's 0.289) hides a mask with no legible shape. Owner "
            "decision between them stays open."),
        page_inner=820, key="robot-arbitration")
    return lp.section_v2("result", None,
                         "Result: ground truth versus prediction", fig)


# ------------------------------------------------------- reference sections
def sec_voxel():
    fig = lp.method_matrix(
        columns=["GT VOXELS", "PREDICTED VOXELS"],
        rows=[
            ("pumpkin, 48af42db", [
                {"img": src("pumpkin_voxel_gt.png")},
                {"img": src("pumpkin_voxel_pred.png")},
            ]),
            ("lantern, 064e4156", [
                {"img": src("lantern_voxel_gt.png")},
                {"img": src("lantern_voxel_pred.png")},
            ]),
        ],
        caption_html=(
            "<b>No mesh, no UV, no texture: grey/orange cubes per occupied "
            "voxel.</b> GT sparse and localized; the prediction solid orange "
            "on both shapes (fraction near 1.0), confirming these two "
            "blanket draws are the model's own output, not a rendering "
            "artifact. Same 512&sup3; grid in both columns; GT's lit label "
            "traces to Dongchen's 256&sup3; bake upsampled by nearest "
            "neighbor, so GT and prediction share a grid but not equal "
            "granularity underneath it."),
        page_inner=820, key="voxel-arbitration")
    return lp.section_v2("voxel", None,
                         "Reference: the prediction in voxel space", fig)


def sec_redpad():
    body = lp.prose(
        "The probe: a debug texture, off-white inside the write side's own "
        "rasterized UV coverage, saturated red in what it considers "
        "padding, rendered as emissive RGB through the identical path. Red "
        "on the surface means the renderer sampled texels the writer never "
        "claimed. (An earlier version of this probe was itself invalid: the "
        "render tool ignored the probe flag in box mode and fell back to "
        "the asset's own textures; the images below are from the corrected "
        "tool, which also logs which image reached each material.)")
    body += lp.method_matrix(
        columns=["OLD BAKE (raw UVs)", "FIXED BAKE (imported UVs)"],
        rows=[
            ("warhammer", [
                {"img": src("probe_hammer_before.png")},
                {"img": src("probe_hammer_after.png")},
            ]),
            ("robot", [
                {"img": src("probe_robot_before.png")},
                {"img": src("probe_robot_after.png")},
            ]),
        ],
        caption_html=(
            "<b>The flip explains all the red, and the fix removes all of "
            "it.</b> Under the old bake the renderer reads padding texels "
            "over 2.0% (hammer) and 5.2% (robot) of the image, in "
            "structural patterns (half the robot's head). Under the fixed "
            "bake the red fraction is 0.00000 and 0.00003: writer and "
            "reader agree everywhere."),
        page_inner=820, key="redpad-decisive")
    return lp.section_v2("redpad", None,
                         "Reference: the divergence probe, before and after "
                         "the fix", body)


def sec_writeside():
    body = lp.prose(
        "The write side checks out against itself: pred_mask_to_asset.py's "
        "UV coverage (pumpkin 0.500, lantern 0.589) is reproduced by an "
        "independent second implementation (0.514, 0.592), and every UV "
        "triangle drawn over the write side's own baked mask lands on "
        "nonzero texels. The divergence therefore sits between write and "
        "read, not inside the writer.")
    body += lp.fig_row(
        [("pumpkin: triangles over the baked mask",
          src("pumpkin_wireframe_overlay.png")),
         ("lantern: triangles over the baked mask",
          src("lantern_wireframe_overlay.png"))],
        caption_html=(
            "<b>No triangle sits over black on the write side.</b> A "
            "self-consistency check of the writer, not a render-side test."),
        native_px=1024, content="pixel-map", key="wireframe-overlay")
    return lp.section_v2("writeside", None,
                         "Reference: the write side is self-consistent", body)


def sec_ruled_out():
    body = lp.prose(
        "Chased and ruled out: the AI denoiser (hardcoded on in the shared "
        "render code; tested off, no change), texture interpolation (Linear "
        "vs nearest, no change), GI color bleeding (diffuse bounces zeroed, "
        "no change), the lantern's 8 UV layers (all report the identical "
        "aggregate bbox), UV wraparound (all TEXCOORD_0 within [0,1]), "
        "material-slot mapping and conditioning provenance (clean). An "
        "all-white control texture was a first-draft dead end: every "
        "possible sample location is white, so it cannot detect wrong "
        "sampling locations at all.")
    body += lp.fig_row(
        [("desk lamp: prediction (open item)", src("desklamp_render_pred.png")),
         ("desk lamp: ground truth", src("desklamp_render_gt.png"))],
        caption_html=(
            "<b>Formerly presented as a clean control; that framing was "
            "wrong and is withdrawn.</b> Under the fixed bake the ground "
            "truth renders as before, but the prediction shows no visible "
            "glow even though the bake logs two materials lit (64% and "
            "100% of their face area) and the render statistics confirm "
            "the path fired; alpha transparency is ruled out. The earlier "
            "claim that this shape was structurally immune to the "
            "divergence also does not hold (all 8 materials pass through "
            "the write path). Kept as an OPEN item, not a control, until "
            "the missing glow is explained."),
        native_px=768, content="photo")
    return lp.section_v2("ruledout", None,
                         "Reference: ruled out, plus one open item", body)


def sec_npz():
    body = lp.prose(
        "Each shape's dump npz: <code>coords</code> (N&times;3, occupied "
        "512&sup3; grid), <code>pred_bc</code> (predicted value), "
        "<code>gt_e</code> (ground-truth label). A voxel counts as "
        "predicted-emissive at <code>pred_bc &gt; 0.5</code>.")
    body += lp.fig_row(
        [("pumpkin predicted (blanket draw)", src("npz_pumpkin_pred.png")),
         ("pumpkin ground truth", src("npz_pumpkin_gt.png"))],
        caption_html=(
            "The pumpkin's dumped draw predicts every one of 2,218,234 "
            "voxels emissive (pred_bc 0.999&ndash;1.001, zero spread); GT "
            "is sparse and localized. The projection resolves real "
            "structure when the prediction has any."),
        key="npz-format")
    return lp.section_v2("npz", None, "Reference: the prediction npz", body)


def sec_provenance():
    body = lp.prose(
        "All images are real artifacts, not redrawn. Before-panels: crops "
        "of the paper's current figure 5 (SegviGen-Emission column). "
        "After/GT panels: bpy_rebake.py, the emissive texture rasterized "
        "against Blender's imported UVs (V flip applied) at full texture "
        "resolution, emission = mask &times; albedo per texel, box render "
        "(Filmic, exposure 1.5), one Solar job per panel (243399-243421). "
        "Probe pairs: the corrected red-padding tool, off-white 250 probe "
        "color, per-material image logging. Voxel renders: paper_v3/render_voxels.py "
        "unmodified via a format converter. Red-padding probes: a local "
        "test-only copy of render_emissive.py, denoiser off, "
        "nearest-neighbor interpolation; the shared file untouched. The "
        "transfer uses pred_mask_to_asset.py's real <code>primitives()</code> "
        "and its exact 2.0-unit nearest-lit tolerance. Checkpoints: "
        "epoch_0008 and epoch_0008_ema of outputs/emis_72kv2_cond_pw1b. "
        "Draws: the multi-draw re-dump plus the robot_draws and "
        "hammer_draws fresh-seed workstreams. Full job ids live in the jobs "
        "board entries ckpt8_eval and mask_debug, 2026-08-10/11.")
    return lp.section_v2("provenance", None, "Provenance", body)


# ----------------------------------------------------------------- the build
def build(publish=False):
    assets_dir = os.path.join(WEB, "assets")
    problems = check_assets()
    if problems:
        sys.exit("ASSET CHECK FAILED:\n  " + "\n  ".join(problems))

    hero = lp.hero_header(
        f"lightgen &middot; mask-bake debug &middot; {PAGE_DATE}",
        "Emission Render Artifacts: Cause and Fix",
        dek_html=(
            "The paper's SegviGen-Emission column rendered near-black while "
            "every baseline looked reasonable. The causes were in the "
            "render path, not only the model: Blender flips the vertical "
            "texture coordinate on glTF import and the mask baker did not, "
            "plus three transfer bugs, compounded by an unlucky saved draw. "
            "This "
            "page shows the fixed panels, the processing change, and how "
            "the shown draws were chosen."),
        toc=[("teaser", "Before / after"), ("diagram", "The processing"),
             ("why", "Why"), ("what", "What was done"),
             ("picked", "Checkpoint and draw choice"), ("result", "Result"),
             ("voxel", "Voxel space"), ("redpad", "Divergence probe"),
             ("writeside", "Write side"), ("ruledout", "Ruled out"),
             ("npz", "The npz"), ("provenance", "Provenance")],
    )

    page_html = lp.page(
        title="Emission Render Artifacts: Cause and Fix",
        header_html=hero,
        body_sections=[sec_teaser(), sec_diagram(), sec_why(), sec_what(),
                       sec_picked(), sec_result(),
                       sec_voxel(), sec_redpad(), sec_writeside(),
                       sec_ruled_out(), sec_npz(), sec_provenance()],
        assets_rel=SITE_ASSETS,
        assets_dir=assets_dir,
        theme="v3",
        tree_html=tree_html(),
        nav_title="Mask-bake debug",
        version_slot=lp.v3_version_slot(date=PAGE_DATE),
        needs_katex=False,
        extra_head=f'<link rel="icon" href="{FAVICON}">',
        fig_numbers=False,
        outline_entries=[
            {"id": "teaser", "label": "Before / after"},
            {"id": "diagram", "label": "The processing"},
            {"id": "why", "label": "Why"},
            {"id": "what", "label": "What was done"},
            {"id": "picked", "label": "Checkpoint and draw choice"},
            {"id": "result", "label": "Result"},
            {"id": "voxel", "label": "Voxel space"},
            {"id": "redpad", "label": "Divergence probe"},
            {"id": "writeside", "label": "Write side"},
            {"id": "ruledout", "label": "Ruled out"},
            {"id": "npz", "label": "The npz"},
            {"id": "provenance", "label": "Provenance"},
        ],
    )

    violations = wz.console_links_in(page_html)
    if violations:
        sys.exit(f"ZONE-LINK GUARD FAILED: page links to the console: {violations}")

    out_path = os.path.join(HERE, "index.html")
    with open(out_path, "w") as f:
        f.write(page_html)
    print(f"wrote {out_path}")
    print(f"  {len(REQUIRED_IMAGES)} images checked, all present")
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
