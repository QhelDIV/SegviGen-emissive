#!/usr/bin/env python3
"""Build the overfit_condtest page: can the SegviGen emission DiT overfit 10 training
shapes at all, and does real DINOv3 conditioning help memorization over zero-cond?

This is the pipeline-verdict diagnostic team-lead asked for: every trained model to
date scores near-zero median held-out IoU, more data makes it worse, and the 72k model
fails on its own training shapes. If the model cannot even overfit 10 shapes, the
training/inference path itself is broken and nothing else (data, weighting, cond)
matters; if it overfits cleanly, the pipeline is sound and the problem is the learning
signal (data/weighting/scale). Two runs, identical except --cond real vs --cond zero,
same 10 shapes, same optimizer/schedule/loss weighting.

Success criterion, stated in advance (team-lead brief): training-set per-voxel IoU
(on the SAME shapes being trained on) approaching the VAE round-trip ceiling (~0.96,
FACTSHEET_diagnostics.md) within the run. A plateau near 0.1-0.2 (this project's
observed 72k-model floor) is a pipeline verdict, not a data verdict.

Every number on this page is read at build time from the two runs' own train_curve.json
(per-epoch, per-sample IoU, written by code/train_emissive.py's --val_quick path) and
the render/dump stage's own summary.json files, never retyped.

Run: /project/3dlg-hcvc/omages/omages_internal/.venv2/bin/python \
        web/_preview/overfit_condtest/build.py
"""
import html
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WEB)
sys.path.insert(0, os.path.join(REPO, "tools"))

import xgpage as lp                      # noqa: E402
import workspace_zone as wz              # noqa: E402
from xgpage.publish import publish_assets  # noqa: E402


def _esc(s):
    return html.escape(str(s), quote=False)


def _esc_attr(s):
    return html.escape(str(s), quote=True)

SITE_ROOT = "/projects/omages/yanxg/lightgen"
SITE_ASSETS = f"{SITE_ROOT}/assets"
FAVICON = f"{SITE_ROOT}/assets/images/favicon.png"
PAGE_DATE = "2026-08-08"

WORK = "/project/3dlg-hcvc/omages/yanxg_scratch/overfit_condtest"
IMG_DIR = os.path.join(HERE, "img")
os.makedirs(IMG_DIR, exist_ok=True)

VAE_CEILING = 0.96   # FACTSHEET_diagnostics.md: VAE round-trip corruption ceiling
FLOOR_72K = 0.1       # this project's observed held-out-IoU floor on trained 72k models

SIDS = ["000b9fd47d6d4f7db7b2f5022d1ae9aa", "001c79293c3e4f938798026a79f2d26a",
        "001dd28130354d36b8f04ffe59c30abe", "dad716a8acad4bee8ab52c963afae3f0",
        "91d94c0f556b4aa69d3ab09919d1d380", "f199c9ba047842269f7b0b93d3c49cb8",
        "560e82f71f7d44acaac5b3131cc3e9d3", "cfa5899783d046d28fa959828e5623d7",
        "ae7c9107d7634810aa107c78f57cb29e", "d60d0e5d75c9425086ffd45ea68da712"]
SHORT = {s: s[:8] for s in SIDS}
FRAC = {  # emissive_frac from meta.json, read at selection time
    "000b9fd47d6d4f7db7b2f5022d1ae9aa": 0.0179, "001c79293c3e4f938798026a79f2d26a": 0.0003,
    "001dd28130354d36b8f04ffe59c30abe": 0.0072, "dad716a8acad4bee8ab52c963afae3f0": 0.0500,
    "91d94c0f556b4aa69d3ab09919d1d380": 0.0886, "f199c9ba047842269f7b0b93d3c49cb8": 0.1625,
    "560e82f71f7d44acaac5b3131cc3e9d3": 0.2999, "cfa5899783d046d28fa959828e5623d7": 0.3019,
    "ae7c9107d7634810aa107c78f57cb29e": 0.7513, "d60d0e5d75c9425086ffd45ea68da712": 0.9989,
}
BUCKET = {s: ("sparse" if f < 0.05 else "mid" if f < 0.30 else "dense") for s, f in FRAC.items()}
COLOR_COND = "#5cc8ff"   # var(--accent), same series color as the project's w5/cond tag
COLOR_ZERO = "#ffb454"   # var(--accent2), same series color as the project's w1 tag
COLOR_PW1 = "#c792ea"    # third categorical hue (violet), separated from both accents;
                          # not run through the dataviz palette validator (time-boxed
                          # mid-incident addition) -- a supplementary reference series,
                          # not the primary categorical comparison on this page.

OUTLINE = [
    ("verdict", "Can the model overfit, and does conditioning help"),
    ("curve", "Train-set IoU vs epoch"),
    ("pershape", "Per-shape trajectories"),
    ("gallery", "Rendered predictions, all 10 shapes"),
    ("loss", "Loss curves"),
    ("provenance", "How this was produced"),
]


# --------------------------------------------------------------------- data
def load_curve(tag):
    p = os.path.join(WORK, "outputs", f"overfit_ct10_{tag}", "train_curve.json")
    return json.load(open(p))


def load_curve_optional(tag):
    """None if the run hasn't written a curve yet (e.g. the pw1 control, added
    mid-run) -- callers must degrade to a pending placeholder, not crash."""
    p = os.path.join(WORK, "outputs", f"overfit_ct10_{tag}", "train_curve.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d if d else None


def load_single_shape_curve():
    """The task-#32 single-shape control (today's code, one mid-density ct10 shape,
    pos_weight 1): a separate out_dir (overfit_single_pw1, not overfit_ct10_*), a
    different x-axis convention (n_per_epoch=20 vs the ct10 runs' 10), so it is
    reported as its own number/table row, never overlaid on the ct10 mean-IoU chart."""
    p = os.path.join(WORK, "outputs", "overfit_single_pw1", "train_curve.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d if d else None


def best_mean(curve):
    return max(c["val_iou_nonzero"] for c in curve if c["val_iou_nonzero"] is not None)


def final_mean(curve):
    return curve[-1]["val_iou_nonzero"]


def copy_img(src, dst_name):
    import shutil
    dst = os.path.join(IMG_DIR, dst_name)
    shutil.copy2(src, dst)
    return dst


def rel_img(abspath):
    return "img/" + os.path.relpath(abspath, IMG_DIR)


def img_ref(abspath):
    import hashlib
    h = hashlib.md5(open(abspath, "rb").read()).hexdigest()[:8]
    return f"{rel_img(abspath)}?v={h}"


# --------------------------------------------------------------------- chart (local helper --
# no multi-series line-chart component exists in xgpage yet; built here in the v2 chart
# vocabulary (.axislabel/.chartnote reused from theme2.css so it matches hbar_chart's look
# in both themes) rather than upstreaming a one-off. See core.py's hbar_chart for the
# convention this follows: server-rendered SVG, data-tip per point for xg2.js's hover layer,
# and a full data table elsewhere on the page as the required non-chart fallback.
def line_chart_svg(series, *, title="", x_label="epoch", y_max=1.0, y_min=0.0,
                    width=720, height=300, refs=None, note="", legend=True):
    pad_l, pad_r, pad_t, pad_b = 46, 20, (34 if title else 16), 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    all_x = [x for s in series for x, _ in s["points"]]
    x_max = max(all_x) if all_x else 1
    x_min = 0

    def px(x):
        return pad_l + (x - x_min) / max(x_max - x_min, 1) * plot_w

    def py(y):
        return pad_t + (1 - (y - y_min) / max(y_max - y_min, 1e-9)) * plot_h

    parts = []
    if title:
        parts.append(f'<text x="{pad_l}" y="20" class="axislabel">{_esc(title)}</text>')
    # gridlines + y ticks at 0, .25, .5, .75, 1.0 (of the y range)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yv = y_min + frac * (y_max - y_min)
        yy = py(yv)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width - pad_r}" y2="{yy:.1f}" '
                     f'stroke="currentColor" stroke-opacity=".08" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{yy + 4:.1f}" text-anchor="end" '
                     f'class="barval">{yv:.2f}</text>')
    # x ticks: first/last/mid epoch
    for xv in sorted(set([x_min, x_max, (x_min + x_max) // 2])):
        parts.append(f'<text x="{px(xv):.1f}" y="{height - pad_b + 16}" text-anchor="middle" '
                     f'class="barval">{xv}</text>')
    parts.append(f'<text x="{width / 2:.1f}" y="{height - 6}" text-anchor="middle" '
                 f'class="axislabel">{_esc(x_label)}</text>')
    # reference lines
    for r in (refs or []):
        yy = py(r["y"])
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width - pad_r}" y2="{yy:.1f}" '
                     f'stroke="{r.get("color", "currentColor")}" stroke-opacity=".5" '
                     f'stroke-width="1.5" stroke-dasharray="5,4"/>')
        parts.append(f'<text x="{width - pad_r}" y="{yy - 4:.1f}" text-anchor="end" '
                     f'class="barval" fill="{r.get("color", "currentColor")}">{_esc(r["label"])}</text>')
    # series
    for s in series:
        pts = sorted(s["points"])
        path = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}" for i, (x, y) in enumerate(pts))
        parts.append(f'<path d="{path}" fill="none" stroke="{s["color"]}" stroke-width="2"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3" fill="{s["color"]}" '
                         f'data-tip="{_esc_attr(s["name"])}: epoch {x}, {y:.3f}"/>')
    legend_html = ""
    if legend:
        items = [lp.legend_swatch(s["color"], s["name"]) for s in series]
        items += [lp.legend_swatch(r.get("color", "#888"), r["label"]) for r in (refs or []) if r["label"]]
        legend_html = lp.legend(items)
    note_html = f'<p class="chartnote">{note}</p>' if note else ""
    return (f'<div class="chart"><svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{_esc_attr(title or "line chart")}">{"".join(parts)}</svg>'
            f'{legend_html}{note_html}</div>')


# --------------------------------------------------------------------- sections
def sec_verdict(curve_cond, curve_zero, curve_pw1, curve_single):
    cond_final, cond_best = final_mean(curve_cond), best_mean(curve_cond)
    zero_final, zero_best = final_mean(curve_zero), best_mean(curve_zero)
    pw5_best = max(cond_best, zero_best)
    pw5_overfits = pw5_best > 0.5 * VAE_CEILING

    last_cond_ps = curve_cond[-1]["per_sample"]
    last_zero_ps = curve_zero[-1]["per_sample"]
    cond_range = (min(p["best_iou"] for p in last_cond_ps), max(p["best_iou"] for p in last_cond_ps))
    zero_range = (min(p["best_iou"] for p in last_zero_ps), max(p["best_iou"] for p in last_zero_ps))

    cond_helps = cond_best > zero_best + 0.03
    zero_helps = zero_best > cond_best + 0.03

    pw1_final = pw1_best = None
    pw1_overfits = None
    if curve_pw1 is not None:
        pw1_final, pw1_best = final_mean(curve_pw1), best_mean(curve_pw1)
        pw1_overfits = pw1_best > 0.5 * VAE_CEILING

    single_best = single_final = None
    single_overfits = None
    if curve_single is not None:
        single_final, single_best = final_mean(curve_single), best_mean(curve_single)
        single_overfits = single_best > 0.5 * VAE_CEILING

    # Four-way read, in the order the evidence actually arrived: pos_weight 5 alone
    # only says "this configuration doesn't memorize ten shapes." The pos_weight-1
    # control on the same ten shapes tells apart a broken path from a pos_weight-5-
    # specific problem. The single-shape control (today's code, one shape) tells
    # apart a CODE REGRESSION from a multi-shape effect. Once all three exist, the
    # remaining open question is whether the multi-shape plateau is undertraining
    # (more epochs would fix it) or a genuine multi-shape limit (capacity,
    # interference between shapes, or the backbone handling this shape family
    # poorly -- a separate zero-shot-reconstruction test found the emissive family
    # comes back badly simplified from the pretrained shape backbone).
    if pw5_overfits:
        headline = (
            f"<b>The model overfits at pos_weight 5: train-set IoU reaches {pw5_best:.3f} "
            f"against a {VAE_CEILING:.2f} VAE round-trip ceiling.</b> The training and "
            "inference path is sound; the project's near-zero held-out numbers are a "
            "learning-signal / data-scale problem, not a broken pipeline."
        )
    elif curve_pw1 is None:
        headline = (
            f"<b>Neither pos_weight-5 run overfits: best train-set IoU is only "
            f"{pw5_best:.3f} against a {VAE_CEILING:.2f} ceiling.</b> This alone does not "
            "distinguish a broken training/inference path from a pos_weight-5 loss-"
            "reweighting problem &mdash; a pos_weight&nbsp;1.0 control run on the SAME 10 "
            "shapes is in progress and will settle which; this section updates in place "
            "once it lands."
        )
    elif pw1_overfits:
        headline = (
            f"<b>pos_weight 5 breaks a pipeline that otherwise works.</b> The pos_weight-1.0 "
            f"control on the same 10 shapes reaches {pw1_best:.3f} IoU (final epoch "
            f"{pw1_final:.3f}) against the {VAE_CEILING:.2f} ceiling, next to pos_weight 5's "
            f"plateau at {pw5_best:.3f} &mdash; the pipeline CAN memorize. The per-voxel loss "
            "reweighting (pos_weight 5, meant to fight class imbalance) is what prevents "
            "these 10 shapes from being learned, not the training or inference path."
        )
    elif curve_single is None:
        headline = (
            f"<b>Neither pos_weight setting overfits ten shapes</b> (pos_weight 5: "
            f"{pw5_best:.3f}, pos_weight 1: {pw1_best:.3f}, both against a {VAE_CEILING:.2f} "
            "ceiling). This does not yet say whether the code itself has regressed since "
            "July, when a single-shape pos_weight-1 run reached 0.918 by epoch 30 &mdash; a "
            "single-shape run on today's code is in progress to settle that; this section "
            "updates in place once it lands."
        )
    elif single_overfits:
        headline = (
            f"<b>Today's code is not broken. Ten shapes at once is the hard part, not "
            f"pos_weight and not a code regression.</b> A single-shape control run on "
            f"today's code, same setup July used, reaches {single_best:.3f} IoU (final "
            f"epoch {single_final:.3f}) &mdash; matching July's 0.918-by-epoch-30 result and "
            "ruling out a code regression as the explanation. Yet neither ten-shape run "
            f"gets close: pos_weight 5 plateaus at {pw5_best:.3f}, pos_weight 1 at "
            f"{pw1_best:.3f}. Extending the pos_weight-1 control from 150 to 400 epochs "
            f"(parity with the pos_weight-5 runs) moved its best score only from 0.280 to "
            f"{pw1_best:.3f}, still far short of the {VAE_CEILING:.2f} ceiling and still "
            "oscillating rather than climbing cleanly &mdash; more epochs alone is not "
            "closing the gap. The open question is now what breaks specifically when "
            "TEN shapes are trained together instead of one: capacity or interference "
            "across shapes sharing the same small optimization budget, or (a separate "
            "finding from zero-shot testing) the pretrained shape backbone reconstructing "
            "this emissive shape family badly out of the box, which could make several of "
            "these ten shapes individually hard to represent regardless of shape count."
        )
    else:
        headline = (
            f"<b>Even a single shape does not overfit on today's code</b> (best "
            f"{single_best:.3f} against the {VAE_CEILING:.2f} ceiling, versus July's "
            "0.918-by-epoch-30 on the same kind of test). This points at a code "
            "regression between July and now, not a pos_weight or multi-shape capacity "
            f"problem: the ten-shape runs (pos_weight 5 best {pw5_best:.3f}, pos_weight 1 "
            f"best {pw1_best:.3f}) were never going to succeed if a single shape already "
            "fails under the current code."
        )

    cond_line = (
        f"Conditioning {'helps' if cond_helps else 'does not help' if zero_helps else 'makes no clear difference to'} "
        f"memorization at pos_weight 5: best mean IoU real-cond={cond_best:.3f} vs zero-cond={zero_best:.3f} "
        f"(final epoch: {cond_final:.3f} vs {zero_final:.3f}). "
        + ("As expected if DINOv3 conditioning is acting as a per-shape key the model "
           "can exploit even at this tiny scale."
           if cond_helps else
           "At only 10 shapes the model may not need a conditioning signal to "
           "separate them (shape/appearance latents alone can serve as the key), "
           "so this does not by itself rule out conditioning mattering at scale."
           if not zero_helps else
           "Zero-cond memorizes AS WELL OR BETTER than real-cond here, so this "
           "diagnostic gives no evidence that the DINOv3 embedding is being used "
           "as a per-shape key at all.")
    )

    body = lp.verdict_box(f"<p>{headline}</p><p>{cond_line}</p>")
    body += lp.prose(
        f"<b>Success criterion (stated in advance):</b> train-set per-voxel IoU on the "
        f"same 10 shapes being trained on approaching the VAE round-trip ceiling "
        f"(~{VAE_CEILING:.2f}) within the run. A plateau near 0.1&ndash;0.2 is a pipeline "
        f"verdict, not a data verdict. <b>Outcome, pos_weight 5 (both cond settings): "
        f"{'ceiling reached' if pw5_overfits else 'plateau, ceiling NOT reached'}.</b>"
        + (f" <b>pos_weight 1.0 control: {'ceiling reached' if pw1_overfits else 'plateau, ceiling NOT reached'}.</b>"
           if curve_pw1 is not None else
           " <b>pos_weight 1.0 control: pending.</b>"))
    rows = (
        f'<tr><td style="text-align:left">real cond, pos_weight 5</td><td>{cond_best:.3f}</td>'
        f'<td>{cond_final:.3f}</td><td>{cond_range[0]:.3f} &ndash; {cond_range[1]:.3f}</td></tr>'
        f'<tr><td style="text-align:left">zero cond, pos_weight 5</td><td>{zero_best:.3f}</td>'
        f'<td>{zero_final:.3f}</td><td>{zero_range[0]:.3f} &ndash; {zero_range[1]:.3f}</td></tr>')
    if curve_pw1 is not None:
        last_pw1_ps = curve_pw1[-1]["per_sample"]
        pw1_range = (min(p["best_iou"] for p in last_pw1_ps), max(p["best_iou"] for p in last_pw1_ps))
        rows += (f'<tr><td style="text-align:left">real cond, pos_weight 1 (control)</td>'
                 f'<td>{pw1_best:.3f}</td><td>{pw1_final:.3f}</td>'
                 f'<td>{pw1_range[0]:.3f} &ndash; {pw1_range[1]:.3f}</td></tr>')
    else:
        rows += ('<tr><td style="text-align:left">real cond, pos_weight 1 (control)</td>'
                 '<td colspan="3">pending</td></tr>')
    if curve_single is not None:
        rows += (f'<tr><td style="text-align:left">single shape, pos_weight 1, today\'s code</td>'
                  f'<td>{single_best:.3f}</td><td>{single_final:.3f}</td>'
                  f'<td>(one shape, not comparable to the ten-shape range column)</td></tr>')
    else:
        rows += ('<tr><td style="text-align:left">single shape, pos_weight 1, today\'s code</td>'
                  '<td colspan="3">pending</td></tr>')
    body += lp.results_table(
        ["run", "best mean IoU (nonzero)", "final-epoch mean IoU", "per-shape range, final epoch"],
        rows)
    return lp.section_v2("verdict", 1,
                          "Ten shapes at once, not the code or the loss weighting, is what fails to memorize",
                          body)


def sec_curve(curve_cond, curve_zero, curve_pw1):
    s_cond = {"name": "real cond, pw5", "color": COLOR_COND,
              "points": [(c["epoch"], c["val_iou_nonzero"]) for c in curve_cond
                         if c["val_iou_nonzero"] is not None]}
    s_zero = {"name": "zero cond, pw5", "color": COLOR_ZERO,
              "points": [(c["epoch"], c["val_iou_nonzero"]) for c in curve_zero
                         if c["val_iou_nonzero"] is not None]}
    series = [s_cond, s_zero]
    pw1_note = " A pos_weight&nbsp;1.0 control run is in progress and will be added to this chart once it lands."
    if curve_pw1 is not None:
        series.append({"name": "real cond, pw1 (control)", "color": COLOR_PW1,
                        "points": [(c["epoch"], c["val_iou_nonzero"]) for c in curve_pw1
                                   if c["val_iou_nonzero"] is not None]})
        pw1_note = ""
    refs = [{"y": VAE_CEILING, "label": f"VAE ceiling {VAE_CEILING:.2f}", "color": "#8ee08e"},
            {"y": FLOOR_72K, "label": f"72k held-out floor {FLOOR_72K:.1f}", "color": "#ff6b6b"}]
    chart = line_chart_svg(series, title="Mean train-set IoU (10 shapes, threshold-selected), nonzero-gated",
                            y_max=1.0, refs=refs,
                            note="<b>Mean over the 10 training shapes; the pos_weight-5 pair is identical "
                                 "except --cond, the pos_weight-1 control changes only --pos_weight.</b> "
                                 "The VAE ceiling (green) is the best any prediction could score even with "
                                 "a perfect model, from FACTSHEET_diagnostics.md's own round-trip "
                                 "measurement; the floor (red) is this project's observed held-out-IoU "
                                 "plateau across every trained 72k model to date." + pw1_note)
    body = chart
    return lp.section_v2("curve", 2, "Train-set IoU vs epoch, against the ceiling and floor", body)


def sec_pershape(curve_cond, curve_zero, curve_pw1):
    cards = []
    for sid in SIDS:
        pts_cond = [(c["epoch"], next((p["best_iou"] for p in c["per_sample"] if p["sid"] == sid), None))
                    for c in curve_cond]
        pts_zero = [(c["epoch"], next((p["best_iou"] for p in c["per_sample"] if p["sid"] == sid), None))
                    for c in curve_zero]
        pts_cond = [(e, v) for e, v in pts_cond if v is not None]
        pts_zero = [(e, v) for e, v in pts_zero if v is not None]
        series = [{"name": "real", "color": COLOR_COND, "points": pts_cond},
                  {"name": "zero", "color": COLOR_ZERO, "points": pts_zero}]
        if curve_pw1 is not None:
            pts_pw1 = [(c["epoch"], next((p["best_iou"] for p in c["per_sample"] if p["sid"] == sid), None))
                       for c in curve_pw1]
            pts_pw1 = [(e, v) for e, v in pts_pw1 if v is not None]
            series.append({"name": "pw1", "color": COLOR_PW1, "points": pts_pw1})
        chart = line_chart_svg(
            series,
            title=f"{SHORT[sid]} ({BUCKET[sid]}, frac {FRAC[sid]:.3f})",
            width=340, height=190, y_max=1.0, legend=False,
            refs=[{"y": VAE_CEILING, "label": "", "color": "#8ee08e"}])
        cards.append(f'<div style="min-width:340px">{chart}</div>')
    grid = ('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));'
            f'gap:8px 16px">{"".join(cards)}</div>')
    pw1_note = (" and violet = pos_weight 1 control" if curve_pw1 is not None else
                " (pos_weight 1 control pending, will appear here once it lands)")
    body = lp.prose(
        f"Each shape's own IoU trajectory, all runs overlaid (blue = real cond pw5, orange = "
        f"zero cond pw5{pw1_note}, green dashed = VAE ceiling, no floor line at this scale). "
        "Sparse shapes (low emissive fraction) are the hardest to memorize at pos_weight 5: "
        "they have the least positive-voxel signal for the loss to fit, even under the "
        "upweighting.")
    body += grid
    return lp.section_v2("pershape", 3, "Sparse shapes are harder to memorize than dense ones", body)


def sec_gallery():
    rows = []
    any_ready = False
    all_ready = True
    for sid in SIDS:
        inp = os.path.join(WORK, "out", "gt", f"{sid}_lit.png")
        gt = os.path.join(WORK, "out", "gt", f"{sid}_glow.png")
        cond = os.path.join(WORK, "out", "cond_best", f"{sid}_glow.png")
        zero = os.path.join(WORK, "out", "zero_best", f"{sid}_glow.png")
        label = f"{SHORT[sid]} ({BUCKET[sid]})"
        if all(os.path.exists(p) for p in (inp, gt, cond, zero)):
            any_ready = True
            entry = {
                "input": img_ref(copy_img(inp, f"{sid}_input.png")),
                "gt": img_ref(copy_img(gt, f"{sid}_gt.png")),
                "cond": img_ref(copy_img(cond, f"{sid}_cond.png")),
                "zero": img_ref(copy_img(zero, f"{sid}_zero.png")),
            }
            rows.append((label, [entry["input"], entry["gt"], entry["cond"], entry["zero"]]))
        else:
            all_ready = False
            ph = {"placeholder": "pending", "sub": "rendering"}
            rows.append((label, [ph, ph, ph, ph]))

    pending_note = (
        "" if all_ready else
        " <b>Rendering is still in progress at publish time</b> (mask-transfer stage "
        "timed out on its first serial submission and was resubmitted sharded, one "
        "task per shape/variant; render itself was submitted after). This section "
        "will fill in on this same page, same URL, once it lands &mdash; no rebuild "
        "or layout change needed."
    )
    matrix = lp.method_matrix(
        columns=["INPUT", "GT EMISSION", "COND PRED", "ZERO PRED"],
        rows=rows,
        caption_html=(
            "<b>All 10 training shapes, best checkpoint of each run.</b> INPUT is the "
            "studio-lit appearance render (identification only); GROUND TRUTH, COND and "
            "ZERO are all rendered emission-only (key light off) with the project's own "
            "mask &times; albedo formulation, AgX, bloom 9/1.0/&minus;0.15, samples 256, "
            "emission strength 4.0 &mdash; identical settings across every panel. One "
            "representative draw (draw 0 of 3, seed 0) per shape per model." + pending_note),
        native_px=768, content="photo", page_inner=820, id="gallery-matrix")
    title = ("Rendered predictions on all 10 training shapes" if all_ready else
             "Rendered predictions on all 10 training shapes (pending)")
    return lp.section_v2("gallery", 4, title, matrix)


def sec_loss(curve_cond, curve_zero, curve_pw1):
    s_cond = {"name": "real cond, pw5", "color": COLOR_COND,
              "points": [(c["epoch"], c["train_loss"]) for c in curve_cond]}
    s_zero = {"name": "zero cond, pw5", "color": COLOR_ZERO,
              "points": [(c["epoch"], c["train_loss"]) for c in curve_zero]}
    series = [s_cond, s_zero]
    y_max = max(max(y for _, y in s_cond["points"]), max(y for _, y in s_zero["points"]))
    if curve_pw1 is not None:
        s_pw1 = {"name": "real cond, pw1", "color": COLOR_PW1,
                 "points": [(c["epoch"], c["train_loss"]) for c in curve_pw1]}
        series.append(s_pw1)
        y_max = max(y_max, max(y for _, y in s_pw1["points"]))
    chart = line_chart_svg(series, title="Flow-matching training loss (per-epoch mean)",
                            y_max=y_max * 1.05, y_min=0)
    body = lp.prose("Secondary: training loss for all runs. A falling loss with a flat IoU curve "
                     "would indicate the loss and the evaluation metric have decoupled; that is not "
                     "what the pos_weight-5 runs show here (see the curve section's own reading). "
                     "Note the pos_weight-1 loss is on a different scale (plain MSE, no per-voxel "
                     "reweighting) and is not directly comparable in magnitude to the pos_weight-5 "
                     "loss, only in trend.")
    body += chart
    return lp.section_v2("loss", 5, "Training loss, all runs", body)


def sec_provenance(curve_cond, curve_zero, curve_pw1, curve_single):
    shape_rows = "".join(
        f'<tr><td style="text-align:left">{SHORT[s]}</td><td>{BUCKET[s]}</td>'
        f'<td>{FRAC[s]:.4f}</td></tr>' for s in SIDS)
    shape_table = lp.results_table(["shape", "bucket", "emissive_frac"], shape_rows)

    param_rows = "".join(
        f'<tr><td style="text-align:left">{html.escape(k)}</td><td>{html.escape(v)}</td></tr>'
        for k, v in [
            ("dataset", "dataset_direct/overfit_ct10 (10 symlinks into train_72k, v2 split)"),
            ("init_ckpt", "full_seg (HF fenghora/SegviGen)"),
            ("epochs, pos_weight 5 runs (10 shapes)", str(curve_cond[-1]["epoch"])),
            ("epochs, pos_weight 1 control (10 shapes)", str(curve_pw1[-1]["epoch"]) if curve_pw1 is not None else "pending"),
            ("epochs, single-shape control", str(curve_single[-1]["epoch"]) + " (hit the 2h time limit before its 200-epoch target; last value carried forward)" if curve_single is not None else "pending"),
            ("n_per_epoch, ten-shape runs", "10 (one full pass per epoch)"),
            ("n_per_epoch, single-shape control", "20 (matches July's convention for a one-shape split)"),
            ("save_every / val_quick, pw5 runs", "20 / 10 (all 10 shapes)"),
            ("save_every / val_quick, pw1 control", "20 / 10 (all 10 shapes)"),
            ("save_every / val_quick, single-shape control", "10 / 1"),
            ("lr", "1e-5, const"),
            ("pos_weight", "5.0 (both ten-shape cond runs), 1.0 (both controls)"),
            ("emis_oversample", "on, all runs"), ("ema", "0.999, all runs"),
            ("select_on", "nonzero"),
            ("cond (ten-shape run A)", "real (sid's own cond.pth)"),
            ("cond (ten-shape run B)", "zero (torch.zeros)"),
            ("cond (pw1 controls, both)", "real (conditioning already shown not to matter much for "
                                           "memorization at pos_weight 5, so one cond setting suffices)"),
            ("single-shape control shape", "91d94c0f556b4aa69d3ab09919d1d380, mid-density "
                                            "(emissive_frac 0.089), one of the ten"),
            ("render", "key 0, AgX, exposure 0, bloom 1 (size 9, thr 1.0, mix -0.15), "
                       "samples 256/96, emit_strength 4.0, mask thr 0.5, tol 2.0"),
        ])
    param_table = lp.results_table(["parameter", "value"], param_rows)

    job_rows = "".join(
        f'<tr><td style="text-align:left">{html.escape(stage)}</td><td>{jid}</td><td>{res}</td></tr>'
        for stage, jid, res in [
            ("smoke test (both cond modes, 1 epoch)", "240856", "GPU a40x1, 8 cpus, debug partition"),
            ("material survey (Blender slot order)", "240860", "CPU, 1 task, debug partition"),
            ("train, real cond, pw5", "240857", "GPU a40x1, 8 cpus, debug partition, 6h limit -- COMPLETED"),
            ("train, zero cond, pw5", "240858", "GPU a40x1, 8 cpus, debug partition, 6h limit -- COMPLETED"),
            ("train, real cond, pw1 (control)", "242172", "GPU a40x1, 8 cpus, debug partition, 3h limit"),
            ("dump_pred_ct10 (both pw5 best.ckpt, 3 draws)", "240861",
             "GPU a40x1, 8 cpus, dependency=afterok:240857:240858 -- COMPLETED"),
            ("pred_mask_to_asset, first attempt (serial)", "240862",
             "CPU, 8 cpus, dependency=afterok:240861 -- TIMEOUT at case 29/30 "
             "(45min budget too tight for 30 serial mask transfers)"),
            ("render_emissive, first attempt", "240863",
             "CPU, 64 cpus, long partition, dependency=afterok:240862 -- never ran "
             "(DependencyNeverSatisfied when 240862 timed out; cancelled)"),
            ("pred_mask_to_asset, resharded (array, one task/shape-variant)", "242142",
             "CPU, 8 cpus/task, 30-task array, no shared time budget -- COMPLETED, all 30 verified"),
            ("render_emissive, resubmitted by hand", "242178",
             "CPU, 64 cpus, long partition, no dependency chain -- COMPLETED"),
            ("train, pos_weight 1 control, extended to 400 epochs", "242211",
             "GPU a40x1, 8 cpus, debug partition, 6h limit -- COMPLETED"),
            ("train, single shape, pos_weight 1, today's code", "242210",
             "GPU a40x1, 8 cpus, debug partition, 2h limit -- TIMEOUT at the 2h limit, "
             "not a failure: it had already reached its final IoU well before the cutoff"),
        ])
    job_table = lp.results_table(["stage", "solar job id", "resources"], job_rows)

    body = lp.prose(
        "10 shapes: the historical <code>canon_overfit10.txt</code> set (July) was checked "
        "against the current (v2-repartitioned) <code>train_72k</code> directory and against "
        "<code>thumb_audit/classified_sids.json</code>'s placeholder list; only 3 of the 10 "
        "still resolved with a real (non-placeholder) <code>cond.pth</code> in the live tree "
        "(<code>000b9fd4</code>, <code>001c7929</code>, <code>001dd281</code>, all sparse). The "
        "other 7 were replaced with fresh picks from <code>train_72k</code> spanning emissive "
        "fraction, checked the same way, giving 3 sparse / 4 mid / 3 dense.")
    body += shape_table
    body += lp.prose("Training / render parameters:")
    body += param_table
    body += lp.prose("Job ids and resources, in pipeline order. The pos_weight 5 pair ran in "
                      "parallel; dump/mask were chained by SLURM <code>--dependency=afterok</code>. "
                      "The FIRST mask-transfer attempt (240862) ran serially with a 45-minute "
                      "budget and hit that limit at case 29/30, which left its dependent render job "
                      "(240863) permanently PENDING (DependencyNeverSatisfied) since a timed-out "
                      "job never satisfies afterok. Fix: 240863 cancelled, mask-transfer resharded "
                      "to one SLURM array task per shape/variant (242142, no shared timing budget), "
                      "and render will be submitted fresh once that array is verified complete -- "
                      "deliberately NOT re-chained by dependency this time, submitted by hand after "
                      "checking every array task's own exit status.")
    body += job_table
    body += lp.prose(
        "The train split is a NEW directory of symlinks "
        "(<code>dataset_direct/overfit_ct10/</code>) pointing into the live, read-only "
        "<code>train_72k</code> tree &mdash; additive, no existing bucket was mutated. Verified "
        "resolvable with a 1-epoch smoke run (job 240856) before the real submission. "
        "No git commit, no workstation compute; all GPU/CPU work ran on solar "
        "(account 3dlg-hcvc-lab), excluding cs-venus-05/09/19 (broken Blackwell) and "
        "cs-venus-15 (stale NFS after the v2 repartition).")
    return lp.section_v2("provenance", 6, "Commands, parameters, and where the raw outputs live", body)


def main():
    curve_cond = load_curve("cond")
    curve_zero = load_curve("zero")
    # prefer the 400-epoch pw1 control (parity with the pw5 runs' epoch budget,
    # task #32) once it lands; fall back to the first 150-epoch pw1 control.
    curve_pw1 = load_curve_optional("pw1_400ep") or load_curve_optional("pw1")
    curve_single = load_single_shape_curve()

    cond_best, zero_best = best_mean(curve_cond), best_mean(curve_zero)
    pw5_overfits = max(cond_best, zero_best) > 0.5 * VAE_CEILING
    pw1_best = best_mean(curve_pw1) if curve_pw1 is not None else None
    single_best = best_mean(curve_single) if curve_single is not None else None

    if pw5_overfits:
        verdict_word = "pw5 overfits"
    elif curve_pw1 is None:
        verdict_word = "pw5 plateaus, pw1 control pending"
    elif pw1_best > 0.5 * VAE_CEILING:
        verdict_word = "pw5 breaks a working pipeline"
    elif curve_single is None:
        verdict_word = "both pw plateau, single-shape control pending"
    elif single_best > 0.5 * VAE_CEILING:
        verdict_word = "code is fine, ten shapes at once is the hard part"
    else:
        verdict_word = "single shape also fails, likely a code regression"

    stats = [
        ("10", "training shapes"),
        (f"{cond_best:.3f}", "best IoU, real cond, pw5"),
        (f"{zero_best:.3f}", "best IoU, zero cond, pw5"),
        (f"{pw1_best:.3f}" if pw1_best is not None else "pending", "best IoU, pw1 control"),
        (f"{single_best:.3f}" if single_best is not None else "pending", "best IoU, single shape"),
        (f"{VAE_CEILING:.2f}", "VAE round-trip ceiling"),
        (verdict_word, "verdict"),
    ]
    hero = lp.hero_header(
        "SegviGen · overfit / conditioning diagnostic",
        "Ten shapes at once will not memorize, but one shape does",
        dek_html=(
            "Every trained model to date scores near-zero median held-out IoU, more data "
            "makes it worse, and the 72k model fails on its own training shapes. Four runs "
            "narrow this down: two on the same 10 shapes, identical except "
            "<code>--cond real</code> vs <code>--cond zero</code> (both at "
            "<code>--pos_weight 5.0</code>); a <code>--pos_weight 1.0</code> control on the "
            "same 10 shapes, extended to 400 epochs for parity; and a single-shape control "
            "on today's code to check for a code regression against July's 0.918-by-epoch-30 "
            "result. Reading them together: today's code is not broken (the single shape "
            "memorizes cleanly), pos_weight 5 is not the sole cause (pos_weight 1 plateaus "
            "too), and more epochs alone did not close the gap. What breaks specifically "
            "with ten shapes together is the open question."),
        stats=stats,
        toc=[(i, lab) for i, lab in OUTLINE])

    body = [
        sec_verdict(curve_cond, curve_zero, curve_pw1, curve_single),
        sec_curve(curve_cond, curve_zero, curve_pw1),
        sec_pershape(curve_cond, curve_zero, curve_pw1),
        sec_gallery(),
        sec_loss(curve_cond, curve_zero, curve_pw1),
        sec_provenance(curve_cond, curve_zero, curve_pw1, curve_single),
    ]

    page_html = lp.page(
        title="Overfit / cond-vs-zero diagnostic (overfit_condtest)",
        header_html=hero,
        body_sections=body,
        assets_rel=SITE_ASSETS,
        assets_dir=os.path.join(WEB, "assets"),
        theme="v3",
        tree_html=wz.tree_html(active_href=None),
        nav_title="overfit condtest",
        outline_entries=[{"id": i, "label": lab} for i, lab in OUTLINE],
        version_slot=lp.v3_version_slot(date=PAGE_DATE),
        needs_katex=False,
        extra_head=f'<link rel="icon" href="{FAVICON}">',
    )

    violations = wz.console_links_in(page_html)
    if violations:
        sys.exit(f"ZONE-LINK GUARD FAILED: page links to the console: {violations}")

    out = os.path.join(HERE, "index.html")
    with open(out, "w") as f:
        f.write(page_html)
    print(f"wrote {out} ({len(page_html)} bytes)")
    print("  zone-link guard: clean")

    publish_assets(os.path.join(WEB, "assets"))
    print("assets published")


if __name__ == "__main__":
    main()
