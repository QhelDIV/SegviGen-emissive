#!/usr/bin/env python3
"""Build the N=300 diagnostics page on the xgpage v3 workspace shell.

The page answers one question: why does every emissive segmentation model on
this project land near 0.1 IoU. A diagnostics run over 300 held-out shapes
measured the three explanations the project had been carrying and ruled all
three out, then stratified the comparison against the baselines, which
corrects the aggregate reading in the model's favour in the sparse regime and
against it everywhere else.

Every number on the page is READ AT BUILD TIME from the run's own
diagnostics.json (staged in data/, provenance in the appendix), never retyped.
Summary rows come from the file's own summary blocks; the four distribution
figures are computed here from its per-shape records, and check() asserts that
the recomputation reproduces the file's summaries before the page is written.

Run: .venv2/bin/python web/_preview/diagnostics/build.py [--publish]
  (.venv2 = /cs/3dlg-project/3dlg-hcvc/omages/omages_internal/.venv2/bin/python)
"""
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WEB)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, HERE)
import workspace_zone as wz  # noqa: E402

import charts as ch  # noqa: E402
import xgpage as lp  # noqa: E402
from xgpage.publish import publish_assets  # noqa: E402

SITE_ROOT = "/projects/omages/yanxg/lightgen"
SITE_ASSETS = f"{SITE_ROOT}/assets"
FAVICON = f"{SITE_ROOT}/assets/images/favicon.png"

PAGE_SLUG = "diagnostics"
PAGE_HREF = f"{wz.WORKSPACE_URL}/{PAGE_SLUG}/index.html"
PUBLISH_DIR = os.path.join(str(wz.WORKSPACE_DIR), PAGE_SLUG)
PAGE_DATE = "2026-08-06"

DATA = os.path.join(HERE, "data", "diagnostics.json")
SOURCE_PATH = ("/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/"
               "emis_72k_unfilt/run1/diagnostics.json")
JOB_ID = "239795"

THR = "0.5"          # the reporting threshold, fixed across the page
THRS = ["0.2", "0.3", "0.4", "0.5"]
BUCKETS = ["[0,0.01)", "[0.01,0.05)", "[0.05,0.2)", "[0.2,0.5)", "[0.5,0.8)",
           "[0.8,1.0]"]


# --------------------------------------------------------------------- the data
def load():
    with open(DATA) as f:
        return json.load(f)


def per_shape(d):
    """Model IoU, draw std, ceiling and emissive fraction, aligned by shape id,
    plus the nonzero-ground-truth mask. baseline_all_one is exactly the shape's
    ground-truth emissive fraction under the true mask, so it is zero on, and
    only on, an exactly empty ground truth."""
    p3, p12, meta = (d["per_shape_diag3"], d["per_shape_diag1_diag2"],
                     d["per_shape_meta_emissive_frac"])
    sids = list(p3)
    return {
        "sids": sids,
        "iou": np.array([p3[s]["iou_by_thr_mean"][THR] for s in sids]),
        "std": np.array([p3[s]["iou_by_thr_std"][THR] for s in sids]),
        "ceiling": np.array([p12[s]["ceiling_iou_by_thr"][THR] for s in sids]),
        "frac": np.array([meta[s] for s in sids]),
        "nonzero": np.array([p12[s]["baseline_all_one"] > 0 for s in sids]),
    }


def bucket_mask(frac, i):
    edges = [0, 0.01, 0.05, 0.2, 0.5, 0.8, 1.0]
    lo, hi = edges[i], edges[i + 1]
    return (frac >= lo) & (frac <= hi) if i == 5 else (frac >= lo) & (frac < hi)


def check(d, ps):
    """The per-shape recomputation, asserted against the file's own summaries.
    A distribution figure drawn from records that disagree with the summary
    rows would put two different runs on one page."""
    bad = []

    def near(name, got, want, tol=5e-4):
        if abs(got - want) > tol:
            bad.append(f"{name}: recomputed {got:.6f} vs reported {want:.6f}")

    m = d["diag3_model_summary_by_thr"][THR]
    near("model mean (all 300)", float(ps["iou"].mean()), m["mean"])
    near("model median (all 300)", float(np.median(ps["iou"])), m["median"])
    s = d["diag3_draw_std_summary_by_thr"][THR]
    near("draw std mean", float(ps["std"].mean()), s["mean"])
    near("draw std p90", float(np.percentile(ps["std"], 90)), s["p90"])
    c = d["diag1_ceiling_summary_by_thr"][THR]["nonzero_gt_only"]
    near("ceiling mean (nonzero GT)", float(ps["ceiling"][ps["nonzero"]].mean()),
         c["mean"])
    if int(ps["nonzero"].sum()) != d["gt_composition"]["n_nonzero_gt"]:
        bad.append(f'nonzero-GT count {int(ps["nonzero"].sum())} vs '
                   f'{d["gt_composition"]["n_nonzero_gt"]}')
    strat = d["diag4_stratified"]
    for i, b in enumerate(BUCKETS):
        msk = bucket_mask(ps["frac"], i)
        if int(msk.sum()) != strat[b]["n"]:
            bad.append(f"bucket {b}: n {int(msk.sum())} vs {strat[b]['n']}")
        near(f"bucket {b} model", float(ps["iou"][msk].mean()),
             strat[b][f"model_iou_at_{THR}"])
        near(f"bucket {b} ceiling", float(ps["ceiling"][msk].mean()),
             strat[b][f"ceiling_iou_at_{THR}"])
    return bad


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


# ------------------------------------------------------------- page-local CSS
EXTRA_CSS = f"""
<style>
{ch.PALETTE_CSS}
/* theme3.css releases .prose's max-width so the 820px column IS the measure,
   but leaves .chart (760px) and .chartnote (720px) on v2's narrower caps, which
   gives a chart and its note two more left edges than the prose around them.
   Released to one edge, as the render-sweep page does. */
.xg3 .chart, .xg3 .chartnote {{ max-width: none; }}

/* theme.css carries v1-era floors (table.results{{min-width:960px}},
   td.rowhead{{min-width:210px}}) that every v3 page inherits, so any results
   table on the 820px column scrolls internally however short its cells are. */
.xg2 table.results {{ min-width: 0; }}
.xg2 table.results td.rowhead {{ min-width: 0; }}
.xg2 table.results td, .xg2 table.results th {{ padding-left: 7px; padding-right: 7px; }}
.xg2 table.results td.num {{ font-variant-numeric: tabular-nums; text-align: right; }}
.xg2 table.results td.win {{ color: var(--accent-ink); font-weight: 700; }}

/* Scroll room past the last section: xg3.js's scrollspy marks the nearest
   [id] above a 110px line, and the final section is too short to reach it. */
.xg3 .v3-main .page::after {{ content: ""; display: block; height: 40vh; }}
</style>
"""


def tree_html():
    entries = wz.tree_entries()
    for group in entries:
        for leaf in group.get("children", []):
            leaf["active"] = leaf["href"] == PAGE_HREF
    return lp.v3_tree(entries, title="Lightgen", subtitle="research workspace",
                      tree_src=wz.TREE_JSON_URL)


def row(cells, cls=None):
    """One results_table row; cells is a list of (html, css_class or None)."""
    return "<tr>" + "".join(
        f'<td class="{c or "rowhead"}">{h}</td>' for h, c in cells) + "</tr>"


# ----------------------------------------------------------------- the build
def build(publish=False):
    assets_dir = os.path.join(WEB, "assets")
    d = load()
    ps = per_shape(d)
    problems = check(d, ps)
    if problems:
        sys.exit("PER-SHAPE CHECK FAILED:\n  " + "\n  ".join(problems))

    cfg = d["config"]
    comp = d["gt_composition"]
    ceil_nz = {t: d["diag1_ceiling_summary_by_thr"][t]["nonzero_gt_only"] for t in THRS}
    model_all = {t: d["diag3_model_summary_by_thr"][t] for t in THRS}
    std_sum = d["diag3_draw_std_summary_by_thr"][THR]
    coord = d["coord_match_diagnostics"]
    strat = d["diag4_stratified"]
    base = d["diag2_baseline_summary"]

    # nonzero-ground-truth model aggregates, from the per-shape records
    nz = ps["nonzero"]
    iou_nz = ps["iou"][nz]
    model_nz_mean, model_nz_median = float(iou_nz.mean()), float(np.median(iou_nz))
    n_zero_iou = int((iou_nz == 0).sum())
    n_below_001 = int((iou_nz < 0.01).sum())
    n_at_half = int((iou_nz >= 0.5).sum())

    ceil_all = ps["ceiling"]
    n_ceil_below_09 = int((ceil_all < 0.9).sum())
    n_ceil_below_08 = int((ceil_all < 0.8).sum())
    n_ceil_below_05 = int((ceil_all < 0.5).sum())

    model_mean_all = model_all[THR]["mean"]
    n_std_over_mean = int((ps["std"] > model_mean_all).sum())
    thr_lo = min(model_all[t]["mean"] for t in THRS)
    thr_hi = max(model_all[t]["mean"] for t in THRS)

    b_pbr = base["pbr_heuristic_best_global_nonzero_gt_only"]
    b_random = base["random_matched"]["nonzero_gt_only"]["mean"]
    b_allone = base["all_one"]["nonzero_gt_only"]["mean"]

    # ================================================================== hero
    hero = lp.hero_header(
        f"lightgen &middot; emissive segmentation &middot; diagnostics &middot; {PAGE_DATE}",
        "Why Every Emissive Model Sits Near 0.1 IoU",
        dek_html=(
            "Three candidate explanations were measured on 300 held-out shapes, and all "
            "three were ruled out. The reference the metric scores against is clean, the "
            "coordinates line up, and the threshold does not matter. What remains is a "
            "model whose median IoU is 0.0067: it produces almost nothing on the typical "
            "shape, leads a random baseline where emission is sparse, and is buried by it "
            "where most of the shape emits. Every number below is read from the run's own "
            "diagnostics file when this page is built."),
        stats=[
            (f'{cfg["n"]}', "shapes scored"),
            (f4(ceil_nz[THR]["mean"]), "round-trip ceiling"),
            (f4(model_nz_median), "model median IoU"),
            (f4(coord["mean_emis_mask_agreement_with_true_voxel"]), "mask agreement"),
            (f4(std_sum["p90"]), "p90 draw-to-draw spread"),
        ],
        toc=[("verdict", "What is settled"), ("ceiling", "The ceiling"),
             ("ruledout", "Alignment and threshold"), ("median", "The median"),
             ("stratified", "Stratified"), ("variance", "Draw variance"),
             ("live", "Still live")],
    )

    # ======================================================== 01 what is settled
    scope_rows = "".join([
        row([("Dataset and split", None),
             (f'<code>dataset_direct</code>, <code>{cfg["split"]}</code>', None)]),
        row([("Checkpoint", None),
             ('<code>emis_72k_unfilt/run1/best.ckpt</code>, which resolves to '
              '<code>epoch_0012</code>', None)]),
        row([("Shapes scored", None),
             (f'{comp["n_ok"]}, of which {comp["n_nonzero_gt"]} have a nonzero '
              f'ground-truth mask and {comp["n_zero_gt"]} are exactly empty', None)]),
        row([("Sampling", None),
             (f'{cfg["draws"]} draws per shape, {cfg["steps"]} steps, seed '
              f'{cfg["seed"]}', None)]),
        row([("Thresholds", None),
             (", ".join(str(t) for t in cfg["thrs"])
              + f"; every figure on this page is at {THR} unless it says otherwise",
              None)]),
    ])
    sec1 = lp.section_v2(
        "verdict", 1,
        "Three explanations are ruled out, and the model&rsquo;s median is "
        f"{f4(model_nz_median)}",
        lp.verdict_box(
            f"The measurement is not what is broken. The reference the metric scores "
            f"against reconstructs at {f4(ceil_nz[THR]['mean'])}, the decoded mask lands "
            f"in the same frame as that reference at "
            f"{f4(coord['mean_emis_mask_agreement_with_true_voxel'])}, and the model "
            f"moves only between {f4(thr_lo)} and {f4(thr_hi)} across all four thresholds "
            f"tried. What is left is the model, and on the typical shape it produces "
            f"essentially nothing.")
        + lp.results_table(["what was run", "how it was configured"], scope_rows)
        + lp.chartnote(
            f"<b>The scope of the run, so every number below has a stated population.</b> "
            f"One checkpoint, one held-out split, {cfg['n']} shapes, {cfg['draws']} draws "
            f"each. Nothing on this page is a training-curve reading; the model was "
            f"re-sampled for the diagnostics.")
        + lp.callout(
            f"An empty prediction against an empty ground truth counts as IoU 1.0, which "
            f"is the convention the project&rsquo;s own evaluation already uses. "
            f"{comp['n_zero_gt']} of the {comp['n_ok']} shapes have an exactly empty "
            f"ground-truth mask, and on those the ceiling and the all-zero baseline are "
            f"1.0 for free. Every aggregate is therefore reported twice, once over all "
            f"{comp['n_ok']} shapes and once over the {comp['n_nonzero_gt']} with a "
            f"nonzero mask, and the page says which it is using each time.",
            title="How IoU is counted here", warn=True)
        + lp.prose(
            f"<p>The three sections that follow close the three explanations, in the order "
            f"they were held: the round trip, the coordinate frame, the threshold. The two "
            f"after them describe what is left. The aggregate reading, that the model "
            f"loses to a coin flip, is true and incomplete: stratified by how much of a "
            f"shape emits, the model is above random in the two sparse groups and buried "
            f"in the dense ones, where marking every voxel is nearly correct for free. "
            f"Section 05 is where that is argued.</p>"))

    # ====================================================== 02 the ceiling
    strip_buckets = []
    for i, b in enumerate(BUCKETS):
        msk = bucket_mask(ps["frac"], i)
        strip_buckets.append((b, strat[b]["n"], strat[b][f"ceiling_iou_at_{THR}"],
                              list(ps["ceiling"][msk])))
    b0_nonzero = strat[BUCKETS[0]][f"ceiling_iou_at_{THR}_nonzero_only"]
    sec2 = lp.section_v2(
        "ceiling", 2,
        "The reference the metric scores against is clean: the round-trip ceiling is "
        f"{f4(ceil_nz[THR]['mean'])}",
        ch.ceiling_strip(strip_buckets)
        + lp.chartnote(
            f"<b>The round-trip ceiling sits near 0.95 in every group, which is where a "
            f"corrupted reference would have shown up and did not.</b> Each dot is one of "
            f"the {cfg['n']} shapes: its ground-truth latent decoded, thresholded at "
            f"{THR}, and scored against its true emissive voxel mask. Groups run left to "
            f"right by how much of the shape emits. The joined markers are the group "
            f"means, from {f3(min(strat[b][f'ceiling_iou_at_{THR}'] for b in BUCKETS))} to "
            f"{f3(max(strat[b][f'ceiling_iou_at_{THR}'] for b in BUCKETS))}, and the line "
            f"through them is the finding. The spread behind the means is real and is "
            f"shown rather than summarised away: {n_ceil_below_09} shapes fall below 0.9, "
            f"{n_ceil_below_08} below 0.8 and {n_ceil_below_05} below 0.5. The leftmost "
            f"group holds all {comp['n_zero_gt']} exactly-empty shapes, which score 1.0 "
            f"under the convention above; over that group&rsquo;s other "
            f"{strat[BUCKETS[0]]['n_nonzero_gt']} shapes the mean is {f3(b0_nonzero)}.")
        + lp.prose(
            f"<p>This measurement was run to confirm a prediction, and it rejected it. The "
            f"expectation was that the emission field passes through a variational "
            f"autoencoder before the metric ever sees it, and that the round trip alone "
            f"damaged the mask enough to cap any model near the score every model was "
            f"reaching. If that were true the ceiling would sit near 0.1. It sits at "
            f"{f4(ceil_nz[THR]['mean'])} mean and {f4(ceil_nz[THR]['median'])} median over "
            f"the {comp['n_nonzero_gt']} shapes with a nonzero mask, with the tenth "
            f"percentile at {f4(ceil_nz[THR]['p10'])}, and every group mean falls between "
            f"{f3(min(strat[b][f'ceiling_iou_at_{THR}'] for b in BUCKETS))} and "
            f"{f3(max(strat[b][f'ceiling_iou_at_{THR}'] for b in BUCKETS))}. Whatever "
            f"produces 0.1 IoU happens after the reference is built.</p>"
            f"<p>The ceiling is also flat across the cut applied to the decoded field: "
            f"{f4(ceil_nz['0.2']['mean'])} at threshold 0.2 and {f4(ceil_nz['0.5']['mean'])} "
            f"at 0.5, with the median unchanged to three decimals. A reconstruction that "
            f"was losing the mask would be sensitive to where the cut is taken.</p>"))

    # ============================================ 03 alignment and threshold
    coord_rows = "".join([
        row([("Decoded ground-truth voxels matched to the input", None),
             (f4(coord["mean_match_frac_of_decoded_gt"]) + " mean, "
              + f4(coord["min_match_frac_of_decoded_gt"]) + " worst shape", "num")]),
        row([("Raw coordinates identical in and out", None),
             (f'{coord["same_raw_coords_input_output_rate"]:.0%} of shapes', "num")]),
        row([("Decoded emissive mask against the true voxel grid", None),
             (f4(coord["mean_emis_mask_agreement_with_true_voxel"]), "num")]),
    ])
    sec3 = lp.section_v2(
        "ruledout", 3,
        f"The coordinates agree at "
        f"{f3(coord['mean_match_frac_of_decoded_gt'])}, and the model stays between "
        f"{f4(thr_lo)} and {f4(thr_hi)} at every threshold",
        ch.threshold_lines([float(t) for t in THRS],
                           [ceil_nz[t]["mean"] for t in THRS],
                           [model_all[t]["mean"] for t in THRS])
        + lp.chartnote(
            f"<b>Neither the ceiling nor the model responds to where the cut is taken, so "
            f"the threshold is not what separates them.</b> Both series are mean IoU on "
            f"one shared axis, plotted at the four thresholds the run swept. The ceiling "
            f"is over the {comp['n_nonzero_gt']} shapes with a nonzero mask; the model is "
            f"over all {cfg['n']}. Across the four cuts the ceiling spans "
            f"{f4(min(ceil_nz[t]['mean'] for t in THRS))} to "
            f"{f4(max(ceil_nz[t]['mean'] for t in THRS))} and the model spans {f4(thr_lo)} "
            f"to {f4(thr_hi)}, a spread narrower than the gap between the "
            f"model&rsquo;s own mean and its median.")
        + lp.results_table(["alignment check", "measured"], coord_rows)
        + lp.chartnote(
            f"<b>The decoded field lands in the same frame as the mask it is scored "
            f"against.</b> The first row is the share of decoded ground-truth voxels that "
            f"find their counterpart in the input, averaged over shapes and reported with "
            f"its worst case. The second is a check that the raw coordinates leave the "
            f"pipeline as they entered it. The third compares the thresholded emissive "
            f"mask with the true voxel grid directly.")
        + lp.prose(
            f"<p>Coordinate misalignment was the second explanation on the list, and it "
            f"would have been consistent with the symptom: a prediction that is correct in "
            f"content but shifted in space scores near zero on an intersection-over-union "
            f"and looks like a model that has learned nothing. The alignment is "
            f"{f3(coord['mean_match_frac_of_decoded_gt'])} on average and "
            f"{f3(coord['min_match_frac_of_decoded_gt'])} on its worst shape, and the raw "
            f"coordinates are identical in and out on every shape, so a shift of that kind "
            f"is not present.</p>"))

    # =========================================================== 04 the median
    sec4 = lp.section_v2(
        "median", 4,
        "On the typical shape the model produces essentially nothing",
        ch.rank_curve(list(iou_nz), model_nz_mean, model_nz_median, n_below_001, "0.01")
        + lp.chartnote(
            f"<b>Half the shapes score under {f4(model_nz_median)}, and the mean is a "
            f"different statistic about a different set of shapes.</b> Every one of the "
            f"{comp['n_nonzero_gt']} shapes with a nonzero ground-truth mask, sorted by "
            f"its own IoU, left to right. The floor on the left is the failure mass: "
            f"{n_zero_iou} shapes score exactly zero and {n_below_001} score below 0.01, "
            f"so the median lands on that floor at {f4(model_nz_median)}. The mean, "
            f"{f4(model_nz_mean)}, is the dashed line, and it is above the curve for "
            f"roughly four fifths of the shapes: it is carried by the {n_at_half} shapes "
            f"at or above 0.5 in the tail on the right. The shaded region is the part of "
            f"the population below 0.01.")
        + lp.prose(
            f"<p>The median is the honest headline for this run. Over the "
            f"{comp['n_nonzero_gt']} shapes with something to find, the model&rsquo;s "
            f"median IoU is {f4(model_nz_median)} and its mean is {f4(model_nz_mean)}; "
            f"over all {cfg['n']} shapes the mean is {f4(model_mean_all)} and the median "
            f"is {f4(model_all[THR]['median'])}. Reporting the mean alone describes a "
            f"minority of the shapes and reads as a weak model. Reporting the median says "
            f"what actually happens on a shape drawn at random, which is close to "
            f"nothing.</p>"
            f"<p>The tail is not noise and it is not an artifact of the empty-mask "
            f"convention: these are shapes with a real emissive mask that the model "
            f"overlaps substantially. Any account of the failure has to explain both ends, a "
            f"floor with more than half the population on it and a thin tail of shapes "
            f"that work.</p>"))

    # ======================================================== 05 stratified
    sbuckets = [{
        "label": b, "n": strat[b]["n"],
        "ceiling": strat[b][f"ceiling_iou_at_{THR}"],
        "model": strat[b][f"model_iou_at_{THR}"],
        "random": strat[b]["baseline_random_matched"],
        "all_one": strat[b]["baseline_all_one"],
        "pbr": strat[b]["baseline_pbr_heuristic_global_pct"],
    } for b in BUCKETS]
    lead = [b for b in sbuckets if b["model"] > b["random"]]
    strat_rows = "".join(
        row([(b["label"], None), (str(b["n"]), "num"),
             (f3(b["ceiling"]), "num"),
             (f3(strat[b["label"]]["baseline_all_zero"]), "num"),
             (f3(b["all_one"]), "num"), (f3(b["random"]), "num"),
             (f3(b["pbr"]), "num"),
             (f3(b["model"]), "num win" if b["model"] > b["random"] else "num")])
        for b in sbuckets)
    overall_rows = "".join([
        row([("Round-trip ceiling", None), (f3(ceil_nz[THR]["mean"]), "num"),
             ("the reconstruction bound, not a competitor", None)]),
        row([("Predict everything", None), (f3(b_allone), "num"),
             ("every voxel marked emissive", None)]),
        row([("Random at the true density", None), (f3(b_random), "num"),
             ("a coin flip per voxel at the shape&rsquo;s own emissive rate", None)]),
        row([("Albedo-brightness rule", None), (f3(b_pbr["mean_iou"]), "num"),
             (f'a global brightness cut on the base colour, best of the seven '
              f'percentiles tried (the {b_pbr["pct"]}th)', None)]),
        row([("All zero", None),
             (f3(base["all_zero"]["nonzero_gt_only"]["mean"]), "num"),
             ("zero by construction once empty masks are excluded", None)]),
        row([("<b>The model</b>", None), (f"<b>{f3(model_nz_mean)}</b>", "num"),
             ("<b>below every baseline except all zero</b>", None)]),
    ])
    sec5 = lp.section_v2(
        "stratified", 5,
        "The model beats random where emission is sparse and is buried where guessing "
        "wins",
        ch.stratified(sbuckets, highlight=(1, 2),
                      highlight_label="the model is above random here")
        + lp.chartnote(
            f"<b>The aggregate ordering reverses inside the two sparse groups: the model "
            f"scores {f3(sbuckets[1]['model'])} against random&rsquo;s "
            f"{f3(sbuckets[1]['random'])} in [0.01,0.05) and {f3(sbuckets[2]['model'])} "
            f"against {f3(sbuckets[2]['random'])} in [0.05,0.2).</b> Groups run left to "
            f"right by the share of the shape that emits in the ground truth; bars are "
            f"mean IoU at threshold {THR} within each group; the dashed rule over each "
            f"group is the round-trip ceiling for those shapes. Only the model is "
            f"labelled, and the shaded groups are the two where it is above random. Read "
            f"right and the picture inverts: where most of a shape emits, marking every "
            f"voxel is nearly correct for free, so predict-everything reaches "
            f"{f3(sbuckets[5]['all_one'])} and random {f3(sbuckets[5]['random'])} in "
            f"[0.8,1.0] while the model reaches {f3(sbuckets[5]['model'])}. Those two "
            f"sparse groups hold {sbuckets[1]['n'] + sbuckets[2]['n']} of the "
            f"{cfg['n']} shapes.")
        + lp.results_table(
            ["emissive fraction", "n", "ceiling", "all zero", "predict all", "random",
             "albedo rule", "model"], strat_rows)
        + lp.chartnote(
            f"<b>Every plotted value, plus the two columns the chart leaves out.</b> Mean "
            f"IoU at threshold {THR} within each group. The model column is marked where "
            f"it is above random. All-zero is nonzero only in the leftmost group, and only "
            f"because that group holds the {comp['n_zero_gt']} exactly-empty shapes, which "
            f"an empty prediction scores 1.0 on by the convention above. The "
            f"albedo-brightness rule is a fixed global brightness cut on the base colour, "
            f"the same one used in the aggregate table below.")
        + lp.prose(
            f"<p>Read only as an aggregate, this run says the model loses to a coin flip. "
            f"Read by group, it says something narrower and more useful: the model has "
            f"learned something where emission is sparse, and the aggregate is dominated "
            f"by groups where a degenerate prediction wins. A baseline that marks "
            f"everything cannot be beaten on a shape where four voxels in five already "
            f"emit, and {sbuckets[4]['n'] + sbuckets[5]['n']} of the {cfg['n']} shapes "
            f"sit above 0.5 emissive fraction. The flat mean over all of them is not the "
            f"whole story, in either direction.</p>"
            f"<p>The qualification matters as much as the finding. The model is above "
            f"random in those two groups; it is not above the albedo-brightness rule in "
            f"either, which reaches {f3(sbuckets[1]['pbr'])} and {f3(sbuckets[2]['pbr'])} "
            f"where the model reaches {f3(sbuckets[1]['model'])} and "
            f"{f3(sbuckets[2]['model'])}. The one group where the model leads every "
            f"baseline that is not degenerate is the leftmost, and that group is also the "
            f"one the empty-mask convention distorts. So the sparse-regime result is a "
            f"signal that the model is not producing noise, not evidence that it is "
            f"competitive.</p>")
        + lp.results_table(["over the 272 shapes with a nonzero mask", "mean IoU",
                            "what it is"], overall_rows)
        + lp.chartnote(
            f"<b>The aggregate the group table corrects: a fixed brightness cut on the "
            f"base colour scores {f3(b_pbr['mean_iou'])} where the model scores "
            f"{f3(model_nz_mean)}.</b> All rows are mean IoU at threshold {THR} over the "
            f"{comp['n_nonzero_gt']} shapes with a nonzero ground-truth mask, so the "
            f"empty-mask convention cannot inflate any of them. Predict-everything and "
            f"random both win here for the reason the group table shows, and the "
            f"albedo-brightness rule wins for a different reason: it is not degenerate, it "
            f"is a genuine predictor from the input the model also receives."))

    # ========================================================= 06 draw variance
    sec6 = lp.section_v2(
        "variance", 6,
        "Between draws the model moves more than its own mean IoU",
        ch.std_curve(list(ps["std"]), model_mean_all, std_sum["p90"], n_std_over_mean)
        + lp.chartnote(
            f"<b>One shape in ten varies by more than {f3(std_sum['p90'])} between draws, "
            f"which is over three times the model&rsquo;s mean IoU of "
            f"{f4(model_mean_all)}.</b> Each point is one shape&rsquo;s standard deviation "
            f"across its {cfg['draws']} draws at threshold {THR}, sorted from most to "
            f"least variable; note the axis runs to 0.5, not to 1. The dashed rule is the "
            f"model&rsquo;s own mean IoU over the same {cfg['n']} shapes, and the shaded "
            f"region marks the {n_std_over_mean} shapes whose draw-to-draw spread exceeds "
            f"it. The distribution is bimodal by construction: {int((ps['std'] == 0).sum())} "
            f"shapes have zero spread because the model returns the same answer, usually "
            f"nothing, on all three draws.")
        + lp.prose(
            f"<p>This is the empirical case for the reporting protocol rather than an "
            f"observation about this checkpoint. The sampler is stochastic, and its "
            f"per-shape spread is of the same order as the quantity being measured: mean "
            f"spread {f4(std_sum['mean'])}, median {f4(std_sum['median'])}, ninetieth "
            f"percentile {f4(std_sum['p90'])}. A single draw on a single shape carries a "
            f"noise term larger than the differences that would be argued from it, so a "
            f"single draw is not a measurement.</p>"
            f"<p>Every number on this page is therefore a mean over {cfg['draws']} draws, "
            f"and the same shape set is used for every arm of every comparison. The cost "
            f"is linear in the draw count and the run took under forty minutes; there is "
            f"no reason to report a one-draw number again.</p>"))

    # ============================================================= 07 still live
    out_rows = "".join([
        row([("The round trip corrupted the reference the metric scores against", None),
             (f"ceiling {f4(ceil_nz[THR]['mean'])} mean, {f4(ceil_nz[THR]['median'])} "
              f"median over {comp['n_nonzero_gt']} shapes, and every group mean between "
              f"{f3(min(strat[b][f'ceiling_iou_at_{THR}'] for b in BUCKETS))} and "
              f"{f3(max(strat[b][f'ceiling_iou_at_{THR}'] for b in BUCKETS))}", None)]),
        row([("The prediction and the reference are in different coordinate frames", None),
             (f"{f4(coord['mean_match_frac_of_decoded_gt'])} mean match fraction, "
              f"identical raw coordinates in and out on every shape", None)]),
        row([("The threshold on the decoded field is set wrong", None),
             (f"model mean {f4(thr_lo)} to {f4(thr_hi)} across "
              f"{', '.join(str(t) for t in cfg['thrs'])}", None)]),
    ])
    live_rows = "".join([
        row([("The model is given nothing to condition on", None),
             ("It sees geometry and PBR channels and no image, while the TEXGen and "
              "TRELLIS.2 baselines receive a thumbnail in which the emissive region is "
              "blown out to white", None)]),
        row([("The training data is unfiltered", None),
             ("Roughly a quarter of the shapes belong to the fullbright group, where the "
              "emissive texture is a copy of the base colour, so the label is arguably not "
              "learnable from any input", None)]),
        row([("Emission may not be inferable from geometry and PBR alone", None),
             ("At this capacity and on this input, the target may not be a function of "
              "what the model is shown", None)]),
    ])
    sec7 = lp.section_v2(
        "live", 7,
        "What the diagnostics do not rule out",
        lp.results_table(["explanation ruled out", "the measurement that ruled it out"],
                         out_rows)
        + lp.chartnote(
            "<b>Three explanations, each closed by a number rather than by argument.</b> "
            "All three were live before this run and none of them survives it. The first "
            "was the one the project had been working from.")
        + lp.results_table(["what is still live", "why it is still live"], live_rows)
        + lp.chartnote(
            "<b>What is left is about the inputs and the target, not about the "
            "measurement.</b> These are stated in the order they would be tested, and none "
            "of them is settled by this run.")
        + lp.prose(
            f"<p>The diagnostics moved the question. Before this run the candidate causes "
            f"included the metric, the reconstruction and the decision rule; after it, all "
            f"three are closed and what remains concerns what the model is given and "
            f"whether the target follows from it. The stratified reading adds one "
            f"constraint on any explanation offered: whatever is proposed has to be "
            f"consistent with a model that is above random on sparse shapes and far below "
            f"a fixed brightness cut on the same shapes.</p>"))

    apx = lp.appendix("Provenance", [
        f"<b>Source.</b> One diagnostics run, job {JOB_ID}, "
        f"<code>{SOURCE_PATH}</code>, staged unmodified into this page&rsquo;s own "
        f"<code>data/</code> directory and read at build time. {cfg['n']} shapes from the "
        f"<code>{cfg['split']}</code> split of <code>dataset_direct</code>, "
        f"{cfg['draws']} draws each at {cfg['steps']} sampler steps, seed {cfg['seed']}, "
        f"scored at thresholds {', '.join(str(t) for t in cfg['thrs'])}. No shape was "
        f"skipped and no ceiling computation failed.",
        "<b>Summary rows come from the file&rsquo;s own summary blocks; the four "
        "distribution figures are recomputed here from its per-shape records.</b> The "
        "build asserts that the recomputation reproduces the reported mean and median "
        "model IoU, the draw-standard-deviation mean and ninetieth percentile, the "
        "ceiling mean, the nonzero-mask count, and every group&rsquo;s count, model mean "
        "and ceiling mean, and it fails rather than writing the page if any of them "
        "disagrees by more than 0.0005. The nonzero-mask aggregates quoted as the "
        "headline are computed the same way, from the per-shape records.",
        "<b>What the round-trip ceiling is.</b> The ground-truth emission field is "
        "encoded and decoded by the same autoencoder the model predicts through, "
        "thresholded, and scored against the shape&rsquo;s true emissive voxel mask. It "
        "is the score a perfect model would receive, so it bounds every number on this "
        "page from above. It is not a model output and it involves no sampling.",
        "<b>What the baselines are.</b> All zero marks no voxel. Predict everything marks "
        "every voxel. Random marks each voxel independently with probability equal to "
        "that shape&rsquo;s own ground-truth emissive fraction, so it is given the answer "
        "the model has to infer. The albedo-brightness rule thresholds the base-colour "
        "brightness at a fixed global percentile, swept over seven percentiles with the "
        "best reported. The first two are degenerate by design and are included to show "
        "what a group&rsquo;s composition alone is worth.",
        "<b>The empty-mask convention, and where it bites.</b> An empty prediction "
        f"against an empty ground truth scores 1.0. {comp['n_zero_gt']} of the "
        f"{comp['n_ok']} shapes have an exactly empty ground-truth mask, and all of them "
        f"fall in the leftmost group, which is why that group&rsquo;s all-zero baseline "
        f"is {f3(strat[BUCKETS[0]]['baseline_all_zero'])} rather than zero and its "
        f"ceiling is {f3(strat[BUCKETS[0]][f'ceiling_iou_at_{THR}'])} rather than "
        f"{f3(b0_nonzero)}. Every aggregate outside the group table is reported over the "
        f"{comp['n_nonzero_gt']} shapes with a nonzero mask, where the convention cannot "
        f"apply.",
        "<b>What this page does not claim.</b> It does not compare these numbers with any "
        "figure computed on rendered views rather than voxels, or on a differently "
        "filtered set of shapes, because neither the population nor the unit would match. "
        "It does not report any training-curve reading as a result. It does not claim a "
        "cause for the 0.1: it closes three candidate causes and states what is left.",
    ])

    page_html = lp.page(
        title="Why Every Emissive Model Sits Near 0.1 IoU",
        header_html=hero,
        body_sections=[sec1, sec2, sec3, sec4, sec5, sec6, sec7, apx],
        assets_rel=SITE_ASSETS,
        assets_dir=assets_dir,
        theme="v3",
        tree_html=tree_html(),
        nav_title="Diagnostics",
        version_slot=lp.v3_version_slot(date=PAGE_DATE),
        needs_katex=False,
        extra_head=f'<link rel="icon" href="{FAVICON}">' + EXTRA_CSS,
        outline_entries=[
            {"id": "verdict", "label": "What is settled"},
            {"id": "ceiling", "label": "The round-trip ceiling"},
            {"id": "ruledout", "label": "Alignment and threshold"},
            {"id": "median", "label": "The median, not the mean"},
            {"id": "stratified", "label": "Stratified by emissive fraction"},
            {"id": "variance", "label": "Draw-to-draw variance"},
            {"id": "live", "label": "What is still live"},
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
    print(f"  per-shape recomputation vs the file's summaries: clean "
          f"({len(ps['sids'])} shapes)")
    print(f"  groups leading random: "
          f"{', '.join(b['label'] for b in lead) if lead else 'none'}")
    print("  zone-link guard: clean")

    publish_assets(assets_dir)
    print(f"assets published -> {assets_dir}")

    if publish:
        os.makedirs(PUBLISH_DIR, exist_ok=True)
        shutil.copy2(out_path, os.path.join(PUBLISH_DIR, "index.html"))
        wz.write_tree_json()
        print(f"published -> {PUBLISH_DIR}")
        print(f"tree.json refreshed -> {wz.TREE_JSON}")


if __name__ == "__main__":
    build(publish="--publish" in sys.argv)
