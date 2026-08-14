#!/usr/bin/env python3
"""Build the SegviGen emissive-segmentation experiment overview (xgpage v2
editorial): the method, the data, the state, and what is open, in one page.

Every number on this page comes from
segvigen_emissive/FACTSHEET_experiment_overview.md (assembled and verified by
the master session 2026-07-30/31) or from
web/_preview/data_compare/data/aggregate_stats.json (the same verified source
the data_compare page draws on). Nothing here is invented, re-derived, or
extrapolated; where a number was not available, the page says so.

Run: .venv2/bin/python web/_preview/experiment_overview/build.py
  (.venv2 = /cs/3dlg-project/3dlg-hcvc/omages/omages_internal/.venv2/bin/python)
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import diagrams as D

import xgpage as lp
from xgpage.publish import publish_assets

WEB = "/local-scratch2/xya120/studio/misc/lightgen/web"
OUT_DIR = HERE
SITE_ASSETS = "/projects/omages/yanxg/lightgen/assets"


def code(text):
    return f'<code>{text}</code>'


def row(*cells):
    return '<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'


def img(name):
    return f"img/{name}"


def pct(x, n=1):
    return f"{x * 100:.{n}f}%"


def load_stats():
    stats = json.load(open(os.path.join(WEB, "_preview", "data_compare", "data", "aggregate_stats.json")))
    gallery = json.load(open(os.path.join(HERE, "gallery_sids.json")))
    return stats, gallery


def build():
    assets_dir = os.path.join(WEB, "assets")
    stats, gallery = load_stats()
    occ = stats["occ_ratio"]
    gap = stats["gap_frac"]
    geo = stats["geometry_agreement"]

    # ================================================================ hero
    hero = lp.hero_header(
        "lightgen SegviGen &middot; emissive segmentation &middot; experiment overview &middot; 2026-07-31",
        "Fine-Tuning SegviGen to Predict Which Voxels Emit Light",
        dek_html=(
            "The current experiment fine-tunes TRELLIS.2's sparse DiT, warm-started from "
            "SegviGen's <code>full_seg.ckpt</code>, to predict a per-voxel binary emissive "
            "mask directly on Dongchen's re-extracted 512&sup3; occupancy grid, with no "
            "somage/GLB round-trip. This page replaces thirteen pages of three different "
            "vintages and a month-stale WORKLOG with one current picture: the method, the "
            "data as actually built, what has been validated, and what is still open."
        ),
        stats=[
            ("72,546", "shapes built, of 74,503 in the split"),
            ("32,050", "survivors if the emissive filter is applied (not yet)"),
            ("~20 h", "measured cost per epoch, 57,968 train shapes"),
            ("512³", "our locked grid, the pretrained encoder's native resolution"),
        ],
        toc=[
            ("predicting", "What is being predicted"),
            ("ladder", "The resolution ladder"),
            ("hijack", "The channel hijack"),
            ("built", "The data, as actually built"),
            ("validated", "Is the data correct?"),
            ("contains", "What the data contains"),
            ("filter", "The filter we are not applying"),
            ("training", "Training setup and measured cost"),
            ("baselines", "Where this sits against the baselines"),
            ("superseded", "Prior results are superseded"),
            ("open", "Open decisions"),
        ],
    )

    sections = []

    # ============================================================ 01 predicting
    fig_channels = D.svg_figure(
        *D.diagram_three_channels(),
        caption_html=(
            "<b>Three token-aligned latent channels; only one is predicted.</b> "
            "<code>shape_slat</code> and <code>input_tex_slat</code> are given to the model "
            "as conditioning (geometry and reflectance); <code>output_tex_slat</code>, the "
            "binarized emission mask, is what the model is trained to produce. All three "
            "share identical coordinates, which is what lets the model read one geometry "
            "token and one appearance token and predict the corresponding emission token at "
            "the same voxel."
        ),
        width_px=980,
        id="fig-channels",
    )
    sec1 = lp.section_v2(
        "predicting", "01", "The model predicts a per-voxel binary emissive mask, not an emission color",
        lp.prose(
            "<p>For every occupied voxel on a shape's surface, the question is binary: does "
            "it emit light, or not. The experiment fine-tunes TRELLIS.2's 1.3B sparse DiT "
            f"({code('slat_flow_imgshape2tex_dit_1_3B_512_bf16')}), warm-started from "
            f"SegviGen's {code('full_seg.ckpt')}, to answer that question over three latent "
            "channels:</p>"
        ) + fig_channels,
    )
    sections.append(sec1)

    # ============================================================ 02 ladder
    fig_ladder = D.svg_figure(
        *D.diagram_resolution_ladder(),
        caption_html=(
            "<b>The three grids nest exactly: 512 / 256 = 2&times; per axis, "
            "512 / 32 = 16&times; per axis.</b> One latent token (the accent outer square) "
            "spans a 16&sup3; block of 512&sup3; cells, 4,096 cells, the same footprint as "
            "an 8&sup3; block of 256&sup3; cells, 512 distinct baked attribute values. A 2D "
            "cross-section is shown for legibility; the same factors apply along all three "
            "axes."
        ),
        width_px=820,
        id="fig-ladder",
    )
    sec2 = lp.section_v2(
        "ladder", "02", "Three grid resolutions are live at once, and confusing them is the most common mistake on this project",
        lp.prose(
            "<p>256&sup3; is Dongchen's attribute bake (<code>pbr_voxels_256</code>, "
            "<code>emission_voxels_256</code>). 512&sup3; is our grid, locked because the "
            "pretrained TRELLIS.2 encoder's contract is 512. 32&sup3; is the latent. Attribute "
            "values are upsampled 256&nbsp;&rarr;&nbsp;512 by parent lookup "
            f"({code('coords512 // 2')}) with a cKDTree fallback for cells with no exact "
            "parent, never re-baked at 512&sup3;.</p>"
        ) + fig_ladder,
    )
    sections.append(sec2)

    # ============================================================ 03 hijack
    fig_hijack = D.svg_figure(
        *D.diagram_channel_hijack(),
        caption_html=(
            "<b>The prediction target is written into the base_color slot, reusing the "
            "pretrained PBR encoder byte-identically.</b> Emission is binarized at "
            "&gt;1/255 (any nonzero emission counts) and written where base_color normally "
            "lives; metallic, roughness, and alpha are fixed module constants "
            f"({code('OUT_METALLIC_U8=0')}, {code('OUT_ROUGHNESS_U8=255')}, "
            f"{code('OUT_ALPHA_U8=255')}), not derived per voxel. No architecture change, "
            "no new encoder was trained."
        ),
        width_px=940,
        id="fig-hijack",
    )
    sec3 = lp.section_v2(
        "hijack", "03", "No new encoder was trained: emission rides in a slot built for reflectance",
        lp.prose(
            "<p>Because the pretrained encoder expects four PBR channels per voxel "
            "(base_color, metallic, roughness, alpha), the cheapest way to encode a new "
            "binary target with the same encoder is to make it look like one of those four "
            "channels. The build script does exactly that:</p>"
        ) + fig_hijack,
    )
    sections.append(sec3)

    # ============================================================ 04 built
    fig_funnel = D.svg_figure(
        *D.diagram_data_funnel(),
        caption_html=(
            "<b>72,546 of 74,503 split shapes are built; the 1,957 missing decompose "
            "exactly into three causes, only one of which is still actionable.</b> 1,036 "
            "never had a source to build from (permanent). 584 are on Dongchen's rebake "
            "list; his rebake job completed but produced no output for them, a question for "
            "him, not a bug in this pipeline. 337 are buildable right now (source present) "
            "but recommended skip: finishing them moves the built count by +0.6%, not "
            "enough to change any result."
        ),
        width_px=980,
        id="fig-funnel",
    )
    sec4 = lp.section_v2(
        "built", "04", "72,546 of 74,503 split shapes are built; two different missing counts are both correct",
        lp.prose(
            "<p>The full split (<code>data_splits_74k.json</code>, indices into the "
            "emissive_thumbnails parquet): train 59,602, val 7,450, test 7,451, total "
            "74,503. Built and on disk: train 57,968, val 7,290, test 7,288, total 72,546.</p>"
        ) + fig_funnel + lp.callout(
            "&ldquo;1,957 missing&rdquo; and &ldquo;~921 missing&rdquo; are both correct, "
            "under different denominators: 1,957 against the full 74,503 split, ~921 "
            "against the 73,467 shapes expected to be buildable after excluding the 1,036 "
            "with no source at all. Neither number is wrong; they answer different "
            "questions.",
            title="Two missing counts, two denominators",
        ) + lp.prose(
            "<p>Build failures logged: 1,791 <code>FileNotFoundError</code> + 70 "
            "<code>CUDA OutOfMemoryError</code>. Of the 1,791: 1,032 are on the permanent "
            "list, 759 on the rebake list, 0 unexplained. Of the 759 rebake-list failures, "
            "only 177 had their source reappear after Dongchen's jobs cleared. One shard "
            "(211 of 375) was killed by host OOM and wrote no manifest, though 103 of its "
            "200 shapes landed on disk anyway; 374 of 375 manifests exist.</p>"
        ),
    )
    sections.append(sec4)

    # ============================================================ 05 validated
    occ_rows = [
        {"label": "p10", "value": occ["p10"], "display": f'{occ["p10"]:.2f}&times;'},
        {"label": "median", "value": occ["median"], "display": f'{occ["median"]:.2f}&times;'},
        {"label": "mean", "value": occ["mean"], "display": f'{occ["mean"]:.2f}&times;'},
        {"label": "p90", "value": occ["p90"], "display": f'{occ["p90"]:.2f}&times;'},
    ]
    geo_rows = [
        {"label": "mean", "value": geo["mean_dist_in_256voxels"], "display": f'{geo["mean_dist_in_256voxels"]:.3f}'},
        {"label": "p90", "value": geo["p90_dist_in_256voxels"], "display": f'{geo["p90_dist_in_256voxels"]:.2f}'},
        {"label": "max", "value": geo["max_dist_in_256voxels"], "display": f'{geo["max_dist_in_256voxels"]:.2f}'},
    ]
    sec5 = lp.section_v2(
        "validated", "05", "Two independently-computed geometries agree to 0.12 voxel widths, and occupancy scales as a surface, not a volume",
        lp.prose(
            f"<p>Our 512&sup3; build was checked against Dongchen's 256&sup3; bake over "
            f"{geo['n_shapes']} shapes, {geo['n_vertex_pairs_compared_total']:,} dual-vertex "
            "pairs (our 512&sup3; dual vertices, averaged down, vs his "
            f"{code('dual_grid_256')}):</p>"
        ) + lp.hbar_chart(geo_rows, title="dual-vertex disagreement, in 256³ voxel widths",
                          note=(
                              f"<b>Mean disagreement is {geo['mean_dist_in_256voxels']:.3f} voxel "
                              f"widths, with no systematic offset.</b> The mean signed per-axis "
                              "offset is "
                              f"[{geo['mean_signed_offset_per_axis_256voxels'][0]:+.4f}, "
                              f"{geo['mean_signed_offset_per_axis_256voxels'][1]:+.4f}, "
                              f"{geo['mean_signed_offset_per_axis_256voxels'][2]:+.4f}] voxel "
                              "widths, near zero on every axis, so the disagreement is "
                              "noise, not a frame or offset bug between the two extractions."
                          )) + lp.prose(
            "<p>Occupied-voxel count, our 512&sup3; grid vs the 256&sup3; bake:</p>"
        ) + lp.hbar_chart(occ_rows, title="occupied-voxel ratio (512/256)",
                          note=(
                              f"<b>The ratio clusters at {occ['median']:.2f}&times; median, "
                              "close to the 4&times; a surface predicts.</b> Occupied cells form "
                              "a 2D shell: halving voxel width doubles resolution per axis, so a "
                              "surface's occupied count should scale ~2&sup2;=4&times;, not the "
                              "2&sup3;=8&times; a solid volume would give. The observed median "
                              f"({occ['median']:.2f}&times;) matches that surface prediction, not "
                              "the volume one."
                          )) + lp.callout(
            f"{pct(gap['frac_zero'])} of shapes have <code>gap_frac</code> exactly zero: the "
            "512&sup3; dual-grid occupancy and the 256&sup3; attribute-bake occupancy nest "
            f"exactly for the large majority of shapes. Median gap_frac is "
            f"{gap['median']:.6f}. But there is a genuine long tail: {gap['n_over_10pct']} of "
            f"17,894 shapes exceed 10% gap, and one small shape reaches 100% (every one of its "
            "voxels needed the nearest-occupied fallback). This is consistent with a small-shape "
            "edge case, not a systematic pipeline bug.",
            title="The two occupancy sets nest almost exactly, with a real small-shape tail",
        ),
    )
    sections.append(sec5)

    # ============================================================ 06 contains
    ef_rows = [
        {"label": "&gt; 0", "value": 90.8, "display": "90.8%"},
        {"label": "&gt; 0.001", "value": 79.7, "display": "79.7%"},
        {"label": "&gt; 0.01", "value": 59.3, "display": "59.3%"},
        {"label": "&gt; 0.1", "value": 36.9, "display": "36.9%"},
        {"label": "&gt; 0.5", "value": 22.9, "display": "22.9%"},
    ]
    gallery_panels = []
    for g in gallery:
        sid = g["sid"]
        ef = g["emissive_frac_512"]
        gallery_panels.append((f'ef={ef:.2f}', img(f'{sid}_geom512.png')))
    gallery_panels_emis = []
    for g in gallery:
        sid = g["sid"]
        ef = g["emissive_frac_512"]
        gallery_panels_emis.append((f'ef={ef:.2f}', img(f'{sid}_emis512.png')))

    sec6 = lp.section_v2(
        "contains", "06", "emissive_frac is bimodal: most shapes have a small emissive region, and a large minority is almost entirely emissive",
        lp.prose(
            f"<p>Sampled n=1,998 random train shapes, reading each shape's "
            f"{code('meta.json')} {code('emissive_frac')} (fraction of a shape's occupied "
            "voxels that are emissive):</p>"
        ) + lp.hbar_chart(ef_rows, title="% of shapes with emissive_frac above threshold",
                          note=(
                              "<b>Median 0.025, mean 0.244: the mean-vs-median gap IS the "
                              "finding.</b> Most shapes carry a small emissive region (a lamp's "
                              "bulb, a sign's face), which is why the median is low; but a large "
                              "minority is almost entirely emissive, largely &ldquo;fullbright&rdquo; "
                              "content where the emissive texture equals the base color, which "
                              "pulls the mean far above the median. Reading this as a smooth "
                              "distribution would be wrong: it is two populations, not one."
                          )) + lp.prose(
            "<p>The gallery below spans that range directly: 12 shapes at "
            "emissive_frac 0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.97, "
            "and 1.0, rendered geometry (base_color) and emission mask side by side at "
            "512&sup3;. White in the emission render marks emissive voxels; a small-region "
            "shape (0.02, second panel) and a fullbright shape (1.0, last panel) sit at "
            "opposite ends of the same gallery, not two hand-picked extremes.</p>"
        ) + lp.fig_grid(gallery_panels, cols=4,
                         caption_html=(
                             "<b>Geometry (base_color) renders across the full emissive_frac range, "
                             "0.0 to 1.0.</b> Twelve shapes, evenly spaced by emissive_frac, all "
                             "rendered at 512&sup3;. Panel labels give each shape's exact "
                             "emissive_frac."
                         ),
                         native_px=460, content='photo') + lp.fig_grid(
            gallery_panels_emis, cols=4,
            caption_html=(
                "<b>The matching emission-mask renders for the same 12 shapes, same order.</b> "
                "White voxels are emissive (&gt;1/255); dark gray is non-emissive occupied "
                "surface. At ef=0.0 the mask render is dark throughout; by ef=1.0 it is "
                "uniformly white. Compare against the geometry row above panel for panel."
            ),
            native_px=460, content='photo',
        ),
    )
    sections.append(sec6)

    # ============================================================ 07 filter
    sec7 = lp.section_v2(
        "filter", "07", "A filter exists and is deliberately not applied: this is a baseline-first choice, not an oversight",
        lp.prose(
            f"<p>{code('vis_data/emissive_filtering/stage1_survivors.txt')} holds 32,121 "
            "sids. Intersected with what is built: train 25,547, val 3,262, test 3,241, "
            "total 32,050 survivors built. That filter is real, computed, and ready to use, "
            "and the current run does not use it.</p>"
        ) + lp.callout(
            "The owner has decided to train on the unfiltered set first: a baseline over "
            "everything, with filtering held as a planned later ablation. This is a "
            "deliberate ordering choice, not a recommendation being ignored.",
            title="Unfiltered first, filtered later",
        ),
    )
    sections.append(sec7)

    # ============================================================ 08 training
    cfg_rows = "".join([
        row("model", code("slat_flow_imgshape2tex_dit_1_3B_512_bf16") + " (TRELLIS.2-4B)"),
        row("warm start", code("full_seg.ckpt") + " (SegviGen)"),
        row("gradient checkpointing", "enabled on 31 modules (fits the 46GB L40S)"),
        row("conditioning", code("--cond zero") + " (no image conditioning, see below)"),
        row("learning rate", code("--lr 1e-5") + ", " + code("--lr_schedule const")),
        row("loss weighting", code("--pos_weight 5.0")),
        row("EMA", code("--ema 0.999")),
        row("token selection", code("--select_on nonzero")),
        row('<code style="white-space:nowrap">--emis_oversample</code>', "OFF, deliberately: on unfiltered data it would "
            "preferentially sample the fullbright group (emissive_frac near 1.0), "
            "amplifying the noise. Every prior run had it ON, so this breaks comparability "
            "with the pilots."),
    ])
    sec8a = lp.section_v2(
        "training", "08", "Every prior pilot ran with oversampling on; this run turns it off, breaking direct comparability",
        lp.prose("<p>Configuration as launched / to launch:</p>") +
        lp.results_table(["parameter", "value / note"], cfg_rows) +
        lp.callout(
            "Zero of the 72,546 built shapes have cond.pth, and build_dataset_direct.py has "
            "no conditioning code path at all. DINOv3-L is now downloaded (1.2G, "
            "model.safetensors present), so this is no longer a licensing block; it is "
            "simply unbuilt.",
            title="Conditioning is unbuilt, not blocked",
        ) + lp.prose(
            "<p>Measured cost, from smoke test job 237741 (3 epochs &times; 24 shapes): "
            "~1.25 s/shape (epoch 2: 27 s / 24 shapes; epoch 3: 34 s / 24 shapes), plus "
            "~7.5 minutes of dataset init (~350k Lustre metadata operations: listdir + "
            "~5 stats + a meta.json open per shape, single-threaded). At that rate, one "
            "full epoch over the 57,968 built train shapes costs approximately "
            "<b>20 hours</b>.</p>"
        ) + lp.callout(
            "Loss went 0.292 &rarr; 0.225 across 3 epochs; warm-start, checkpointing, EMA, "
            "and quick-val all ran without error. Quick-val IoU was 0.0003 at n=2, 72 total "
            "training samples seen. This is a plumbing check only: do not compare it to "
            "0.259 or 0.203, and do not read it as a result.",
            title="Smoke test: pipeline verified functional, IoU not meaningful",
            warn=True,
        ),
    )
    sections.append(sec8a)

    # ============================================================ 09 baselines
    sec9 = lp.section_v2(
        "baselines", "09", "DiffusionNet's 0.259 IoU is real, but it is a per-face metric being compared to a per-voxel one",
        lp.prose(
            "<p>The paper's related-work baselines (sec/5_experiment.tex): TEXGen (works in "
            "UV space), TRELLIS.2 (3D latent; two variants, replace albedo with emission or "
            "replace all PBR with a single emission channel), and DiffusionNet (works on the "
            "surface).</p>"
        ) + lp.callout(
            "DiffusionNet val IoU 0.259 is real and reproducible on a clean split. But it is "
            "a PER-FACE metric, and SegviGen's is PER-VOXEL. These are incommensurable: "
            "eval_emissive.py already calls itself a &ldquo;proxy.&rdquo; The 0.259 number "
            "must not be read as a target SegviGen is chasing on equal terms. This is the "
            "single most important honesty point on this page.",
            title="Per-face and per-voxel IoU are not the same metric",
            warn=True,
        ),
    )
    sections.append(sec9)

    # ============================================================ 10 superseded
    sec10 = lp.section_v2(
        "superseded", "10", "Every published result page describes a pipeline this experiment no longer uses",
        lp.prose(
            "<p>Every published result page, <code>finetune_binary_v1</code>, "
            "<code>results_2k_v1</code>, <code>training_curves_v1</code>, "
            "<code>fullseg_canon10*</code>, <code>gt_vs_pred_canon10</code>, "
            "<code>official_repro</code>, is &ldquo;Path A&rdquo;: the somage/GLB "
            "round-trip pipeline that the direct-ovoxel design (this page) replaced. Their "
            "numbers do not describe the current experiment.</p>"
        ) + lp.results_table(["run", "result", "status"], "".join([
            row("best zero-cond pilot, 232 shapes", "val IoU 0.203 at epoch 25", "Path A, superseded"),
            row("v2, 232 shapes, oversample", "0.095 at ep10, 0.068 at ep20 (overfit)", "Path A, superseded"),
            row("best honest held-out, all Path A runs", "~0.15 and declining", "Path A, superseded"),
        ])),
    )
    sections.append(sec10)

    # ============================================================ 11 open
    open_items = [
        "Real run budget: <code>--n_per_epoch</code> (proposed 3,000, roughly 65 min/epoch) "
        "and <code>--epochs</code>; needs a non-debug partition.",
        "<code>--emis_oversample</code>: currently off; confirm or flip.",
        "Mop-up of the 337 buildable shapes: recommended skip (+0.6%, cannot move a result).",
        "The 584 shapes Dongchen's rebake did not produce output for: a question for him.",
    ]
    sec11 = lp.section_v2(
        "open", "11", "Four decisions are open and unresolved as of this page",
        lp.prose("<ul>" + "".join(f"<li>{it}</li>" for it in open_items) + "</ul>"),
    )
    sections.append(sec11)

    # ============================================================ appendix
    apx = lp.appendix("Provenance", [
        "Every number on this page is sourced from "
        "<code>segvigen_emissive/FACTSHEET_experiment_overview.md</code> (assembled and "
        "verified by the master session, 2026-07-30/31) or from "
        "<code>web/_preview/data_compare/data/aggregate_stats.json</code> (built by "
        "<code>aggregate_stats.py</code> from <code>manifest_sample.parquet</code> + "
        "<code>deepquant_400.jsonl</code>, produced by scripts run on solar against the "
        "real .vxz files).",
        "&sect;05 geometry-agreement and occupancy-ratio numbers: "
        f"{geo['n_shapes']} shapes, {geo['n_vertex_pairs_compared_total']:,} vertex pairs "
        "(geometry_agreement), and the full manifest scan (occ_ratio, gap_frac) in "
        "aggregate_stats.json, identical source to the data_compare page.",
        "&sect;06 gallery: 12 shapes from <code>gallery_sids.json</code>, chosen to span "
        "emissive_frac 0.0&ndash;1.0 in even steps; renders copied from "
        "<code>web/_preview/data_compare/img/</code> (48 PNGs, "
        "<code>&lt;sid&gt;_{geom,emis}{256,512}.png</code>).",
        "Dataset build: array 237094, <code>data_splits_74k.json</code>, "
        "<code>build_dataset_direct.py</code>.",
        "Smoke test: job 237741, 3 epochs &times; 24 shapes.",
        "The channel-hijack diagram (&sect;03) and the resolution-ladder diagram (&sect;02) "
        "are adapted from <code>web/_preview/data_pipeline/diagrams.py</code>, which is "
        "grounded in file:line citations against <code>build_dataset_direct.py</code> and "
        "<code>uv_voxel_pipeline/voxelize.py</code>; see that page for the citations.",
        "The mip bug affecting emissive textures &ge;512&times;512 pixels "
        "(<code>uv_voxel_pipeline/voxelize.py</code>) is orthogonal to grid resolution and "
        "is not restated on this page; see <code>data_pipeline</code> &sect;07 for the full "
        "correction.",
        "Prior published pages (&sect;10) are Path A (somage/GLB round-trip); this page "
        "describes the direct-ovoxel design that replaced it.",
        "Every figure on this page is a computed schematic (box/arrow/grid layout drawn "
        "from a coordinate table in code) or a real rendered PNG from the dataset; no "
        "hand-drawn geometry is used for any evidence claim.",
    ])

    page_html = lp.page(
        title="SegviGen Emissive Segmentation: Experiment Overview (lightgen)",
        header_html=hero,
        body_sections=sections + [apx],
        assets_rel=SITE_ASSETS,
        assets_dir=assets_dir,
        theme="v2",
        needs_katex=False,
    )
    out_path = os.path.join(OUT_DIR, "index.html")
    with open(out_path, "w") as f:
        f.write(page_html)
    print(f"wrote {out_path}")

    publish_assets(assets_dir)
    print(f"assets published -> {assets_dir}")


if __name__ == "__main__":
    build()
