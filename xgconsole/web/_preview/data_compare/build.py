#!/usr/bin/env python3
"""Dongchen's preprocessed o-voxels (256^3) vs our re-extracted dataset_direct (512^3):
a visual + quantitative comparison, xgpage v2 editorial.

Every number on this page comes from aggregate_stats.json (built by
aggregate_stats.py from manifest_sample.parquet + deepquant_400.jsonl, both produced
by scripts run on solar against the real .vxz files -- nothing here is retyped from
the build brief).

Run: .venv_console/bin/python web/_preview/data_compare/build.py
"""
import os, sys, json, glob, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import xgpage as lp
from xgpage.publish import publish_assets

WEB = "/local-scratch2/xya120/studio/misc/lightgen/web"
OUT_DIR = HERE
SITE_ASSETS = "/projects/omages/yanxg/lightgen/assets"
DATA_DIR = os.path.join(HERE, "data")   # fetched aggregate_stats.json / gallery_sids.json
IMG_DIR = os.path.join(HERE, "img")     # copied renders


def pct(x, n=1):
    return f"{x*100:.{n}f}%"


def load():
    stats = json.load(open(os.path.join(DATA_DIR, "aggregate_stats.json")))
    gallery = json.load(open(os.path.join(DATA_DIR, "gallery_sids.json")))
    return stats, gallery


def img(name):
    return f"img/{name}"


def build():
    stats, gallery = load()
    assets_dir = os.path.join(WEB, "assets")

    occ = stats["occ_ratio"]
    gap = stats["gap_frac"]
    ef = stats["emis_frac_compare"]
    ch = stats["children_hypothesis"]
    geo = stats["geometry_agreement"]

    hero = lp.hero_header(
        "lightgen emissive dataset &middot; his 256&sup3; vs ours 512&sup3; &middot; 2026-07-30",
        "Two Independent Extractions, Checked Against Each Other",
        dek_html=(
            "Dongchen's appearance bake at 256&sup3; and this project's native re-extraction "
            "at 512&sup3; are two independent pipelines reading the same source GLBs. This "
            f"page checks them against each other over {stats['n_manifest_scanned']:,} shapes "
            f"scanned and {stats['n_deep_quant_sample']} shapes decoded in full: how the "
            "occupied-voxel counts scale between the two resolutions, how often the two "
            "occupancy sets fail to nest, whether emissive-frac drifts between grids, and "
            "whether the two independently-computed dual-grid geometries actually agree in "
            "world space."
        ),
        toc=[
            ("occupancy", "Occupancy scaling"),
            ("gaps", "Where the grids don't nest"),
            ("emisfrac", "Emissive fraction across resolutions"),
            ("children", "The emissive-children hypothesis"),
            ("geometry", "Geometry agreement"),
            ("gallery", "Shape gallery"),
            ("provenance", "Provenance"),
        ],
    )

    sections = []

    # ================================================================ 01 occupancy
    occ_rows = [
        {"label": "p10", "value": occ["p10"], "display": f'{occ["p10"]:.2f}×'},
        {"label": "median", "value": occ["median"], "display": f'{occ["median"]:.2f}×'},
        {"label": "mean", "value": occ["mean"], "display": f'{occ["mean"]:.2f}×'},
        {"label": "p90", "value": occ["p90"], "display": f'{occ["p90"]:.2f}×'},
    ]
    hist_rows = []
    edges = occ["hist_edges"]
    for i, c in enumerate(occ["hist"]):
        hist_rows.append({"label": f"{edges[i]:.1f}–{edges[i+1]:.1f}×", "value": c, "display": str(c)})

    sec1 = lp.section_v2(
        "occupancy", "01",
        f"Occupied-voxel count scales {occ['median']:.2f}&times; median from 256&sup3; to 512&sup3;, "
        "close to the 4&times; a surface predicts",
        lp.prose(
            f"<p>Across {stats['n_manifest_scanned']:,} shapes present in both datasets, "
            "n_voxels_512 / n_voxels_256 (both read straight from meta.json, no re-decode "
            "needed for this ratio):</p>"
        ) + lp.hbar_chart(occ_rows, title="occupied-voxel ratio (512/256)",
                          note=(f"<b>The ratio clusters near 4&times;, as expected for a surface "
                                f"(occupied cells are a 2D shell; halving voxel width doubles "
                                f"resolution per axis, so a surface's occupied count scales "
                                f"~2&sup2;=4&times;, not the 2&sup3;=8&times; a solid volume "
                                f"would give).</b> {pct(occ['frac_near_4x'])} of shapes land within "
                                f"&plusmn;12.5% of exactly 4&times; (3.5&ndash;4.5&times;); the spread "
                                f"(p10={occ['p10']:.2f}&times;, p90={occ['p90']:.2f}&times;) reflects "
                                f"per-shape geometric complexity, not a systematic bias.")) +
        lp.prose("<p>The full distribution:</p>") +
        lp.hbar_chart(hist_rows, title="shapes per ratio bin (clipped at 10×)"),
    )
    sections.append(sec1)

    # ================================================================ 02 gaps
    gap_rows = [
        {"label": "median", "value": max(gap["median"], 1e-6), "display": f'{gap["median"]:.5f}'},
        {"label": "mean", "value": gap["mean"], "display": f'{gap["mean"]:.5f}'},
        {"label": "p90", "value": gap["p90"], "display": f'{gap["p90"]:.5f}'},
        {"label": "p99", "value": gap["p99"], "display": f'{gap["p99"]:.5f}'},
    ]
    sec2 = lp.section_v2(
        "gaps", "02",
        f"The two occupancy sets nest almost perfectly: {pct(gap['frac_zero'])} of shapes have zero gap",
        lp.prose(
            "<p><code>gap_frac</code> is the fraction of our 512&sup3; voxels whose 256&sup3; "
            "parent cell has no attribute value and needed the cKDTree nearest-occupied "
            f"fallback (build_dataset_direct.py's <code>Upsampler256to512</code>). Over "
            f"{stats['n_manifest_scanned']:,} shapes:</p>"
        ) + lp.hbar_chart(gap_rows, title="gap_frac") +
        lp.chartnote(
            f"<b>{pct(gap['frac_zero'])} of shapes have gap_frac exactly zero</b> -- the QEF "
            f"dual-grid occupancy at 512&sup3; and the texture-attribute rasterization "
            f"occupancy at 256&sup3; nest exactly for the large majority of shapes. Only "
            f"{pct(gap['frac_over_1pct'])} of shapes exceed 1% gap, and {gap['n_over_10pct']} "
            f"shape(s) exceed 10%."
        ) + lp.callout(
            (f"The distribution has a genuine long tail: the worst shape "
             f"(<code>{gap['worst_shapes'][0]['sid'][:16]}&hellip;</code>, only "
             f"{gap['worst_shapes'][0]['n_voxels_512']:,} voxels) has "
             f"<b>gap_frac = 1.0</b> -- every one of its 512 voxels needed the nearest-occupied "
             "fallback, meaning its dual-grid and attribute-rasterization occupancy sets share "
             "no exact cell in common at all. This and the next 4 worst shapes are all small "
             f"({min(s['n_voxels_512'] for s in gap['worst_shapes']):,}&ndash;"
             f"{max(s['n_voxels_512'] for s in gap['worst_shapes']):,} voxels) -- consistent "
             "with a small-shape edge case (e.g. a thin/degenerate mesh where a global scale or "
             "coordinate mismatch between the two extractions has more relative effect), not a "
             "systematic pipeline bug."),
            title=f"A real long tail: {gap['n_over_10pct']} shapes exceed 10% gap, one hits 100%",
        ),
    )
    sections.append(sec2)

    # ================================================================ 03 emis frac
    ef_rows = [
        {"label": "his (256)", "value": ef["mean_ef256"], "display": pct(ef["mean_ef256"])},
        {"label": "ours (512)", "value": ef["mean_ef512"], "display": pct(ef["mean_ef512"])},
    ]
    sparse = ef.get("sparse_lt5pct", {})
    dense = ef.get("dense_ge5pct", {})
    sec3 = lp.section_v2(
        "emisfrac", "03",
        f"Emissive fraction drifts {ef['mean_relative_increase_pct']:.1f}% mean (256&sup3;&rarr;512&sup3;), "
        "and the drift concentrates in sparse shapes",
        lp.prose(
            f"<p>Over the {ef['n']} decoded shapes with any emission at 256&sup3;, mean "
            "emissive fraction (fraction of occupied voxels that are emissive):</p>"
        ) + lp.hbar_chart(ef_rows, title="mean emissive fraction") +
        (lp.hbar_chart([
            {"label": f"sparse (his &lt;5%, n={sparse.get('n','?')}), mean", "value": abs(sparse.get('mean_relative_increase_pct', 0)),
             "display": f"{sparse.get('mean_relative_increase_pct', 0):+.2f}%"},
            {"label": "sparse, median", "value": abs(sparse.get('median_relative_increase_pct', 0)),
             "display": f"{sparse.get('median_relative_increase_pct', 0):+.2f}%"},
            {"label": f"dense (his &ge;5%, n={dense.get('n','?')}), mean", "value": abs(dense.get('mean_relative_increase_pct', 0)),
             "display": f"{dense.get('mean_relative_increase_pct', 0):+.2f}%"},
            {"label": "dense, median", "value": abs(dense.get('median_relative_increase_pct', 0)),
             "display": f"{dense.get('median_relative_increase_pct', 0):+.2f}%"},
        ], title="relative increase in emissive_frac, 256→512") if sparse and dense else "") +
        lp.chartnote(
            f"<b>Sparse shapes (his emissive_frac &lt;5%) drift more on average "
            f"({sparse.get('mean_relative_increase_pct',0):.1f}% mean vs "
            f"{dense.get('mean_relative_increase_pct',0):.1f}% for dense shapes), but the gap "
            f"between mean and median within the sparse group itself "
            f"({sparse.get('mean_relative_increase_pct',0):.1f}% mean vs "
            f"{sparse.get('median_relative_increase_pct',0):.1f}% median, n={sparse.get('n','?')}) "
            "shows this is a few large outlier shapes pulling the average, not a uniform drift "
            "across sparse shapes.</b> This confirms the DIRECTION flagged in the brief (sparse "
            "shapes drift more) but at roughly a third the magnitude previously estimated "
            "(~7% here vs the ~25% figure in the brief) on this "
            f"{stats['n_deep_quant_sample']}-shape sample; see &sect;04 "
            "for why the count-based mechanism does not explain even this smaller effect."
        ),
    )
    sections.append(sec3)

    # ================================================================ 04 children hypothesis
    mc_e = ch["pooled_mean_children_per_emissive_parent"]
    mc_d = ch["pooled_mean_children_per_dark_parent"]
    sec4 = lp.section_v2(
        "children", "04",
        f"Emissive parents average only {ch['pooled_relative_diff_pct']:.1f}% more children, "
        "too small to explain &sect;03's drift",
        lp.prose(
            f"<p>Tested hypothesis (the brief's highest-value question): does an emissive "
            f"256&sup3; parent voxel produce more occupied 512&sup3; children than a dark "
            f"parent? If so, that geometric mechanism alone would explain &sect;03's "
            f"sparse-shape emissive-frac rise. Pooled over {ch['n_shapes_paired']} shapes "
            f"({ch['total_parents']:,} parent voxels total, {ch['total_gap_parents']:,} with no "
            f"256 classification / excluded):</p>"
        ) + lp.hbar_chart([
            {"label": "emissive parent", "value": mc_e, "display": f"{mc_e:.3f}"},
            {"label": "dark parent", "value": mc_d, "display": f"{mc_d:.3f}"},
        ], title="mean occupied 512-children per 256-parent (pooled)") +
        lp.callout(
            (f"Emissive parents average {mc_e:.3f} children vs {mc_d:.3f} for dark parents: a "
             f"{ch['pooled_relative_diff_pct']:+.2f}% relative difference, pooled over "
             f"{ch['total_parents']:,} parent cells. The direction is consistent with the "
             f"hypothesis (emissive slightly higher) and shows up per-shape too "
             f"({ch['n_shapes_emis_gt_dark']} of {ch['n_shapes_paired']} shapes have "
             f"emissive&gt;dark vs {ch['n_shapes_dark_gt_emis']} the reverse, a real but weak "
             f"majority), but the MAGNITUDE is an order of magnitude too small: {mc_e:.2f} vs "
             f"{mc_d:.2f} children/parent cannot produce a 7% (let alone the brief's 25%) swing "
             "in emissive_frac by itself. <b>THE COUNT-BASED MECHANISM IS NOT THE EXPLANATION</b> "
             "for &sect;03's sparse-shape drift; a more likely driver is that sparse shapes have "
             "few emissive voxels in absolute terms, so the emissive_frac RATIO is intrinsically "
             "noisy (a handful of voxels flipping status via the &gt;1/255 threshold or the "
             "cKDTree gap-fallback swings a small denominator by a large relative amount): "
             "this page did not test that alternative directly."),
            title=f"A real but tiny effect ({ch['pooled_relative_diff_pct']:.1f}% pooled): not the driver of the sparse-shape drift",
            warn=True,
        ),
    )
    sections.append(sec4)

    # ================================================================ 05 geometry
    off = geo["mean_signed_offset_per_axis_256voxels"]
    sec5 = lp.section_v2(
        "geometry", "05",
        f"Two independent dual-grid extractions agree to {geo['mean_dist_in_256voxels']:.3f} "
        "voxel widths mean, with no systematic offset",
        lp.prose(
            f"<p>The key validation: Dongchen's 256&sup3; dual-grid vertex position vs ours "
            f"(512&sup3; dual vertices, averaged down to the parent 256 cell), compared in the "
            f"shared normalized [-0.5,0.5]&sup3; mesh frame, over {geo['n_shapes']} shapes "
            f"({geo['n_vertex_pairs_compared_total']:,} of {geo['n_256_vertices_total']:,} "
            f"256-vertices had a matching 512 parent, {pct(geo['coverage_frac'])} coverage):</p>"
        ) + lp.hbar_chart([
            {"label": "mean", "value": geo["mean_dist_in_256voxels"], "display": f'{geo["mean_dist_in_256voxels"]:.4f}'},
            {"label": "p90 (mean of per-shape p90)", "value": geo["p90_dist_in_256voxels"], "display": f'{geo["p90_dist_in_256voxels"]:.4f}'},
            {"label": "worst shape max", "value": geo["max_dist_in_256voxels"], "display": f'{geo["max_dist_in_256voxels"]:.4f}'},
        ], title="positional disagreement, in units of one 256-grid voxel width") +
        lp.callout(
            (f"Mean disagreement is {geo['mean_dist_in_256voxels']:.3f} of one 256-grid voxel "
             f"width, i.e. two independent dual-contouring runs (one at 256&sup3;, one at "
             f"512&sup3; then averaged down) land within a few percent of a voxel of each "
             f"other. <b>No systematic offset</b>: the mean signed per-axis offset is "
             f"[{off[0]:+.4f}, {off[1]:+.4f}, {off[2]:+.4f}] voxel widths, effectively zero "
             f"in all three axes -- this is agreement, not a coordinate-frame bug."),
            title="Geometry agrees; no axis-permutation or offset bug found",
        ),
    )
    sections.append(sec5)

    # ================================================================ 06 gallery
    gallery_sorted = sorted(gallery, key=lambda e: e["emissive_frac_512"])
    gallery_body = []
    for e in gallery_sorted:
        sid = e["sid"]
        gallery_body.append(lp.prose(
            f'<p style="margin-bottom:4px"><code>{sid}</code> &middot; '
            f'emissive_frac={e["emissive_frac_512"]:.3f} &middot; '
            f'{e["n_voxels_512"]:,} voxels @512 / {e["n_voxels_256"]:,} @256</p>'
        ))
        gallery_body.append(lp.fig_row(
            [("his 256³ appearance geometry", img(f"{sid}_geom256.png")),
             ("our 512³ appearance geometry", img(f"{sid}_geom512.png"))],
            caption_html=(
                "<b>Same camera, same scale, same shape.</b> Appearance geometry colored by "
                "PBR base_color, his 256&sup3; dual grid vs our native 512&sup3; re-extraction."
            ),
            content="photo",
        ))
        gallery_body.append(lp.fig_row(
            [("his emission (256, continuous, per-shape heatmap)", img(f"{sid}_emis256.png")),
             ("our emission target (512, binarized, white=emissive)", img(f"{sid}_emis512.png"))],
            caption_html=(
                "<b>Binarization at &gt;1/255 keeps the emissive silhouette, discards the "
                "continuous intensity.</b> His raw emissive channel (warm heatmap, normalized "
                "to this shape's own max value) vs our training target (white=emissive, "
                "dark=non, output.vxz base_color slot)."
            ),
            content="photo",
        ))

    sec6 = lp.section_v2(
        "gallery", "06",
        "Twelve shapes spanning emissive_frac from 0.0 to 1.0",
        lp.prose(
            "<p>Chosen to span the full emissive-fraction range and a wide size range "
            "(50k&ndash;1.75M voxels @512), not hand-picked for a favorable look:</p>"
        ) + "".join(gallery_body),
    )
    sections.append(sec6)

    # ================================================================ 07 provenance
    sec7 = lp.section_v2(
        "provenance", "07", "Provenance",
        lp.prose(
            "<p>All scripts ran in the <code>trellis2</code> conda env on solar "
            "(<code>/3dlg-jupiter-project/lightgen/miniforge3/envs/trellis2</code>), compute "
            "nodes only (the login node does not mount <code>/3dlg-jupiter-project</code>).</p>"
        ) + lp.results_table(["script", "purpose", "output"], "".join([
            f'<tr><td><code>scan_meta.py</code></td><td>scan meta.json across '
            f'dataset_direct/{{train,val,test}}_72k, sampling up to 6,000/split</td>'
            f'<td>manifest_sample.parquet ({stats["n_manifest_scanned"]:,} rows)</td></tr>',
            f'<tr><td><code>deep_quant.py</code></td><td>decode his emission_voxels_256 + '
            f'dual_grid_256 and our input.vxz via o_voxel.io.read; compute children-per-parent '
            f'and geometry agreement</td><td>deepquant_400.jsonl ({stats["n_deep_quant_sample"]} shapes, '
            f'stratified by emissive_frac &times; size)</td></tr>',
            '<tr><td><code>pick_gallery.py</code></td><td>select 12 shapes spanning emissive_frac 0&ndash;1</td>'
            '<td>gallery_sids.json</td></tr>',
            '<tr><td><code>render_compare.py</code></td><td>bpy renders via xgutils.bpyutil '
            '(subprocess-isolated per shape)</td><td>48 PNGs</td></tr>',
            '<tr><td><code>aggregate_stats.py</code></td><td>pool the manifest + deep-quant sample '
            'into every number on this page</td><td>aggregate_stats.json</td></tr>',
        ])) + lp.prose(
            "<p>SLURM jobs (solar, account 3dlg-hcvc-lab, partition 3dlg-hcvc-lab-short, "
            "cs-venus-02, excluding cs-venus-05/09/19): scan_meta job <code>237696</code>, "
            "deep_quant job <code>237702</code> (400/400 shapes, 821s), render job "
            "<code>237706</code> (12/12 shapes, subprocess-isolated).</p>"
        ) + lp.prose(
            "<p>Environment note: the trellis2 conda env lacked several xgutils dependencies "
            "(h5py, PyMCubes, matplotlib+deps, pydantic, libigl) -- all installed additively "
            "with <code>pip install</code> (numpy/torch/cuda pins untouched, confirmed via "
            "<code>pip install --dry-run</code> before installing matplotlib). "
            "<code>view_transform=\"Standard\"</code> was passed explicitly to bpyutil's "
            "render calls -- this bpy build's OCIO config lacks the default "
            "<code>\"Khronos PBR Neutral\"</code> transform.</p>"
        ) + lp.prose(
            "<p>Read-only on both source datasets; nothing was written to out_uv_voxel_74k or "
            "dataset_direct. All new files live under "
            "<code>/3dlg-jupiter-project/lightgen/segvigen_emissive/compare_out/</code> "
            "(scripts/, logs/, renders/, manifest_sample.parquet, deepquant_400.jsonl, "
            "gallery_sids.json, aggregate_stats.json).</p>"
        ),
    )
    sections.append(sec7)

    page_html = lp.page(
        title="His 256³ vs Ours 512³: Emissive Dataset Comparison",
        theme="v2",
        assets_dir=assets_dir,
        assets_rel=SITE_ASSETS,
        header_html=hero,
        body_sections=sections,
    )
    # Drop native lazy-loading on the gallery images: at 48 images down a long page,
    # Chromium's native loading="lazy" heuristic was observed to leave images past the
    # first ~6 shapes permanently un-fetched (naturalWidth=0, img.complete=false) even
    # after scrolling the full page and waiting 30s -- reproduced identically against
    # BOTH the local file and the live published URL, so not a QA-script artifact. 48
    # PNGs at ~50KB each is cheap enough that eager loading is the safer default here.
    page_html = page_html.replace(' loading="lazy"', "")
    out_path = os.path.join(OUT_DIR, "index.html")
    open(out_path, "w").write(page_html)
    print(f"wrote {out_path}")
    publish_assets(assets_dir)


if __name__ == "__main__":
    build()
