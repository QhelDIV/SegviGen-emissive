"""
Assemble vis_data/results_2k_html/index.html — the visual review page for the 2k
emission-filtered fine-tune result: a clean negative result (no fine-tune beats the
zero-shot SegviGen oracle). Four sections (Overview / Results / The tiny-glow wall /
Predictions), built on tools/xgpage.py.

All headline numbers are VERBATIM from notes/2026-07-07_autonomous_run_state.md
(the honest full-111-val, K=4-averaged, @0.5, voxel-bucket protocol) and cross-verified
directly against the source Slurm logs on the cluster:
  eval_231621.log = 2k-W5 best (outputs/emis_2k_w5/best.ckpt = epoch_0006.ckpt)
  eval_231622.log = 2k-W5 EMA  (epoch_0006_ema.ckpt)
  eval_231623.log = 2k-balanced best (outputs/emis_2k_bal/best.ckpt = epoch_0008.ckpt)
  eval_231624.log = 2k-balanced EMA  (epoch_0008_ema.ckpt)
The 8 Predictions-section shapes are a FRESH eval + --dump_vis rerun on a 8-shape
subset (val_pred8_2k, a symlink split), draws=4 for the honest per-shape IoU@0.5
caption number (dump8_231636.log = W5-best, dump8_231637.log = balanced-best); the
rendered voxel image itself is the first of those 4 draws (eval_emissive.py only
dumps draw index 0), same convention as finetune_binary_v1/build_finetune_page.py.

  python build_results_2k_page.py
"""
import os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "vis_data", "results_2k_v1")
OUT = os.path.join(ROOT, "vis_data", "results_2k_html")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "..", "tools"))
import xgpage as lp

SEGVIGEN_ROOT = "/3dlg-jupiter-project/lightgen/segvigen_emissive"

# ---------------------------------------------------------------------------------
# Palette (validated: node scripts/validate_palette.js "#c98500,#3987e5,#199e70"
# --mode dark -> ALL CHECKS PASS; the oracle/threshold line #6b7480 is the project's
# existing --ref reference-line gray (training_curves_v1), intentionally NOT run
# through the categorical validator since it's a threshold marker, not a data series).
PAGE_STYLE = """
  :root {
    --old1k: #c98500;   /* categorical slot 3 -- the 1k baseline being compared against */
    --w5: #3987e5;       /* categorical slot 1 -- 2k + W5 (fixed 5x) weighting */
    --bal: #199e70;      /* categorical slot 2 -- 2k + balanced ((1-p)/p) weighting */
    --ref: #6b7480;      /* reference/threshold line -- not a data series */
  }
  .page { max-width: 1320px; }
  a { color: #7db8f5; }
  .tag.w5 { color: var(--w5); border-color: rgba(57,135,229,.4); }
  .tag.bal { color: var(--bal); border-color: rgba(25,158,112,.4); }
  .tag.old1k { color: var(--old1k); border-color: rgba(201,133,0,.4); }
  .legend .ln { border-top-color: var(--ref); }
  table.results td.num, table.results th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .oracle-row td { color: #fff; font-weight: 600; }
"""

# ---------------------------------------------------------------------------------
# ------------------------------------------------------------- VERIFIED NUMBERS --
# Honest protocol: full 111-val, 4 samples/shape averaged, @0.5, voxel buckets.
ORACLE_NZ = 0.219
OLD1K_NZ = 0.117
W5_BEST_NZ, W5_EMA_NZ = 0.103, 0.107
BAL_BEST_NZ, BAL_EMA_NZ = 0.114, 0.112

# stratified by GT coverage (bucket_by=voxel; IoU @ global best_thr=0.2), verbatim from
# eval_231621/622/623/624.log's own "stratified by GT coverage" section
BUCKET_LABELS = ["zero", "tiny (0,5%]", "medium (5,30%]", "large (>30%)"]
STRAT = {
    "w5best": [0.3200, 0.0547, 0.1145, 0.3191],
    "w5ema": [0.1600, 0.0361, 0.1380, 0.4260],
    "balbest": [0.1100, 0.0492, 0.1215, 0.3607],
    "balema": [0.0600, 0.0400, 0.1446, 0.4190],
}

# noise callout: outputs/emis_2k_w5/train_curve.json epoch 6 (== best.ckpt) 16-sample
# quick-val vs eval_231621.log's full 111-shape K=4-averaged honest number
QUICKVAL_PEAK = 0.1791
HONEST_AVG = 0.1029

# ---------------------------------------------------------------------------------
# ---------------------------------------------------------- SVG bar chart helpers
def comparison_bar_svg():
    """oracle (dashed ref line) vs old-1k / 2k-W5 (best+EMA) / 2k-balanced (best+EMA),
    nonzero-glow IoU@0.5. Groups: 1 + 2 + 2 = 5 bars."""
    W, H = 640, 320
    ML, MR, MT, MB = 46, 14, 20, 42
    plot_w, plot_h = W - ML - MR, H - MT - MB
    ymax = 0.26

    def y(v):
        return MT + plot_h * (1 - v / ymax)

    groups = [
        ("old-1k\n(W5-EMA)", [("best", OLD1K_NZ, "var(--old1k)")]),
        ("2k + W5", [("best", W5_BEST_NZ, "var(--w5)"), ("EMA", W5_EMA_NZ, "var(--w5)")]),
        ("2k + balanced", [("best", BAL_BEST_NZ, "var(--bal)"), ("EMA", BAL_EMA_NZ, "var(--bal)")]),
    ]
    bar_w = 46
    pair_gap = 6
    group_widths = [len(bars) * bar_w + (len(bars) - 1) * pair_gap for _, bars in groups]
    n_groups = len(groups)
    total_bars_w = sum(group_widths)
    group_gap = (plot_w - total_bars_w) / (n_groups + 1)

    parts = []
    for gv in [0, 0.05, 0.10, 0.15, 0.20]:
        gy = y(gv)
        parts.append(f'<line x1="{ML}" y1="{gy:.1f}" x2="{W-MR}" y2="{gy:.1f}" stroke="#2a3038" stroke-width="1"/>')
        parts.append(f'<text x="{ML-8}" y="{gy+4:.1f}" text-anchor="end" fill="#8b96a5" font-size="10">{gv:.2f}</text>')

    # oracle dashed reference line
    oy = y(ORACLE_NZ)
    parts.append(f'<line x1="{ML}" y1="{oy:.1f}" x2="{W-MR}" y2="{oy:.1f}" stroke="var(--ref)" stroke-width="1.5" stroke-dasharray="5,4"/>')
    parts.append(f'<text x="{W-MR}" y="{oy-6:.1f}" text-anchor="end" fill="#c7cdd6" font-size="11" font-weight="600">{ORACLE_NZ:.3f} zero-shot oracle</text>')

    x = ML + group_gap
    for label, bars in groups:
        gw = len(bars) * bar_w + (len(bars) - 1) * pair_gap
        bx = x
        for sub, val, color in bars:
            by = y(val)
            base = y(0)
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{base-by:.1f}" rx="3" fill="{color}"{" fill-opacity=\"0.55\"" if sub=="EMA" else ""}/>')
            parts.append(f'<text x="{bx+bar_w/2:.1f}" y="{by-6:.1f}" text-anchor="middle" fill="#d8dde6" font-size="10.5">{val:.3f}</text>')
            parts.append(f'<text x="{bx+bar_w/2:.1f}" y="{base+14:.1f}" text-anchor="middle" fill="#8b96a5" font-size="9.5">{sub}</text>')
            bx += bar_w + pair_gap
        cx = x + gw / 2
        for i, line in enumerate(label.split("\n")):
            parts.append(f'<text x="{cx:.1f}" y="{H-MB+30+i*12:.1f}" text-anchor="middle" fill="#aeb6c2" font-size="10.5">{line}</text>')
        x += gw + group_gap

    parts.append(f'<line x1="{ML}" y1="{y(0):.1f}" x2="{W-MR}" y2="{y(0):.1f}" stroke="#3a4048" stroke-width="1.2"/>')
    return f'<svg viewBox="0 0 {W} {H}">{"".join(parts)}</svg>'


def stratified_bar_svg():
    """4 GT-coverage buckets x 4 checkpoints (w5best/w5ema/balbest/balema), grouped."""
    W, H = 640, 340
    ML, MR, MT, MB = 46, 14, 20, 50
    plot_w, plot_h = W - ML - MR, H - MT - MB
    ymax = 0.46
    series = [
        ("w5best", "var(--w5)", 1.0, "W5 best"),
        ("w5ema", "var(--w5)", 0.55, "W5 EMA"),
        ("balbest", "var(--bal)", 1.0, "bal best"),
        ("balema", "var(--bal)", 0.55, "bal EMA"),
    ]
    n_buckets = len(BUCKET_LABELS)
    bar_w = 20
    pair_gap = 2
    group_w = len(series) * bar_w + (len(series) - 1) * pair_gap
    group_gap = (plot_w - n_buckets * group_w) / (n_buckets - 1) if n_buckets > 1 else 0

    def y(v):
        return MT + plot_h * (1 - v / ymax)

    parts = []
    for gv in [0, 0.1, 0.2, 0.3, 0.4]:
        gy = y(gv)
        parts.append(f'<line x1="{ML}" y1="{gy:.1f}" x2="{W-MR}" y2="{gy:.1f}" stroke="#2a3038" stroke-width="1"/>')
        parts.append(f'<text x="{ML-8}" y="{gy+4:.1f}" text-anchor="end" fill="#8b96a5" font-size="10">{gv:.1f}</text>')

    x = ML
    base = y(0)
    for bi, blabel in enumerate(BUCKET_LABELS):
        bx = x
        for key, color, op, _ in series:
            val = STRAT[key][bi]
            by = y(val)
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{base-by:.1f}" rx="2.5" fill="{color}" fill-opacity="{op}"/>')
            if val >= 0.03:
                parts.append(f'<text x="{bx+bar_w/2:.1f}" y="{by-4:.1f}" text-anchor="middle" fill="#c7cdd6" font-size="8.5">{val:.2f}</text>')
            bx += bar_w + pair_gap
        cx = x + group_w / 2
        parts.append(f'<text x="{cx:.1f}" y="{H-MB+18:.1f}" text-anchor="middle" fill="#aeb6c2" font-size="10.5">{blabel}</text>')
        x += group_w + group_gap

    parts.append(f'<line x1="{ML}" y1="{base:.1f}" x2="{W-MR}" y2="{base:.1f}" stroke="#3a4048" stroke-width="1.2"/>')
    return f'<svg viewBox="0 0 {W} {H}">{"".join(parts)}</svg>'


COMPARISON_SVG = comparison_bar_svg()
STRATIFIED_SVG = stratified_bar_svg()

# ---------------------------------------------------------------------------------
# ------------------------------------------------------------ Predictions section
# sid -> (gt_frac, w5best_iou, balbest_iou, bucket) -- from dump8_231636.log (W5-best)
# / dump8_231637.log (balanced-best), the fresh 8-shape draws=4 dump rerun.
ROWS = [
    ("e9e31994a53d4fa68308f745c682a0b9", 1.000, 0.523, 0.288, "large"),
    ("ff6c2c51f7b040279200f8154a376841", 0.976, 0.058, 0.414, "large"),
    ("8f674e4c66c646528902d831941de942", 0.207, 0.071, 0.085, "medium"),
    ("1f5d1ee530ee4e91a05c1ca1e3c67db4", 0.284, 0.174, 0.095, "medium"),
    ("05bfab8ee60947bb8b02ff1519ffc121", 0.050, 0.055, 0.072, "tiny"),
    ("091fe81b15a9439996dee0ac9f19fed9", 0.046, 0.053, 0.021, "tiny"),
    ("0314e1cf6b8042be8381926e0f5cadbb", 0.033, 0.021, 0.008, "tiny"),
    ("089b371b8e8f4f11b89b58c8ff54311e", 0.000, 0.000, 0.002, "zero"),
]

def cp_all():
    import shutil
    missing = []
    for sid, *_ in ROWS:
        d = os.path.join(SRC, sid)
        for kind, dst_suffix in [("render_input.png", "input"), ("render_emissive.png", "emissive"),
                                  ("render_pred_w5best.png", "predw5"), ("render_pred_balbest.png", "predbal"),
                                  ("render_meshpred_w5.png", "meshpred_w5")]:
            s = os.path.join(d, kind)
            dst = os.path.join(OUT, f"{sid}_{dst_suffix}.png")
            if os.path.exists(s):
                shutil.copy(s, dst)
            else:
                missing.append((sid, kind))
    # interactive-3D preview GLBs (lightweight; loaded on click via model-viewer)
    glb_missing = []
    for sid, *_ in ROWS:
        d = os.path.join(SRC, sid)
        for kind in ["viewer_app.glb", "viewer_gt.glb", "viewer_pred_w5.glb"]:
            s = os.path.join(d, kind)
            if os.path.exists(s):
                shutil.copy(s, os.path.join(OUT, f"{sid}_{kind}"))
            else:
                glb_missing.append((sid, kind))
    if missing:
        print(f"\n{len(missing)} MISSING panels: {missing}\n")
    else:
        print("all panels present for all 8 sids")
    if glb_missing:
        print(f"{len(glb_missing)} MISSING viewer GLBs: {glb_missing}")
    else:
        print("all 3 viewer GLBs present for all 8 sids")

cp_all()


def iou_class(v):
    if v >= 0.3:
        return "iou-hi"
    if v <= 0.05:
        return "iou-lo"
    return ""


def build_row(sid, gt, w5, bal, bucket):
    def iou_span(v):
        return f'<span class="{iou_class(v)}">{v:.3f}</span>'
    return f"""
      <tr class="bucket-{bucket}">
        <td class="rowhead">
          <span class="sid">{sid}</span>
          <span class="frac">{gt:.3f}</span><span class="bucketlbl">{bucket} glow</span>
          <div class="iourow">
            <div><span class="tag w5">W5 best</span> IoU {iou_span(w5)}</div>
            <div><span class="tag bal">balanced best</span> IoU {iou_span(bal)}</div>
          </div>
        </td>
        <td><img src="{sid}_input.png"><div class="cap">appearance</div></td>
        <td><img src="{sid}_emissive.png"><div class="cap">GT target</div></td>
        <td><img src="{sid}_predw5.png"><div class="cap">2k+W5 best &mdash; IoU {w5:.3f}</div></td>
        <td><img src="{sid}_predbal.png"><div class="cap">2k+balanced best &mdash; IoU {bal:.3f}</div></td>
      </tr>"""


rows_html = "".join(build_row(*r) for r in ROWS)


def build_mesh_row(sid, gt, w5, bal, bucket):
    def iou_span(v):
        return f'<span class="{iou_class(v)}">{v:.3f}</span>'
    return f"""
      <tr class="bucket-{bucket}">
        <td class="rowhead">
          <span class="sid">{sid}</span>
          <span class="frac">{gt:.3f}</span><span class="bucketlbl">{bucket} glow</span>
          <div class="iourow">
            <div><span class="tag w5">W5 best</span> IoU {iou_span(w5)}</div>
          </div>
        </td>
        <td>{lp.viewer_img(f"{sid}_input.png", f"{sid}_viewer_app.glb", cap="appearance", title=f"{sid} · appearance")}</td>
        <td>{lp.viewer_img(f"{sid}_emissive.png", f"{sid}_viewer_gt.glb", cap="GT emissive (mesh)", title=f"{sid} · GT emissive")}</td>
        <td>{lp.viewer_img(f"{sid}_meshpred_w5.png", f"{sid}_viewer_pred_w5.glb", cap=f"2k+W5 best pred (mesh) &mdash; IoU {w5:.3f}", title=f"{sid} · 2k+W5 pred · IoU {w5:.3f}")}</td>
      </tr>"""


mesh_rows_html = "".join(build_mesh_row(*r) for r in ROWS)

# ==================================================================== Overview
overview_body = f"""
    <p>Two levers on top of the earlier 1k emissive fine-tune, both aimed at the same
      majority-class collapse (most surface voxels are non-emissive, so a model drifts
      toward "paint nothing"): <strong>bigger, cleaner training data</strong>
      (<code>train_2k_ef</code> &mdash; 2,000 emission-filtered PBR shapes, ~2&times; the
      old 1k set) and <strong>balanced per-shape loss weighting</strong>
      (W<sub>shape</sub>&asymp;(1&minus;p)/p, capped, vs. the old flat 5&times;). Full
      111-shape val set, 4 generations per shape averaged (single draws vary
      &plusmn;0.09 &mdash; see the noise callout below), voxel-based IoU, @0.5 fixed
      threshold, reported on shapes that actually glow (nonzero-glow).</p>
    {lp.verdict_box(
        "Bigger clean data + balanced per-shape weighting &mdash; neither beats zero-shot "
        f"SegviGen ({ORACLE_NZ:.3f}). All four 2k checkpoints land &asymp;0.10&ndash;0.11, tied "
        f"with the old 1k model ({OLD1K_NZ:.3f}). The wall is tiny-glow regions, untouched by "
        "data volume or loss weighting.")}
    {lp.hero_figs([
        f'<div class="hero-fig"><div class="hf-title">Nonzero-glow IoU @0.5 &mdash; everything sits below the oracle</div>{lp.chart_wrap(COMPARISON_SVG)}</div>',
        f'<div class="hero-fig"><div class="hf-title">The same result, split by how much of each shape glows</div>{lp.chart_wrap(STRATIFIED_SVG)}</div>',
    ])}
"""

# ==================================================================== Results
results_body = f"""
    <p class="sub">Honest protocol: full 111-shape val set, <strong>K=4</strong> generations
      per shape averaged, voxel-based IoU @ fixed threshold 0.5, reported on the 86 shapes
      with nonzero ground-truth glow (a checkpoint can't win by predicting all-black on the
      25 zero-glow shapes). Draw-std at the reported threshold averages
      &plusmn;0.09&ndash;0.11 per checkpoint &mdash; single-draw comparisons at this scale
      are not trustworthy (see the callout below).</p>

    {lp.chart_wrap(COMPARISON_SVG)}

    {lp.results_table(
        ["model", "nonzero IoU @0.5", "zero-glow bucket", "source log"],
        f'''
      <tr class="oracle-row"><td>zero-shot SegviGen oracle (frozen + label parts)</td><td class="num">{ORACLE_NZ:.3f}</td><td class="num">1.00</td><td>{lp.filepath("oracle_val96.json", f"{SEGVIGEN_ROOT}/dataset/oracle_val96.json")}</td></tr>
      <tr><td><span class="tag old1k">old 1k fine-tune</span> W5-EMA</td><td class="num">{OLD1K_NZ:.3f}</td><td class="num">0.03</td><td>{lp.filepath("eval_231582.log", f"{SEGVIGEN_ROOT}/eval_231582.log")}</td></tr>
      <tr><td><span class="tag w5">2k + W5</span> best (epoch 6)</td><td class="num">{W5_BEST_NZ:.3f}</td><td class="num">0.32</td><td>{lp.filepath("eval_231621.log", f"{SEGVIGEN_ROOT}/eval_231621.log")}</td></tr>
      <tr><td><span class="tag w5">2k + W5</span> EMA (epoch 6)</td><td class="num">{W5_EMA_NZ:.3f}</td><td class="num">0.16</td><td>{lp.filepath("eval_231622.log", f"{SEGVIGEN_ROOT}/eval_231622.log")}</td></tr>
      <tr><td><span class="tag bal">2k + balanced</span> best (epoch 8)</td><td class="num">{BAL_BEST_NZ:.3f}</td><td class="num">0.11</td><td>{lp.filepath("eval_231623.log", f"{SEGVIGEN_ROOT}/eval_231623.log")}</td></tr>
      <tr><td><span class="tag bal">2k + balanced</span> EMA (epoch 8)</td><td class="num">{BAL_EMA_NZ:.3f}</td><td class="num">0.06</td><td>{lp.filepath("eval_231624.log", f"{SEGVIGEN_ROOT}/eval_231624.log")}</td></tr>
        ''')}

    {lp.callout(
        '<strong>Why we average 4 samples, not 1:</strong> a 16-sample quick-val on the '
        '<span class="tag w5">2k + W5</span> checkpoint briefly showed IoU '
        f'<strong>{QUICKVAL_PEAK:.3f}</strong> &mdash; a +53% jump that would have been the '
        'headline. It was small-sample noise: the exact same checkpoint '
        f'({lp.filepath("epoch_0006.ckpt", f"{SEGVIGEN_ROOT}/outputs/emis_2k_w5/epoch_0006.ckpt")}, '
        f'read from {lp.filepath("train_curve.json", f"{SEGVIGEN_ROOT}/outputs/emis_2k_w5/train_curve.json")} '
        f'epoch 6) averages <strong>{HONEST_AVG:.3f}</strong> on the full 111-shape set '
        f'({lp.filepath("eval_231621.log", f"{SEGVIGEN_ROOT}/eval_231621.log")}). The multi-sample '
        'eval caught a false +53% before it reached anyone &mdash; this is why every number on '
        'this page is a K=4 average, never a single generation.',
        warn=True)}
"""

# ==================================================================== Tiny-glow wall
wall_body = f"""
    <p class="sub">Same 4 new checkpoints, IoU@ global-best-threshold (0.2), broken down by how
      much of each shape's surface is actually marked emissive in ground truth
      (<code>bucket_by=voxel</code>: real surface-voxel occupancy, not tessellation-biased face
      area). Buckets: <strong>zero</strong> (n=25), <strong>tiny</strong> (0&ndash;5%, n=53),
      <strong>medium</strong> (5&ndash;30%, n=18), <strong>large</strong> (&gt;30%, n=15).</p>

    {lp.chart_wrap(STRATIFIED_SVG)}

    <p><strong>Every model works on large glow (0.32&ndash;0.43 IoU) and fails on tiny glow
      (0.04&ndash;0.06 IoU).</strong> Tiny-glow shapes are also the largest bucket by far
      (53 of 111 &mdash; the median val shape is only &asymp;1.4% emissive), so the aggregate
      nonzero score is pinned down by exactly the failure mode neither lever touches. Data
      volume (1k&rarr;2k) and loss weighting (flat 5&times; &rarr; per-shape balanced) both
      shift the <em>zero-glow</em> and <em>large-glow</em> numbers around, but the tiny-glow
      column barely moves &mdash; 0.055/0.036/0.049/0.040, all within noise of each other.</p>
"""

# ==================================================================== Mesh view (headline)
mesh_body = f"""
    <p class="sub">The <span class="tag w5">2k+W5 best</span> prediction shown the way the
      paper will show it: decoded to a real surface via the official SegviGen
      <code>slat_to_glb</code> path (predicted texture-latent &rarr; res-512 remesh &rarr;
      4096&sup2; base-color bake), not coarsened to voxel cubes. Same 8 shapes, ordered
      large&rarr;tiny&rarr;zero glow. Within each row the appearance, GT emissive, and
      predicted mesh share orientation, scale, and camera &mdash; flipping between columns is
      a coloring change on the same object. The predicted base color is hard-thresholded at
      0.5 to white (emissive) / black (non-emissive), matching the headline metric and the
      GT's crisp look. <strong>Click any panel</strong> (&#128269; 3D) to open it in an
      interactive orbit/zoom viewer.</p>

    {lp.callout(
        'The predicted mesh is <strong>one representative generation</strong> (fixed seed); '
        'the captioned IoU is the <strong>K=4 average</strong> for that shape+checkpoint '
        f'({lp.filepath("dump8_231636.log", f"{SEGVIGEN_ROOT}/dump8_231636.log")}), so a row '
        'can look a little better or worse than its own caption number &mdash; the same '
        'stochasticity documented above. Decoded with '
        f'{lp.filepath("make_pred_glb.py", f"{SEGVIGEN_ROOT}/code/make_pred_glb.py")} from '
        f'{lp.filepath("epoch_0006.ckpt", f"{SEGVIGEN_ROOT}/outputs/emis_2k_w5/epoch_0006.ckpt")}. '
        'The same predictions, coarsened to voxel cubes in metric space, are in the section below.')}

    {lp.results_table(
        ["shape &middot; GT emissive_frac", "appearance", "GT emissive (mesh)", "2k+W5 best pred (mesh)"],
        mesh_rows_html)}
    {lp.legend([
        'Paper-style smooth mesh via official <code>slat_to_glb</code> (remesh + base-color bake)',
        '<span class="tag w5">2k+W5 best</span> fixed 5&times; emissive-voxel loss weight, epoch 6',
        'mesh = one representative generation; IoU shown is the K=4 average for that shape',
    ])}
"""

# ==================================================================== Predictions (voxel)
predictions_body = f"""
    <p class="sub">The same predictions as the paper-style mesh view above, but as the
      <strong>coarse voxel cubes in the space the IoU is actually computed in</strong> &mdash;
      and with the <span class="tag bal">2k+balanced best</span> checkpoint alongside
      <span class="tag w5">2k+W5 best</span> for the model comparison. 8 real val_96 shapes,
      ordered large&rarr;tiny&rarr;zero glow. <span class="tag w5">2k+W5 best</span> = epoch 6
      checkpoint, fixed 5&times; emissive-voxel loss weight. <span class="tag bal">2k+balanced
      best</span> = epoch 8 checkpoint, per-shape W<sub>shape</sub>&asymp;(1&minus;p)/p loss
      weight (capped). Appearance, GT target, and both predictions within a row share the same
      orientation, scale, and camera &mdash; flipping between columns is a texture change on a
      fixed object, not a different pose. Both predictions are thresholded at 0.5, matching the
      headline metric above.</p>

    {lp.callout(
        'Captioned IoU is the <strong>K=4 average</strong> for that exact shape+checkpoint '
        f'({lp.filepath("dump8_231636.log", f"{SEGVIGEN_ROOT}/dump8_231636.log")} W5-best / '
        f'{lp.filepath("dump8_231637.log", f"{SEGVIGEN_ROOT}/dump8_231637.log")} balanced-best) &mdash; '
        'the RENDERED voxels are the first of those 4 stochastic draws (the eval script only '
        'dumps draw 0 for visualization), so a single row can look slightly better or worse '
        'than its own caption number. This is the same stochasticity documented in the Results '
        'callout above, made visible: <strong>ff6c2c51&hellip;</strong> below is a large-glow '
        'shape (97.6% emissive) where 2k+W5 nonetheless collapses to near-total miss (0.058) '
        'while 2k+balanced recovers most of it (0.414) &mdash; even "large glow works" is a '
        'tendency across the bucket, not a guarantee on every shape.')}

    {lp.results_table(
        ["shape &middot; GT emissive_frac", "appearance", "GT target", "2k+W5 best pred", "2k+balanced best pred"],
        rows_html)}
    {lp.legend([
        '<span class="tag w5">2k+W5 best</span> fixed 5&times; emissive-voxel loss weight, epoch 6',
        '<span class="tag bal">2k+balanced best</span> per-shape (1&minus;p)/p loss weight (capped), epoch 8',
        'IoU shown is K=4-averaged for that shape; rendered voxels are draw 0 of those 4',
    ])}
"""

# ==================================================================== assemble
body_sections = [
    lp.section("overview", 1, "Overview", body_html=overview_body, preview_rem=None),
    lp.section("results", 2, "Results: a clean negative result",
               takeaway=f"All four 2k checkpoints land {W5_BEST_NZ:.3f}–{BAL_BEST_NZ:.3f} nonzero IoU, "
                        f"below the old 1k model ({OLD1K_NZ:.3f}) and well below the {ORACLE_NZ:.3f} "
                        "zero-shot oracle — and a 16-sample quick-val briefly overstated the 2k+W5 "
                        "result by 53% before the K=4 average corrected it.",
               body_html=results_body, preview_rem=46),
    lp.section("tiny-glow-wall", 3, "The tiny-glow wall",
               takeaway="Every checkpoint nails large-glow shapes (0.32–0.43 IoU) and fails on "
                        "tiny-glow shapes (0.04–0.06 IoU) — and tiny-glow shapes are the "
                        "majority of the val set, so this single failure mode caps the aggregate "
                        "score regardless of data volume or loss weighting.",
               body_html=wall_body, preview_rem=40),
    lp.section("mesh-view", 4, "Paper-style mesh view: predictions as real surfaces",
               takeaway="Decoded through the official slat_to_glb path (remesh + base-color "
                        "bake), the 2k+W5 predictions are smooth surfaces, not voxel cubes: "
                        "large-glow shapes come back as clean white emissive regions on a real "
                        "mesh, and tiny/zero-glow shapes come back mostly black — the same "
                        "tiny-glow wall, now visible at surface quality.",
               body_html=mesh_body, preview_rem=52),
    lp.section("predictions", 5, "The same predictions as voxels (metric space) &mdash; W5 vs balanced",
               takeaway="Large-glow shapes are visually close to ground truth; tiny-glow shapes "
                        "come back mostly black or with the model painting the wrong region "
                        "entirely — the tiny-glow wall made visible, not just charted.",
               body_html=predictions_body, preview_rem=49),
    ('<footer>Lightgen war room &middot; segvigen_emissive/vis_data/results_2k_html &middot;\n'
     '    Honest full-val (K=4, @0.5, voxel buckets): ' +
     ", ".join(lp.filepath(f"eval_{j}.log", f"{SEGVIGEN_ROOT}/eval_{j}.log")
               for j in ["231621", "231622", "231623", "231624"]) +
     ' &middot; 8-shape prediction dump: ' +
     ", ".join(lp.filepath(f"dump8_{j}.log", f"{SEGVIGEN_ROOT}/dump8_{j}.log")
               for j in ["231636", "231637"]) + '\n'
     '    &middot; <a href="../training_curves_v1/index.html">training curves</a> &middot; '
     '<a href="../finetune_binary_v1/index.html">1k fine-tune (data + predictions)</a> &middot; '
     '<a href="../index.html">&uarr; all lightgen visuals</a>\n'
     '  </footer>'),
]

html = lp.page(
    title="2k fine-tune results: a clean negative result",
    header_html=lp.header(
        "2k fine-tune results: a clean negative result",
        'Bigger clean data (<code>train_2k_ef</code>, 2,000 emission-filtered PBR shapes) and '
        'balanced per-shape loss weighting were both meant to fix the emissive fine-tune\'s '
        'majority-class collapse. Neither beats zero-shot SegviGen. Here is the honest '
        'full-val comparison, why the failure concentrates on tiny-glow regions, and what the '
        'predictions actually look like on 8 real shapes.'),
    body_sections=body_sections,
    outline_entries=[
        {"id": "overview", "label": "Overview"},
        {"id": "results", "label": "Results"},
        {"id": "tiny-glow-wall", "label": "The tiny-glow wall"},
        {"id": "mesh-view", "label": "Paper-style mesh view"},
        {"id": "predictions", "label": "Predictions (voxels)"},
    ],
    needs_katex=False,
    assets_dir=os.path.join(ROOT, "..", "web", "assets"),  # cache-bust theme.css/ui.js
    extra_head=f"<style>{PAGE_STYLE}</style>\n" + lp.model_viewer_head(),
    extra_body_end=lp.model_viewer_modal(),
)

with open(os.path.join(OUT, "index.html"), "w") as f:
    f.write(html)
print(f"\nwrote {OUT}/index.html")
