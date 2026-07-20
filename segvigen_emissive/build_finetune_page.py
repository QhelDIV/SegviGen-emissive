"""
Assemble vis_data/finetune_examples_html/index.html from the rendered PNGs in
vis_data/finetune_examples/<sid>/ (render_input.png, render_emissive.png, img.png,
render_pred_w5ema.png, render_pred_w1ema.png) + the per-sid IoU numbers (hardcoded from
eval96_231379.log / eval96_231380.log — the --dump_vis reruns that actually produced the
rendered voxel predictions).

IMPORTANT: generation is stochastic (documented in notes/metrics_explainer.html), so this
--dump_vis rerun's per-sample IoUs differ from the eval96_231258/9.log run originally used
to pick these 8 sids for their story (e.g. a "flip" case). The 8 sids and bucket labels
still come from the original selection log (gt_frac is stable — it's ground truth); the
per-sample IoU numbers shown here are from THIS rerun since they must match what's
actually rendered. Global-best fixed thresholds for this rerun: W5-EMA @0.2, W1-EMA @0.5
(read off each log's own "BEST (fixed thr)" line).

2026-07-06: rewritten on top of tools/xgpage.py (the shared component module) as part
of the design-system extraction — same verified content, composed from shared components
instead of a hand-rolled STYLE/JS blob. Report-style structure (Overview -> Results ->
Method -> Data with outline + Medium-style preview/expand) unchanged from the previous
hand-built version; see PIXEL PARITY verification notes in the extraction report.

  python build_finetune_page.py
"""
import os, shutil, hashlib, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "vis_data", "finetune_examples")
OUT = os.path.join(ROOT, "vis_data", "finetune_examples_html")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "..", "tools"))
import xgpage as lp

# Cluster path root (verified 2026-07-07 via `ssh solar` directory listing) for the
# filepath() hover/click-copy component. Per-sample files use <sid> as a literal
# placeholder in the copyable path.
SEGVIGEN_ROOT = "/3dlg-jupiter-project/lightgen/segvigen_emissive"

# sid -> (gt_frac, w5ema_iou@0.2, w1ema_iou@0.5, bucket) — from eval96_231379.log
# (W5-EMA dump rerun) / eval96_231380.log (W1-EMA dump rerun); see module docstring re:
# stochastic sampling vs the original selection log.
ROWS = [
    ("e9e31994a53d4fa68308f745c682a0b9", 1.000, 0.405, 0.036, "large"),
    ("a82965cbfbe3470eae134efdccf15011", 1.000, 1.000, 0.000, "large"),
    ("f65a020ba69c47e2a66f635ee0e6f8c2", 0.548, 0.687, 0.631, "large"),
    ("d5fb4f19d4164612b165caac5471555c", 0.180, 0.180, 0.984, "medium"),
    ("f52e9b616c0a4075a70e5eb844f07bb3", 0.079, 0.169, 0.137, "medium"),
    ("10b7ad59f3bc4851a86d7f165ecd4c16", 0.015, 0.001, 0.473, "tiny"),
    ("bbeccdb222e74d99812cd2bd892222a8", 0.027, 0.027, 0.000, "tiny"),
    ("0414e54cda324108a7a51615f5cfd376", 0.000, 0.000, 1.000, "zero"),
]

PIPELINE_SID = "d5fb4f19d4164612b165caac5471555c"  # green/yellow fish — recognizable, clear
                                                    # white eye/fin/belly emissive markings.
                                                    # Rendered separately by render_section_a.py
                                                    # at a 3/4 camera + light backdrop (secA_*.png);
                                                    # NOT reused from the vis_data/finetune_examples
                                                    # per-sid renders (those stay dark-bg, B+C only).
HERO_ROW_SID = "a82965cbfbe3470eae134efdccf15011"  # large glow, IoU 1.000 (W5) vs 0.000 (W1) —
                                                    # the clearest single-row "eager vs timid" story.
# Real counts from render_section_a.py's secA_stats.txt (this sid's actual npz dump):
PIPE_DENSE_VOXELS = "~476k"
PIPE_LATENT_TOKENS = "1,734"

# ---- copy/flatten images into the html dir ----
def cp(src, dst):
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, dst))
        return True
    print(f"  [MISSING] {src}")
    return False

missing = []
for sid, *_ in ROWS:
    d = os.path.join(SRC, sid)
    if not cp(os.path.join(d, "render_input.png"), f"{sid}_input.png"): missing.append((sid, "input"))
    if not cp(os.path.join(d, "render_emissive.png"), f"{sid}_emissive.png"): missing.append((sid, "emissive"))
    if not cp(os.path.join(d, "img.png"), f"{sid}_photo.png"): missing.append((sid, "photo"))
    if not cp(os.path.join(d, "render_pred_w5ema.png"), f"{sid}_pred_w5ema.png"): missing.append((sid, "pred_w5ema"))
    if not cp(os.path.join(d, "render_pred_w1ema.png"), f"{sid}_pred_w1ema.png"): missing.append((sid, "pred_w1ema"))

if missing:
    print(f"\n{len(missing)} MISSING panels: {missing}\n")
else:
    print("all panels present for all 8 sids")

# ---- pipeline strip images: rendered by render_section_a.py directly into OUT
# (secA_1..5), NOT copied from vis_data/finetune_examples — Section A uses its own
# 3/4-camera + light-backdrop renders so shapes read against the dark page. Just verify
# they exist (run render_section_a.py first if not).
SECA = ["secA_1_source.png", "secA_2_target.png", "secA_3_voxel.png",
        "secA_4_latent.png", "secA_5_photo.png"]
for fn in SECA:
    if not os.path.exists(os.path.join(OUT, fn)):
        print(f"  [MISSING] {fn} — run render_section_a.py first")

# hero_curve.png (Overview's 2nd hero figure — a screenshot of training_curves_v1's
# "Quick-val IoU" chart) is a static captured asset, not rendered by this repo; just verify.
if not os.path.exists(os.path.join(OUT, "hero_curve.png")):
    print("  [MISSING] hero_curve.png — re-capture from the live training_curves_v1 page if needed")

# KaTeX now lives at the shared web/assets/katex/ (moved 2026-07-06); no per-page copy.


def cache_bust(fn):
    """Content-hash query string (?v=<hash8>) so a browser cache from a previous publish
    can't show a stale image after a re-render — the owner hit this after the last
    republish. Falls back to a bare filename if the file is missing (already warned above)."""
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        return fn
    h = hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    return f"{fn}?v={h}"


def iou_class(v):
    if v >= 0.5:
        return "iou-hi"
    if v <= 0.05:
        return "iou-lo"
    return ""


def loss_math_html():
    """The 'exact loss math' subsection. Built from lp.equation()/lp.inline_katex() calls
    (raw LaTeX strings, r-prefixed so backslashes aren't Python escapes) rather than
    hand-written <span class="kx"> HTML — same rendered result, less duplication."""
    setup_p = (
        '<p style="font-size:.82rem;margin:0">Setup (both arms) &mdash; per sample, target latent '
        + lp.inline_katex(r'z \in \mathbb{R}^{N\times32}')
        + ' (the encoded white/black repaint), noise '
        + lp.inline_katex(r'\varepsilon \sim \mathcal{N}(0,I)') + ', '
        + lp.inline_katex(r't \sim \mathcal{U}(0,1)') + ':</p>'
    )
    combined_eq = lp.equation(
        r'x_t = t\,\varepsilon + (1-t)\,z \qquad v^* = \varepsilon - z \qquad '
        r'\hat v = f_\theta(x_t,\ \text{tex}_{\text{in}},\ \text{shape},\ t,\ \text{cond})'
    )
    w1_p = ('<p style="font-size:.82rem;margin:1rem 0 .3rem"><span class="tag w1">W1</span>'
            ' (plain) &mdash; every token equal:</p>')
    w1_eq = lp.equation(r'\mathcal{L} = \tfrac{1}{32N}\textstyle\sum_{i,c}(\hat v_{ic} - v^*_{ic})^2')
    w5_p = ('<p style="font-size:.82rem;margin:1rem 0 .3rem"><span class="tag w5">W5</span>'
            ' ("eager") &mdash; each token i has '
            + lp.inline_katex(r'm_i \in [0,1]')
            + ' = the emissive fraction of its 16&sup3; block ('
            + lp.filepath("emis_mask.pth", f"{SEGVIGEN_ROOT}/dataset/train_1k/<sid>/emis_mask.pth")
            + '). With W=5:</p>')
    w5_eq1 = lp.equation(
        r'w_i = 1 + (W{-}1)\,m_i',
        comment='fully-emissive block &rarr; <strong>5&times;</strong>, half &rarr; 3&times;, none &rarr; 1&times;')
    w5_eq2 = lp.equation(
        r'\tilde w_i = w_i \big/ \tfrac{1}{N}\textstyle\sum_j w_j',
        comment='renormalized so weights average exactly 1 &mdash; keeps loss '
                'scale / lr meaning; real sample: weights &isin; [0.91, 4.55], mean 1.0000')
    w5_eq3 = lp.equation(r'\mathcal{L} = \tfrac{1}{32N}\textstyle\sum_{i,c} \tilde w_i\,(\hat v_{ic} - v^*_{ic})^2')

    means_p = '<p style="font-size:.82rem;margin:1rem 0 .4rem"><strong>What this means:</strong></p>'
    bullet3_frac = lp.inline_katex(r'\frac{Wp}{Wp + (1-p)}')
    bullets = f'''<ol class="loss-bullets">
      <li>It re-prices <strong>locations</strong>, not label mistakes &mdash; the loss lives in
        latent space on the flow velocity; both error directions at an emissive token cost
        5&times;. No classification happens at training time.</li>
      <li>The renormalization makes background tokens cost slightly <strong>less</strong> than
        1&times; (&asymp;0.91&ndash;0.95) &mdash; false-whites become mildly discounted, which
        is the mechanism behind W5's "eager" over-painting personality.</li>
      <li>Why 5&times; is too weak for tiny glow: with p = emissive-token fraction, the
        emissive share of total loss is
        {bullet3_frac}.
        For the median shape (p=0.014): &asymp;6.6% &mdash; up from 1.4%, still a rounding
        error to the optimizer. Giving positives ~half the loss on such a shape needs
        W&asymp;70 &mdash; motivates the planned per-shape balanced weighting
        (W<sub>shape</sub> &asymp; (1&minus;p)/p, capped).</li>
    </ol>'''

    return f'''
      <h4 id="loss-math">The exact loss math</h4>
      <div class="loss-wrap">
        {setup_p}
        {combined_eq}
        {w1_p}
        {w1_eq}
        {w5_p}
        {w5_eq1}
        {w5_eq2}
        {w5_eq3}
        {means_p}
        {bullets}
      </div>'''


def build_row(sid, gt, w5, w1, bucket):
    def iou_span(v):
        return f'<span class="{iou_class(v)}">{v:.3f}</span>'
    return f"""
      <tr class="bucket-{bucket}">
        <td class="rowhead">
          <span class="sid">{sid}</span>
          <span class="frac">{gt:.3f}</span><span class="bucketlbl">{bucket} glow</span>
          <div class="iourow">
            <div><span class="tag w5">W5-EMA</span> IoU {iou_span(w5)}</div>
            <div><span class="tag w1">W1-EMA</span> IoU {iou_span(w1)}</div>
          </div>
        </td>
        <td><img src="{sid}_input.png"><div class="cap">appearance</div></td>
        <td class="rawinput"><img src="{sid}_photo.png"><div class="cap"><b>raw pipeline input</b><br>cond photo</div></td>
        <td><img src="{sid}_emissive.png"><div class="cap">GT target</div></td>
        <td><img src="{sid}_pred_w5ema.png"><div class="cap">W5-EMA @ 0.2</div></td>
        <td><img src="{sid}_pred_w1ema.png"><div class="cap">W1-EMA @ 0.5</div></td>
      </tr>"""


rows_html = "".join(build_row(*r) for r in ROWS)


# ==================================================================== Overview
overview_body = f"""
    <p>We fine-tuned SegviGen's <code>full_seg</code> model to paint every surface voxel of a
      3D asset white (emissive/glowing) or black (not) — a binary special case of its usual
      multi-part segmentation — training two identical runs that differ only in how much an
      emissive-voxel mistake costs the loss.</p>
    {lp.verdict_box(
        "Training is stable and neither arm beats the 0.235 zero-shot oracle "
        "on the flat-mean metric — but the weighting changes the model's <em>behavior</em> "
        "(eager vs timid) far more than it changes its score.")}
    <p style="font-size:.85rem;color:var(--muted)">This page covers the original 1k-shape
      fine-tune (W5 vs W1). All three "what's next" items below (multi-draw eval,
      timidity-proof selection, per-shape balanced weighting) have since shipped on a
      bigger 2k-shape dataset — still below the zero-shot oracle, and the reason why is now
      well understood. See <a href="../results_2k_v1/index.html">latest 2k results
      &rarr;</a>.</p>
    {lp.hero_figs([
        lp.hero_fig_row(
            "Aligned results row &mdash; a82965cb&hellip; (large glow, IoU 1.000)",
            [(f"{HERO_ROW_SID}_input.png", "appearance"),
             (f"{HERO_ROW_SID}_emissive.png", "GT target"),
             (f"{HERO_ROW_SID}_pred_w5ema.png", "W5-EMA &mdash; IoU 1.000"),
             (f"{HERO_ROW_SID}_pred_w1ema.png", "W1-EMA &mdash; IoU 0.000")]),
        lp.hero_fig_image(
            "Quick-val IoU over training (from training_curves_v1)",
            "hero_curve.png",
            alt="W5 vs W1 quick-val IoU curve, both below the 0.235 oracle and 0.230 prior-best reference lines"),
    ])}
    <p><strong>What's next:</strong></p>
    {lp.next_bullets([
        '<strong>Multi-draw eval</strong> — average several stochastic samples per shape; '
        'right now the same checkpoint can score 0.0 or 0.5 on the same shape between runs '
        '(see the Results stochasticity note below).',
        '<strong>Timidity-proof checkpoint selection</strong> — the flat-mean metric currently '
        'rewards a model that predicts nothing on zero-glow shapes; select on the non-zero-glow '
        "subset instead so a \"timid\" checkpoint can't win on free points.",
        '<strong>Per-shape balanced weighting</strong> — replace the fixed W=5 with '
        'W<sub>shape</sub> &asymp; (1&minus;p)/p (capped), so tiny-glow shapes get a loss weight '
        'that actually matches their rarity instead of a one-size-fits-all multiplier '
        "(see the Method section's loss math).",
    ])}
"""

# ==================================================================== Results
results_body = f"""
    <p>All 8 are from the <strong>val_96</strong> held-out set, never seen during training.
      Rows are ordered by ground-truth glow size (large &rarr; medium &rarr; tiny &rarr; zero)
      so the glow-size story reads top to bottom. <span class="tag w5">W5-EMA</span> =
      "eager" (mistakes on emissive voxels weighted 5&times;, epoch 14 EMA).
      <span class="tag w1">W1-EMA</span> = "timid" (identical run, plain 1&times; loss,
      epoch 16 EMA). Each checkpoint's voxels are thresholded at its own global-best fixed
      cutoff on the full val_96 sweep (W5-EMA @ 0.2, W1-EMA @ 0.5 &mdash; log-verified).
      Appearance, GT target, and both predictions within a row share the same orientation,
      scale, and camera &mdash; flipping between those four columns is a texture change on
      a fixed object, not a different pose. The <b>cond photo</b> column (dashed border) is
      the one exception: it's the data pipeline's own raw render, kept as-is. For the training
      loss/IoU curves behind these two checkpoints, see
      <a href="../training_curves_v1/index.html">training_curves_v1</a>.</p>

    {lp.callout(
        '<strong>Note on the numbers below:</strong> sampling is stochastic, so these 8 sids '
        'were chosen from an earlier eval pass (' +
        lp.filepath("eval96_231258.log", f"{SEGVIGEN_ROOT}/eval96_231258.log") + ' / ' +
        lp.filepath("eval96_231259.log", f"{SEGVIGEN_ROOT}/eval96_231259.log") +
        ') for their '
        'glow-size story, but the voxel renders and IoUs shown here come from a second pass '
        '(' + lp.filepath("eval96_231379.log", f"{SEGVIGEN_ROOT}/eval96_231379.log") + ' W5-EMA / ' +
        lp.filepath("eval96_231380.log", f"{SEGVIGEN_ROOT}/eval96_231380.log") + ' W1-EMA) run '
        'specifically to dump per-voxel predictions for rendering &mdash; so the picture and '
        "the number in each cell always agree, even where a shape's score shifted between "
        'runs (ground-truth <code>emissive_frac</code> is fixed; predictions are not).',
        warn=True)}

    {lp.results_table(
        ["shape &middot; GT emissive_frac", "appearance", "cond photo", "GT target", "W5-EMA pred", "W1-EMA pred"],
        rows_html)}
    {lp.legend([
        '<span class="tag w5">W5-EMA</span> "eager" &mdash; mistakes on emissive voxels count 5&times;',
        '<span class="tag w1">W1-EMA</span> "timid" &mdash; plain loss, same data/schedule',
        'IoU shown per checkpoint at its own global-best fixed threshold, this dump run',
    ])}
"""

# ==================================================================== Method
concat_diagram = f'''<div class="concat-diagram">
        <div class="concat-col">
          <div class="cblock">{"".join('<div class="ccell noisy"></div>' for _ in range(14))}</div>
          <div class="clabel">x_t (N noisy target tokens)</div>
        </div>
        <div style="font-size:1.1rem;color:var(--muted)">+</div>
        <div class="concat-col">
          <div class="cblock">{"".join('<div class="ccell ref"></div>' for _ in range(14))}</div>
          <div class="clabel">input_tex_slat (N appearance-reference tokens)</div>
        </div>
        <div style="font-size:1.1rem;color:var(--muted)">&rarr;</div>
        <div class="concat-col">
          <div class="clabel" style="margin-top:0">2N tokens into the transformer<br>(self-attention across both halves)</div>
        </div>
      </div>'''

tensor_rows = [
    {"name": "coords", "shape": "(N,4) int",
     "role": "shared sparse coordinate set &mdash; [batch,x,y,z] at the 32&sup3; grid scale; "
             "N ranges 66&ndash;6,148 across the dataset, median &asymp;2,053. All three slats "
             "below share this same N."},
    {"name": "shape_slat.feats", "shape": "(N,32)", "role": "geometry latent (shape VAE)"},
    {"name": "input_tex_slat.feats", "shape": "(N,32)",
     "role": "PBR appearance latent (texture VAE on the original textures)"},
    {"name": "output_tex_slat.feats", "shape": "(N,32)", "cls": "target",
     "role": "<strong>the TARGET</strong> &mdash; texture VAE on the white/black repaint"},
    {"name": "cond", "shape": "(1,~1024,1024)",
     "role": "DINOv3-L patch tokens of the background-removed rendered photo (see the Data "
             "section's stage 5; token count varies slightly per image, e.g. 1029). Zeros = "
             "the unconditional/CFG branch. Stored per-sample as " +
             lp.filepath("cond.pth", f"{SEGVIGEN_ROOT}/dataset/train_1k/<sid>/cond.pth") + "."},
    {"name": "emis_mask", "shape": "(N,)", "cls": "aux",
     "role": "per-token emissive-coverage weight, used only by the W5 loss arm &mdash; "
             "training-only, never fed to the model as input. Stored per-sample as " +
             lp.filepath("emis_mask.pth", f"{SEGVIGEN_ROOT}/dataset/train_1k/<sid>/emis_mask.pth") + "."},
]

method_body = f"""
    <p style="font-size:.85rem">Here's the exact tensor inventory for one training sample
      (verified against the code and a real sample, N=1203 tokens):</p>
    {lp.tensor_table(tensor_rows, id="tensor-inventory", extra_html=(
        loss_math_html() + f'''
      <p style="font-size:.82rem;margin-top:.9rem">The actual SegviGen trick: Gen3DSeg
        <strong>concatenates</strong> <code>x_t</code> with <code>input_tex_slat</code> along the
        token axis (<code>shape_slat</code> duplicated to match) &mdash; the transformer processes
        2N tokens: N carrying the noisy answer, N carrying the real appearance for reference,
        connected by self-attention.</p>
      {concat_diagram}'''))}
"""

# ==================================================================== Data
strip_stages = f"""
      <div class="strip-stage">
        <span class="n">1</span>
        <img src="{cache_bust('secA_1_source.png')}" alt="somage record">
        <div class="t">Somage record (the pipeline's source)</div>
        <div class="d">A texture-map tensor + mesh (Dongchen's "somage" format) — the
          pipeline's actual starting point, itself built from an artist-made TexVerse
          asset upstream (the artist's original file never enters this pipeline). A
          somage has no picture of its own; shown here is its reconstructed appearance
          (the {lp.filepath(f"{PIPELINE_SID}_input.glb", f"{SEGVIGEN_ROOT}/dataset/val_96/{PIPELINE_SID}/glb/{PIPELINE_SID}_input.glb")}
          built in stage 2), including which surfaces were
          marked "emissive" (glowing) in that record.</div>
      </div>
      <div class="strip-arrow">&rarr;</div>
      <div class="strip-stage">
        <span class="n">2</span>
        <div class="imgpair"><img src="{cache_bust('secA_1_source.png')}"><img src="{cache_bust('secA_2_target.png')}"></div>
        <div class="t">Two derived GLBs</div>
        <div class="d">From the somage record, one export step builds two GLBs sharing
          the same geometry: original textures (left, what the model sees — a
          reconstruction, not the artist's original file) and a solid white/black
          recoloring (right) — white wherever the artist's material said "emissive" (the
          training TARGET). Notice the eye and belly patch turn white; everything else —
          including the pectoral fin — goes black, since only the eye and belly were
          marked emissive by the artist.</div>
      </div>
      <div class="strip-arrow">&rarr;</div>
      <div class="strip-stage">
        <span class="n">3</span>
        <img src="{cache_bust('secA_3_voxel.png')}" alt="voxelized target">
        <div class="t">Voxelize at 512³</div>
        <div class="d">Both GLBs are converted into a dense grid of colored surface
          voxels at 512³ resolution ({PIPE_DENSE_VOXELS} occupied voxels for this shape) —
          geometry becomes a big regular array. Shown: the target's voxels, colored by
          the actual GT emissive mask — eye and belly patch survive clearly, matching
          stage 2.</div>
      </div>
      <div class="strip-arrow">&rarr;</div>
      <div class="strip-stage">
        <span class="n">4</span>
        <img src="{cache_bust('secA_4_latent.png')}" alt="sparse latent tokens">
        <div class="t">Encode to sparse 32³ latents</div>
        <div class="d">A VAE compresses the 512³ grid down to a sparse latent — for this
          shape, {PIPE_LATENT_TOKENS} tokens (geometry + input-appearance + target-color
          codes each) — this is what the flow model is actually trained to denoise. Same
          shape, same eye/belly markings, now a handful of coarse blocks.</div>
      </div>
      <div class="strip-arrow">&rarr;</div>
      <div class="strip-stage">
        <span class="n">5</span>
        <img src="{cache_bust('secA_5_photo.png')}" alt="rendered photo">
        <div class="t">Photo &rarr; DINOv3 tokens</div>
        <div class="d">Separately, the input GLB is rendered to a photo (background
          removed) and run through a DINOv3 vision encoder &mdash; an optional extra hint
          of what the object actually looks like. This photo looks "laid down" compared
          to stages 1&ndash;4 because the data-build pipeline's own renderer imports the
          GLB the same uncorrected way stages 1&ndash;2 originally did &mdash; consistent
          across every sample in the dataset (so not a training-correctness bug), but left
          as-is here rather than re-oriented like the other four stages.</div>
      </div>"""

data_body = f"""
    <p>Illustrated here with one real val_96 example
      (<code>{PIPELINE_SID[:8]}…</code>, a reef fish with two emissive markings — eye and a
      belly patch) rendered upright (dorsal ridge up, belly down) from the same 3/4 camera
      angle at every stage so the transformation reads as one continuous story:</p>
    <div class="strip" id="pipeline-strip">{strip_stages}
    </div>
    <p style="font-size:.85rem;color:var(--muted)">The model's job: given (4) and (5) for
      the <em>input</em> appearance, predict the target latent whose decoded color is
      white/black in the right places &mdash; without ever seeing the artist's original
      material flag directly.</p>
    <p style="font-size:.85rem">This is one hand-picked example. For the full dataset with
      statistics and 48 random examples:
      <a href="../dataset_gallery_v1/index.html">the dataset gallery</a>.</p>
"""

# ==================================================================== assemble
body_sections = [
    lp.section("overview", 1, "Overview", body_html=overview_body, preview_rem=None),
    lp.section("results", 2, "Results: eight examples, appearance &rarr; prediction vs ground truth",
               takeaway='W5 ("eager") reliably captures large-glow shapes but over-paints tiny '
                        'ones; W1 ("timid") sometimes matches large glow well but often collapses to '
                        'near-nothing, and racks up free points on zero-glow shapes &mdash; no single number '
                        'tells this whole story.',
               body_html=results_body, preview_rem=49),
    lp.section("method", 3, "Method: what the model actually receives",
               takeaway="The model denoises a flow-matching velocity in latent space and never "
                        "sees a classification label &mdash; W5 vs W1 is entirely about which locations' errors "
                        "the loss weights more, not a different task.",
               body_html=method_body, preview_rem=26.5),
    lp.section("data", 4, "Data: how one training sample is made",
               takeaway='One artist-made TexVerse asset becomes a handful of tokens (median '
                        '&asymp;2,053) via a raw "somage" record and 512&sup3; voxelization &mdash; the artist\'s '
                        'original file itself never enters this pipeline.',
               body_html=data_body, preview_rem=66),
    ('<footer>Lightgen war room &middot; segvigen_emissive/vis_data/finetune_examples_html &middot;\n'
     '    8 hand-picked val_96 shapes &middot; ' +
     lp.filepath("eval96_231379.log", f"{SEGVIGEN_ROOT}/eval96_231379.log") + ' (W5-EMA) / ' +
     lp.filepath("eval96_231380.log", f"{SEGVIGEN_ROOT}/eval96_231380.log") + ' (W1-EMA)</footer>'),
]

html = lp.page(
    title="SegviGen binary-emissive fine-tune — data + predictions",
    header_html=lp.header(
        "SegviGen binary-emissive fine-tune — data + predictions",
        'How one training sample gets built, and what the two fine-tuned '
        'checkpoints (<span class="tag w5">W5-EMA</span> <span class="tag w1">W1-EMA</span>) '
        'actually predict against ground truth on 8 real val_96 shapes spanning the glow-size '
        'spectrum.'),
    body_sections=body_sections,
    outline_entries=[
        {"id": "overview", "label": "Overview"},
        {"id": "results", "label": "Results"},
        {"id": "method", "label": "Method", "sub": [
            {"id": "tensor-inventory", "label": "Tensor inventory"},
            {"id": "loss-math", "label": "Loss math"},
        ]},
        {"id": "data", "label": "Data", "sub": [
            {"id": "pipeline-strip", "label": "Pipeline strip"},
        ]},
    ],
    needs_katex=True,
)

with open(os.path.join(OUT, "index.html"), "w") as f:
    f.write(html)
print(f"\nwrote {OUT}/index.html")
