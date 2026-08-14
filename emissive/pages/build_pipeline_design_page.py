"""
build_pipeline_design_page.py — "Direct-ovoxel emissive fine-tune — method & pipeline"
xgpage v2. A TIGHT, canonical method/pipeline reference (a 2-minute read), the shared
source-of-truth for the experiment design. LIVING page: update as decisions lock. Not a
results page (no statband). New URL: _preview/pipeline_design.
  /local-scratch2/xya120/studio/misc/lightgen/.venv_console/bin/python build_pipeline_design_page.py
"""
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "vis_data", "pipeline_design_html")
os.makedirs(OUT, exist_ok=True)
import xgpage as lp

ASSETS_REL = "/projects/omages/yanxg/lightgen/assets"
ASSETS_DIR = os.path.join(ROOT, "..", "web", "assets")
UPDATED = "2026-07-23"

# ================================================================ hero
hero = lp.hero_header(
    f"lightgen · method · pipeline &nbsp;·&nbsp; living, updated {UPDATED}",
    "Direct-ovoxel emissive fine-tune &mdash; method &amp; pipeline",
    dek_html=(
        "The canonical description of the emissive fine-tune, so we work from one mental model. "
        "<b>Goal:</b> fine-tune SegviGen <code>full_seg</code> (Trellis.2 slat flow, initialised "
        "from <code>full_seg.ckpt</code>) to predict a <b>per-voxel binary emissive mask</b>, "
        "trained <b>directly on Dongchen&rsquo;s corrected ovoxels</b> &mdash; not the old "
        "somage&nbsp;&rarr;&nbsp;GLB&nbsp;&rarr;&nbsp;re-voxelize round-trip. Labels come straight "
        "from the voxels. All three design decisions are now locked (below)."),
    stats=None,
    toc=[
        ("open", "Decisions (locked)"),
        ("pipeline", "The pipeline"),
        ("data", "Data"),
        ("filter", "Shape filter"),
        ("model", "Resolution"),
        ("loader", "Data loader"),
        ("eval", "Train &amp; eval"),
        ("dropped", "Superseded"),
    ],
)

# ================================================================ open decisions (prominent, top)
def decard(n, q, a, b):
    return (f'<div class="odec"><div class="od-n">{n}</div><div class="od-body">'
            f'<div class="od-q">{q}</div><div class="od-opts">{a}<span class="od-or">vs</span>{b}</div>'
            f'</div></div>')
def rz(n, title, text):
    return (f'<div class="rz-row"><span class="rz-check">&check;</span>'
            f'<div><b>{n}. {title}:</b> {text}</div></div>')
resolved = (
    rz("1", "Resolution &amp; encoder",
       "<b>upsample 256&sup3;&rarr;512&sup3;</b> to reuse the pretrained encoder. "
       "<span class=\"rz-why\">Keeps the <code>full_seg</code> pretrained init; 256&nbsp;=&nbsp;"
       "&frac12;&middot;512, a clean 2&times;. Native-256 (B) declined &mdash; no pretrained "
       "weights at that config.</span>")
    + rz("2", "Conditioning",
         "<b>zero-cond</b> (image conditioning fed as zeros). "
         "<span class=\"rz-why\">DINOv3 conditioning is deferred to the future; revisit "
         "later.</span>")
    + rz("3", "Emissive threshold",
         "<b>any nonzero (&gt; 0) emission voxel = emissive</b>. "
         "<span class=\"rz-why\">Not &gt; 1/255 &mdash; any authored emission at all counts as a "
         "positive label.</span>"))
s_open = lp.section_v2("open", "01",
    "All three decisions are locked",
    resolved
    + lp.prose(
        "The design is fully specified. This page stays as the canonical reference; the only "
        "number still pending is the exact ovoxel-native trainable count (&sect;4).")
)

# ================================================================ 02 pipeline (centerpiece)
def pstage(title, sub, cls=""):
    return (f'<div class="pstage {cls}"><div class="ps-t">{title}</div>'
            f'<div class="ps-s">{sub}</div></div>')
pipe = (
    '<div class="pipe">'
    + pstage("Dongchen ovoxels", "out_uv_voxel_74k &middot; 256&sup3;", "start")
    + '<div class="parrow">&rarr;</div>'
    + pstage("Shape filter", "drop zero-emission + albedo-lit")
    + '<div class="parrow">&rarr;</div>'
    + pstage("Data loader", "frame + geometry + resolution")
    + '<div class="parrow">&rarr;</div>'
    + pstage("full_seg fine-tune", "init full_seg.ckpt")
    + '<div class="parrow">&rarr;</div>'
    + pstage("Glow-stratified eval", "IoU vs baselines", "end")
    + '</div>')
s_pipeline = lp.section_v2("pipeline", "02",
    "One pass: corrected ovoxels in, emissive mask out",
    pipe
    + lp.prose(
        "No somage, no GLB rebuild, no second voxelization. The corrected ovoxels are the input "
        "and the label source; the model learns the binary emissive mask directly on them.")
)

# ================================================================ 03 data
s_data = lp.section_v2("data", "03",
    "Data: PBR voxels in, binarized emission voxels as target",
    '<table class="mtab"><tbody>'
    '<tr><th>Input</th><td><code>pbr_voxels_256</code> &mdash; base_color, metallic, roughness, '
    'alpha, at 256&sup3;.</td></tr>'
    '<tr><th>Target</th><td><code>emission_voxels_256</code>, <b>binarized &gt; 0</b> '
    '(any nonzero emission voxel = emissive). Labels straight from the voxels &mdash; no somage.</td></tr>'
    '<tr><th>Base corpus</th><td>~72k shapes that have Dongchen ovoxels.</td></tr>'
    '</tbody></table>')

# ================================================================ 04 shape filter
filter_flow = (
    '<div class="pipe">'
    + pstage("Base corpus", "~72k with ovoxels", "start")
    + '<div class="parrow">&rarr;</div>'
    + pstage("&minus; zero-emission", "no authored emission")
    + '<div class="parrow">&rarr;</div>'
    + pstage("&minus; albedo-lit", "whole-surface = albedo")
    + '<div class="parrow">&rarr;</div>'
    + pstage("Kept", "trainable set", "end")
    + '</div>')
s_filter = lp.section_v2("filter", "04",
    "Shape filter: one ovoxel-native pass",
    filter_flow
    + '<table class="mtab"><tbody>'
    '<tr><th>Exclude &mdash; zero</th><td>zero-emission shapes (nothing to learn).</td></tr>'
    '<tr><th>Exclude &mdash; albedo-lit</th><td><b>whole-surface emission = albedo</b>: the emission '
    'voxel &asymp; the base_color voxel over essentially the entire surface (a fullbright copy = '
    '<b>input == target leakage</b>). <span class="gloss">&ldquo;Albedo-lit&rdquo;: the emission is a '
    'copy of the albedo, so the shape is lit by its own albedo &mdash; more precise than '
    '&ldquo;self-lit,&rdquo; which a genuine emitter also is.</span> The test is whole-surface-only.</td></tr>'
    '<tr><th>Keep</th><td>everything else &mdash; real emitters, and <b>localized copy-emitters</b> '
    '(a copy on a small sub-mesh, e.g. a screen or a warning light). These are not whole-surface, '
    'so the albedo-lit test does not exclude them; they are simply kept.</td></tr>'
    '</tbody></table>'
    + lp.prose(
        "Everything is <b>ovoxel-native</b> &mdash; computed on <code>emission_voxels</code> vs. "
        "<code>pbr_voxels</code>, no somage. The exact kept (trainable) count is pending a recompute "
        "on the voxel base.")
)

# ================================================================ 05 model reconciliation (OPEN)
s_model = lp.section_v2("model", "05",
    "Resolution: upsample 256&sup3; &rarr; 512&sup3; to reuse the pretrained encoder",
    lp.callout(
        "The pretrained <code>full_seg</code> encoder is <b>f16, 512&sup3; surface &rarr; 32&sup3; "
        "latent</b>; Dongchen&rsquo;s ovoxels are <b>256&sup3;</b>. <b>Resolved (A):</b> upsample "
        "the ovoxels <b>256&sup3;&nbsp;&rarr;&nbsp;512&sup3;</b> and reuse the pretrained encoder "
        "unchanged &mdash; a clean 2&times; (256&nbsp;=&nbsp;&frac12;&middot;512, both divide 32), "
        "which <b>keeps the <code>full_seg</code> pretrained init</b>. Option&nbsp;B "
        "(fine-tune at native 256&sup3;, adapt the encoder f8&nbsp;&rarr;&nbsp;32&sup3;) was "
        "declined &mdash; there are <b>no pretrained weights</b> at that config.",
        title="&check; Resolved: upsample, keep the pretrained encoder")
)

# ================================================================ 06 data loader
s_loader = lp.section_v2("loader", "06",
    "Data loader: four steps to a training pair",
    '<div class="steps">'
    '<div class="stp"><span class="stp-n">1</span> read <code>pbr</code> + <code>emission</code> voxels</div>'
    '<div class="stp"><span class="stp-n">2</span> frame transform <code>dong = (x, z, 1&minus;y)</code></div>'
    '<div class="stp"><span class="stp-n">3</span> derive structure geometry (dual-vertex grid) from occupancy</div>'
    '<div class="stp"><span class="stp-n">4</span> upsample 256&sup3;&rarr;512&sup3;, then the pretrained f16 encoder &rarr; 32&sup3; latent; emission voxels as the target</div>'
    '</div>')

# ================================================================ 07 train / eval
s_eval = lp.section_v2("eval", "07",
    "Train &amp; eval: init from full_seg, glow-stratified IoU",
    '<table class="mtab"><tbody>'
    '<tr><th>Init</th><td><code>full_seg.ckpt</code> (Trellis.2 slat flow).</td></tr>'
    '<tr><th>Conditioning</th><td><b>zero-cond</b> (locked) &mdash; image conditioning fed as '
    'zeros; DINOv3 conditioning is deferred to the future.</td></tr>'
    '<tr><th>Metric</th><td>glow-stratified <b>IoU</b>; tiny-glow is the known hard regime.</td></tr>'
    '<tr><th>Baselines</th><td>0.219 zero-shot oracle &middot; 0.259 DiffusionNet &middot; ~0.20 prior '
    'fine-tune plateau.</td></tr>'
    '</tbody></table>')

# ================================================================ 08 superseded
s_dropped = lp.section_v2("dropped", "08",
    "Superseded &mdash; do not drift back",
    '<div class="dropwrap"><div class="drop-banner">DROPPED &mdash; replaced by the direct-ovoxel path above</div>'
    '<table class="mtab drop"><tbody>'
    '<tr><th>Path A</th><td>somage emission label &rarr; recolor GLB &rarr; voxelize at 512&sup3;. '
    'The old round-trip; replaced by training directly on the ovoxels.</td></tr>'
    '<tr><th>somage labels</th><td>dropped <b>entirely</b> as the label source; the emission voxels '
    'are the labels now.</td></tr>'
    '<tr><th>glb-texture-image check</th><td>no longer the <b>operative</b> filter &mdash; kept only '
    'as a validation cross-check.</td></tr>'
    '</tbody></table></div>')

# ================================================================ css
extra_css = """
/* open decisions (prominent) */
.odec-wrap{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin:8px auto 4px;max-width:var(--breakout-max,972px);}
.odec-wrap.two{grid-template-columns:1fr 1fr;}
@media(max-width:760px){.odec-wrap,.odec-wrap.two{grid-template-columns:1fr;}}
/* resolved row */
.rz-row{display:flex;gap:12px;align-items:flex-start;margin:6px auto 4px;max-width:var(--breakout-max,972px);
  border-radius:12px;padding:13px 16px;background:color-mix(in srgb,#4a9d6a 12%,transparent);
  border:1px solid color-mix(in srgb,#4a9d6a 42%,transparent);}
.rz-check{font-size:1.2rem;line-height:1.3;color:#3a8d5a;flex:0 0 auto;font-weight:700;}
.rz-why{opacity:.72;font-size:.9rem;}
@media(prefers-color-scheme:dark){.rz-check{color:#6cc48c;}}
:root[data-theme="dark"] .rz-check{color:#6cc48c;} :root[data-theme="light"] .rz-check{color:#3a8d5a;}
.odec{display:flex;gap:12px;border-radius:12px;padding:14px 16px;
  background:color-mix(in srgb,var(--accent) 8%,transparent);border:1px solid color-mix(in srgb,var(--accent) 42%,transparent);}
.od-n{font:700 1.3rem/1 var(--serif,Georgia,serif);color:var(--accent);flex:0 0 auto;}
.od-q{font-weight:700;font-size:.98rem;margin-bottom:6px;}
.od-opts{font-size:.86rem;line-height:1.5;opacity:.9;}
.od-or{display:inline-block;margin:0 6px;font:700 .66rem/1 ui-monospace,monospace;text-transform:uppercase;opacity:.5;}
/* pipeline diagram */
.pipe{display:flex;align-items:stretch;justify-content:center;gap:6px;flex-wrap:wrap;margin:10px auto;max-width:var(--breakout-max,972px);}
.pstage{flex:1 1 150px;min-width:132px;border-radius:11px;padding:14px 14px;text-align:center;
  background:color-mix(in srgb,var(--ink) 5%,transparent);border:1px solid color-mix(in srgb,var(--ink) 14%,transparent);}
.pstage.start{background:color-mix(in srgb,var(--ink) 7%,transparent);}
.pstage.end{background:color-mix(in srgb,var(--accent) 10%,transparent);border-color:color-mix(in srgb,var(--accent) 45%,transparent);}
.ps-t{font-weight:700;font-size:.96rem;} .pstage.end .ps-t{color:var(--accent);}
.ps-s{opacity:.62;font-size:.78rem;margin-top:4px;line-height:1.35;}
.parrow{display:flex;align-items:center;color:var(--accent);font-size:1.3rem;flex:0 0 auto;}
@media(max-width:720px){.pipe{flex-direction:column;}.parrow{transform:rotate(90deg);height:22px;}}
/* method tables */
.mtab{width:100%;border-collapse:collapse;margin:6px auto;max-width:var(--breakout-max,972px);font-size:.95rem;}
.mtab th{text-align:left;vertical-align:top;white-space:nowrap;padding:9px 14px 9px 0;width:1%;
  font:700 .8rem/1.4 ui-monospace,monospace;color:var(--accent);border-bottom:1px solid color-mix(in srgb,var(--ink) 10%,transparent);}
.mtab td{padding:9px 0;line-height:1.55;border-bottom:1px solid color-mix(in srgb,var(--ink) 8%,transparent);}
.mtab.drop th{color:var(--ink);opacity:.7;}
.gloss{opacity:.62;font-size:.9em;font-style:italic;}
/* loader steps */
.steps{display:flex;flex-direction:column;gap:8px;margin:8px auto;max-width:var(--breakout-max,972px);}
.stp{display:flex;align-items:baseline;gap:10px;font-size:.95rem;line-height:1.5;
  border-left:3px solid var(--accent);padding:7px 12px;background:color-mix(in srgb,var(--ink) 4%,transparent);border-radius:0 8px 8px 0;}
.stp-n{font:700 .8rem/1 ui-monospace,monospace;color:var(--accent);flex:0 0 auto;}
/* dropped */
.dropwrap{margin:6px auto;max-width:var(--breakout-max,972px);}
.drop-banner{font:700 .68rem/1 ui-monospace,monospace;letter-spacing:.05em;color:#b06;
  background:color-mix(in srgb,#c66 14%,transparent);border:1px solid color-mix(in srgb,#c66 40%,transparent);
  border-radius:7px;padding:7px 12px;margin-bottom:8px;text-align:center;}
.mtab.drop{opacity:.82;}
"""

html = lp.page(
    title="Direct-ovoxel emissive fine-tune — method & pipeline — lightgen",
    header_html=hero,
    body_sections=[s_open, s_pipeline, s_data, s_filter, s_model, s_loader, s_eval, s_dropped],
    theme="v2", assets_rel=ASSETS_REL, assets_dir=ASSETS_DIR,
    extra_head=f"<style>{extra_css}</style>",
)
with open(os.path.join(OUT, "index.html"), "w") as f:
    f.write(html)
print("wrote", os.path.join(OUT, "index.html"), f"({len(html)} bytes)")
