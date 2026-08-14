"""
build_uvvox_report_page.py — "The corrected emission data (out_uv_voxel_74k)"
xgpage v2 editorial. Understanding Dongchen's uv_voxel_pipeline output and how
it differs from the old somage -> glb -> o-voxel path.

Data: /cs/3dlg-jupiter-project/lightgen/uv_voxel_pipeline/out_uv_voxel_74k
Doc of record: dongchen-yang/lightgen data_processing/uv_voxel_pipeline/GENERATION.md
Emission validated live (atlas.npz emission_color) on teddy/creature/sword.
Runs on the standalone xgpage package (uv pip install -e ~/studio/xgpage).
"""
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "vis_data", "uvvox_report_html")
IMG = os.path.join(OUT, "img")
os.makedirs(IMG, exist_ok=True)
import xgpage as lp

ASSETS_REL = "/projects/omages/yanxg/lightgen/assets"
ASSETS_DIR = os.path.join(ROOT, "..", "web", "assets")

N_SHAPES = "72,374"
# emissive VOXEL fraction (256^3, emissive iff emission > 1/255 = any authored emission)
# validated live on out_uv_voxel_74k/<sid>/emission_voxels_256/<sid>.vxz
VOX = [
    ("9a4b8c391dee410f860e62bdb5c80550", "Teddy bear", "emits nothing", 0.000),
    ("7d46d70a984e4accb7bdd0c53d2b4d7f", "Red creature", "glows", 0.765),
    ("0e0e2c2000fc4f01944f87af441bd1f3", "Sword", "flames + runes", 0.375),
    ("f57e883babaa4369aa0ecf09bbea04b0", "Sign panel", "dim glow", 0.604),
    ("e1d56eefe58f49fc81a68e71c9a2fc57", "Factor-only", "no emission", 0.000),
]

# ---------------------------------------------------------------- hero
hero = lp.hero_header(
    "lightgen · data · emission ground truth",
    "The corrected emission data is ready: 72k shapes, direct from GLB",
    dek_html=(
        "Dongchen&rsquo;s <code>uv_voxel_pipeline</code> has processed the corpus into "
        "aligned UV atlases and voxels with <b>correct per-voxel emission</b> baked in "
        "(the o-voxel mipmap fix, plus factor-only materials the old bake dropped). This "
        "is the direct-GLB approach we validated, productionized. It replaces the "
        "somage&nbsp;&rarr;&nbsp;glb&nbsp;&rarr;&nbsp;o-voxel path for good, and it is the "
        "ground truth our emissive fine-tune should now train on."),
    stats=[
        (N_SHAPES, "shapes processed (of 74,503)"),
        ("512&sup2; + 256&sup3;", "UV atlas + voxel grid"),
        ("&ge; 0.99", "atlas&ndash;voxel alignment (fp_voxel)"),
        ("fixed", "emission (teddy 96%&rarr;0)"),
    ],
    toc=[
        ("what", "What the data is"),
        ("correct", "The emission is correct"),
        ("how", "How it is made"),
        ("vs", "vs. somage → glb → o-voxel"),
        ("us", "What it means for us"),
    ],
)

# ---------------------------------------------------------------- 01 what
artifact_table = """
<table class="art-table">
<thead><tr><th>artifact</th><th>size</th><th>what it holds</th></tr></thead>
<tbody>
<tr><td><code>atlas.npz</code></td><td>5.5 MB</td><td>512&sup2; UV atlas: color, <b>emission_color</b>, metal, rough, position, objnormal, occupancy</td></tr>
<tr><td><code>&lt;sha&gt;.coords.npz</code></td><td>2.8 MB</td><td>voxel coordinates</td></tr>
<tr><td><code>emission_voxels_256/</code></td><td>0.37 MB</td><td>per-voxel <b>emission</b> @ 256&sup3; (.vxz)</td></tr>
<tr><td><code>pbr_voxels_256/</code></td><td>0.37 MB</td><td>per-voxel PBR @ 256&sup3; (.vxz)</td></tr>
</tbody></table>"""
s_what = lp.section_v2("what", "01", "Four aligned artifacts per shape, keyed by Sketchfab ID",
    lp.prose(
        f"<code>out_uv_voxel_74k/</code> holds <b>{N_SHAPES} shapes</b> (of the pinned "
        "74,503 split; ~97%, the rest failed the quality gate). Each shape is a directory "
        "named by its SHA &mdash; and that <b>SHA is the Sketchfab model UID</b>, so "
        "<code>sketchfab.com/models/&lt;sha&gt;/embed</code> opens any shape with no "
        "mapping table. Per shape:")
    + artifact_table
    + lp.callout(
        "Why both a UV atlas and voxels? The project compares two baselines: <b>TEXGen</b> "
        "works in UV texture space (it consumes <code>atlas.npz</code>), while <b>TRELLIS.2 / "
        "SegviGen</b> &mdash; our emissive segmentation &mdash; works in voxel space (it "
        "consumes <code>emission_voxels_256</code>). <b>For us the voxels are the data; the "
        "voxel output is UV-free 3D.</b> The atlas is TEXGen&rsquo;s input. (UV is still used "
        "transiently inside the bake &mdash; to read a voxel&rsquo;s emission you sample the "
        "emissive texture, which lives in UV &mdash; but it is not attached to the output "
        "voxels.)", title="Two representations, for two baselines"),
)

# ---------------------------------------------------------------- 02 correct
vox_cells = ""
for sid, label, note, frac in VOX:
    badge = f"{frac*100:.0f}%"
    accent = " emis" if frac > 0.01 else ""
    vox_cells += (
        f'<figure class="vox-cell{accent}">'
        f'<img src="img/{sid}_vox.png" alt="{label} emission voxels" loading="lazy">'
        f'<figcaption><span class="vl">{label}</span>'
        f'<span class="vf">{badge} emissive</span></figcaption></figure>')
vox_grid = f'<div class="vox-grid">{vox_cells}</div>'

s_correct = lp.section_v2("correct", "02", "The emission is correct on the actual voxels",
    lp.prose(
        "These are the shipped 256&sup3; emission voxels (<code>emission_voxels_256/*.vxz</code>), "
        "rendered directly &mdash; <b>grey = surface, orange = emissive</b>: a voxel is labelled "
        "emissive if it carries <b>any authored emission</b> (value &gt; 1/255, dropping only the "
        "single darkest floor where encoding noise lives). This is the corrected binarization: "
        "the earlier 0.04 threshold discarded the large dim gradient real materials carry. Same "
        "shapes as our pilot:")
    + vox_grid
    + lp.prose(
        "<b>The orange lands on exactly the glowing parts, and nowhere else.</b> The teddy "
        "bear is entirely grey (0% of 166k voxels) &mdash; the broken voxelizer had painted "
        "it 96%. The creature&rsquo;s appendages, the sword&rsquo;s blade and runes, and the "
        "sign&rsquo;s strip carry real, localized emission; the factor-only sphere is "
        "correctly blank. This is the corrected signal, in the production voxels we would "
        "actually train on.")
    + lp.callout(
        "These are Dongchen&rsquo;s voxels read straight from the corpus, not our "
        "re-voxelization. Teddy 0%, glowers localized &mdash; the fix is in the data. "
        "(256&sup3; reads coarser than our 512&sup3; pilot panels; that is the corpus "
        "resolution, see &sect;5.)", title="Read from the shipped .vxz"),
)

# ---------------------------------------------------------------- 03 how
s_how = lp.section_v2("how", "03", "One aligned pass from the original GLB, no Blender",
    lp.prose(
        "The generator is Blender-free and runs one fresh process per shape "
        "(<code>load &rarr; atlas &rarr; voxelize &rarr; validate</code>, GPU, resumable, "
        "staging-then-promote). From the <b>original</b> TexVerse GLB it produces, in a "
        "single aligned pass: <code>xatlas</code> UV-unwraps the mesh, <code>nvdiffrast</code> "
        "bakes the material channels into the 512&sup2; atlas, and the fixed "
        "<code>o_voxel</code> voxelizes to 256&sup3;. Alignment is gated at "
        "<code>fp_voxel &ge; 0.99</code>.")
    + lp.callout(
        "Two corrections over the somage bake are deliberate: (1) the o-voxel emissive "
        "mipmap fix (the one-line bug), and (2) baking <b>factor &times; texture for every "
        "channel including constant factors with no texture</b> &mdash; the factor-only "
        "materials the somage bake used to drop as black atlases now carry their emission.",
        title="What it fixes"),
)

# ---------------------------------------------------------------- 04 vs
vs_html = """
<div class="vs-grid">
<div class="vs-col old"><div class="vs-h">somage &rarr; glb &rarr; o-voxel <span>(old, ours)</span></div>
<ul>
<li>glb &rarr; somage bake &rarr; rebuild glb &rarr; voxelize &mdash; <b>four hops</b></li>
<li>Blender in the somage creation</li>
<li>two independent voxelizations, <b>alignment not guaranteed</b> (pilot saw 1&ndash;10% overlap)</li>
<li>emission via 512&sup2; atlas; <b>factor-only dropped</b> (black); o-voxel mipmap bug</li>
</ul></div>
<div class="vs-col new"><div class="vs-h">uv_voxel_pipeline <span>(new, Dongchen)</span></div>
<ul>
<li>glb &rarr; atlas + voxels, <b>one pass</b></li>
<li><b>Blender-free</b> (xatlas + nvdiffrast + o_voxel)</li>
<li><b>aligned by construction</b> (fp_voxel &ge; 0.99)</li>
<li>emission correct: <b>fixed o-voxel + factor&times;texture</b>; somage-format drop-in</li>
</ul></div>
</div>"""
s_vs = lp.section_v2("vs", "04", "It replaces the somage detour end to end",
    lp.prose(
        "The old path we had been on took the original asset through a lossy somage bake and "
        "a mesh rebuild before voxelizing. The new pipeline goes straight from the GLB:")
    + vs_html
    + lp.prose(
        "In one line: the <b>direct-GLB approach we validated, productionized</b> &mdash; no "
        "somage detour, no double voxelization, no dropped factor-only emission, no mipmap "
        "bug, both representations aligned in a single pass."),
)

# ---------------------------------------------------------------- 05 us
s_us = lp.section_v2("us", "05", "What it means for the emissive fine-tune",
    lp.prose(
        "<b>This is the ground truth to train on.</b> Our own <code>build_dataset_direct.py</code> "
        "and the pilot machinery are superseded &mdash; we consume this corpus. The per-voxel "
        "emission is continuous, so for our binary emissive segmentation we threshold it into "
        "the white/black target, as before.")
    + lp.callout(
        "One thing to reconcile first: resolution. This data is <b>256&sup3;</b>; our SegviGen "
        "emissive fine-tune ran at <b>512&sup3;</b> (glb_to_vxz grid 512 &rarr; 32&sup3; latent). "
        "Before restarting the fine-tune we need to confirm whether SegviGen takes 256&sup3; "
        "emission voxels directly, or whether we need a 512&sup3; target from this pipeline. "
        "That is the first concrete question.",
        title="Open: 256&sup3; here vs. 512&sup3; before", warn=True),
)

extra_css = """
.vs-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:6px auto;
  max-width:var(--breakout-max,972px);}
.vs-col{border-radius:10px;padding:12px 16px;min-width:0;}
.vs-col.old{background:color-mix(in srgb,var(--ink) 5%,transparent);}
.vs-col.new{background:color-mix(in srgb,var(--accent) 8%,transparent);}
.vs-h{font-weight:650;margin-bottom:6px;} .vs-h span{font-weight:400;opacity:.6;font-size:.85em;}
.vs-col ul{margin:0;padding-left:18px;} .vs-col li{margin:4px 0;font-size:.92rem;line-height:1.5;}
@media(max-width:640px){.vs-grid{grid-template-columns:1fr;}}
.art-table{width:100%;border-collapse:collapse;margin:6px auto;font-size:.92rem;}
.art-table th{text-align:left;font:600 .72rem/1 ui-monospace,monospace;letter-spacing:.06em;
  text-transform:uppercase;opacity:.6;padding:6px 10px;border-bottom:1px solid color-mix(in srgb,var(--ink) 18%,transparent);}
.art-table td{padding:8px 10px;border-bottom:1px solid color-mix(in srgb,var(--ink) 8%,transparent);vertical-align:top;}
.art-table td:nth-child(2){white-space:nowrap;opacity:.7;}
.vox-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;
  margin:8px auto;max-width:var(--breakout-max,972px);}
.vox-cell{margin:0;min-width:0;border-radius:10px;overflow:hidden;
  background:#111113;border:1px solid color-mix(in srgb,var(--ink) 12%,transparent);}
.vox-cell img{display:block;width:100%;aspect-ratio:1;object-fit:cover;}
.vox-cell figcaption{display:flex;justify-content:space-between;align-items:baseline;gap:6px;
  padding:7px 10px;background:#111113;color:#e8e8ea;}
.vox-cell .vl{font-weight:600;font-size:.82rem;}
.vox-cell .vf{font:600 .72rem/1 ui-monospace,monospace;opacity:.6;white-space:nowrap;}
.vox-cell.emis .vf{color:#e8863f;opacity:1;}
"""

html = lp.page(
    title="The corrected emission data (uv_voxel_74k) — lightgen",
    header_html=hero,
    body_sections=[s_what, s_correct, s_how, s_vs, s_us],
    theme="v2", assets_rel=ASSETS_REL, assets_dir=ASSETS_DIR,
    extra_head=f"<style>{extra_css}</style>",
)
with open(os.path.join(OUT, "index.html"), "w") as f:
    f.write(html)
print("wrote", os.path.join(OUT, "index.html"))
