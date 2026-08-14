"""
build_uvvox_gallery_page.py — "Emission examples: 700 from the corrected corpus"
A gallery of per-voxel emission renders (emission_voxels_256) from Dongchen's
uv_voxel_pipeline. Grey = surface, orange = emissive (lum>0.04). Sorted by
emissive fraction, filterable by glow amount, each thumbnail links to Sketchfab
(sid == model UID). Renders: pipelineworker software voxel rasterizer.
Runs on the standalone xgpage package.
"""
import os, json
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "vis_data", "uvvox_gallery_html")
IMG = os.path.join(OUT, "img")
os.makedirs(IMG, exist_ok=True)
import xgpage as lp

ASSETS_REL = "/projects/omages/yanxg/lightgen/assets"
ASSETS_DIR = os.path.join(ROOT, "..", "web", "assets")
GAL = os.path.join(ROOT, "direct_pilot", "uvvox_gallery")

man = json.load(open(os.path.join(GAL, "gallery_manifest.json")))
if isinstance(man, dict):
    man = man.get("rows") or man.get("data") or list(man.values())[0]
man = sorted(man, key=lambda r: -float(r["emissive_frac"]))
n = len(man)
n_glow = sum(1 for r in man if r["stratum"] == "glow")
n_tiny = sum(1 for r in man if r["stratum"] == "tiny")
n_zero = sum(1 for r in man if r["stratum"] == "zero")

# ---------------------------------------------------------------- hero
hero = lp.hero_header(
    "lightgen · data · emission gallery",
    f"{n} emission examples from the corrected corpus",
    dek_html=(
        "Per-voxel emission from Dongchen&rsquo;s <code>uv_voxel_pipeline</code> "
        "(<code>emission_voxels_256</code>), rendered directly: <b>grey = surface, orange = "
        "emissive</b> &mdash; a voxel counts as emissive if it carries <b>any authored emission</b> "
        "(value &gt; 1/255). A glow-weighted sample across the corpus, sorted brightest-first, "
        "filter by amount below. Every thumbnail links to its source model on Sketchfab."),
    stats=[
        (f"{n}", "shapes"),
        (f"{n_glow}", "clearly glowing (&gt;5%)"),
        (f"{n_tiny}", "tiny glow (0&ndash;5%)"),
        (f"{n_zero}", "zero glow (correctly grey)"),
    ],
    toc=[],
)

# ---------------------------------------------------------------- filter + grid
filt = (
    '<div class="galfilt" role="tablist">'
    f'<button class="gf on" data-f="all">All &middot; {n}</button>'
    f'<button class="gf" data-f="glow">Glowing &middot; {n_glow}</button>'
    f'<button class="gf" data-f="tiny">Tiny &middot; {n_tiny}</button>'
    f'<button class="gf" data-f="zero">Zero &middot; {n_zero}</button>'
    '</div>')

cells = []
for r in man:
    sid = r["sid"]; frac = float(r["emissive_frac"]); strat = r["stratum"]
    pct = f"{frac*100:.0f}%" if frac >= 0.005 else "0%"
    emis_cls = " e" if frac > 0.01 else ""
    cells.append(
        f'<a class="gcell{emis_cls}" data-s="{strat}" '
        f'href="https://sketchfab.com/models/{sid}" target="_blank" rel="noopener" '
        f'title="{sid} · {pct} emissive · open on Sketchfab">'
        f'<img loading="lazy" src="img/{sid}.png" alt="{pct} emissive" width="300" height="300">'
        f'<span class="gfrac">{pct}</span></a>')
grid = f'<div class="galgrid" id="galgrid">{"".join(cells)}</div>'

intro = lp.prose(
    "The orange lands on exactly the emissive regions and nowhere else &mdash; filaments, "
    "screens, runes, lava, glowing trim &mdash; while non-emissive shapes stay fully grey. "
    "This is the corrected signal (the o-voxel mipmap fix + factor&times;texture baking) "
    "across a real slice of the corpus, not a hand-picked few. Click any shape to see the "
    "original on Sketchfab.")

body = f'<section class="xg2 galsec">{intro}{filt}{grid}</section>'

# ---------------------------------------------------------------- css + js
extra_css = """
.galsec{max-width:1180px;margin:0 auto;padding:0 clamp(16px,4vw,40px);}
.galfilt{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 16px;position:sticky;top:0;
  background:var(--bg);padding:10px 0;z-index:5;}
.gf{font:600 .82rem/1 inherit;padding:7px 14px;border-radius:999px;cursor:pointer;
  border:1px solid color-mix(in srgb,var(--ink) 20%,transparent);background:transparent;
  color:var(--ink);opacity:.7;}
.gf.on{background:var(--accent);border-color:var(--accent);color:#fff;opacity:1;}
.galgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px;}
.gcell{position:relative;display:block;border-radius:9px;overflow:hidden;background:#16161a;
  border:1px solid color-mix(in srgb,var(--ink) 12%,transparent);text-decoration:none;
  aspect-ratio:1;transition:transform .12s,box-shadow .12s;}
.gcell:hover{transform:translateY(-2px);box-shadow:0 6px 18px rgba(0,0,0,.35);
  border-color:var(--accent);}
.gcell img{display:block;width:100%;height:100%;object-fit:cover;}
.gfrac{position:absolute;right:6px;bottom:6px;font:600 .7rem/1 ui-monospace,monospace;
  padding:3px 6px;border-radius:5px;background:rgba(0,0,0,.55);color:#cfcfd2;}
.gcell.e .gfrac{color:#f0994e;}
.galgrid.f-glow .gcell:not([data-s="glow"]),
.galgrid.f-tiny .gcell:not([data-s="tiny"]),
.galgrid.f-zero .gcell:not([data-s="zero"]){display:none;}
"""
extra_js = """
<script>
(function(){
 var g=document.getElementById('galgrid');
 document.querySelectorAll('.gf').forEach(function(b){
   b.addEventListener('click',function(){
     document.querySelectorAll('.gf').forEach(function(x){x.classList.remove('on');});
     b.classList.add('on');
     g.className='galgrid'+(b.dataset.f==='all'?'':' f-'+b.dataset.f);
   });
 });
})();
</script>"""

html = lp.page(
    title=f"{n} emission examples (uv_voxel corpus) — lightgen",
    header_html=hero,
    body_sections=[body],
    theme="v2", assets_rel=ASSETS_REL, assets_dir=ASSETS_DIR,
    extra_head=f"<style>{extra_css}</style>",
    extra_body_end=extra_js,
)
with open(os.path.join(OUT, "index.html"), "w") as f:
    f.write(html)
print(f"wrote {OUT}/index.html  ({n} cells)")
