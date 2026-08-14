"""
build_emissive_filtering_page.py — "The filter is self-lit shading, not material type"
xgpage v2 editorial. FINAL (2026-07-23, 3rd pass): filter LOCKED (rescue all NA + all-copy
multi-material rule). Exact counts from the fixed full scan over all 80,735 shapes; the
"provisional" framing is removed. The emissive-data filter criterion is SELF-LIT / SHADELESS
shading (emission is a copy of albedo), which cross-cuts pbrType. New trainable = 48,628
(+23,271 vs the old pbrType rule). Old pbrType funnel shown SUPERSEDED.

Assets: detector sanity renders (direct_pilot/sanity_renders/{texcmp_flagged,
contact_metal_flagged,contact_na_kept}.png), NA voxel gallery + lit strip. Runs on the
standalone xgpage package:
  /local-scratch2/xya120/studio/misc/lightgen/.venv_console/bin/python build_emissive_filtering_page.py
"""
import os, json
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "vis_data", "emissive_filtering_html")
os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
import xgpage as lp

ASSETS_REL = "/projects/omages/yanxg/lightgen/assets"
ASSETS_DIR = os.path.join(ROOT, "..", "web", "assets")
A = json.load(open(os.path.join(OUT, "page_assets.json")))
SKF = "https://sketchfab.com/models/"
def cm(n): return f"{n:,}"

# ---- final numbers (fixed full scan, all 80,735) ----
N_PARQ = 80735
PARQ_METAL, PARQ_SPEC, PARQ_NA = 33091, 1815, 45829
SELF_LIT = 12946                         # 16.0%
SL_METAL, SL_SPEC, SL_NA = 2354, 480, 10112
LIT_METAL, LIT_SPEC, LIT_NA, LIT_ALL = 7.1, 26.4, 22.1, 16.0
NA_REAL = 100 - LIT_NA                    # 77.9%
LIT_METAL_BUG = 56.5                      # the retired bug number (kept for the methods story)
# funnel base = WORKING CORPUS (Path-A: emissive-glb INTERSECT somage-bake); 80,735 is the
# detector corpus only. Numbers from direct_pilot/funnel_reconciliation.json.
N_WORK = 73470                            # working corpus (funnel base)
ZERO_EMIT = 13057                         # 73,470 -> 60,413
NONZERO = 60413
SELF_LIT_DROP = 11785                     # 60,413 -> 48,628 (19.5% of nonzero)
SL_RATE_WORK = 19.5
N_NEW = 48628                             # pre-rescue trainable
RESCUE = 4538
RES_METAL, RES_SPEC, RES_NA = 1306, 113, 3119
TRAIN_FINAL = 53166                       # post coverage-rescue (final)
PBR_CLEAN, NA_RESCUE = 22833, 25795
NO_SOMAGE = 7265                          # emissive glbs without a somage bake
DONGCHEN_VOX = 72471
OLD_TRAIN, OLD_SL = 25357, 2524           # old pbrType rule on the working corpus
GAIN = TRAIN_FINAL - OLD_TRAIN            # +27,809

# ================================================================ hero
hero = lp.hero_header(
    "lightgen · data · emissive filtering",
    "The filter is self-lit shading, not material type",
    dek_html=(
        "We had been filtering the emissive corpus by <b>pbrType</b> &mdash; drop the shadeless "
        "<code>&lt;NA&gt;</code> class, keep metalness + specular. That criterion is <b>wrong "
        "both ways.</b> The property that actually decides whether a shape is usable is its "
        "<b>shading</b>: is it self-lit (its &lsquo;emission&rsquo; is just a copy of its albedo, "
        "a fullbright trick) or a genuine localized emitter? That distinction <b>cross-cuts "
        "pbrType</b> &mdash; contamination inside the pbr set, real emitters inside "
        "<code>&lt;NA&gt;</code>. There is no shading flag to read; it is detected from the GLB. "
        "The fixed full-corpus scan is complete and the filter is locked: the numbers below are "
        "final."),
    stats=[
        ("shading", "the real filter"),
        (cm(TRAIN_FINAL), f"trainable (+{cm(GAIN)})"),
        (f"~{NA_REAL:.0f}%", "of &lt;NA&gt; are real emitters"),
        (f"{LIT_METAL}%", "metalness self-lit (not 56%)"),
    ],
    toc=[
        ("reframe", "The reframe"),
        ("noflag", "No shading flag"),
        ("detector", "The detector"),
        ("bug", "A methods lesson"),
        ("rates", "Self-lit rates"),
        ("newfilter", "The new filter &amp; set"),
        ("naclass", "The &lt;NA&gt; class"),
        ("super", "Superseded funnel"),
    ],
)

# ================================================================ 01 reframe
def rate_bar(pct, lab, sub, cls=""):
    return (f'<div class="rbar {cls}"><div class="rb-track"><div class="rb-fill" '
            f'style="width:{pct}%"></div></div><div class="rb-lab"><b>{pct}%</b> {lab}'
            f'<span>{sub}</span></div></div>')
reframe_vis = (
    '<div class="twoway">'
    '<div class="tw-col"><div class="tw-h">inside the <b>kept</b> pbr set</div>'
    + rate_bar(LIT_METAL, "of metalness is self-lit", f"contamination kept ({cm(SL_METAL)} shapes)", "bad")
    + '</div>'
    '<div class="tw-col"><div class="tw-h">inside the <b>dropped</b> &lt;NA&gt; set</div>'
    + rate_bar(NA_REAL, "of &lt;NA&gt; are real emitters", "usable data dropped", "good")
    + '</div></div>')
s_reframe = lp.section_v2("reframe", "01",
    "The criterion is self-lit shading, and it cross-cuts pbrType",
    lp.prose(
        "A <b>self-lit</b> (fullbright / shadeless) asset copies its color texture into the "
        "emission channel so it renders at full brightness with no lighting. Its emission is "
        "not authored light &mdash; it is the albedo, wearing the emission slot. A <b>real "
        "emitter</b> instead has a distinct, localized emission map (a filament, a screen, a "
        "rune). Only real emitters are usable emissive training data. The mistake was using "
        "<b>pbrType</b> as a proxy for that distinction:")
    + reframe_vis
    + lp.prose(
        "Dropping the whole <code>&lt;NA&gt;</code> class threw away a large pool of real "
        "emitters, while keeping self-lit contamination inside the metalness set. The right "
        "split is by <b>shading</b>, detected per shape, regardless of material type. That "
        "criterion is now locked, and the pbrType two-filter funnel (&sect;8) is superseded.")
)

# ================================================================ 02 no flag
s_noflag = lp.section_v2("noflag", "02",
    "There is no shading flag to read &mdash; it must be detected",
    lp.prose(
        "The owner suspected the source data might carry a lit/unlit flag. It does not. "
        "<code>pbrType</code> has only three values &mdash; <b>metalness, specular, "
        "&lt;NA&gt;</b> &mdash; in <i>both</i> the 80,735-row emissive parquet and the full "
        "<b>858,669-row</b> TexVerse manifest. There is no &lsquo;lit&rsquo;/&lsquo;unlit&rsquo;/"
        "&lsquo;shadeless&rsquo; value and no shading column anywhere, and the GLBs do not use "
        "the <code>KHR_materials_unlit</code> extension.")
    + lp.callout(
        "Fullbright is authored <b>by hand</b>: the artist copies the base-color image into the "
        "<code>emissiveTexture</code> so the object self-lights. Nothing in the metadata records "
        "that they did. So shading cannot be read &mdash; it has to be <b>detected</b> from the "
        "GLB&rsquo;s own textures.",
        title="Fullbright is a hand-authored convention, not a flag")
)

# ================================================================ 03 detector
s_detector = lp.section_v2("detector", "03",
    "The detector: is the emissive texture a copy of the albedo?",
    lp.prose(
        "A material is <b>self-lit</b> when its <code>emissiveTexture</code> is the same image as "
        "its <code>baseColorTexture</code>. A shape is self-lit when <b>all</b> its emitting "
        "materials are copies. The detector is a structural GLB check, per material:")
    + '<div class="deteqn">'
      '<div class="de-row"><span class="de-k">(a) same image</span> the emissive and base-color '
      'textures point at the <b>same glTF image index</b> &mdash; an exact copy, unambiguous.</div>'
      '<div class="de-row"><span class="de-k">(b) content fit</span> different indices, but the '
      'emissive is a scaled copy of the albedo: <b>corr(emission, albedo) &gt; 0.9</b> AND '
      'scale-fit residual <b>&lt; 0.05</b> (emission &asymp; k&middot;albedo).</div>'
      '</div>'
    + lp.prose(
        "Both conditions are needed &mdash; (b) alone has a failure mode, which is the next "
        "section.")
)

# ================================================================ 04 bug vignette
bug_fig = lp.fig("img/detect/texcmp_flagged.png",
    caption_html=(
        "<b>Six shapes the first detector called &lsquo;copy&rsquo; &mdash; every one is a real "
        "emitter.</b> Columns: <code>baseColorTexture</code> | <code>emissiveTexture</code> | "
        "|diff|&times;4. The base color is a full atlas; the emission is <b>mostly black with a "
        "few localized glows</b> (yellow lamps, blue screens, filament strips) &mdash; plainly "
        "<i>not</i> a copy of the albedo. Yet the scale-fit residual reads &asymp;0 (headers), "
        "so residual-alone flagged them. The diff column (bright = differs) shows how far the "
        "two images actually are."),
    native_px=768, content="photo")
s_bug = lp.section_v2("bug", "04",
    "A methods lesson: the first detector reported 56% and was wrong",
    lp.prose(
        f"The first build used the residual test <i>alone</i> (residual &lt; 0.05) and reported "
        f"<b>{LIT_METAL_BUG}% of metalness as fullbright</b>. That was a bug. A near-black real "
        f"emission map is fit by <b>k&nbsp;&asymp;&nbsp;0</b> in emission&nbsp;&asymp;&nbsp;"
        f"k&middot;albedo, driving the residual to &asymp;0 &mdash; a <b>k&rarr;0 degeneracy</b> "
        f"that reads &lsquo;copy&rsquo; for any dim emitter.")
    + bug_fig
    + lp.prose(
        "A ground-truth image comparison (base-color vs. emissive texture) exposed it: all six "
        "ambiguous flagged shapes were real localized emitters with distinct black emission maps "
        "&mdash; <b>zero</b> true copies. The fix adds the correlation gate "
        "(<b>corr&nbsp;&gt;&nbsp;0.9</b>): a real emitter&rsquo;s emission is uncorrelated with "
        "its albedo, so it no longer passes.")
    + lp.callout(
        f"With the fixed detector, the old pbrType-kept set turns out to be only <b>~10% "
        f"self-lit</b> ({cm(OLD_SL)} shapes), never {LIT_METAL_BUG}%. The bug scare is fully "
        f"retired. The lasting lesson: an emit-only render cannot tell a dark-albedo fullbright "
        f"asset from a genuine faint emitter &mdash; both look dim. The <b>texture images are "
        f"the ground truth</b>.",
        title="The kept set was ~10% self-lit, not 56%")
)

# ================================================================ 05 rates
rate_rows = [
    {"label": "metalness", "value": LIT_METAL, "display": f"{LIT_METAL}% &middot; {cm(SL_METAL)}",
     "tip": f"metalness self-lit: {LIT_METAL}% ({cm(SL_METAL)} of {cm(PARQ_METAL)})"},
    {"label": "specular", "value": LIT_SPEC, "display": f"{LIT_SPEC}% &middot; {cm(SL_SPEC)}",
     "tip": f"specular self-lit: {LIT_SPEC}% ({cm(SL_SPEC)} of {cm(PARQ_SPEC)})"},
    {"label": "&lt;NA&gt;", "value": LIT_NA, "display": f"{LIT_NA}% &middot; {cm(SL_NA)}",
     "tip": f"<NA> self-lit: {LIT_NA}% ({cm(SL_NA)} of {cm(PARQ_NA)}) -> {NA_REAL:.0f}% real emitters"},
    {"label": "overall", "value": LIT_ALL, "display": f"{LIT_ALL}% &middot; {cm(SELF_LIT)}",
     "tip": f"overall self-lit: {LIT_ALL}% ({cm(SELF_LIT)} of {cm(N_PARQ)})"},
]
rates_chart = lp.hbar_chart(rate_rows, title="self-lit rate by pbrType (detector corpus, all 80,735 GLBs)",
    label_w=96,
    note=(f"<b>Self-lit is a minority everywhere, and it cross-cuts pbrType.</b> Metalness is "
          f"only {LIT_METAL}% self-lit ({cm(SL_METAL)} shapes) &mdash; the pbr set was mostly "
          f"clean already; and <code>&lt;NA&gt;</code> is only {LIT_NA}% self-lit, so "
          f"<b>~{NA_REAL:.0f}% of it ({cm(PARQ_NA - SL_NA)} shapes) are real emitters</b> the "
          f"old rule discarded. {cm(SELF_LIT)} shapes ({LIT_ALL}%) are self-lit in total."))
s_rates = lp.section_v2("rates", "05",
    "Self-lit is a minority in every class",
    rates_chart
    + lp.prose(
        f"The headline is <code>&lt;NA&gt;</code>: at {LIT_NA}% self-lit, roughly <b>four in "
        f"five</b> of the shapes the old filter dropped are real emitters. And metalness, once "
        f"the {LIT_METAL_BUG}% bug is removed, is only {LIT_METAL}% self-lit &mdash; the kept pbr "
        f"set was never badly contaminated.")
)

# ================================================================ 06 new filter + set
def fnode(n, label, sub, cls=""):
    return (f'<div class="fnode {cls}"><div class="fn-n">{cm(n)}</div>'
            f'<div class="fn-l">{label}</div><div class="fn-s">{sub}</div></div>')
new_funnel = (
    '<div class="efun efun4">'
    + fnode(N_WORK, "working corpus", "emissive-glb &cap; somage bake", "start")
    + f'<div class="farrow">&rarr;<span class="fa-drop">&minus;{cm(ZERO_EMIT)}<br>no emission</span></div>'
    + fnode(NONZERO, "nonzero emission", "has authored emission", "mid")
    + f'<div class="farrow">&rarr;<span class="fa-drop">&minus;{cm(SELF_LIT_DROP)}<br>self-lit &middot; {SL_RATE_WORK}%</span></div>'
    + fnode(N_NEW, "real emitters", "pre-rescue", "mid")
    + f'<div class="farrow">&rarr;<span class="fa-add">+{cm(RESCUE)}<br>rescued</span></div>'
    + fnode(TRAIN_FINAL, "trainable set", "final", "end")
    + '</div>'
    '<div class="newcomp"><b>' + cm(TRAIN_FINAL) + '</b> = ' + cm(N_NEW) +
    ' (self-lit filter) &nbsp;+&nbsp; ' + cm(RESCUE) +
    ' localized copy-emitters (coverage guard) &nbsp;&middot;&nbsp; <span class="gain">+' + cm(GAIN) +
    '</span> vs the old pbrType rule (' + cm(OLD_TRAIN) + ')</div>')

metal_fig = lp.fig("img/detect/contact_metal_flagged.png",
    caption_html=(
        "<b>Self-lit shapes the filter drops: emission copies the whole albedo.</b> Each pair is "
        "the lit appearance and its emission-only render (<code>std | emit_only</code>). A rusty "
        "car, a skeleton figure, coral, a tablet, an Iron Man model &mdash; in every one the "
        "emit-only render reproduces the <i>entire textured surface</i>, because the emissive "
        "texture is a copy of the base color (n_copy = n_emit). That whole-surface signature "
        "&mdash; distinct from the localized emission of &sect;4&rsquo;s real emitters &mdash; is "
        "what the detector keys on."),
    native_px=1360, content="photo")
na_kept_fig = lp.fig("img/detect/contact_na_kept.png",
    caption_html=(
        "<b>&lt;NA&gt; shapes the filter rescues: real localized emitters.</b> A "
        "chandelier&rsquo;s candle flames, a character&rsquo;s glowing visor, lava veins, a "
        "car&rsquo;s lights &mdash; ordinary shaded objects with a distinct, <i>localized</i> "
        f"emitter (emission &ne; albedo). These are the kind of shape the old pbrType rule "
        f"discarded and the new filter keeps ({cm(NA_RESCUE)} rescued in all)."),
    native_px=1360, content="photo")

s_newfilter = lp.section_v2("newfilter", "06",
    f"The new filter: {cm(TRAIN_FINAL)} trainable shapes, +{cm(GAIN)}",
    lp.prose(
        "The locked filter, in shading terms and independent of pbrType: <b>drop a shape if it "
        "is self-lit</b> (all its emitting materials are albedo copies); <b>keep it if it has "
        "nonzero emission and at least one real (non-copy) emitter.</b> The trainable funnel runs "
        f"over the <b>working corpus</b> ({cm(N_WORK)}), and a coverage guard then rescues "
        f"{cm(RESCUE)} localized copy-emitters (see the detector report):")
    + new_funnel
    + lp.callout(
        f"<b>{cm(N_PARQ)}</b> = the detector corpus (all emissive-declared GLBs; 16.0% read "
        f"self-lit corpus-wide). <b>{cm(N_WORK)}</b> = the Path-A <b>working corpus</b> "
        f"&mdash; those that also have a somage bake (the label source); the {cm(NO_SOMAGE)} "
        f"without a bake cannot be Path-A-trained, so the trainable funnel starts at "
        f"{cm(N_WORK)}, not {cm(N_PARQ)}. Within the working nonzero corpus, {SL_RATE_WORK}% "
        f"are self-lit.",
        title="Detector corpus (80,735) vs. working corpus (73,470)")
    + '<div class="nf-grid">'
      f'<div class="nf-card"><div class="nf-n">{cm(SL_METAL)}</div><div class="nf-l">metalness '
      f'dropped as self-lit ({LIT_METAL}%)</div><div class="nf-s">the pbr keep-set was mostly '
      'clean; little changes here</div></div>'
      f'<div class="nf-card win"><div class="nf-n">{cm(NA_RESCUE)}</div><div class="nf-l">real '
      'emitters rescued from <code>&lt;NA&gt;</code></div><div class="nf-s">the big win: usable '
      'data the old rule discarded</div></div>'
      '</div>'
    + lp.prose(
        "The two ends of the decision, from the fixed detector &mdash; self-lit copies the filter "
        "drops, and real emitters it rescues:")
    + metal_fig
    + na_kept_fig
)

# ================================================================ 07 the <NA> class
NG = A["nagal"]; nng = len(NG)
def nband(f): return "hi" if f >= 0.5 else "mid" if f >= 0.05 else "lo"
nHi = sum(1 for g in NG if nband(g["frac"]) == "hi")
nMid = sum(1 for g in NG if nband(g["frac"]) == "mid")
nLo = sum(1 for g in NG if nband(g["frac"]) == "lo")
nafilt = ('<div class="galfilt" data-target="nagalgrid" role="tablist">'
    f'<button class="gf on" data-f="all">All &middot; {nng}</button>'
    f'<button class="gf" data-f="hi">&ge;50% emissive &middot; {nHi}</button>'
    f'<button class="gf" data-f="mid">5&ndash;50% &middot; {nMid}</button>'
    f'<button class="gf" data-f="lo">&lt;5% &middot; {nLo}</button>'
    '</div>')
nacells = []
for g in NG:
    s = g["sid"]; f = g["frac"]; b = nband(f); p = f"{f*100:.0f}%" if f >= 0.005 else "0%"
    nacells.append(
        f'<a class="gcell" data-b="{b}" href="{SKF}{s}" target="_blank" rel="noopener" '
        f'title="{s} · {p} emissive · pbrType &lt;NA&gt; · open on Sketchfab">'
        f'<img loading="lazy" src="img/nagal/{s}.png" alt="{p} emissive" width="300" height="300">'
        f'<span class="gfrac">{p}</span></a>')
nagrid_full = f'<div class="galgrid" id="nagalgrid">{"".join(nacells)}</div>'
LT = A["nalit"]
lit_cells = "".join(
    f'<a class="litcell" href="{SKF}{r["sid"]}" target="_blank" rel="noopener" '
    f'title="{r["sid"]} · {r["frac"]*100:.1f}% emissive · pbrType &lt;NA&gt; · open on Sketchfab">'
    f'<img loading="lazy" src="img/nalit/{r["sid"]}.png" alt="lit appearance">'
    f'<span>{r["frac"]*100:.1f}%</span></a>' for r in LT)
lit_grid = f'<div class="litgrid">{lit_cells}</div>'
s_naclass = lp.section_v2("naclass", "07",
    "The &lt;NA&gt; class up close: one pbrType label over a mixed bag",
    lp.prose(
        f"Here are 150 of the {cm(PARQ_NA)} <code>&lt;NA&gt;</code> shapes, their emission voxels "
        f"rendered directly (<b>grey = surface, orange = emissive</b>), sorted brightest-first. "
        f"The top is unmistakable fullbright &mdash; whole surfaces orange &mdash; but it fades "
        f"to ordinary grey objects at the bottom. pbrType gave this entire spectrum one label; "
        f"the shading detector is what separates the ~{LIT_NA:.0f}% self-lit from the "
        f"~{NA_REAL:.0f}% real emitters.")
    + nafilt + nagrid_full
    + lp.prose(
        "The low-emission shapes at the bottom, rendered with normal lighting &mdash; ordinary, "
        "usable content the old pbrType rule discarded, most of it now rescued:")
    + lit_grid
    + lp.chartnote(
        "Planets, characters, furniture, vehicles, signs (each reads under 5% emissive; frac "
        "shown). Most are shaded objects with a small emitter or none &mdash; not fullbright.")
)

# ================================================================ 08 superseded funnel
sup_flow = (
    '<div class="efun">'
    + fnode(N_WORK, "working corpus", "same base", "start")
    + '<div class="farrow">&rarr;<span class="fa-drop">drop &lt;NA&gt;<br>+ no-glow</span></div>'
    + fnode(OLD_TRAIN, "old trainable", "metalness + specular, superseded", "end")
    + '</div>')
s_super = lp.section_v2("super", "08",
    "Superseded: the pbrType two-filter funnel",
    '<div class="superwrap"><div class="super-banner">SUPERSEDED &mdash; replaced by the '
    'shading detector (&sect;3&ndash;6)</div>'
    + sup_flow + '</div>'
    + lp.prose(
        f"The earlier pipeline dropped the <code>&lt;NA&gt;</code> class by pbrType, then dropped "
        f"zero-emission survivors, yielding {cm(OLD_TRAIN)} &lsquo;trainable&rsquo; shapes. It "
        f"was the wrong split: it discarded ~{NA_REAL:.0f}% of <code>&lt;NA&gt;</code> that are "
        f"real emitters ({cm(NA_RESCUE)} now rescued), and of the {cm(OLD_TRAIN)} it kept, "
        f"{cm(OLD_SL)} (~10%) were actually self-lit contamination now dropped. The "
        f"<i>criterion</i> is retired; the shading detector replaces it, for a net "
        f"<b>+{cm(GAIN)}</b> ({cm(TRAIN_FINAL)} trainable). Both funnels run over the same "
        f"{cm(N_WORK)} working corpus; the zero-emission condition (per-channel value "
        f"&gt; 1/255) survives unchanged.")
)

# ================================================================ provenance
prov = lp.prose(
    "<b>Corpus.</b> Emission per shape from <code>uv_voxel_pipeline/out_uv_voxel_74k/</code> "
    "(<code>emission_voxels_256/&lt;sha&gt;.vxz</code>, 256&sup3;). Material metadata + pbrType "
    f"from <code>emissive_thumbnails_obj_ids_df.parquet</code> ({cm(N_PARQ)} rows = the detector "
    f"corpus). The trainable funnel base is the <b>Path-A working corpus</b> = {cm(N_WORK)} "
    f"(emissive GLB &cap; somage bake); {cm(NO_SOMAGE)} emissive GLBs lack a bake. "
    f"(Reconciliation: <code>direct_pilot/funnel_reconciliation.json</code>.)")
prov += lp.prose(
    "<b>No shading flag.</b> <code>pbrType</code> &isin; {metalness, specular, &lt;NA&gt;} in both "
    f"the {cm(N_PARQ)}-row parquet and the 858,669-row TexVerse manifest; no shading column; GLBs "
    "carry no <code>KHR_materials_unlit</code>.")
prov += lp.prose(
    "<b>Shading detector.</b> Per material, self-lit iff <code>emissiveTexture</code> == "
    "<code>baseColorTexture</code>: same glTF image index, OR "
    "<code>corr(emission,&nbsp;albedo)&nbsp;&gt;&nbsp;0.9</code> AND scale-fit "
    "residual&nbsp;&lt;&nbsp;0.05; a shape is self-lit iff all its emitting materials are copies. "
    f"The correlation gate was added after a residual-only build mis-flagged {LIT_METAL_BUG}% of "
    f"metalness (k&rarr;0 degeneracy). Full scan over {cm(N_PARQ)}: {cm(SELF_LIT)} self-lit "
    f"(metalness {cm(SL_METAL)}, specular {cm(SL_SPEC)}, &lt;NA&gt; {cm(SL_NA)}).")
prov += lp.prose(
    "<b>Filter.</b> Over the working corpus: drop {z} zero-emission, then drop {sl} self-lit "
    "({r}% of the {nz} nonzero), leaving <b>{pre}</b> ({pc} pbr-clean + {na} &lt;NA&gt;-rescued); "
    "a coverage guard then rescues {res} localized copy-emitters for a final "
    "<b>{fin}</b>, +{g} vs the old pbrType rule ({old}).".format(
        z=cm(ZERO_EMIT), sl=cm(SELF_LIT_DROP), r=SL_RATE_WORK, nz=cm(NONZERO), pre=cm(N_NEW),
        pc=cm(PBR_CLEAN), na=cm(NA_RESCUE), res=cm(RESCUE), fin=cm(TRAIN_FINAL), g=cm(GAIN),
        old=cm(OLD_TRAIN)))
prov += lp.prose(
    "<b>Renders.</b> Emission voxels: the project&rsquo;s software voxel rasterizer. Detector "
    "sanity sheets: <code>direct_pilot/sanity_renders/</code> "
    "(<code>texcmp_flagged</code>, <code>contact_metal_flagged</code>, <code>contact_na_kept</code>).")
s_prov = lp.section_v2("prov", "09", "How to reproduce", prov)

# ================================================================ css
extra_css = """
.twoway{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:10px auto 4px;max-width:var(--breakout-max,972px);}
@media(max-width:640px){.twoway{grid-template-columns:1fr;}}
.tw-col{border-radius:12px;padding:14px 16px;background:color-mix(in srgb,var(--ink) 4%,transparent);
  border:1px solid color-mix(in srgb,var(--ink) 12%,transparent);}
.tw-h{font-size:.9rem;margin-bottom:10px;opacity:.85;}
.rbar{margin:0;} .rb-track{height:16px;border-radius:8px;background:color-mix(in srgb,var(--ink) 10%,transparent);overflow:hidden;}
.rb-fill{height:100%;border-radius:8px;background:var(--accent);}
.rbar.bad .rb-fill{background:#c66;} .rbar.good .rb-fill{background:#4a9d6a;}
.rb-lab{font-size:.9rem;margin-top:7px;} .rb-lab b{font-size:1.05rem;}
.rb-lab span{display:block;opacity:.6;font-size:.8rem;margin-top:1px;}
.deteqn{margin:10px auto;max-width:var(--breakout-max,972px);display:flex;flex-direction:column;gap:8px;}
.de-row{border-left:3px solid var(--accent);padding:8px 12px;font-size:.95rem;line-height:1.5;
  background:color-mix(in srgb,var(--ink) 4%,transparent);border-radius:0 8px 8px 0;}
.de-k{display:inline-block;font:700 .72rem/1 ui-monospace,monospace;text-transform:uppercase;letter-spacing:.05em;color:var(--accent);margin-right:8px;}
.nf-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:10px auto;max-width:var(--breakout-max,972px);}
@media(max-width:640px){.nf-grid{grid-template-columns:1fr;}}
.nf-card{border-radius:12px;padding:16px 18px;background:color-mix(in srgb,var(--ink) 5%,transparent);
  border:1px solid color-mix(in srgb,var(--ink) 12%,transparent);}
.nf-card.win{background:color-mix(in srgb,var(--accent) 10%,transparent);border-color:color-mix(in srgb,var(--accent) 45%,transparent);}
.nf-n{font:650 2rem/1 var(--serif,Georgia,serif);} .nf-card.win .nf-n{color:var(--accent);}
.nf-l{font-weight:650;margin-top:4px;} .nf-s{opacity:.62;font-size:.82rem;margin-top:3px;line-height:1.4;}
.newcomp{text-align:center;font-size:.92rem;margin:2px auto 6px;max-width:var(--breakout-max,972px);opacity:.9;}
.newcomp b{font-size:1.05rem;} .newcomp .gain{color:var(--accent);font-weight:700;}
.superwrap{position:relative;margin:6px auto 16px;max-width:var(--breakout-max,972px);}
.super-banner{font:700 .68rem/1 ui-monospace,monospace;letter-spacing:.05em;color:#b06;
  background:color-mix(in srgb,#c66 14%,transparent);border:1px solid color-mix(in srgb,#c66 40%,transparent);
  border-radius:7px;padding:7px 12px;margin-bottom:10px;text-align:center;}
.superwrap .efun{opacity:.52;filter:grayscale(.5);}
.efun{display:flex;align-items:stretch;gap:10px;flex-wrap:wrap;justify-content:center;margin:6px auto 8px;max-width:var(--breakout-max,972px);}
.fnode{flex:1 1 170px;min-width:150px;border-radius:12px;padding:16px 18px;
  background:color-mix(in srgb,var(--ink) 5%,transparent);border:1px solid color-mix(in srgb,var(--ink) 12%,transparent);}
.fnode.end{background:color-mix(in srgb,var(--accent) 10%,transparent);border-color:color-mix(in srgb,var(--accent) 45%,transparent);}
.fn-n{font:650 1.7rem/1 var(--serif,Georgia,serif);letter-spacing:-.01em;} .fnode.end .fn-n{color:var(--accent);}
.superwrap .fnode.end .fn-n{color:inherit;}
.fn-l{font-weight:650;margin-top:3px;font-size:.95rem;} .fn-s{opacity:.62;font-size:.8rem;margin-top:3px;line-height:1.4;}
.farrow{display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:1.5rem;color:var(--accent);flex:0 0 auto;padding:0 2px;width:82px;}
.farrow .fa-drop,.farrow .fa-add{font-size:.7rem;color:var(--ink);opacity:.6;font-weight:600;text-align:center;margin-top:4px;line-height:1.3;white-space:nowrap;}
.farrow .fa-add{color:#4a9d6a;opacity:1;}
.efun4 .fnode{flex:1 1 138px;min-width:126px;padding:13px 14px;} .efun4 .farrow{width:64px;}
.efun4 .fn-n{font-size:1.45rem;}
@media(max-width:720px){.efun{flex-direction:column;}
  .farrow{flex-direction:row;gap:8px;min-width:0;transform:rotate(90deg);height:34px;}
  .farrow .fa-drop,.farrow .fa-add{transform:rotate(-90deg);white-space:nowrap;}}
.galfilt{display:flex;flex-wrap:wrap;gap:8px;margin:16px auto 14px;max-width:var(--breakout-max,972px);
  position:sticky;top:0;background:var(--bg);padding:10px 0;z-index:5;}
.gf{font:600 .82rem/1 inherit;padding:7px 14px;border-radius:999px;cursor:pointer;
  border:1px solid color-mix(in srgb,var(--ink) 20%,transparent);background:transparent;color:var(--ink);opacity:.7;}
.gf.on{background:var(--accent);border-color:var(--accent);color:#fff;opacity:1;}
.galgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(128px,1fr));gap:9px;margin:0 auto;max-width:var(--breakout-max,972px);}
.gcell{position:relative;display:block;border-radius:9px;overflow:hidden;background:#16161a;
  border:1px solid color-mix(in srgb,var(--ink) 12%,transparent);text-decoration:none;aspect-ratio:1;transition:transform .12s,box-shadow .12s;}
.gcell:hover{transform:translateY(-2px);box-shadow:0 6px 18px rgba(0,0,0,.35);border-color:var(--accent);}
.gcell img{display:block;width:100%;height:100%;object-fit:cover;}
.gfrac{position:absolute;right:6px;bottom:6px;font:600 .7rem/1 ui-monospace,monospace;padding:3px 6px;border-radius:5px;background:rgba(0,0,0,.55);color:#f0994e;}
.galgrid.f-hi .gcell:not([data-b="hi"]),.galgrid.f-mid .gcell:not([data-b="mid"]),.galgrid.f-lo .gcell:not([data-b="lo"]){display:none;}
.litgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin:14px auto 0;max-width:var(--breakout-max,972px);}
.litcell{position:relative;display:block;border-radius:9px;overflow:hidden;background:#111113;
  border:1px solid color-mix(in srgb,var(--ink) 12%,transparent);text-decoration:none;aspect-ratio:1;transition:transform .12s,box-shadow .12s;}
.litcell:hover{transform:translateY(-2px);box-shadow:0 6px 18px rgba(0,0,0,.35);border-color:var(--accent);}
.litcell img{display:block;width:100%;height:100%;object-fit:cover;}
.litcell span{position:absolute;left:6px;bottom:6px;font:600 .68rem/1 ui-monospace,monospace;padding:3px 6px;border-radius:5px;background:rgba(0,0,0,.55);color:#bcd0e0;}
"""
extra_js = """
<script>
(function(){
 document.querySelectorAll('.galfilt').forEach(function(bar){
   var grid=document.getElementById(bar.dataset.target); if(!grid)return;
   bar.querySelectorAll('.gf').forEach(function(b){b.addEventListener('click',function(){
     bar.querySelectorAll('.gf').forEach(function(x){x.classList.remove('on');});
     b.classList.add('on');
     grid.className='galgrid'+(b.dataset.f==='all'?'':' f-'+b.dataset.f);
   });});
 });
})();</script>"""

html = lp.page(
    title="Emissive filtering: the criterion is self-lit shading — lightgen",
    header_html=hero,
    body_sections=[s_reframe, s_noflag, s_detector, s_bug, s_rates, s_newfilter,
                   s_naclass, s_super, s_prov],
    theme="v2", assets_rel=ASSETS_REL, assets_dir=ASSETS_DIR,
    extra_head=f"<style>{extra_css}</style>",
    extra_body_end=extra_js,
)
with open(os.path.join(OUT, "index.html"), "w") as f:
    f.write(html)
print("wrote", os.path.join(OUT, "index.html"), f"({len(html)} bytes)")
