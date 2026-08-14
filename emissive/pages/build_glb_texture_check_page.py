"""
build_glb_texture_check_page.py — "The glb texture-image self-lit detector"
xgpage v2 editorial. A dedicated report on the per-material emissiveTexture-vs-baseColorTexture
image check that produced the 48,628 trainable set (see the emissive-filtering page). Separate
URL: _preview/glb_texture_check. Coverage-guard counts (§7) are FINAL (threshold 0.5:
rescue 4,538 -> refined trainable 53,166). Includes the 192-shape base|emissive|diff gallery.

Assets: direct_pilot/sanity_renders/{texcmp_truecopies,texcmp_flagged,contact_metal_flagged,
contact_na_kept}.png, direct_pilot/ovoxel_detector/ovdis_triptych.png. Runs on the standalone
xgpage package:
  /local-scratch2/xya120/studio/misc/lightgen/.venv_console/bin/python build_glb_texture_check_page.py
"""
import os, json
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "vis_data", "glb_texture_check_html")
os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
import xgpage as lp

ASSETS_REL = "/projects/omages/yanxg/lightgen/assets"
ASSETS_DIR = os.path.join(ROOT, "..", "web", "assets")
GALMAN = json.load(open(os.path.join(OUT, "glb_check_gallery_manifest.json")))
SKF = "https://sketchfab.com/models/"
def cm(n): return f"{n:,}"

N_PARQ = 80735
SELF_LIT = 12946
LIT_ALL = 16.0
SAME_IDX, CONTENT = 8982, 3964
SAME_PCT, CONTENT_PCT = 69.4, 30.6
SL_METAL, SL_SPEC, SL_NA = 2354, 480, 10112
LIT_METAL, LIT_SPEC, LIT_NA = 7.1, 26.4, 22.1
PARQ_METAL, PARQ_SPEC, PARQ_NA = 33091, 1815, 45829
LIT_METAL_BUG = 56.5
OLD_SL = 2524
OMAGE_UNDER = 3.1
# trainable funnel base = WORKING CORPUS (funnel_reconciliation.json); 80,735 = detector corpus
N_WORK = 73470                              # working corpus (emissive-glb INTERSECT somage bake)
ZERO_EMIT, NONZERO = 13057, 60413
SELF_LIT_DROP = 11785                       # 19.5% of the nonzero working corpus
SL_RATE_WORK = 19.5
NO_SOMAGE = 7265
# coverage guard (final)
N_NEW = 48628                              # pre-rescue trainable
COV_LOCAL_PCT, COV_WHOLE_PCT = 27, 57      # distribution: <=0.1 localized, >0.9 whole
COV_CORR, COV_AGREE = 0.914, 94.7           # glb-area vs ovoxel-emission cross-validation
COV_DROP, COV_RESCUE = 7247, 4538           # at threshold 0.5
RES_METAL, RES_SPEC, RES_NA = 1306, 113, 3119
N_REFINED = N_NEW + COV_RESCUE              # 53,166
OLD_DROPS = 11785
OVERDROP_PCT = 38.5

# ================================================================ hero
hero = lp.hero_header(
    "lightgen · data · self-lit detector",
    "The glb texture-image self-lit detector",
    dek_html=(
        "This is the detector behind the emissive training set: a per-material comparison of "
        "each material&rsquo;s <code>emissiveTexture</code> image against its "
        "<code>baseColorTexture</code> image, read straight from the GLB. When the two are the "
        "same image, the &lsquo;emission&rsquo; is just the albedo &mdash; a fullbright "
        "(self-lit) trick, not authored light. It reads authoring intent from the raw texture "
        "images, not from a UV atlas or a render. Over the whole corpus it flags "
        f"<b>{LIT_ALL}%</b> of shapes self-lit."),
    stats=[
        (f"{LIT_ALL}%", f"flagged self-lit ({cm(SELF_LIT)}/{cm(N_PARQ)})"),
        (f"{SAME_PCT:.0f}%", "exact same-image copies"),
        (f"{CONTENT_PCT:.0f}%", "re-encoded content copies"),
        ("2", "copy tests, one gate"),
    ],
    toc=[
        ("method", "The method"),
        ("mech", "Two copy mechanisms"),
        ("bug", "A methods lesson"),
        ("findings", "Findings"),
        ("valid", "Validation"),
        ("gallery", "Browse the proof"),
        ("coverage", "Coverage guard"),
        ("limit", "Limitation"),
        ("prov", "Provenance"),
        ("relation", "The three checks"),
    ],
)

# ================================================================ 01 method
s_method = lp.section_v2("method", "01",
    "The method: compare each material&rsquo;s emissive and base-color images",
    lp.prose(
        "For every material in the GLB, resolve two images from the glTF header: the "
        "<code>baseColorTexture</code> (or the <code>KHR_pbrSpecularGlossiness</code> diffuse "
        "texture, for that workflow) and the <code>emissiveTexture</code>. A material is a "
        "<b>self-lit copy</b> if those two images are the same, by either test:")
    + '<div class="deteqn">'
      '<div class="de-row"><span class="de-k">(a) same index</span> the emissive and base-color '
      'textures reference the <b>same glTF image index</b> &mdash; an exact copy, unambiguous.</div>'
      '<div class="de-row"><span class="de-k">(b) content match</span> different indices, but the '
      'images are the same picture: decode both at <b>32&times;32</b>, fit '
      'emission&nbsp;&asymp;&nbsp;k&middot;base, and call it a copy iff <b>corr&nbsp;&gt;&nbsp;0.9</b> '
      'AND <b>residual&nbsp;&lt;&nbsp;0.05</b>.</div>'
      '</div>'
    + lp.prose(
        "A shape is <b>self-lit</b> only when <b>all</b> of its emitting materials are copies "
        "(conservative: a single genuine emitter keeps the shape). The check parses the glTF "
        "header and the two texture images per material &mdash; <b>no mesh load, no UV atlas, no "
        "render</b>. It reads the artist&rsquo;s authoring intent directly from the pixels they "
        "shipped.")
    + lp.callout(
        "Why images and not renders: a render bakes lighting, view, and geometry into the "
        "answer, and an emit-only render of a dark-albedo copy looks the same as a faint real "
        "emitter. The raw texture images are the ground truth for &lsquo;is the emission a copy "
        "of the albedo?&rsquo; &mdash; and that is exactly the question.",
        title="It reads authoring intent, not appearance")
)

# ================================================================ 02 two mechanisms
mech_chart = lp.hbar_chart(
    [{"label": "same index", "value": SAME_IDX, "display": f"{cm(SAME_IDX)} &middot; {SAME_PCT:.0f}%",
      "tip": f"same glTF image index: {cm(SAME_IDX)} ({SAME_PCT:.0f}%)"},
     {"label": "content match", "value": CONTENT, "display": f"{cm(CONTENT)} &middot; {CONTENT_PCT:.0f}%",
      "tip": f"content-matched (different slot, same picture): {cm(CONTENT)} ({CONTENT_PCT:.0f}%)"}],
    title=f"how the {cm(SELF_LIT)} self-lit shapes copy", label_w=110,
    note=(f"<b>Nearly a third of copies are re-encoded, not shared.</b> {SAME_PCT:.0f}% point the "
          f"emissive and base-color slots at the <i>same</i> glTF image; but {CONTENT_PCT:.0f}% "
          f"({cm(CONTENT)} shapes) store the <i>same picture</i> at a different image slot &mdash; "
          f"a tuned or re-encoded duplicate. A same-index test alone, and a pixel-identity check "
          f"on a rebaked atlas, both miss these."))
truecopy_fig = lp.fig("img/texcmp_truecopies.png",
    caption_html=(
        "<b>Confirmed copies, both mechanisms.</b> Columns: <code>baseColorTexture</code> | "
        "<code>emissiveTexture</code> | |diff|&times;4 (black = identical). Top three "
        "<span class=\"tag-si\">SAME-INDEX</span>: the two slots are literally the same image "
        "(diff is black). Bottom three <span class=\"tag-cm\">CONTENT-MATCH</span>: different "
        "image slots (<code>baseImg=0, emitImg=1/2</code>) holding the <i>same picture</i> "
        "&mdash; corr = 1.000, residual &asymp; 0, diff still black. The content-match test is "
        "what catches the re-encoded copies."),
    native_px=768, content="photo")
s_mech = lp.section_v2("mech", "02",
    "Two copy mechanisms &mdash; and the re-encoded ones are the payoff",
    mech_chart + truecopy_fig)

# ================================================================ 03 bug
bug_fig = lp.fig("img/texcmp_flagged.png",
    caption_html=(
        "<b>Six shapes the residual-only build called &lsquo;copy&rsquo; &mdash; all real "
        "emitters.</b> Same three columns. The base color is a full atlas; the emission is "
        "<b>mostly black with a few localized glows</b> &mdash; plainly not a copy. Yet the "
        "scale-fit residual reads &asymp;0 (headers), so residual-alone flagged them. The diff "
        "column (bright = differs) shows how far the two images actually are."),
    native_px=768, content="photo")
s_bug = lp.section_v2("bug", "03",
    "A methods lesson: residual alone reported 56% and was wrong",
    lp.prose(
        f"The first build used the residual test <i>alone</i> and reported <b>{LIT_METAL_BUG}% "
        f"of metalness as fullbright</b>. That was a bug: a near-black real emission map is fit "
        f"by <b>k&nbsp;&asymp;&nbsp;0</b> in emission&nbsp;&asymp;&nbsp;k&middot;base, driving "
        f"the residual to &asymp;0 &mdash; a <b>k&rarr;0 degeneracy</b> that reads "
        f"&lsquo;copy&rsquo; for any dim emitter.")
    + bug_fig
    + lp.prose(
        "A ground-truth image comparison exposed it: all six ambiguous flagged shapes were real "
        "localized emitters, <b>zero</b> true copies. The fix adds the correlation gate "
        "(<b>corr&nbsp;&gt;&nbsp;0.9</b>): a real emitter&rsquo;s emission is uncorrelated with "
        f"its albedo, so it no longer passes, and the metalness rate falls to {LIT_METAL}%.")
    + lp.callout(
        "The texture images are the ground truth; an emit-only render cannot decide it (a "
        "dark-albedo copy and a faint real emitter look the same). Correlation, not residual, "
        "is what separates a copy from a dim distinct map.",
        title="Textures decide it, renders can't")
)

# ================================================================ 04 findings
rate_rows = [
    {"label": "metalness", "value": LIT_METAL, "display": f"{LIT_METAL}% &middot; {cm(SL_METAL)}",
     "tip": f"metalness: {cm(SL_METAL)} of {cm(PARQ_METAL)} ({LIT_METAL}%)"},
    {"label": "specular", "value": LIT_SPEC, "display": f"{LIT_SPEC}% &middot; {cm(SL_SPEC)}",
     "tip": f"specular: {cm(SL_SPEC)} of {cm(PARQ_SPEC)} ({LIT_SPEC}%)"},
    {"label": "&lt;NA&gt;", "value": LIT_NA, "display": f"{LIT_NA}% &middot; {cm(SL_NA)}",
     "tip": f"<NA>: {cm(SL_NA)} of {cm(PARQ_NA)} ({LIT_NA}%)"},
    {"label": "overall", "value": LIT_ALL, "display": f"{LIT_ALL}% &middot; {cm(SELF_LIT)}",
     "tip": f"overall: {cm(SELF_LIT)} of {cm(N_PARQ)} ({LIT_ALL}%)"},
]
rates_chart = lp.hbar_chart(rate_rows, title="self-lit rate by pbrType (full scan, 80,735 shapes)",
    label_w=96,
    note=(f"<b>Self-lit cross-cuts pbrType.</b> There is contamination inside the metalness set "
          f"({cm(SL_METAL)}, {LIT_METAL}%) and a great many real emitters inside "
          f"<code>&lt;NA&gt;</code> (only {LIT_NA}% self-lit, so ~{100-LIT_NA:.0f}% are real). "
          f"Neither the &lsquo;keep pbr&rsquo; nor the &lsquo;drop &lt;NA&gt;&rsquo; half of the "
          f"old rule was right."))
s_findings = lp.section_v2("findings", "04",
    "Findings: self-lit is a minority, and it cross-cuts pbrType",
    rates_chart
    + lp.prose(
        f"With the fixed detector the old metalness + specular keep-set turns out to be only "
        f"<b>~10% self-lit</b> ({cm(OLD_SL)} shapes), never {LIT_METAL_BUG}% &mdash; the bug "
        f"scare is fully retired. The material type was simply the wrong signal: it is wrong "
        f"both ways, and the texture-image check is what gets the split right.")
)

# ================================================================ 05 validation
metal_fig = lp.fig("img/contact_metal_flagged.png",
    caption_html=(
        "<b>Flagged self-lit: emission copies the whole albedo.</b> Lit appearance | emission-only "
        "(<code>std | emit_only</code>). A rusty car, a skeleton, a tablet, coral &mdash; the "
        "emit-only render reproduces the <i>entire textured surface</i> (n_copy = n_emit). The "
        "whole-surface signature confirms these are fullbright."),
    native_px=1360, content="photo")
nakept_fig = lp.fig("img/contact_na_kept.png",
    caption_html=(
        "<b>Not flagged: real localized emitters.</b> A chandelier&rsquo;s flames, a glowing "
        "visor, lava veins, a car&rsquo;s lights &mdash; the emit-only render lights only a small "
        "region (emission &ne; albedo). These are kept."),
    native_px=1360, content="photo")
s_valid = lp.section_v2("valid", "05",
    "Validation: whole-surface copies vs. localized emitters",
    lp.prose(
        "Rendering the two outcomes confirms the check keys on the right thing: flagged shapes "
        "emit their whole albedo surface; kept shapes emit only a localized region.")
    + metal_fig + nakept_fig)

# ================================================================ 06 gallery
def gband(r): return (("si" if r["mechanism"] == "same-index" else "cm"),
                      ("whole" if r["coverage_class"] == "whole" else "local"))
gmech = ('<div class="gfilt2" data-target="glbgal" data-group="m" role="tablist">'
    '<span class="gfl">mechanism</span>'
    '<button class="gf on" data-f="all">All &middot; 192</button>'
    '<button class="gf" data-f="si">same-index &middot; 96</button>'
    '<button class="gf" data-f="cm">content-matched &middot; 96</button></div>')
gcov = ('<div class="gfilt2" data-target="glbgal" data-group="c" role="tablist">'
    '<span class="gfl">coverage</span>'
    '<button class="gf on" data-f="all">All &middot; 192</button>'
    '<button class="gf" data-f="whole">whole-surface &middot; 96</button>'
    '<button class="gf" data-f="local">localized &middot; 96</button></div>')
gcells = []
for r in GALMAN:
    s = r["sha"]; mech, cov = gband(r)
    covp = f"{r['coverage']*100:.0f}%"
    lab = "whole-surface (dropped)" if cov == "whole" else "localized (rescued)"
    gcells.append(
        f'<a class="gcellw" data-mech="{mech}" data-cov="{cov}" href="{SKF}{s}" '
        f'target="_blank" rel="noopener" '
        f'title="{s} · {r["mechanism"]} · coverage {covp} · {r["pbrType"]} · open on Sketchfab">'
        f'<img loading="lazy" src="img/glbgal/{s}.png" alt="base | emissive | diff" width="600" height="222">'
        f'<span class="gcw-tag">{r["mechanism"]} &middot; cov {covp} &middot; {lab}</span></a>')
glbgrid = f'<div class="ggrid2" id="glbgal">{"".join(gcells)}</div>'
s_gallery = lp.section_v2("gallery", "06",
    "Browse the proof: 192 flagged shapes, base vs. emissive vs. diff",
    lp.prose(
        "Every cell is one flagged shape&rsquo;s <b>base color | emissive | |diff|</b> textures "
        "&mdash; the copy is visible directly (base == emissive, diff black). Sorted by coverage, "
        "whole-surface first. Filter by <b>mechanism</b> (same-index vs. re-encoded content match) "
        "and by <b>coverage</b> (<b>whole-surface</b> = genuine fullbright, dropped; "
        "<b>localized</b> = a copy-emitter on a small sub-mesh, rescued by the coverage guard "
        "below). Click any cell for its model on Sketchfab.")
    + '<div class="gfilts">' + gmech + gcov + '</div>' + glbgrid
    + lp.chartnote(
        "48 shapes in each of the four buckets (mechanism &times; coverage). The localized "
        "columns are the copy-emitters the shape-level rule over-drops and the coverage guard "
        "rescues; the whole-surface columns are the fullbright it correctly drops.")
)

# ================================================================ 07 coverage guard (FINAL)
cov_fig = lp.fig("img/ovdis_triptych.png",
    caption_html=(
        "<b>glb says fullbright, ovoxel says keep.</b> Columns: appearance | glb emission-only "
        "render | Dongchen&rsquo;s composited ovoxel emission. Each shape has an all-copy material, "
        "so the shape-level rule drops it &mdash; but the copy sits on a <i>small</i> sub-mesh: a "
        "warning light, a kiosk screen, lit house windows, a TV fire. The emission-only and "
        "ovoxel columns show the glow is localized, not whole-surface. Coverage (<code>cov</code>, "
        "headers) is tiny for these and &asymp;1.0 for genuine fullbright."),
    native_px=630, content="photo")
cov_chart = lp.hbar_chart(
    [{"label": "localized (&le;0.1)", "value": COV_LOCAL_PCT, "display": f"{COV_LOCAL_PCT}%",
      "tip": f"localized copy-emitters, coverage <=0.1: {COV_LOCAL_PCT}%"},
     {"label": "whole (&gt;0.9)", "value": COV_WHOLE_PCT, "display": f"{COV_WHOLE_PCT}%",
      "tip": f"whole-surface fullbright, coverage >0.9: {COV_WHOLE_PCT}%"}],
    title="flagged shapes by copy-material coverage (clear valley between)", label_w=120,
    note=(f"<b>Coverage is cleanly bimodal.</b> {COV_LOCAL_PCT}% of flagged shapes have the copy "
          f"on &le;10% of their surface (a localized emitter), {COV_WHOLE_PCT}% cover &gt;90% "
          f"(genuine fullbright), with a clear valley between &mdash; so a threshold is robust."))
def fnode(n, label, sub, cls=""):
    return (f'<div class="fnode {cls}"><div class="fn-n">{cm(n)}</div>'
            f'<div class="fn-l">{label}</div><div class="fn-s">{sub}</div></div>')
train_funnel = (
    '<div class="efun efun4">'
    + fnode(N_WORK, "working corpus", "emissive-glb &cap; somage bake", "start")
    + f'<div class="farrow">&rarr;<span class="fa-drop">&minus;{cm(ZERO_EMIT)}<br>no emission</span></div>'
    + fnode(NONZERO, "nonzero emission", "has authored emission", "mid")
    + f'<div class="farrow">&rarr;<span class="fa-drop">&minus;{cm(SELF_LIT_DROP)}<br>self-lit &middot; {SL_RATE_WORK}%</span></div>'
    + fnode(N_NEW, "real emitters", "pre-rescue", "mid")
    + f'<div class="farrow">&rarr;<span class="fa-add">+{cm(COV_RESCUE)}<br>rescued</span></div>'
    + fnode(N_REFINED, "trainable set", "final", "end")
    + '</div>')

s_coverage = lp.section_v2("coverage", "07",
    f"The coverage guard rescues {cm(COV_RESCUE)} localized copy-emitters",
    lp.prose(
        "A cross-check against the ovoxels found one failure mode of the shape-level "
        "<b>all-copy</b> rule: it <b>over-drops localized copy-emitters</b>. An artist sometimes "
        "builds a small emitter &mdash; a warning light, a screen, a lit window &mdash; as a "
        "copy material on a tiny sub-mesh. Every emitting material is technically a copy, so the "
        "shape is flagged, but it is really a real localized emitter.")
    + cov_fig
    + lp.prose(
        "The signal is <b>glb-native coverage</b>: copy-material triangle area / total mesh area. "
        "It is cleanly bimodal, and it holds up under an independent check &mdash; glb "
        f"triangle-area coverage vs. ovoxel emission coverage agree at <b>corr {COV_CORR}</b> "
        f"({COV_AGREE}% localized/whole agreement).")
    + cov_chart
    + '<div class="nf-grid">'
      f'<div class="nf-card"><div class="nf-n">{cm(COV_DROP)}</div><div class="nf-l">whole-surface '
      'fullbright, still dropped</div><div class="nf-s">coverage &gt; 0.5</div></div>'
      f'<div class="nf-card win"><div class="nf-n">+{cm(COV_RESCUE)}</div><div class="nf-l">localized '
      'copy-emitters rescued</div><div class="nf-s">metalness {m} + specular {s} + &lt;NA&gt; {na}</div></div>'
      '</div>'.format(m=cm(RES_METAL), s=cm(RES_SPEC), na=cm(RES_NA))
    + lp.prose(
        f"At a threshold of <b>0.5</b> (in the valley, robust &mdash; the rescue moves only "
        f"34&rarr;40.5% across 0.3&ndash;0.7): drop {cm(COV_DROP)} genuine fullbright, "
        f"<b>rescue {cm(COV_RESCUE)}</b> localized copy-emitters. This only <b>adds</b> to the "
        f"keep set &mdash; the KEEP side of the detector is 100% concordant with the ovoxel "
        f"check &mdash; and {OVERDROP_PCT}% of the {cm(OLD_DROPS)} self-lit drops turn out to "
        f"have been over-drops. The trainable funnel, over the Path-A working corpus:")
    + train_funnel
    + lp.callout(
        f"The trainable funnel runs over the <b>working corpus</b> ({cm(N_WORK)} = emissive GLB "
        f"&cap; somage bake), <i>not</i> the {cm(N_PARQ)} detector corpus &mdash; the "
        f"{cm(NO_SOMAGE)} GLBs without a somage bake have no Path-A label and cannot be trained. "
        f"The {LIT_ALL}% self-lit headline is a corpus-wide detector statistic; within the "
        f"working nonzero corpus the self-lit rate is <b>{SL_RATE_WORK}%</b> "
        f"({cm(SELF_LIT_DROP)}/{cm(NONZERO)}). Net trainable: <b>{cm(N_REFINED)}</b>.",
        title="Detector corpus (80,735) vs. trainable funnel (73,470 &rarr; 53,166)")
    + lp.callout(
        f"Coverage uses glb triangle area under node transforms; heavy skinning could shift the "
        f"true covered fraction slightly. The {COV_CORR} agreement with the independent ovoxel "
        f"coverage says that is not material here.",
        title="Caveat: triangle-area coverage")
)

# ================================================================ 07 limitation
s_limit = lp.section_v2("limit", "08",
    "Limitation: a strongly tinted copy can slip the residual",
    lp.prose(
        "The content-match test fits emission&nbsp;&asymp;&nbsp;k&middot;base with a single "
        "scalar k. A copy that is <b>strongly tinted</b> (emission = a colored multiple of the "
        "albedo, not a grey scale) can push the residual above the fit and read as "
        "not-a-copy &mdash; a rare, small <b>under-count</b> (a few genuine fullbright shapes "
        "kept). Same-index copies are immune (identity needs no fit), and the correlation gate "
        "still holds for most tints. A per-channel scale fit would close the gap if the "
        "under-count ever matters.")
)

# ================================================================ 08 provenance
prov = lp.prose(
    "<b>Input.</b> glTF/GLB header parse only for the image check (no mesh load): per material, "
    "the <code>baseColorTexture</code> / <code>KHR_pbrSpecularGlossiness</code> diffuse image and "
    "the <code>emissiveTexture</code> image. Mesh triangle areas are loaded only for the coverage "
    "signal (&sect;6).")
prov += lp.prose(
    "<b>Thresholds.</b> Content match: decode 32&times;32, <code>corr&nbsp;&gt;&nbsp;0.9</code> "
    "AND scale-fit <code>residual&nbsp;&lt;&nbsp;0.05</code>. Shape self-lit iff all emitting "
    "materials are copies. Coverage guard: copy-area / mesh-area, threshold <b>0.5</b> (in the "
    "bimodal valley; cross-validated against ovoxel emission coverage at corr {c}).".format(c=COV_CORR))
prov += lp.prose(
    f"<b>Scan.</b> All {cm(N_PARQ)} shapes of "
    f"<code>emissive_thumbnails_obj_ids_df.parquet</code>. Result: {cm(SELF_LIT)} self-lit "
    f"({cm(SAME_IDX)} same-index + {cm(CONTENT)} content-match); by pbrType metalness "
    f"{cm(SL_METAL)}, specular {cm(SL_SPEC)}, &lt;NA&gt; {cm(SL_NA)}.")
prov += lp.prose(
    "<b>Saved lists.</b> <code>direct_pilot/lit_shadeless_shas.txt</code> (the flagged self-lit "
    "SHAs) and <code>direct_pilot/train_keep_48628.txt</code> (the kept trainable set). Sanity "
    "sheets: <code>direct_pilot/sanity_renders/</code> and "
    "<code>direct_pilot/ovoxel_detector/</code>.")
s_prov = lp.section_v2("prov", "09", "Provenance", prov)

# ================================================================ 09 the three checks
s_relation = lp.section_v2("relation", "10",
    "The three checks: this one is authoritative",
    '<div class="three">'
    '<div class="tc-card win"><div class="tc-h">glb texture-image <span>(this)</span></div>'
    '<div class="tc-b">Per-material emissive-vs-base image identity, straight from the GLB. '
    'Catches both same-index and re-encoded copies; reads authoring intent, not appearance. '
    '<b>Authoritative</b> for the keep/drop split.</div></div>'
    '<div class="tc-card"><div class="tc-h">omage atlas <span>(demoted)</span></div>'
    f'<div class="tc-b">Pixel identity on the somage-rebaked atlas. <b>Undercounts by '
    f'{OMAGE_UNDER}%</b> and misses the {CONTENT_PCT:.0f}% re-encoded copies entirely (the rebake '
    'changes pixels), so a same-picture copy no longer reads identical.</div></div>'
    '<div class="tc-card"><div class="tc-h">ovoxel <span>(cross-check)</span></div>'
    '<div class="tc-b">Copy status composited into Dongchen&rsquo;s voxels. Surfaced the '
    'coverage refinement (&sect;6), but is <b>not a replacement</b>: its strict gate '
    'under-flags genuine fullbright. Complementary, not authoritative.</div></div>'
    '</div>'
    + lp.callout(
        "<code>KHR_materials_unlit</code> is present on <b>0</b> shapes in this "
        "emissive-declared subset &mdash; it would only serve as the shading flag if the scan "
        "were widened beyond emissive-declared shapes. Within this corpus, the hand-authored "
        "copy is the only signal, and the texture-image check is how to read it.",
        title="No KHR_materials_unlit in the subset")
)

# ================================================================ css
extra_css = """
.deteqn{margin:10px auto;max-width:var(--breakout-max,972px);display:flex;flex-direction:column;gap:8px;}
.de-row{border-left:3px solid var(--accent);padding:8px 12px;font-size:.95rem;line-height:1.5;
  background:color-mix(in srgb,var(--ink) 4%,transparent);border-radius:0 8px 8px 0;}
.de-k{display:inline-block;font:700 .72rem/1 ui-monospace,monospace;text-transform:uppercase;
  letter-spacing:.05em;color:var(--accent);margin-right:8px;}
.tag-si,.tag-cm{font:700 .64rem/1 ui-monospace,monospace;letter-spacing:.03em;padding:2px 5px;border-radius:4px;}
.tag-si{background:color-mix(in srgb,#4a9d6a 22%,transparent);color:#2f7d4f;}
.tag-cm{background:color-mix(in srgb,var(--accent) 20%,transparent);color:var(--accent);}
.prelim-banner,.reconcile-banner{font:600 .74rem/1.45 inherit;color:#a76b00;
  background:color-mix(in srgb,#e0a030 14%,transparent);border:1px solid color-mix(in srgb,#e0a030 42%,transparent);
  border-radius:8px;padding:9px 14px;margin:2px auto 12px;max-width:var(--breakout-max,972px);}
.reconcile-banner b{color:inherit;}
@media(prefers-color-scheme:dark){.prelim-banner,.reconcile-banner{color:#e0b060;}}
:root[data-theme="dark"] .prelim-banner,:root[data-theme="dark"] .reconcile-banner{color:#e0b060;}
:root[data-theme="light"] .prelim-banner,:root[data-theme="light"] .reconcile-banner{color:#a76b00;}
.three{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin:8px auto 6px;max-width:var(--breakout-max,972px);}
@media(max-width:760px){.three{grid-template-columns:1fr;}}
.tc-card{border-radius:12px;padding:14px 16px;background:color-mix(in srgb,var(--ink) 5%,transparent);
  border:1px solid color-mix(in srgb,var(--ink) 12%,transparent);}
.tc-card.win{background:color-mix(in srgb,var(--accent) 10%,transparent);border-color:color-mix(in srgb,var(--accent) 45%,transparent);}
.tc-h{font-weight:700;font-size:1rem;} .tc-h span{font-weight:400;opacity:.6;font-size:.82em;}
.tc-b{font-size:.9rem;line-height:1.55;margin-top:6px;opacity:.9;}
/* rescue/drop cards (shared) */
.nf-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:12px auto;max-width:var(--breakout-max,972px);}
@media(max-width:640px){.nf-grid{grid-template-columns:1fr;}}
.nf-card{border-radius:12px;padding:16px 18px;background:color-mix(in srgb,var(--ink) 5%,transparent);
  border:1px solid color-mix(in srgb,var(--ink) 12%,transparent);}
.nf-card.win{background:color-mix(in srgb,var(--accent) 10%,transparent);border-color:color-mix(in srgb,var(--accent) 45%,transparent);}
.nf-n{font:650 2rem/1 var(--serif,Georgia,serif);} .nf-card.win .nf-n{color:var(--accent);}
.nf-l{font-weight:650;margin-top:4px;} .nf-s{opacity:.62;font-size:.82rem;margin-top:3px;line-height:1.4;}
/* funnel */
.efun{display:flex;align-items:stretch;gap:10px;flex-wrap:wrap;justify-content:center;margin:10px auto;max-width:var(--breakout-max,972px);}
.fnode{flex:1 1 138px;min-width:126px;border-radius:12px;padding:13px 14px;
  background:color-mix(in srgb,var(--ink) 5%,transparent);border:1px solid color-mix(in srgb,var(--ink) 12%,transparent);}
.fnode.end{background:color-mix(in srgb,var(--accent) 10%,transparent);border-color:color-mix(in srgb,var(--accent) 45%,transparent);}
.fn-n{font:650 1.45rem/1 var(--serif,Georgia,serif);letter-spacing:-.01em;} .fnode.end .fn-n{color:var(--accent);}
.fn-l{font-weight:650;margin-top:3px;font-size:.92rem;} .fn-s{opacity:.62;font-size:.78rem;margin-top:3px;line-height:1.35;}
.farrow{display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:1.4rem;color:var(--accent);flex:0 0 auto;padding:0 2px;width:64px;}
.farrow .fa-drop,.farrow .fa-add{font-size:.68rem;color:var(--ink);opacity:.6;font-weight:600;text-align:center;margin-top:4px;line-height:1.3;white-space:nowrap;}
.farrow .fa-add{color:#4a9d6a;opacity:1;}
@media(max-width:720px){.efun{flex-direction:column;}
  .farrow{flex-direction:row;gap:8px;min-width:0;transform:rotate(90deg);height:34px;}
  .farrow .fa-drop,.farrow .fa-add{transform:rotate(-90deg);white-space:nowrap;}}
/* wide triptych gallery */
.gfilts{display:flex;flex-wrap:wrap;gap:10px 24px;margin:16px auto 14px;max-width:var(--breakout-max,972px);
  position:sticky;top:0;background:var(--bg);padding:10px 0;z-index:5;}
.gfilt2{display:flex;flex-wrap:wrap;gap:8px;align-items:center;}
.gfl{font:700 .66rem/1 ui-monospace,monospace;text-transform:uppercase;letter-spacing:.06em;opacity:.5;margin-right:2px;}
.gf{font:600 .8rem/1 inherit;padding:6px 12px;border-radius:999px;cursor:pointer;
  border:1px solid color-mix(in srgb,var(--ink) 20%,transparent);background:transparent;color:var(--ink);opacity:.7;}
.gf.on{background:var(--accent);border-color:var(--accent);color:#fff;opacity:1;}
.ggrid2{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px;margin:0 auto;max-width:var(--breakout-max,972px);}
.gcellw{display:block;border-radius:9px;overflow:hidden;text-decoration:none;background:#111113;
  border:1px solid color-mix(in srgb,var(--ink) 12%,transparent);transition:transform .12s,box-shadow .12s;}
.gcellw:hover{transform:translateY(-2px);box-shadow:0 6px 18px rgba(0,0,0,.35);border-color:var(--accent);}
.gcellw img{display:block;width:100%;height:auto;}
.gcw-tag{display:block;font:600 .66rem/1.3 ui-monospace,monospace;color:#cfcfd2;padding:5px 8px;background:#111113;}
.ggrid2.m-si .gcellw:not([data-mech="si"]),.ggrid2.m-cm .gcellw:not([data-mech="cm"]),
.ggrid2.c-whole .gcellw:not([data-cov="whole"]),.ggrid2.c-local .gcellw:not([data-cov="local"]){display:none;}
"""
extra_js = """
<script>
(function(){
 document.querySelectorAll('.gfilt2').forEach(function(bar){
   var grid=document.getElementById(bar.dataset.target), g=bar.dataset.group; if(!grid)return;
   bar.querySelectorAll('.gf').forEach(function(b){b.addEventListener('click',function(){
     bar.querySelectorAll('.gf').forEach(function(x){x.classList.remove('on');});
     b.classList.add('on');
     Array.prototype.slice.call(grid.classList).forEach(function(c){
       if(c.indexOf(g+'-')===0)grid.classList.remove(c);});
     if(b.dataset.f!=='all')grid.classList.add(g+'-'+b.dataset.f);
   });});
 });
})();</script>"""

html = lp.page(
    title="The glb texture-image self-lit detector — lightgen",
    header_html=hero,
    body_sections=[s_method, s_mech, s_bug, s_findings, s_valid, s_gallery, s_coverage,
                   s_limit, s_prov, s_relation],
    theme="v2", assets_rel=ASSETS_REL, assets_dir=ASSETS_DIR,
    extra_head=f"<style>{extra_css}</style>",
    extra_body_end=extra_js,
)
with open(os.path.join(OUT, "index.html"), "w") as f:
    f.write(html)
print("wrote", os.path.join(OUT, "index.html"), f"({len(html)} bytes)")
