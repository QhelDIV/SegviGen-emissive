"""
Assemble vis_data/dataset_gallery_html/index.html: dataset statistics (train_1k=1123,
val_96=111) + a 48-sample gallery (12 per emissive_frac bucket, seed 42, from
train_1k) with appearance | GT-target renders.

Reads dataset_stats.json (built by compute_dataset_stats.py on the cluster) and the
per-sid renders in vis_data/dataset_gallery/<sid>/{render_input,render_emissive}.png.

  python build_dataset_gallery_page.py
"""
import os, json, shutil, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "tools"))
import xgpage as lp
STATS_PATH = os.path.join(ROOT, "dataset_stats.json")
SRC = os.path.join(ROOT, "vis_data", "dataset_gallery")
OUT = os.path.join(ROOT, "vis_data", "dataset_gallery_html")
os.makedirs(OUT, exist_ok=True)

# Cluster path roots (verified 2026-07-07 via `ssh solar` directory listings — not
# guessed) for the filepath() hover/click-copy component. Per-sample files use
# <sid> as a literal placeholder in the copyable path (there's no one right sid).
SEGVIGEN_ROOT = "/3dlg-jupiter-project/lightgen/segvigen_emissive"
DIFFUSIONNET_ROOT = "/3dlg-jupiter-project/lightgen/diffusionnet_xg"
TEXVERSE_ROOT = "/3dlg-falas/project/omages/datasets/TexVerse/lightgen"

D = json.load(open(STATS_PATH))
SEED = D["seed"]
PSS = D["per_split_stats"]
FRACS = D["per_split_fracs"]
GALLERY = D["gallery_sids"]
POOL_SIZES = D["gallery_bucket_pool_sizes"]
TOKEN_COUNTS = sorted(D["token_counts_subsample"])

# validated categorical pair (dataviz skill, dark-mode column, slots 1 & 8)
COL_TRAIN = "#3987e5"   # blue
COL_VAL = "#d95926"     # orange
COL_SEQ = "#199e70"     # aqua (single-hue sequential; distinct 3rd hue for the token chart)

BIN_DEFS = [
    ("0", lambda f: f == 0),
    ("(0,1%]", lambda f: 0 < f <= 0.01),
    ("(1,3%]", lambda f: 0.01 < f <= 0.03),
    ("(3,10%]", lambda f: 0.03 < f <= 0.10),
    ("(10,30%]", lambda f: 0.10 < f <= 0.30),
    ("(30,60%]", lambda f: 0.30 < f <= 0.60),
    ("(60,100%]", lambda f: 0.60 < f <= 1.0001),
]

BUCKET_ORDER = ["0", "(0,0.05]", "(0.05,0.3]", ">0.3"]
BUCKET_LABEL = {"0": "zero glow", "(0,0.05]": "tiny glow", "(0.05,0.3]": "medium glow", ">0.3": "large glow"}


# ---------------------------------------------------------------- copy gallery imgs
missing = []
for bucket in BUCKET_ORDER:
    for sid in GALLERY[bucket]:
        d = os.path.join(SRC, sid)
        for kind in ["input", "emissive"]:
            s = os.path.join(d, f"render_{kind}.png")
            dst = os.path.join(OUT, f"{sid}_{kind}.png")
            if os.path.exists(s):
                shutil.copy(s, dst)
            else:
                missing.append((sid, kind))
if missing:
    print(f"MISSING {len(missing)}: {missing}")
else:
    print(f"all {sum(len(v) for v in GALLERY.values())*2} gallery panels present")


# ---------------------------------------------------------------- SVG bar chart helpers
def frac_hist_svg():
    W, H = 620, 300
    ML, MR, MT, MB = 46, 10, 16, 40
    plot_w, plot_h = W - ML - MR, H - MT - MB
    n_bins = len(BIN_DEFS)
    bar_w = 20
    pair_gap = 2
    group_gap = (plot_w - n_bins * (2 * bar_w + pair_gap)) / (n_bins - 1)
    ymax = 30.0  # % headroom above the observed max (~24.5%)

    def y(v):
        return MT + plot_h * (1 - v / ymax)

    pct = {}
    for split in ["train_1k", "val_96"]:
        fracs = FRACS[split]
        n = len(fracs)
        pct[split] = [100.0 * sum(1 for f in fracs if pred(f)) / n for _, pred in BIN_DEFS]

    parts = []
    # gridlines at 0/10/20/30%
    for gv in [0, 10, 20, 30]:
        gy = y(gv)
        parts.append(f'<line x1="{ML}" y1="{gy:.1f}" x2="{W-MR}" y2="{gy:.1f}" stroke="#2a3038" stroke-width="1"/>')
        parts.append(f'<text x="{ML-8}" y="{gy+4:.1f}" text-anchor="end" fill="#8b96a5" font-size="10">{gv}%</text>')

    x = ML
    for i, (label, _) in enumerate(BIN_DEFS):
        tv = pct["train_1k"][i]
        vv = pct["val_96"][i]
        ty, vy = y(tv), y(vv)
        base = y(0)
        parts.append(f'<rect x="{x:.1f}" y="{ty:.1f}" width="{bar_w}" height="{base-ty:.1f}" rx="3" fill="{COL_TRAIN}"/>')
        if tv >= 3:
            parts.append(f'<text x="{x+bar_w/2:.1f}" y="{ty-5:.1f}" text-anchor="middle" fill="#d8dde6" font-size="9.5">{tv:.0f}</text>')
        x2 = x + bar_w + pair_gap
        parts.append(f'<rect x="{x2:.1f}" y="{vy:.1f}" width="{bar_w}" height="{base-vy:.1f}" rx="3" fill="{COL_VAL}"/>')
        if vv >= 3:
            parts.append(f'<text x="{x2+bar_w/2:.1f}" y="{vy-5:.1f}" text-anchor="middle" fill="#d8dde6" font-size="9.5">{vv:.0f}</text>')
        cx = (x + x2 + bar_w) / 2
        parts.append(f'<text x="{cx:.1f}" y="{H-MB+16:.1f}" text-anchor="middle" fill="#8b96a5" font-size="9.5">{label}</text>')
        x += 2 * bar_w + pair_gap + group_gap

    parts.append(f'<line x1="{ML}" y1="{y(0):.1f}" x2="{W-MR}" y2="{y(0):.1f}" stroke="#3a4048" stroke-width="1.2"/>')
    return f'<svg viewBox="0 0 {W} {H}">{"".join(parts)}</svg>'


def token_hist_svg():
    W, H = 620, 260
    ML, MR, MT, MB = 46, 10, 16, 34
    plot_w, plot_h = W - ML - MR, H - MT - MB
    edges = [0, 250, 500, 1000, 1500, 2000, 2750, 3500, 4500, 6200]
    counts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        counts.append(sum(1 for t in TOKEN_COUNTS if lo <= t < hi or (hi == edges[-1] and t == hi)))
    n_bins = len(counts)
    ymax = max(counts) * 1.25
    bar_w = (plot_w - (n_bins - 1) * 4) / n_bins

    def y(v):
        return MT + plot_h * (1 - v / ymax)

    parts = []
    for gv in range(0, int(ymax) + 1, 10):
        gy = y(gv)
        parts.append(f'<line x1="{ML}" y1="{gy:.1f}" x2="{W-MR}" y2="{gy:.1f}" stroke="#2a3038" stroke-width="1"/>')
        parts.append(f'<text x="{ML-8}" y="{gy+4:.1f}" text-anchor="end" fill="#8b96a5" font-size="10">{gv}</text>')

    x = ML
    base = y(0)
    for i, c in enumerate(counts):
        cy = y(c)
        parts.append(f'<rect x="{x:.1f}" y="{cy:.1f}" width="{bar_w:.1f}" height="{base-cy:.1f}" rx="3" fill="{COL_SEQ}"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{cy-5:.1f}" text-anchor="middle" fill="#d8dde6" font-size="9.5">{c}</text>')
        lo, hi = edges[i], edges[i+1]
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{H-MB+16:.1f}" text-anchor="middle" fill="#8b96a5" font-size="9">{lo}&ndash;{hi}</text>')
        x += bar_w + 4

    parts.append(f'<line x1="{ML}" y1="{base:.1f}" x2="{W-MR}" y2="{base:.1f}" stroke="#3a4048" stroke-width="1.2"/>')
    parts.append(f'<text x="{(ML+W-MR)/2:.1f}" y="{H-4}" text-anchor="middle" fill="#8b96a5" font-size="10">sparse latent tokens per sample</text>')
    return f'<svg viewBox="0 0 {W} {H}">{"".join(parts)}</svg>'


def bucket_bar_svg(split, color_muted="#232a34"):
    """4-bucket 100%-stacked bar (style matches notes/metrics_explainer.html section 6)."""
    W, H = 620, 44
    counts = [PSS[split]["bucket_counts"][b] for b in BUCKET_ORDER]
    n = sum(counts)
    ML, MR = 4, 4
    plot_w = W - ML - MR
    shades = ["#1b2028", "#22384a", "#1c527a", "#0f6aab"]
    x = ML
    parts = []
    for (b, c), shade in zip(zip(BUCKET_ORDER, counts), shades):
        w = plot_w * c / n
        parts.append(f'<rect x="{x:.1f}" y="6" width="{max(w-1.5,0):.1f}" height="28" fill="{shade}" stroke="#383f4a" stroke-width="0.6"/>')
        if w > 46:
            parts.append(f'<text x="{x+w/2:.1f}" y="24" text-anchor="middle" fill="#d8dde6" font-size="10.5">{100*c/n:.0f}%</text>')
        x += w
    return f'<svg viewBox="0 0 {W} {H}">{"".join(parts)}</svg>'


def pick_list_flow_html():
    """'How the pick list was made' funnel, from the full 80,735-asset corpus down to
    train_1k/val_96. All counts below are as verified on the cluster 2026-07-05 (parquet +
    split file + labels_uv) -- not re-derived here, per instruction."""

    SHA = "e111814b271864c5f246944a3766718e90b0120c"
    SEGVIGEN = f"https://github.com/QhelDIV/SegviGen/blob/{SHA}"

    def stage(num, lbl, sub="", highlight=False, code=""):
        cls = "flow-stage highlight" if highlight else "flow-stage"
        subhtml = f'<div class="fsub">{sub}</div>' if sub else ""
        codehtml = f'<div class="fcode">code: {code}</div>' if code else ""
        return f'<div class="{cls}"><div class="fnum">{num}</div><div class="flbl">{lbl}</div>{subhtml}{codehtml}</div>'

    def arrow(label):
        return f'<div class="flow-arrow"><span class="arrowglyph">&darr;</span>{label}</div>'

    html = f"""
    <div class="flow-wrap">
    <div class="flow">
      {stage("858K", "TexVerse", 'unique Sketchfab models, &ge;1024px textures, CC-licensed; 158K with PBR '
             'materials (<a href="https://arxiv.org/abs/2508.10868">arXiv 2508.10868</a>)')}
      {arrow("somage processing (Dongchen)")}
      {stage("854,287", "somage-processed master", lp.filepath(
                 "df_SomgProc_final.parquet",
                 f"{TEXVERSE_ROOT}/somages_corresp_dc80k/df_SomgProc_final.parquet") +
             '; success==True: 824,858 = 96.6%; the ~3.7k gap to 858K = items that never entered somage processing',
             code='&#128274; <code>omage_encoderv3</code> in '
                  '<a href="https://github.com/QhelDIV/xgutils">xgutils</a> (private repo)')}
      {arrow("emissive flag pass (Dongchen)")}
      <div class="flow-side">Criterion: iterate all ~850K GLBs, keep any model where
        <b>at least one material declares non-zero emissive strength</b> &rarr; ~80.7K flagged;
        thumbnails grabbed for manual inspection &mdash; that's why the parquet is named
        {lp.filepath("emissive_thumbnails_obj_ids_df.parquet", f"{TEXVERSE_ROOT}/emissive_thumbnails_obj_ids_df.parquet")}.</div>
      {stage("80,735", "emissive-flagged corpus", "every shape here DECLARES emission on some material",
             code='&#128274; <a href="https://github.com/dongchen-yang/lightgen/blob/main/data_processing/'
                  'render_based/check_emission.py">check_emission.py</a> (team repo, private) &mdash; '
                  'check_emission() at ~L90 classifies each GLB by max declared emission strength')}
      {arrow("data_splits_74k.json (seed-42 shuffle)")}
      {stage("74,503", lp.filepath("data_splits_74k.json", f"{DIFFUSIONNET_ROOT}/data/data_splits_74k.json"))}
      <div class="honesty-box">&#9888; <b>Honesty box:</b> the 74k tier is murky. The live
        emissive-complete pool is <b>74,353</b> (criterion inferred: emissive-flagged + complete
        somage data; the generating script for <code>emissive_complete_obj_ids.txt</code> was not
        found &mdash; unresolved). Our split file says <b>74,503</b> (150-shape discrepancy,
        unexplained). The team's newer PINNED split is <b>73,472</b> (val/test = Dongchen's 1099
        baseline held out). We used the oldest, unpinned variant.
        <div class="fcode">code: pinned-split generator &mdash; &#128274;
          <a href="https://github.com/dongchen-yang/lightgen/blob/main/data_processing/create_splits_74k_pinned.py">create_splits_74k_pinned.py</a>
          (team repo, private); <code>emissive_complete_obj_ids.txt</code> generator &mdash; NOT FOUND;
          our data_splits_74k.json generator &mdash; not located, file lives at &#9106;
          {lp.filepath(f"{DIFFUSIONNET_ROOT}/data/data_splits_74k.json", f"{DIFFUSIONNET_ROOT}/data/data_splits_74k.json")} (cluster)</div>
      </div>
      {arrow("80/10/10 partition (a split, not a filter)")}
      <div class="flow-branch">
        {stage("59,602", "train")}
        {stage("7,450", "val")}
        <div class="flow-stage test"><div class="fnum">7,451</div><div class="flbl">test</div>
          <div class="fsub">never touched by this project</div></div>
      </div>
      {arrow("PBR filter &mdash; train partition only")}
      <div class="pbr-legend">
        <span>corpus-level <code>pbrType</code> tag counts:</span>
        <span class="pass">metalness 33,091 &#10003;</span>
        <span class="pass">specular 1,815 &#10003;</span>
        <span class="fail">&lt;NA&gt; 45,829 &#10007;</span>
        <span>(no PBR workflow tagged &mdash; typically fully-lit/baked/shadeless; this is the ONLY criterion)</span>
        <div class="fcode">code: &#128279; <code>_load_split_sids</code> in
          <a href="{SEGVIGEN}/build_dataset.py#L66">build_dataset.py#L66</a></div>
        <div class="fcode">what this keeps vs drops &rarr;
          <a href="../pbr_filter_v1/index.html">pbr_filter_v1</a></div>
      </div>
      {stage("26,264 / 59,602 (44.1%)", lp.filepath("train_pbr_sids_all.json", f"{SEGVIGEN_ROOT}/dataset/train_pbr_sids_all.json"),
             "order preserved from the seed-42 shuffle")}
      {arrow("selection: reuse + deterministic prefix")}
      {stage("224 + 900", "reused (May 512-build subset, zero fresh GPU) + fresh candidates",
             "fresh = train_pbr_sids_all[224:1124] &mdash; a deterministic PREFIX of the (already "
             "seed-42-shuffled) filtered order, statistically equivalent to a random sample but NOT a "
             "fresh random draw. The pseudocode block below the funnel spells this selection out line by line.",
             code=(lp.filepath("BUILD_INFO.md", f"{SEGVIGEN_ROOT}/dataset/train_1k/BUILD_INFO.md") +
                   ' (val_96 has its own sibling copy: ' +
                   lp.filepath("BUILD_INFO.md", f"{SEGVIGEN_ROOT}/dataset/val_96/BUILD_INFO.md") + ') '
                   '+ saved artifacts (' +
                   lp.filepath("train_pbr_sids_all.json", f"{SEGVIGEN_ROOT}/dataset/train_pbr_sids_all.json") +
                   ', <code>chunks/*.txt</code>) (cluster) &mdash; no repo link exists for this step'))}
      {arrow("build (render+RMBG+DINOv3 cond, GLB&rarr;vxz&rarr;3 slats) + completeness prune")}
      <div class="flow-side"><b>Build stage code</b> (pinned SHA <code>e111814</code>, all &#128279; public):
        <a href="{SEGVIGEN}/somage_to_glb.py#L144">somage_to_glb.py#L144</a> (<code>convert_one()</code> &mdash;
        targets+input GLBs; calls <code>build_input_glb</code> L97 + <code>build_emissive_target_glb</code> L130),
        <a href="{SEGVIGEN}/build_dataset.py">build_dataset.py</a> (main loop, cond render),
        <a href="{SEGVIGEN}/data_toolkit/vxz_to_slat.py">data_toolkit/vxz_to_slat.py</a> (encode).</div>
      {stage("1123", "train_1k", ('224 reused + 899 fresh (1 candidate dropped &mdash; missing ' +
             lp.filepath("cond.pth", f"{SEGVIGEN_ROOT}/dataset/train_1k/<sid>/cond.pth") + ')'),
             highlight=True,
             code=(lp.filepath("train_1k_manifest.json", f"{SEGVIGEN_ROOT}/dataset/train_1k_manifest.json") + ' + ' +
                   lp.filepath("val_96_manifest.json", f"{SEGVIGEN_ROOT}/dataset/val_96_manifest.json") + ' + ' +
                   lp.filepath("train_1k_dcclean_manifest.json", f"{SEGVIGEN_ROOT}/dataset/train_1k_dcclean_manifest.json")))}
      <div class="honesty-box">&#9888; <b>Leakage found (2026-07-06):</b> train_1k contains 7 shapes
        from Dongchen's 1099-set held-outs (3 val + 4 test). Our published numbers are unaffected
        (scored on val_96, which is clean), but any run compared on DC's benchmark must use
        {lp.filepath("train_1k_dcclean_manifest.json", f"{SEGVIGEN_ROOT}/dataset/train_1k_dcclean_manifest.json")}
        (1116 sids, on the cluster next to the original).</div>
      <div class="flow-side">
        <b>Parallel val chain</b> (same rule, val-pbr pool): 3,344 val-pbr candidates &rarr;
        6 reused + 105 fresh &rarr; <b>val_96 = 111</b>, zero attrition.
      </div>
    </div>
    </div>

    <p class="caption">&#128279; public &middot; &#128274; team/private repo &middot; &#9106; cluster-only path</p>
    <p class="caption">74k-era positional indices do NOT resolve against the current (rebuilt)
      master parquet (~9% hit rate) &mdash; all our manifests are literal sid lists for this reason.</p>

    <pre class="flow-code"><span class="c"># mirrors the funnel above</span>
full   = load_parquet("emissive_thumbnails_obj_ids_df")   <span class="c"># 80,735 rows</span>
split  = json.load("data_splits_74k.json")                <span class="c"># 74,503 (seed 42; drop criterion undocumented)</span>
train_idx, val_idx, test_idx = split["train"], split["val"], split["test"]  <span class="c"># 59,602 / 7,450 / 7,451</span>

<span class="c"># PBR filter -- the ONLY criterion, applied to train (and val) only</span>
def pbr_pass(row): return row.pbrType in ("metalness", "specular")   <span class="c"># else &lt;NA&gt; -&gt; drop</span>

train_pbr = [i for i in train_idx if pbr_pass(full[i])]   <span class="c"># 26,264 / 59,602 (44.1%)</span>
save(train_pbr, "train_pbr_sids_all.json")                <span class="c"># order preserved (seed-42 shuffle)</span>

reused = load(may_512_build_pbr_subset)                   <span class="c"># 224, zero new GPU cost</span>
fresh  = train_pbr[<span class="n">224</span>:<span class="n">1124</span>]                                    <span class="c"># deterministic PREFIX, not a new random draw</span>
sids   = reused + fresh                                    <span class="c"># 224 + 900 candidates</span>

built    = [build(sid) for sid in sids]                   <span class="c"># render+RMBG+DINOv3, GLB-&gt;vxz-&gt;3 slats</span>
train_1k = [b for b in built if b.cond_pth_exists]         <span class="c"># prune incomplete -&gt; 1123 (1 dropped)</span>

<span class="c"># val: same rule on the val-pbr pool (3,344) -&gt; 6 reused + 105 fresh -&gt; val_96 = 111 (zero attrition)</span></pre>
"""
    return html


# ---------------------------------------------------------------- gallery cards
def fmt_frac(frac):
    """3 decimals normally; more precision for genuinely-tiny-but-nonzero fractions so
    they don't misleadingly round to 0.000 (which would look like the wrong bucket)."""
    if 0 < frac < 0.0005:
        return f"{frac:.5f}"
    return f"{frac:.3f}"


def gallery_section():
    blocks = []
    for bucket in BUCKET_ORDER:
        sids = GALLERY[bucket]
        pool_n = POOL_SIZES[bucket]
        cards = []
        for sid in sids:
            frac_path = os.path.join(SRC, sid, "meta.json")
            frac = json.load(open(frac_path))["emissive_frac"]
            cards.append(f"""
        <div class="card">
          <div class="pair"><img src="{sid}_input.png"><img src="{sid}_emissive.png"></div>
          <div class="cardcap"><span class="csid">{sid[:10]}&hellip;</span><span class="cfrac">frac {fmt_frac(frac)}</span></div>
        </div>""")
        blocks.append(f"""
    <div class="bucket-block">
      <h3>{BUCKET_LABEL[bucket]} &mdash; 12 of {pool_n} train_1k samples <span class="bucketrange">({bucket} emissive_frac)</span></h3>
      <div class="cardgrid">{"".join(cards)}</div>
    </div>""")
    return "".join(blocks)




def stat_tiles(split, cls):
    s = PSS[split]
    n_label = "train_1k" if split == "train_1k" else "val_96"
    return f"""
    <div class="stat-row">
      <div class="stat {cls}"><b>{s['n']}</b><span>{n_label} samples</span></div>
      <div class="stat {cls}"><b>{s['mean']:.3f}</b><span>mean emissive_frac</span></div>
      <div class="stat {cls}"><b>{s['median']:.3f}</b><span>median emissive_frac</span></div>
      <div class="stat {cls}"><b>{s['pct_zero']:.1f}%</b><span>zero-glow shapes</span></div>
      <div class="stat {cls}"><b>{s['pct_gt0p3']:.1f}%</b><span>&gt;30% glow shapes</span></div>
    </div>"""


def main():
    # ---------------------------------------------------------------- Overview
    overview_body = f"""
    <p>Full population statistics for both dataset splits (not the 48-sample gallery
      subset below) &mdash; every number here comes from all 1123 + 111 samples'
      {lp.filepath("meta.json", f"{SEGVIGEN_ROOT}/dataset/train_1k/<sid>/meta.json")}, not an estimate.</p>
    {stat_tiles("train_1k", "train")}
    {stat_tiles("val_96", "val")}
"""

    # ---------------------------------------------------------------- Distributions
    distributions_body = f"""
    <div id="fracdist">
    <h3>Emissive-fraction distribution</h3>
    <p>What fraction of each shape's surface is marked emissive, computed directly from
      every sample's {lp.filepath("meta.json", f"{SEGVIGEN_ROOT}/dataset/train_1k/<sid>/meta.json")}
      (full population: 1123 + 111 shapes, not a
      subsample). The distribution is heavily right-skewed &mdash; most shapes have a
      little or no glow, with a long tail of mostly/fully-emissive shapes &mdash; so bins
      widen geometrically past zero.</p>
    {lp.callout(f'''
      <strong>How <code>emissive_frac</code> is computed:</strong> it's the fraction of
      <strong>triangles</strong> labeled emissive &mdash; a raw face-COUNT ratio (<code>frac =
      float(emask.mean())</code> in <code>somage_to_glb.py</code>), with no area weighting, no
      UV weighting, and no voxels involved. It's inherited directly from the per-face label
      format ({lp.filepath("labels_uv_74k", f"{DIFFUSIONNET_ROOT}/labels_uv_74k")}).
      <br><br>
      <strong>Consequence:</strong> it's tessellation-dependent &mdash; a shape with a few huge
      emissive triangles under-reports its glow, while a densely-triangulated small glow region
      over-reports. Real observed example (canon10 audit): one shape scores <strong>0.404</strong>
      by this face-count measure vs <strong>0.026</strong> by voxel coverage &mdash; a >15&times;
      difference on the same shape.
      <br><br>
      This is <em>not</em> the same measure as {lp.filepath("emis_mask.pth", f"{SEGVIGEN_ROOT}/dataset/train_1k/<sid>/emis_mask.pth")}
      coverage (the
      voxel-surface fraction, &asymp; true surface-area fraction, used for the W5 loss
      weighting) &mdash; the two correlate but aren't identical: r=0.83 (train) / 0.92 (val).
      <br><br>
      <span style="color:var(--muted);font-size:.88em">Honest limitation: the glow-size buckets
      on this page (and the stratified eval elsewhere) are keyed on this face-count measure,
      while the IoU metric lives in voxel space. Re-bucketing by voxel coverage is a planned
      eval upgrade, not yet done.</span>
    ''')}
    <div class="chart-wrap">
      {frac_hist_svg()}
    </div>
    {lp.legend([
        '<span class="sw" style="background:#3987e5"></span>train_1k (n=1123)',
        '<span class="sw" style="background:#d95926"></span>val_96 (n=111)',
        "bars = % of each split's samples (not raw counts, for comparability)",
    ])}
    <h4>4-bucket breakdown (used to build the gallery below)</h4>
    <p class="caption">train_1k &mdash; 0 / (0,0.05] / (0.05,0.3] / &gt;0.3 emissive_frac, by share of samples:</p>
    <div class="chart-wrap">{bucket_bar_svg("train_1k")}</div>
    <p class="caption">val_96, same buckets:</p>
    <div class="chart-wrap">{bucket_bar_svg("val_96")}</div>
    {lp.legend([
        '<span class="sw" style="background:#1b2028;border:1px solid #383f4a"></span>0% (zero glow)',
        '<span class="sw" style="background:#22384a"></span>tiny (0&ndash;5%]',
        '<span class="sw" style="background:#1c527a"></span>medium (5&ndash;30%]',
        '<span class="sw" style="background:#0f6aab"></span>large (&gt;30%)',
    ])}
    </div>

    <div id="tokendist">
    <h3>Latent size distribution</h3>
    <p>Sparse 32³-latent token count per sample (the <code>coords</code> length the flow
      model actually trains on) &mdash; a <strong>200-sample random estimate</strong>
      pooled across both splits (not the full population; loading every
      {lp.filepath("shape_slat.pth", f"{SEGVIGEN_ROOT}/dataset/train_1k/<sid>/shape_slat.pth")}
      would be needlessly slow for a distribution this stable).</p>
    {lp.stat_tiles([
        (min(TOKEN_COUNTS), "min tokens"),
        (sorted(TOKEN_COUNTS)[len(TOKEN_COUNTS)//2], "median tokens"),
        (max(TOKEN_COUNTS), "max tokens"),
        (200, "samples in this estimate"),
    ])}
    <div class="chart-wrap">
      {token_hist_svg()}
    </div>
    </div>
"""

    # ---------------------------------------------------------------- Gallery
    gallery_body = f"""
    <p>12 samples drawn uniformly at random from each emissive_frac bucket
      (<strong>seed {SEED}</strong>, deterministic &mdash; not hand-picked), rendered at a
      consistent 3/4 camera + light backdrop (same rendering approach as the fine-tune
      predictions page's Section A). Each card: appearance (left) vs. the white/black GT
      target (right).</p>
    {lp.callout('''
      Renders show each asset in its <strong>dataset (somage) frame</strong> &mdash; the
      exact orientation the voxelization/training pipeline and the model itself see, not
      necessarily how a person would naturally pose the object. A minority of source assets
      were authored sideways at that layer, and appear so here too &mdash; that's expected,
      not a rendering bug.
    ''')}
    {gallery_section()}
"""

    # ---------------------------------------------------------------- Provenance
    provenance_body = f"""
    <p><strong>train_1k</strong> = 1123 samples (224 reused from an earlier build + 900
      fresh &minus; 1 dropped for a missing {lp.filepath("cond.pth", f"{SEGVIGEN_ROOT}/dataset/train_1k/<sid>/cond.pth")}, attrition
      1/1005 &asymp; 0.1%). <strong>val_96</strong> = 111 samples (6 reused + 105 fresh).
      Both are drawn from a PBR-passing pool of 26,264 / 59,602 candidate shapes (44%)
      &mdash; the <code>--pbr_only</code> filter drops fully-lit/baked shapes that have no
      clean PBR signal to learn emissive-vs-albedo from.</p>

    <h3>How the pick list was made</h3>
    <p class="caption">From the full 858K-asset TexVerse corpus down to train_1k/val_96 &mdash;
      every stage below, verified on the cluster (not re-derived from memory).</p>
    {pick_list_flow_html()}
    {lp.honesty_box(
        "<b>Why does an all-emissive corpus still contain 24.5% zero-glow shapes?</b> "
        "\"Declared emissive\" &ne; \"effectively emissive\": glTF materials can carry "
        "<code>emissiveFactor&gt;0</code> with an all-black emissive texture, or emission too weak "
        "for the per-face labeling thresholds &rarr; <code>emissive_frac=0</code> at label level. "
        "(Same reason Dongchen's 1099 \"emission_filtered\" split removed 55 zero-emission samples.)")}
"""

    body_sections = [
        lp.section("overview", 1, "Overview", body_html=overview_body, preview_rem=None),
        lp.section("distributions", 2, "Distributions",
                   takeaway="Both splits are heavily right-skewed toward little-to-no glow, with a "
                            "long tail of mostly-glowing shapes &mdash; and emissive_frac itself is a "
                            "raw triangle-count ratio, not area-weighted, so it disagrees with true "
                            "voxel coverage by up to 15&times; on individual shapes.",
                   body_html=distributions_body, preview_rem=42),
        lp.section("gallery", 3, "48 random examples from train_1k",
                   takeaway="These 48 cards are a seeded random draw per glow bucket, not hand-picked "
                            "&mdash; direct evidence the pipeline produces the claimed appearance/target "
                            "pairs at scale, sideways-authored source assets included.",
                   body_html=gallery_body, preview_rem=42.5),
        lp.section("provenance", 4, "Provenance",
                   takeaway="train_1k and val_96 are both drawn from a PBR-passing pool of just "
                            "26,264/59,602 (44%) train candidates &mdash; and the upstream "
                            "80,735&rarr;74,503 pre-filter is itself undocumented.",
                   body_html=provenance_body, preview_rem=42.5),
        '<footer>Lightgen war room &middot; segvigen_emissive/vis_data/dataset_gallery_html &middot;\n'
        '    dataset_stats.json (compute_dataset_stats.py, cluster) &middot; 48/1123 train_1k samples rendered</footer>',
    ]

    html = lp.page(
        title="SegviGen emissive fine-tune — dataset statistics + gallery",
        header_html=lp.header(
            "The fine-tuning dataset: statistics + gallery",
            'Full population statistics for both dataset splits (not just the '
            "gallery subset below), plus 48 randomly-sampled real examples so it's evident the "
            'build pipeline actually produced these at scale. Companion to '
            '<a href="../finetune_binary_v1/index.html">the fine-tune predictions page</a>, '
            'whose Section A walks through how one sample is made.'),
        body_sections=body_sections,
        outline_entries=[
            {"id": "overview", "label": "Overview"},
            {"id": "distributions", "label": "Distributions", "sub": [
                {"id": "fracdist", "label": "Emissive fraction"},
                {"id": "tokendist", "label": "Latent size"},
            ]},
            {"id": "gallery", "label": "Gallery"},
            {"id": "provenance", "label": "Provenance"},
        ],
        needs_katex=False,
    )

    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write(html)
    print(f"\nwrote {OUT}/index.html")


if __name__ == "__main__":
    main()
