"""
Assemble vis_data/pbr_filter_html/index.html: what the `--pbr_only` train filter
keeps vs drops. The filter drops 56% of the 80,735-shape emissive corpus
(45,829 non-PBR-tagged shapes) -- this page shows pbrType is a TOOLING
signature (SubstancePainter vs. Minecraft/voxel/AI-gen tags), not a quality
signal, and that the dropped shapes are actually MORE emissive-labeled than
the kept ones (86.1% vs 75.9% nonzero-emissive).

Reads the 48 thumbnails + sids.json (kept/dropped, 24 each, seed 42) and
names.json (name/tags/categories per sid, pulled from TexVerse metadata.json.1
on the cluster) in vis_data/pbr_filter_thumbs/.

  python build_pbr_filter_page.py
"""
import os, json, shutil, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
import xgpage as lp  # the installed package (uv pip install -e ~/studio/xgpage); migrated 2026-07-22

SRC = os.path.join(ROOT, "vis_data", "pbr_filter_thumbs")
OUT = os.path.join(ROOT, "vis_data", "pbr_filter_html")
os.makedirs(OUT, exist_ok=True)

TEXVERSE_META = "/3dlg-falas/project/omages/datasets/TexVerse/TexVerse/metadata.json.1"

SIDS = json.load(open(os.path.join(SRC, "sids.json")))
NAMES = json.load(open(os.path.join(SRC, "names.json")))

# ---------------------------------------------------------------- copy thumbnails
missing = []
for group in ["kept", "dropped"]:
    for sid in SIDS[group]:
        fn = f"{group}_{sid}.png"
        s = os.path.join(SRC, fn)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(OUT, fn))
        else:
            missing.append(fn)
if missing:
    print(f"MISSING {len(missing)} thumbnails: {missing}")
else:
    print(f"all {len(SIDS['kept']) + len(SIDS['dropped'])} thumbnails present")


# ---------------------------------------------------------------- gallery cards
def group_cards(group, cls):
    cards = []
    for sid in SIDS[group]:
        info = NAMES.get(sid, {"name": "(unknown)", "tags": [], "cats": []})
        cards.append(lp.thumb_card(
            f"{group}_{sid}.png",
            info["name"],
            tags=info["tags"][:3],
            cls=cls,
        ))
    return cards


def main():
    # ---------------------------------------------------------------- Overview
    overview_body = f"""
    <p>The train split's <code>--pbr_only</code> filter keeps only shapes tagged with a
      PBR material workflow (<code>pbrType</code> in {{metalness, specular}}) and drops
      everything else. This page asks: what does that filter actually select for?</p>
    {lp.verdict_box(
        'The <code>--pbr_only</code> filter drops 56% of the emissive corpus '
        '(45,829 of 80,735). Investigation shows <code>pbrType</code> is a '
        '<strong>TOOLING signature, not a quality signal</strong> &mdash; and the dropped '
        'shapes are actually <strong>MORE emissive-labeled</strong> than the kept ones. The '
        "filter's real justification (baked-lit input confusing albedo) was never "
        'cleanly tested.')}
    {lp.stat_tiles([
        ("80,735", "emissive corpus"),
        ("34,906", "PBR-kept"),
        ("45,829 (56%)", "non-PBR-dropped"),
        ("15×", "substancepainter-tag gap"),
        ("86% vs 76%", "dropped-shapes emissive-labeled vs kept"),
    ])}
"""

    # ---------------------------------------------------------------- Findings
    findings_body = f"""
    <ol class="next-bullets">
      <li><strong>pbrType is a tooling signature.</strong> The <code>substancepainter</code>
        tag appears on 22.6% of PBR-tagged shapes vs. just 1.5% of non-PBR shapes &mdash;
        a 15&times; gap. Shapes with clean metalness/specular PBR materials are
        disproportionately ones exported from Substance Painter, not shapes that are
        objectively higher quality.</li>
      <li><strong>The "scans / baked-lit" hypothesis is debunked.</strong> A keyword match
        for scan/bake-adjacent terms lands at 15.6% (PBR) vs. 14.8% (non-PBR) &mdash;
        essentially equal, no signal. What <em>is</em> concentrated in the non-PBR pool is
        a different cluster entirely: <code>minecraft</code> at 3.16% vs. 0.11% (29&times;),
        <code>voxel</code> at 9&times;, <code>createdwithai</code> at 5.7%, and
        <code>fnaf</code> at 2.0%. The non-PBR pool skews toward specific fan/hobbyist/AI
        content categories, not toward degraded scans.</li>
      <li><strong>Non-PBR shapes are denser, not junkier.</strong> Median vertex count is
        23,688 for non-PBR shapes vs. 11,697 for PBR-tagged ones &mdash; more than double.
        Whatever the filter is selecting for, it isn't simplicity or low geometric
        fidelity.</li>
      <li><strong>Non-PBR shapes are less curated by their uploaders</strong>, not
        necessarily lower quality: 46.3% of non-PBR shapes have zero Sketchfab categories
        assigned, vs. 34.0% of PBR-tagged shapes.</li>
    </ol>
    {lp.honesty_box(
        '<strong>The cost.</strong> Non-PBR-labeled shapes are <strong>86.1%</strong> '
        'nonzero-emissive vs. <strong>75.9%</strong> for kept shapes &mdash; the filter '
        'discards roughly <strong>24,900</strong> usable emissive-labeled train shapes to '
        'keep roughly <strong>18,270</strong> (about 35% MORE usable shapes discarded than '
        'retained). Its actual justification (that baked-lit/fully-lit input confuses the '
        "model's albedo signal) was never cleanly tested: a <code>pandas.NA</code> "
        'comparison bug meant no real filtered-vs-unfiltered ablation run ever happened. '
        '<strong>Next experiment:</strong> a proper PBR-filter ablation, and evaluating '
        '<code>#n_pbr_materials</code> as a better-founded proxy than the binary '
        '<code>pbrType</code> tag.')}
"""

    # ---------------------------------------------------------------- Gallery
    gallery_body = f"""
    <p class="caption">Sketchfab-authored thumbnails &mdash; arbitrary lighting/angle/framing,
      shown as-is; these are the artist's own preview renders, not our pipeline's.</p>
    <p>24 seeded-random (seed 42) shapes from each group of the 80,735 corpus.</p>
    {lp.bucket_block("Kept (PBR-workflow) &mdash; 24 of 34,906", group_cards("kept", "kept"))}
    {lp.bucket_block("Dropped (non-PBR) &mdash; 24 of 45,829", group_cards("dropped", "dropped"))}
"""

    body_sections = [
        lp.section("overview", 1, "Overview", body_html=overview_body, preview_rem=None),
        lp.section("findings", 2, "Findings: what pbrType actually measures",
                   takeaway="pbrType tracks which tool an artist used (Substance Painter) far "
                            "more than any quality or renderability signal &mdash; and the "
                            "shapes it discards are denser, less-curated, and MORE often "
                            "emissive-labeled than the ones it keeps.",
                   body_html=findings_body, preview_rem=40),
        lp.section("gallery", 3, "48 examples: kept vs. dropped",
                   takeaway="Side by side, the kept and dropped pools don't look like "
                            "'clean' vs. 'broken' &mdash; they look like two different corners "
                            "of Sketchfab (professional PBR exports vs. Minecraft/voxel/fan/AI "
                            "content).",
                   body_html=gallery_body, preview_rem=44),
        '<footer>Lightgen war room &middot; segvigen_emissive/vis_data/pbr_filter_html &middot;\n'
        f'    48/80,735 shapes &middot; names/tags from TexVerse '
        f'{lp.filepath("metadata.json.1", TEXVERSE_META, "(cluster)")}</footer>',
    ]

    html = lp.page(
        title="PBR filter: what it keeps vs drops",
        header_html=lp.header(
            "What the PBR filter keeps vs drops",
            'The train split\'s <code>--pbr_only</code> filter drops 56% of the 80,735-shape '
            'emissive corpus. This page characterizes what it actually selects for, and what '
            'that costs. Companion to '
            '<a href="../dataset_gallery_v1/index.html">the dataset gallery</a>, whose '
            'provenance funnel shows where this filter sits in the pipeline.'),
        body_sections=body_sections,
        outline_entries=[
            {"id": "overview", "label": "Overview"},
            {"id": "findings", "label": "Findings"},
            {"id": "gallery", "label": "Gallery"},
        ],
        needs_katex=False,
    )

    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write(html)
    print(f"\nwrote {OUT}/index.html")


if __name__ == "__main__":
    main()
