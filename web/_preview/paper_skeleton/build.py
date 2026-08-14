#!/usr/bin/env python3
"""Build the LightGen paper-skeleton page: a claim chain on the v3 workspace shell.

The page is the `.skel` claim-chain component (one atomic claim per numbered
line, grouped under uppercase heads) inside the xgpage v3 workspace shell: left
page tree, 820px content column, right per-page outline tracking the heads.

The content is read at build time from
segvigen_emissive/PAPER_SKELETON_V2_CLAIMS.md and is never restated here, so
the page cannot drift from the file. skel.verify() asserts that every rendered
counter value equals the number written in the file; a markup or CSS change
that shifts the numbering fails the build.

Claims are stated as the paper's TARGET by explicit owner decision. Nothing is
merged, split, expanded, reordered, softened or added.

It publishes to workspace/paper_skeleton/, not under _preview/, because that is
where the zone's tree.json points.

Run: .venv2/bin/python web/_preview/paper_skeleton/build.py
  (.venv2 = /cs/3dlg-project/3dlg-hcvc/omages/omages_internal/.venv2/bin/python)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WEB)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import diagrams as D  # noqa: E402
import skel  # noqa: E402
import workspace_zone as wz  # noqa: E402  (read-only: tree entries + the zone guard)

import xgpage as lp  # noqa: E402
from xgpage.publish import publish_assets  # noqa: E402

SITE_ROOT = "/projects/omages/yanxg/lightgen"
SITE_ASSETS = f"{SITE_ROOT}/assets"
FAVICON = f"{SITE_ROOT}/assets/images/favicon.png"

CLAIMS_MD = os.path.join(REPO, "segvigen_emissive", "PAPER_SKELETON_V2_CLAIMS.md")

# This page lives in the research WORKSPACE zone, not under _preview/.
#
# PLACEMENT IS PROVISIONAL (2026-08-06). The claim chain is a SECOND artifact
# alongside the prose paper skeleton, which keeps `workspace/paper_skeleton/`.
# A distinct slug is used here deliberately: the lead is evaluating a port of
# the somages version tooling, whose immutable `<page>/v/N/` snapshots and
# versions.json manifest have their own directory expectations, so nothing here
# imitates that layout. Re-point this constant when the scheme is decided.
PAGE_SLUG = "paper_skeleton_claims"
PAGE_HREF = f"{wz.WORKSPACE_URL}/{PAGE_SLUG}/index.html"
PUBLISH_DIR = os.path.join(str(wz.WORKSPACE_DIR), PAGE_SLUG)
PROSE_HREF = f"{wz.WORKSPACE_URL}/paper_skeleton/index.html"
PAGE_DATE = "2026-08-06"

# The heads after whose section the claim list is broken so a figure can sit
# between two segments. Three figures, three breaks.
FIGURE_BREAKS = ["Why it is hard", "Relation to concurrent work", "Method"]

# ---------------------------------------------------------------- zone tree
# The zone's own entries (tools/workspace_zone.py) plus a "Paper" group this
# page opens. Kept here rather than edited into workspace_zone.py because that
# module is outside this page's write scope; write_tree_json() there will
# overwrite the published tree.json unless the same group lands in
# wz.tree_entries().
# The group describes the PROSE skeleton, which owns the tree slot. The claim
# chain is deliberately NOT added yet: its placement (and whether it appears as
# a sibling leaf or as a version of the same page) is the lead's call, pending
# the versioning decision. Adding a leaf now would prejudge that.
PAPER_GROUP = {
    "label": "Paper",
    "children": [
        {"label": "Paper skeleton", "href": PROSE_HREF,
         "meta": "plan of record: claims, evidence, what is unearned"},
    ],
}


def tree_entries():
    """wz.tree_entries() with the Paper group inserted after Overview."""
    entries = wz.tree_entries()
    at = next((i for i, g in enumerate(entries) if g.get("label") == "Overview"), -1)
    entries.insert(at + 1, json.loads(json.dumps(PAPER_GROUP)))
    return entries


def tree_html():
    entries = tree_entries()
    for group in entries:
        for leaf in group.get("children", []):
            leaf["active"] = leaf["href"] == PAGE_HREF
    return lp.v3_tree(entries, title="Lightgen", subtitle="research workspace",
                      tree_src=wz.TREE_JSON_URL)


def row_left(*cells):
    """A table row of running text: every cell left-aligned (theme2 .rowhead)."""
    return "<tr>" + "".join(f'<td class="rowhead">{c}</td>' for c in cells) + "</tr>"


# ------------------------------------------------------------ page-local CSS
EXTRA_CSS = """
<style>
/* The .skel claim chain, from the somages semantic_charting reference. One
   change against that source: `def` joins `head` and `open` outside the
   counter, because the content file's numbering runs 3, DEF, 4 and opens the
   Method section with a DEF before claim 25. See skel.py. */
/* The counter is reset on the WRAPPER, not on each list. A counter-reset
   scopes the counter to that element and its descendants, so a `skl` created
   on the first <ol> is out of scope in the next one and every later segment
   would restart at 1 (seen in the pixels; getComputedStyle reports the
   counter-reset honestly but cannot resolve counter() in ::before, so only a
   screenshot catches it). One wrapper puts all four segments in one scope. */
.skelwrap{counter-reset:skl}
.skel{counter-reset:none;list-style:none;padding:0;margin:0}
.skel li{counter-increment:skl;position:relative;padding:.34em 0 .34em 2.1em;
  border-top:1px solid var(--line)}
.skel li:first-child{border-top:0}
.skel li::before{content:counter(skl);position:absolute;left:0;top:.42em;
  font:11px ui-monospace,Menlo,monospace;color:var(--ink-3,var(--ink-2))}
.skel li.def{font-weight:600}
.skel li.open,.tbw{color:var(--ink-2);font-style:italic}
.skel li.open,.skel li.head,.skel li.def{counter-increment:none}
/* ...and print no number at all: stopping the increment still leaves ::before
   showing the PREVIOUS item's value, which reads as a claim number that is not
   in the content file (a def showed a stale 6, an open a stale 11). */
.skel li.open::before,.skel li.head::before,.skel li.def::before{content:none}
.skel li.head{font:11px ui-monospace,Menlo,monospace;letter-spacing:.08em;
  text-transform:uppercase;color:var(--ink-2);padding-left:0;padding-top:1.1em}
.skel li.head:first-child{padding-top:.2em}
/* the head is a scrollspy target: give it room to clear the sticky top bar */
.skel li.head{scroll-margin-top:70px}

/* Diagram labels stay legible at phone width: the shared rule floors the SVG at
   640px, which shrinks an 820-unit viewBox's 12px type under the readable floor.
   These diagrams scroll inside their own figure instead (SKILL.md rule 5). */
.xg2 .diagram svg { min-width: 760px; }

.xg2 .toc .num { display: none; }

/* v3: the 820px content column IS the measure (theme3.css sets .prose and
   .chart to max-width:none), so the list and the figures fill it and share one
   left edge. D12's viewport-centering is superseded here. */

/* The comparison matrix must fit the column, not scroll inside it: its point is
   that both sides are in one glance, and its natural width was 960px against an
   820px column (measured). Fixed layout with explicit shares wraps the prose
   cells instead. */
/* theme.css carries a v1-era `table.results{min-width:960px}` plus
   `td.rowhead{min-width:210px}`, which every v3 page inherits, so ANY results
   table on the 820px column scrolls internally no matter how short its cells
   are. Both floors are lifted here. Worth upstreaming as a v3 override. */
.skelwrap table.results{table-layout:fixed;width:100%;min-width:0}
.skelwrap table.results td.rowhead{min-width:0}
.skelwrap table.results th:nth-child(1),.skelwrap table.results td:nth-child(1){width:20%}
.skelwrap table.results th:nth-child(2),.skelwrap table.results td:nth-child(2),
.skelwrap table.results th:nth-child(3),.skelwrap table.results td:nth-child(3){width:40%}
/* at phone width a 20% axis column is narrower than the word "Representation",
   which then breaks mid-word; give that column a bigger share instead. */
@media (max-width:700px){
  .skelwrap table.results th:nth-child(1),.skelwrap table.results td:nth-child(1){width:30%}
}

/* Scroll room past the end. xg3.js's scrollspy marks the nearest [id] above a
   110px line; the last head sits close enough to the page bottom that scrolling
   all the way down still left it 413px below that line, so "Results" could
   never light up in the outline. Measured, not guessed. */
.xg3 .v3-main .page::after{content:"";display:block;height:42vh}
</style>
"""


def build():
    assets_dir = os.path.join(WEB, "assets")

    items = skel.parse(CLAIMS_MD)
    n_claims = skel.verify(items)      # build fails if numbering diverges
    head_ids = skel.heads(items)
    segments = skel.render(items, breaks=FIGURE_BREAKS)
    assert len(segments) == len(FIGURE_BREAKS) + 1

    # ================================================================== hero
    hero = lp.hero_header(
        "lightgen &middot; paper skeleton &middot; claim chain &middot; " + PAGE_DATE,
        "LightGen Paper Skeleton: The Claim Chain",
        dek_html=(
            "One atomic claim per numbered line, in the order the paper argues them. "
            "Each line becomes one sentence. <b>Bold</b> lines are a definition or the "
            "central claim of their section; <i>italic uncounted</i> lines are questions "
            "we have not settled. Claims are stated as the paper's target. Numbers appear "
            "only where they are verified."
        ),
        toc=[(hid, label) for hid, label in head_ids],
    )

    # ============================================================== figures
    fig_coverage = D.svg_figure(
        *D.diagram_coverage(),
        caption_html=(
            "<b>Two populations, not one: 63.1 percent of shapes are at or below 0.1 "
            "emissive coverage and 22.9 percent are above 0.5, with 14.0 percent "
            "between.</b> Bands are the reported thresholds and are not equal in width, "
            "so the bars are shares of shapes, not a density; the five measured numbers "
            "are the shares above 0, 0.001, 0.01, 0.1 and 0.5, and the six band shares "
            "are their differences. Carries claims 5, 6 and 21."
        ),
        width_px=820, id="fig-coverage")

    cmp_rows = "".join([
        row_left("<b>Input</b>",
                 "a reference image that already shows the object glowing",
                 "geometry and material alone"),
        row_left("<b>Representation</b>",
                 "multi-view, fused to UV",
                 "TRELLIS.2's sparse 3D latent, natively"),
        row_left("<b>Place in a pipeline</b>",
                 "textures an existing mesh from a photograph",
                 "adds a stage to image-to-3D generation"),
    ])
    fig_compare = (
        lp.results_table(["axis", "EmissionGen", "ours"], cmp_rows)
        + lp.chartnote("<b>The two setups differ at the input, before any modeling "
                       "choice.</b> One row per claim: 16, 17, 18.")
    )

    fig_method = D.svg_figure(
        *D.diagram_mask_albedo(),
        caption_html=(
            "<b>The model produces one bit per voxel; the color comes from the input it "
            "was already given.</b> A 9&times;9 cross-section of a lamp, drawn from a cell "
            "table in code. Left, the albedo the PBR stage already produced. Centre, the "
            "binary mask, white where the surface emits. Right, their product: the shade "
            "keeps its own color, everything else is exactly zero. A schematic of the "
            "formula, not a model output. Carries the Method definition and claims 25 "
            "and 26."
        ),
        width_px=820, id="fig-mask-albedo")

    # One wrapper around every segment and figure, so the claim counter has a
    # single scope and numbering runs unbroken across the figure breaks.
    body = ['<div class="skelwrap">' + "".join([
        segments[0], fig_coverage,
        segments[1], fig_compare,
        segments[2], fig_method,
        segments[3],
    ]) + "</div>"]

    apx = lp.appendix("Provenance", [
        "Every line on this page is read at build time from "
        "<code>segvigen_emissive/PAPER_SKELETON_V2_CLAIMS.md</code>, the content of "
        f"record. {n_claims} numbered claims, "
        f"{sum(1 for k, _, _ in items if k == 'def')} definitions, "
        f"{sum(1 for k, _, _ in items if k == 'open')} open questions, "
        f"{len(head_ids)} sections. Nothing is merged, split, expanded, reordered or "
        "added, and the build asserts that every rendered number equals the number "
        "written in the file.",
        "The coverage chart plots the five survival percentages from the content "
        "document; its six band shares are their differences, which the caption states. "
        "The method diagram is a computed schematic: cell coordinates come from a table "
        "in <code>diagrams.py</code>. The comparison matrix restates claims 16, 17 and "
        "18, one per row.",
        "No model output appears anywhere on this page.",
    ])

    page_html = lp.page(
        title="LightGen Paper Skeleton: The Claim Chain",
        header_html=hero,
        body_sections=body + [apx],
        assets_rel=SITE_ASSETS,
        assets_dir=assets_dir,
        theme="v3",
        tree_html=tree_html(),
        nav_title="Paper skeleton",
        # Date-frozen slot: the zone deliberately runs no versioning machinery
        # (workspace_zone.py, LITE scope), and this mode needs no versions.json.
        version_slot=lp.v3_version_slot(date=PAGE_DATE),
        needs_katex=False,
        extra_head=f'<link rel="icon" href="{FAVICON}">' + EXTRA_CSS,
        outline_entries=[{"id": hid, "label": label} for hid, label in head_ids],
    )

    # ZONE-BOUNDARY LAW: nothing in the workspace zone may link to the operator
    # console. The console lives at the site ROOT, so this checks its real
    # hrefs rather than a "/console/" substring. Fail the build, never publish.
    violations = wz.console_links_in(page_html)
    if violations:
        sys.exit(f"ZONE-LINK GUARD FAILED: page links to the console: {violations}")

    out_path = os.path.join(HERE, "index.html")
    with open(out_path, "w") as f:
        f.write(page_html)
    print(f"wrote {out_path}")
    print(f"  {n_claims} claims, numbering verified against the content file")
    print("  zone-link guard: clean")

    publish_assets(assets_dir)
    print(f"assets published -> {assets_dir}")


if __name__ == "__main__":
    build()
