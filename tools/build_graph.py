#!/usr/bin/env python3
"""build_graph.py — the Lightgen page-relationship graph (graph.json + graph.html).

An Obsidian-style graph of the project's published pages: nodes are pages,
directed edges are content links between them ("builds on / cites"), read
straight out of each page's own rendered HTML. The point is orientation —
seeing how the project's pages connect and grew — for the owner and for
future agents, not a demo.

Usage:
    .venv_console/bin/python tools/build_graph.py             # graph.json + graph.html -> live root
    .venv_console/bin/python tools/build_graph.py --check      # rebuild graph.json only, print counts, don't write the page

Doubles as the thin per-artifact rebuild command (same pattern as
build_jobs.py): a full console rebuild (build_console.py) also calls
build_graph_tab(), but an agent who only changed page content can run this
file alone to refresh the graph without a full console rebuild.

CONTRACT (graph.json, the renderer-agnostic output — read this before writing
another renderer against it):
    {
      "generated_at": "<iso8601>",
      "nodes": [{"id", "title", "label", "tier", "url", "created", "modified",
                 "x", "y", "in_degree", "out_degree", "has_upstream"}, ...],
      "edges": [{"source", "target", "type": "link"}, ...]
    }
"id" is the same page identity inventory_pages.py already uses (its "name"
column — bare for root/preview tiers, "updates/<date>" / "workspace/<slug>"
for the two nested tiers). "title" is the page's full rendered title (shown
on hover only, round-3); "label" is what the renderer actually draws under/
beside the node -- web/pages.yaml's `shortname:` if the page registers one,
else the bare id (round-3, owner: "there is a need for a unique shortname
for the job, like an id, not a full name which is very long"). EVERY edge is
now a directed arrow (round-3, owner: "we want directed graph, instead of
undirected graph. show arrows instead" -- source is the arrow's tail,
target is its head, for every type below, no exceptions). "type" is "link"
for a crawled content link, "upstream" for a page.yaml `upstreams:` entry
(round-3 -- source MOTIVATED target's creation; see load_upstream_edges()),
or one of "supersedes"/"evidence-for"/"part-of"/"same-page" for a curated
edge from web/graph_edges.yaml (round-2 addition, extended round-3) --
hand-verified against the actual page content, never crawled or invented;
see xgpage.graph.load_typed_edges(). "has_upstream" is true iff the node is
the TARGET of at least one "upstream" edge -- the completeness signal
(round-3, replaces the old zero-degree "orphan" list): a page with none is
either a genuine root or missing its upstreams: entry, surfaced the same
way the Pages tab surfaces an uncurated page.
x/y are a stable 2D layout in an abstract ~1200x800 coordinate space (not pixels —
the renderer fits it to whatever canvas it has); Timeline mode does not use
x/y at all -- see graph_view.js's computeTimelineLayout() for its own
independent, non-persisted bin/stack positions.

NODES come from inventory_pages.scan_rows() (the exact cached scan that also
backs pages.json — no separate rescan of the published tree). Deviation from
the original architecture note ("nodes from pages.json"): pages.json bakes
its rows as pre-rendered HTML table cells for the itables widget, so this
reads the raw scan function that PRODUCES those rows instead of re-parsing
HTML out of them — same data, one less fragile round-trip, and it can never
drift from what the Pages tab shows since both are the same cached scan.

A REAL GAP this surfaced: scan_rows() only walked WEBDIR's top level (root),
_preview/, and updates/ — the research-workspace zone's own living pages
(workspace/rendering, workspace/render_sweep, workspace/diagnostics,
workspace/paper_skeleton) sit one level deeper and were invisible to the
scan entirely (workspace/index.html itself was the only workspace URL ever
seen, as a root-tier page). Without those four pages as graph NODES the
graph could not represent the very edge the QA brief asks for by name (the
Rendering-setups page linking the Render-sweep page) — so this required
extending inventory_pages.py with a new "workspace" tier (mirrors the
existing "update" tier's one-level-deeper pattern), not just this file.
theme3.css already ships a `.tier-workspace` badge (xgpage skill rule 0(e),
added for cobalt3d's identical gap), so no CSS was needed for the fix to
show up correctly in the Pages tab too.

EXTRACTED 2026-08-10 into xgpage.graph (mirrors the console and jobs-board
extractions): the crawler (chrome exclusion, href resolution), layout
persistence, and the typed-edge loader now live in the package as pure,
GraphConfig-driven functions -- see xgpage.graph's module docstring for the
full account of each (chrome-exclusion class names, the header/citation
false-negative that shaped it, href resolution rules, and the layout
algorithm). This file keeps only what stays PROJECT-OWNED: node discovery
(scan_nodes(), reading inventory_pages.py's own page-inventory scan --
another project's page model may look nothing like this one), reading each
node's HTML off PUBLISH_DEST (the callable handed to xgpage.graph.scan_edges
as html_reader), graph_edges.yaml itself (the curated DATA, never the
loading mechanism), and the graph PAGE (console-shell assembly via
build_console.py, exactly like build_jobs_tab() -- project-owned, not part
of the package).

web/assets/graph_view.js + graph_view.css (the SVG rendering: pan/zoom/
hover/search/timeline mode) and the vendored d3-force UMD files
(web/assets/vendor/d3-*.v3.*.min.js -- physics only, four small files
fetched once from unpkg and checked in, verified to wire up under a plain
<script> tag, no CDN at runtime) also moved into the package's shared
assets this round (previously LIGHTGEN-LOCAL siblings sync_xgpage_assets.py
preserved untouched; now synced FROM the package like theme3.css/xg3.js
already were -- see that script's updated docstring). The interaction
model, timeline mode, and typed-edge rendering documented below are
unchanged; only WHERE the files' canonical copy lives changed.

ROUND 2 (2026-08-10, owner-directed, three additions on top of the above):
1. Interaction state machine: click/hover/search/select were rebuilt around
   an explicit model with a guaranteed reset (background click or Escape
   clears everything) after the owner reported a real bug (a hover-then-
   click sequence could leave the page permanently dimmed until refresh).
   Fully documented in graph_view.js's own "interaction state" comment
   block; the permanent regression test is tools/qa_graph_journeys.js
   (7 scripted journeys, run it after any change to the interaction code).
2. Timeline mode (Map/Timeline toggle in the toolbar): the SAME nodes and
   edges, laid out deterministically by `created` date (x) and tier (y,
   fixed lane order root/workspace/preview/update), no physics, so
   development flow reads left to right instead of as a force-directed
   cluster. See graph_view.js's computeTimelineLayout().
3. Curated typed edges (web/graph_edges.yaml, load_typed_edges() below):
   hand-verified relationships (supersedes/evidence-for/part-of) the
   content-link crawler cannot see, merged into graph.json's edges with
   their real "type" (the field the v1 contract reserved for exactly this).
   Rendered dashed with the relationship word labeled on the edge itself.

ROUND 3 (2026-08-14, owner-directed, on top of the above):
1. Push-out layout + label-reveal-by-interaction: see xgpage.graph's module
   docstring ("PUSH-OUT / DECLUTTER") for the layout mechanism, and
   graph_view.css/js for the hide-by-default label mechanism -- both fixed
   the owner's "attracting behaviour still clutters" report.
2. DIRECTED graph, arrows everywhere (owner: "we want directed graph,
   instead of undirected graph. show arrows instead"): every edge, of
   every type, renders with a visible arrowhead at its target, oriented by
   the edge's own source/target -- no more implicitly-undirected map mode.
3. The `upstreams:` convention (owner: "add a guideline... for each page
   we should write the upstreams: which page motivates the creation of
   this page"): web/pages.yaml's per-page `upstreams:` list is now the
   PRIMARY edge store for motivation relationships, load_upstream_edges()
   below; graph_edges.yaml narrows to relations that are NOT motivation
   (supersedes, same-page). Replaces the old zero-degree "orphan" list
   with a zero-upstream completeness list (has_upstream in the contract).
4. Node labels switch from full titles to each page's short id or
   registered `shortname:` (owner: "a unique shortname for the job, like
   an id, not a full name which is very long"); the title moves to a
   hover tooltip. See node_reach_excess()'s docstring for why this also
   simplified the round-3(1) declutter math.
5. Timeline mode rebuilt around the owner's sketch: no more tier lanes
   (color still encodes tier, via the same legend); pages bin by date
   (algorithmic, deterministic -- see graph_view.js's computeTimelineBins())
   and stack VERTICALLY within a bin; a year/month header sits above the
   bin row; edges render as directed arcs. All of this lives in
   graph_view.js/css (client-side, computed fresh each load from
   graph.json) -- nothing here in the Python layer changed for it beyond
   the "label"/arrow-everywhere contract fields above.
"""
import argparse
import datetime
import hashlib
import html
import json
import pathlib
import sys
from urllib.parse import urlsplit

from xgpage import graph as xg

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import build_console as bc  # noqa: E402
import inventory_pages as ip  # noqa: E402

PUBLISH_DEST = bc.PUBLISH_DEST
BASE_URL = bc.BASE_URL
SITE_ROOT = bc.SITE_ROOT
SITE_HOST = urlsplit(BASE_URL).netloc

GRAPH_EDGES_YAML = REPO / "web" / "graph_edges.yaml"

GRAPH_CONFIG = xg.GraphConfig(
    site_root=SITE_ROOT,
    site_host=SITE_HOST,
    positions_file=REPO / ".console_build" / "graph_positions.json",
    typed_edges_path=GRAPH_EDGES_YAML,
    # "same-page" added round-3: two published URLs that are really the
    # SAME living page (a preview build later version-minted into the
    # workspace zone under a different URL, e.g. paper_v3 / workspace/
    # paper_skeleton) aren't supersedes/evidence-for/part-of -- none of
    # those three claim identity, they claim a relationship BETWEEN two
    # distinct things.
    typed_edge_types=frozenset({"supersedes", "evidence-for", "part-of", "same-page"}),
)

# --------------------------------------------------- label-aware reach --
# Calibration for xg.compute_layout()'s node_reach declutter (round-3).
# FIRST ATTEMPT (kept here as the documented failure, not deleted, so it
# isn't retried the same way): giving every node its full rendered-label
# footprint in world units inflated even ordinary short-titled nodes'
# minimum spacing, which grew the whole bounding box enough that map
# mode's fitView() -- which always fits the entire largest connected
# component to its viewport -- shrank its zoom by very nearly the same
# factor, cancelling almost all of the intended on-screen gain (measured:
# fit zoom went from 0.155 to 0.029, and DIFFERENT pairs still overlapped).
# What actually works: node_reach carries only the EXCESS an outlier node
# needs beyond a baseline every ordinary node already clears at
# config.min_sep=170 (see graph.py's compute_layout docstring for why this
# is additive, not "everyone's full footprint") -- most nodes contribute
# 0, so the bulk of the layout is untouched; only a real outlier (a long
# title, a high in-degree marker) pushes its own pairs further apart.
# BASELINE_CHARS/BASELINE_INDEG: the "ordinary" node config.min_sep=170
# was already tuned around (an ordinary title, in-degree 0-1) -- picked
# from this project's own median, not a formula. FIT_ZOOM here is the
# measured fit zoom from the FIRST (flat min_sep=170, only 2 residual
# overlaps) build, i.e. "the zoom level excess distance actually has to
# clear px on screen at" -- re-measure and update if this project's
# node/edge count changes enough to meaningfully shift the fit zoom (a QA
# overlap check catching zero violations after a rebuild is the signal
# it's still good).
FIT_ZOOM = 0.155
LABEL_PX_PER_CHAR = 4.85
BASELINE_CHARS = 30
BASELINE_RADIUS = 9.0


def _radius_for(in_degree):
    """Mirrors graph_view.js's radiusFor() exactly -- the node marker's own
    rendered radius, part of its screen footprint."""
    r = 6 + min(in_degree, 12) * 1.4
    return max(6.0, min(22.0, r))


def node_reach_excess(label, in_degree):
    # Round-3: the renderer now draws each node's short `label` (a slug or
    # shortname), not its long `title` -- this function must match whatever
    # text is ACTUALLY on screen, or it tunes spacing for a string nobody
    # sees. Slugs are short enough that most nodes now land under
    # BASELINE_CHARS and contribute zero excess, which is correct: the
    # whole reason this became necessary (round-3's first, reverted
    # attempt) was long TITLE-length labels; slug labels mostly don't have
    # that problem, so this mechanism is now a rarely-triggered safety net
    # rather than routine tuning.
    chars = len(label) if len(label) <= 46 else 45  # matches graph_view.js's truncation
    excess_px = (max(0, chars - BASELINE_CHARS) * LABEL_PX_PER_CHAR
                 + max(0.0, _radius_for(in_degree) - BASELINE_RADIUS))
    return excess_px / FIT_ZOOM


# ------------------------------------------------------- upstream edges --
def load_upstream_edges(nodes, curation):
    """web/pages.yaml `upstreams:` entries -> directed edges (round-3, the
    PRIMARY motivation-edge store, replacing free-form edge sweeping for
    anything shaped like "X's existence motivated Y"). {source: upstream_id,
    target: this_id, type: "upstream"} -- source is the arrow's tail (the
    earlier, motivating page), target its head (the page it motivated),
    matching the owner's stated arrow convention directly with no
    reinterpretation needed. Only a REGISTERED page (one with a pages.yaml
    entry) can name upstreams; an unregistered node simply has none, and
    surfaces in the zero-upstream completeness list like any other page
    missing this field -- not a hard failure, mirrors load_typed_edges()'s
    "drop unknown id, warn, don't crash" discipline."""
    ids = {n["id"] for n in nodes}
    out = []
    seen = set()
    for nid in sorted(ids):
        meta = curation.get(nid)
        if not meta:
            continue
        for up in meta.get("upstreams") or []:
            if up not in ids:
                print(f"[pages.yaml] {nid}: dropping upstream {up!r}: unknown node id", file=sys.stderr)
                continue
            if up == nid:
                print(f"[pages.yaml] {nid}: dropping self-referential upstream", file=sys.stderr)
                continue
            key = (up, nid)
            if key in seen:
                continue
            seen.add(key)
            out.append({"source": up, "target": nid, "type": "upstream"})
    return out


# --------------------------------------------------------------- node scan --
def node_dir(url):
    """URL -> the page's directory path relative to PUBLISH_DEST, e.g.
    'workspace/rendering' or '_preview/data_compare'."""
    rel = url[len(BASE_URL) + 1:]
    if rel.endswith("/index.html"):
        rel = rel[: -len("/index.html")]
    return rel


def scan_nodes():
    rows = ip.scan_rows(use_cache=True)
    return [
        {"id": r["name"], "title": r["title"], "tier": r["tier"], "url": r["url"],
         "created": r["created"], "modified": r["modified"], "_dir": node_dir(r["url"])}
        for r in rows
    ]


# --------------------------------------------------------------- edge scan --
def _read_node_html(node):
    """The html_reader callable xgpage.graph.scan_edges() needs: how to get
    a node's rendered HTML off disk is this project's own I/O concern (here,
    PUBLISH_DEST/<node's dir>/index.html); the crawler itself only walks the
    text it's handed."""
    idx = PUBLISH_DEST / node["_dir"] / "index.html"
    return idx.read_text(errors="ignore") if idx.exists() else None


# ------------------------------------------------------------------ build --
def build_graph_data():
    nodes = scan_nodes()
    curation = ip.load_curation()
    edges = xg.scan_edges(nodes, _read_node_html, GRAPH_CONFIG)  # crawled; plain (a, b) tuples
    node_ids = sorted(n["id"] for n in nodes)
    typed_edges = xg.load_typed_edges(node_ids, GRAPH_CONFIG)  # curated; [{"source","target","type"}, ...]
    upstream_edges = load_upstream_edges(nodes, curation)  # pages.yaml `upstreams:`; [{"source","target","type":"upstream"}, ...]

    # LAYOUT and degree counting treat a typed OR upstream edge as a real
    # connection too (pulls nodes together and counts against orphan status
    # exactly like a content link does) -- combined as plain (a, b) pairs,
    # type-blind, for compute_layout() and the position-persistence
    # neighbor-set diff.
    typed_pairs = [(e["source"], e["target"]) for e in typed_edges]
    upstream_pairs = [(e["source"], e["target"]) for e in upstream_edges]
    all_pairs = sorted(set(edges) | set(typed_pairs) | set(upstream_pairs))
    in_deg_pre = {nid: 0 for nid in node_ids}
    for _a, b in all_pairs:
        in_deg_pre[b] += 1
    by_id = {n["id"]: n for n in nodes}
    labels = {nid: (curation.get(nid, {}).get("shortname") or nid) for nid in node_ids}
    node_reach = {nid: node_reach_excess(labels[nid], in_deg_pre[nid]) for nid in node_ids}
    positions = xg.compute_layout(node_ids, all_pairs, GRAPH_CONFIG, node_reach=node_reach)
    xg.save_positions(GRAPH_CONFIG, positions, all_pairs)

    in_deg = {nid: 0 for nid in node_ids}
    out_deg = {nid: 0 for nid in node_ids}
    for a, b in all_pairs:
        out_deg[a] += 1
        in_deg[b] += 1
    has_upstream = {nid: False for nid in node_ids}
    for _a, b in upstream_pairs:
        has_upstream[b] = True

    out_nodes = []
    for nid in node_ids:
        n = by_id[nid]
        x, y = positions[nid]
        out_nodes.append({"id": nid, "title": n["title"], "label": labels[nid], "tier": n["tier"],
                           "url": n["url"], "created": n["created"], "modified": n["modified"],
                           "x": x, "y": y, "in_degree": in_deg[nid], "out_degree": out_deg[nid],
                           "has_upstream": has_upstream[nid]})
    # A pair covered by a curated typed OR upstream edge is strictly more
    # informative than the generic crawled "link" between the same two
    # pages (found live, round 2: glb_direct_pilot_v1 already had a
    # content-link citation to pipeline_glb_direct, and the SAME pair is
    # also the one verified evidence-for relationship, which would
    # otherwise draw two overlapping edges for one relationship) --
    # suppress the plain link edge for any ordered pair a typed or
    # upstream edge already covers.
    covered_pairs = set(typed_pairs) | set(upstream_pairs)
    out_edges = ([{"source": a, "target": b, "type": "link"} for a, b in edges if (a, b) not in covered_pairs]
                 + typed_edges + upstream_edges)
    return {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "nodes": out_nodes, "edges": out_edges}


def write_graph_json(out_dir=None):
    data = build_graph_data()
    dest = pathlib.Path(out_dir or PUBLISH_DEST) / "graph.json"
    dest.write_text(json.dumps(data, indent=1))
    return data


# -------------------------------------------------------------- the page --
TIER_LEGEND = [("root", "root", "var(--good)"),
               ("workspace", "workspace", "var(--accent-ink)"),
               ("preview", "preview", "var(--blue-ink)"),
               ("update", "daily update", "var(--violet-ink, var(--blue-ink))")]


def legend_html():
    items = "".join(
        f'<span class="gl-item"><span class="gl-dot" style="background:{color}"></span>{html.escape(label)}</span>'
        for _, label, color in TIER_LEGEND)
    typed_item = ('<span class="gl-item"><span class="gl-dash"></span>'
                  'curated relationship (labeled on the edge)</span>')
    arrow_item = '<span class="gl-item"><span class="gl-arrow"></span>arrow points motivator &rarr; motivated page</span>'
    return f'<div class="graph-legend" id="graph-legend">{items}{typed_item}{arrow_item}</div>'


def _asset_hash8(path):
    return hashlib.md5(path.read_bytes()).hexdigest()[:8]


def graph_extra_head(base):
    css = REPO / "web/assets/graph_view.css"
    v = _asset_hash8(css) if css.exists() else ""
    qs = f"?v={v}" if v else ""
    return f'<link rel="stylesheet" href="{bc.ASSETS_REL}/graph_view.css{qs}">'


def graph_scripts(base, data):
    js = REPO / "web/assets/graph_view.js"
    v = _asset_hash8(js) if js.exists() else ""
    qs = f"?v={v}" if v else ""
    vendor = bc.ASSETS_REL + "/vendor"
    tags = "".join(
        f'<script src="{vendor}/{name}"></script>\n'
        for name in ("d3-quadtree.v3.0.1.min.js", "d3-dispatch.v3.0.1.min.js",
                      "d3-timer.v3.0.1.min.js", "d3-force.v3.0.0.min.js"))
    return (tags +
            f'<script src="{bc.ASSETS_REL}/graph_view.js{qs}" '
            f'data-graph-src="{SITE_ROOT}/graph.json"></script>')


def build_graph_tab(out_dir):
    data = write_graph_json(out_dir)
    n_nodes = len(data["nodes"])
    n_edges = len(data["edges"])
    n_no_upstream = sum(1 for n in data["nodes"] if not n["has_upstream"])
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    base = bc.xc.console_base(bc.CONFIG, out_dir)

    body = f'''
    <section class="graph-page" data-graph-src="{SITE_ROOT}/graph.json">
      <p class="sub">built {now} &middot; {n_nodes} pages, {n_edges} edges, {n_no_upstream} with no known upstream</p>
      <p>Every published page is a node; every edge is a directed arrow, tail at the page that
      motivated it, head at the page it motivated. <code>web/pages.yaml</code>'s <code>upstreams:</code>
      list is the primary source for that (a page states what motivated its own creation); plain
      arrows are content links crawled straight out of each page's rendered HTML (page-tree links,
      outlines, and the theme toggle excluded); dashed, labeled arrows are curated relationships
      (supersedes, evidence-for, part-of, same-page) checked by hand against the pages, for anything
      that isn't a motivation link. Built by <code>tools/build_graph.py</code>, which reads the same
      page inventory as the Pages tab; Map-mode positions persist in
      <code>.console_build/graph_positions.json</code> so a rebuild never reshuffles pages you
      already know the layout of, and Timeline mode recomputes its own bin/stack layout from the
      data fresh each load. Node labels are each page's short id (or its registered
      <code>shortname:</code>); hover a node for its full title. Click a node to select it (its
      neighborhood stays lit, everything else dims); click empty space or press Escape to clear the
      selection. Double-click a node to open its page.</p>
      <div class="graph-toolbar">
        <input type="search" id="graph-search" placeholder="Search pages&hellip;" aria-label="Search pages">
        <div class="graph-modes" role="group" aria-label="Layout mode">
          <button type="button" data-graph-mode="map" class="active">Map</button>
          <button type="button" data-graph-mode="timeline">Timeline</button>
        </div>
        {legend_html()}
      </div>
      <div class="graph-layout">
        <div class="graph-canvas-wrap">
          <svg id="graph-svg" class="graph-svg" role="img" aria-label="Page relationship graph"></svg>
          <div class="graph-hint">scroll to zoom &middot; drag to pan &middot; click selects, double-click opens &middot; Esc clears</div>
        </div>
        <aside class="graph-orphans">
          <h3>No known upstream</h3>
          <p class="sub">Pages with no <code>upstreams:</code> arrow pointing in: should only be genuine
          starting points. Anything else here is missing its <code>upstreams:</code> entry in
          <code>web/pages.yaml</code>.</p>
          <ul id="graph-orphan-list"></ul>
        </aside>
      </div>
    </section>
    {graph_scripts(base, data)}
    '''
    extra_head = graph_extra_head(base)
    bc.xc.write_page(out_dir, "graph.html",
                     bc.xc.console_page(bc.CONFIG, "Graph — Lightgen Console", "graph", body,
                                        bc.console_tree_entries(base), base, wide=True,
                                        extra_head=extra_head, nav_title="Graph"))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="rebuild graph.json only, print counts, skip the page")
    args = ap.parse_args()
    if args.check:
        data = write_graph_json()
    else:
        data = build_graph_tab(PUBLISH_DEST)
    n_no_upstream = sum(1 for n in data["nodes"] if not n["has_upstream"])
    print(f"graph: {len(data['nodes'])} nodes, {len(data['edges'])} edges, {n_no_upstream} with no upstream "
          f"-> {bc.BASE_URL}/graph.json" + ("" if args.check else f" + {bc.BASE_URL}/graph.html"))


if __name__ == "__main__":
    main()
