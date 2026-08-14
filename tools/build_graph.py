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
      "nodes": [{"id", "title", "tier", "url", "created", "modified",
                 "x", "y", "in_degree", "out_degree"}, ...],
      "edges": [{"source", "target", "type": "link"}, ...]
    }
"id" is the same page identity inventory_pages.py already uses (its "name"
column — bare for root/preview tiers, "updates/<date>" / "workspace/<slug>"
for the two nested tiers). "type" is "link" for a crawled content link, or
one of "supersedes"/"evidence-for"/"part-of" for a curated edge from
web/graph_edges.yaml (round-2 addition, 2026-08-10) -- hand-verified against
the actual page content, never crawled or invented; see
xgpage.graph.load_typed_edges().
x/y are a stable 2D layout in an abstract ~1200x800 coordinate space (not pixels —
the renderer fits it to whatever canvas it has).

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
)
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
    edges = xg.scan_edges(nodes, _read_node_html, GRAPH_CONFIG)  # crawled; plain (a, b) tuples
    node_ids = sorted(n["id"] for n in nodes)
    typed_edges = xg.load_typed_edges(node_ids, GRAPH_CONFIG)  # curated; [{"source","target","type"}, ...]

    # LAYOUT and degree counting treat a typed edge as a real connection too
    # (a "supersedes"/"evidence-for" relationship pulls nodes together and
    # counts against orphan status exactly like a content link does) --
    # combined as plain (a, b) pairs, type-blind, for compute_layout() and
    # the position-persistence neighbor-set diff.
    typed_pairs = [(e["source"], e["target"]) for e in typed_edges]
    all_pairs = sorted(set(edges) | set(typed_pairs))
    positions = xg.compute_layout(node_ids, all_pairs, GRAPH_CONFIG)
    xg.save_positions(GRAPH_CONFIG, positions, all_pairs)

    in_deg = {nid: 0 for nid in node_ids}
    out_deg = {nid: 0 for nid in node_ids}
    for a, b in all_pairs:
        out_deg[a] += 1
        in_deg[b] += 1

    by_id = {n["id"]: n for n in nodes}
    out_nodes = []
    for nid in node_ids:
        n = by_id[nid]
        x, y = positions[nid]
        out_nodes.append({"id": nid, "title": n["title"], "tier": n["tier"], "url": n["url"],
                           "created": n["created"], "modified": n["modified"],
                           "x": x, "y": y, "in_degree": in_deg[nid], "out_degree": out_deg[nid]})
    # A pair covered by a curated typed edge is strictly more informative
    # than the generic crawled "link" between the same two pages (found
    # live: glb_direct_pilot_v1 already had a content-link citation to
    # pipeline_glb_direct, and the SAME pair is also the one verified
    # evidence-for relationship, which would otherwise draw two overlapping
    # edges for one relationship) -- suppress the plain link edge for any
    # ordered pair a typed edge already covers.
    typed_pair_set = set(typed_pairs)
    out_edges = ([{"source": a, "target": b, "type": "link"} for a, b in edges if (a, b) not in typed_pair_set]
                 + typed_edges)
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
    return f'<div class="graph-legend" id="graph-legend">{items}{typed_item}</div>'


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
    n_orphans = sum(1 for n in data["nodes"] if n["in_degree"] == 0 and n["out_degree"] == 0)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    base = bc.xc.console_base(bc.CONFIG, out_dir)

    body = f'''
    <section class="graph-page" data-graph-src="{SITE_ROOT}/graph.json">
      <p class="sub">built {now} &middot; {n_nodes} pages, {n_edges} edges, {n_orphans} not yet connected</p>
      <p>Every published page is a node, positioned once and remembered between rebuilds so the
      map stays navigable by memory. A directed edge means one page's own rendered content links
      the other: page-tree links, outlines, and the theme toggle are excluded, only links inside
      the article body count. Dashed edges are curated relationships (supersedes, evidence-for,
      part-of) checked by hand against the pages, not crawled. Built by
      <code>tools/build_graph.py</code>, which reads the same page inventory as the Pages tab and
      crawls each page's HTML for its real content links; positions persist in
      <code>.console_build/graph_positions.json</code> so a rebuild never reshuffles pages you
      already know the layout of. Only brand-new pages, or pages whose links changed, move.
      Click a node to select it (its neighborhood stays lit, everything else dims); click empty
      space or press Escape to clear the selection. Double-click a node to open its page.</p>
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
          <h3>Not yet connected</h3>
          <p class="sub">Pages with no content links in or out: the maintenance queue for missing cross-links.</p>
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
    n_orphans = sum(1 for n in data["nodes"] if n["in_degree"] == 0 and n["out_degree"] == 0)
    print(f"graph: {len(data['nodes'])} nodes, {len(data['edges'])} edges, {n_orphans} orphans "
          f"-> {bc.BASE_URL}/graph.json" + ("" if args.check else f" + {bc.BASE_URL}/graph.html"))


if __name__ == "__main__":
    main()
