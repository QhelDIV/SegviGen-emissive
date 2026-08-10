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
for the two nested tiers). "type" is reserved for future curated edge kinds
(supersedes, evidence-for); this build only ever emits "link". x/y are a
stable 2D layout in an abstract ~1200x800 coordinate space (not pixels —
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

EDGES are found by reading each node's own PUBLISH_DEST HTML and walking it
with a small stdlib html.parser.HTMLParser subclass (ContentAnchorParser)
that tracks an open-tag stack and marks a subtree as CHROME the moment it
enters one of: <nav class="v3-tree">, <nav class="outline"|"nav-tabs"|
"nav-subtabs"|...toc...>, <aside class="...outline...">, <div class=
"v3-topbar"|"v3-scrim">, or a theme-toggle button. The hero <header> itself
(eyebrow/h1/dek/toc-pills) is deliberately NOT excluded — a first cut
stripped it wholesale and silently dropped a real citation living in a
page's dek/sub paragraph ("Companion to <a href=...>the dataset gallery</a>"
on pbr_filter_v1); the only thing in a header worth excluding is the
toc-pills' own "#section-id" links, and resolve_href() already drops every
fragment-only href on its own, so no header rule is needed at all (see
_is_chrome_start's docstring for the live repro). Anchors found OUTSIDE any
chrome subtree are content links. This covers v1 (floating .outline
sidebar), v2 (hero header, optional .v2-outline-rail), and v3 (.v3-tree/
.v3-outline/.v3-topbar) pages uniformly, without needing a DOM library —
verified against a real page (workspace/rendering/index.html) that has the
SAME href duplicated once in chrome (the tree link) and once in content
(the "the render sweep page" citation inside a .chartnote) to make sure
only the content one counts.

A resolved href becomes an edge only if it points at another known node's
directory (asset paths, external hosts, mailto:/javascript:, and same-page
"#fragment" links all resolve to nothing and are dropped); self-loops are
dropped; an edge exists once per ordered (source, target) pair regardless of
how many times a page links another.

LAYOUT PERSISTENCE lives in this builder, in .console_build/graph_positions.json
(gitignored build-scratch dir — see the .gitignore entry added alongside this
file), NOT in the client: static hosting cannot write back a client-side
drag. A node already present with an UNCHANGED incident-edge set keeps its
exact stored (x, y), rounded to 2 decimals, forever — the owner's spatial
memory of the map is a feature, not a bug to relax away. A brand-new node,
or one whose own edge set changed since the last build, is "movable": it
seeds at the centroid of its current neighbors' positions (or a deterministic
golden-angle ring position around the whole graph's bounding box if it has
no neighbors at all), then a small hand-rolled Fruchterman-Reingold-style
force pass runs a fixed iteration count with EVERY node (movable and frozen
alike) contributing repulsion/attraction forces, but only movable nodes'
positions are ever updated — "light global settling" without ever reshuffling
an untouched node. No randomness anywhere (ring angles and tie-break jitter
come from an md5 of the node id, not random.random()), so a rebuild with no
content change reproduces byte-identical (x, y) for every node.

Renderer stack (per the architecture brief): vendored d3-force (physics
ONLY: d3-force + its d3-quadtree/d3-dispatch/d3-timer dependencies, four
small UMD files fetched once from unpkg and checked in at
web/assets/vendor/d3-*.v3.*.min.js, verified to actually wire up under a
plain <script> tag — no CDN at runtime, no bundler) driving a SESSION-LOCAL
node-drag re-settle client-side only; the SVG rendering itself
(pan/zoom/hover/search/edges/labels) is hand-written vanilla JS,
web/assets/graph_view.js + graph_view.css. Both are LIGHTGEN-LOCAL siblings
under web/assets/ (same precedent as the hand-vendored model-viewer.min.js —
see sync_xgpage_assets.py's module docstring), not part of the xgpage
package: this is a project-specific console tab, not a reusable report
component.
"""
import argparse
import datetime
import hashlib
import html
import json
import math
import pathlib
import posixpath
import sys
from html.parser import HTMLParser
from urllib.parse import urlsplit

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import build_console as bc  # noqa: E402
import inventory_pages as ip  # noqa: E402

PUBLISH_DEST = bc.PUBLISH_DEST
BASE_URL = bc.BASE_URL
SITE_ROOT = bc.SITE_ROOT
SITE_HOST = urlsplit(BASE_URL).netloc

POSITIONS_FILE = REPO / ".console_build" / "graph_positions.json"

CANVAS_W, CANVAS_H = 1200.0, 800.0


# ------------------------------------------------------------- chrome scan --
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input",
             "link", "meta", "param", "source", "track", "wbr"}
_NAV_CHROME_HINTS = ("v3-tree", "outline", "nav-tabs", "nav-subtabs", "toc")


def _is_chrome_start(tag, attrs):
    """True if THIS start tag (not counting ancestors) opens a chrome
    subtree: the page tree, the outline/toc rails (v1 floating sidebar, v2
    outline rail, v3 outline aside), the v3 mobile topbar/scrim, or the theme
    toggle button.

    Deliberately NOT excluded: the hero <header> itself. First cut of this
    parser stripped every <header> wholesale (eyebrow/h1/dek/toc-pills all
    live there) on the theory that it's pure chrome — wrong, found live on
    pbr_filter_v1: its dek paragraph carries a real citation ("Companion to
    <a href='../dataset_gallery_v1/index.html'>the dataset gallery</a>"),
    which the blanket rule silently dropped as an edge. The only thing in a
    header actually worth excluding is the toc-pills nav's own "#section-id"
    links, and resolve_href() already drops every fragment-only href on its
    own (empty path after the "#"), so no special-casing was needed at all —
    removing the header rule fixed the false negative with no new false
    positive (verified: toc pills never carry a non-fragment href anywhere
    in the engine)."""
    cls = attrs.get("class", "").split()
    if tag == "nav" and any(any(h in c for h in _NAV_CHROME_HINTS) for c in cls):
        return True
    if tag == "aside" and any("outline" in c for c in cls):
        return True
    if tag == "div" and any(c in ("v3-topbar", "v3-scrim") for c in cls):
        return True
    if tag == "button" and attrs.get("id") == "xg-theme-btn":
        return True
    return False


class ContentAnchorParser(HTMLParser):
    """Collects every <a href> OUTSIDE a chrome subtree (see
    _is_chrome_start). Tolerant of any real-world tag nesting: an unmatched
    end tag just pops whatever's on top, never raises."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []  # bool per open non-void element: is this subtree chrome?
        self.hrefs = []

    def _handle(self, tag, attrs_list, void):
        attrs = dict(attrs_list)
        parent_chrome = self.stack[-1] if self.stack else False
        is_chrome = parent_chrome or _is_chrome_start(tag, attrs)
        if tag == "a" and not is_chrome and attrs.get("href"):
            self.hrefs.append(attrs["href"])
        if not void and tag not in VOID_TAGS:
            self.stack.append(is_chrome)

    def handle_starttag(self, tag, attrs_list):
        self._handle(tag, attrs_list, void=False)

    def handle_startendtag(self, tag, attrs_list):
        self._handle(tag, attrs_list, void=True)

    def handle_endtag(self, tag):
        if tag in VOID_TAGS:
            return
        if self.stack:
            self.stack.pop()


def extract_content_hrefs(html_text):
    p = ContentAnchorParser()
    try:
        p.feed(html_text)
    except Exception:
        pass
    return p.hrefs


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
def resolve_href(href, from_dir, url_to_id):
    try:
        parts = urlsplit(href)
    except ValueError:
        return None
    if parts.scheme and parts.scheme not in ("http", "https"):
        return None  # mailto:, javascript:, tel:, data:, ...
    if parts.netloc and parts.netloc != SITE_HOST:
        return None  # external host
    path = parts.path
    if not path:
        return None  # "#fragment"-only same-page link, or empty
    if path.startswith(SITE_ROOT + "/"):
        rel = path[len(SITE_ROOT) + 1:]
    elif path.startswith("/"):
        return None  # absolute path outside this site's root
    else:
        rel = posixpath.normpath(posixpath.join(from_dir, path))
    rel = rel.strip("/")
    if rel.endswith("/index.html"):
        rel = rel[: -len("/index.html")]
    elif rel == "index.html":
        rel = ""
    return url_to_id.get(rel)


def scan_edges(nodes):
    url_to_id = {n["_dir"]: n["id"] for n in nodes}
    edges = set()
    for n in nodes:
        idx = PUBLISH_DEST / n["_dir"] / "index.html"
        if not idx.exists():
            continue
        text = idx.read_text(errors="ignore")
        for href in extract_content_hrefs(text):
            target = resolve_href(href, n["_dir"], url_to_id)
            if target and target != n["id"]:
                edges.add((n["id"], target))
    return sorted(edges)


# ---------------------------------------------------------------- layout ---
def _load_positions():
    try:
        data = json.loads(POSITIONS_FILE.read_text())
        return data.get("positions", {}), [tuple(e) for e in data.get("edges", [])]
    except (OSError, ValueError):
        return {}, []


def _save_positions(positions, edges):
    POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"positions": {k: list(positions[k]) for k in sorted(positions)},
               "edges": [list(e) for e in edges]}
    POSITIONS_FILE.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def _neighbor_sets(edges):
    out = {}
    for a, b in edges:
        out.setdefault(a, set()).add(b)
        out.setdefault(b, set()).add(a)
    return out


def _det01(node_id):
    """Deterministic pseudo-random value in [0, 1) from an md5 of the id —
    NEVER random.random(): a rebuild with no content change must reproduce
    byte-identical positions, so nothing here may depend on process state."""
    h = hashlib.md5(node_id.encode()).hexdigest()[:8]
    return int(h, 16) / 0xFFFFFFFF


def compute_layout(node_ids, edges):
    """Returns {id: (x, y)}. See the module docstring's LAYOUT PERSISTENCE
    section for the algorithm; this is the whole thing, one function.

    A node is MOVABLE iff it is brand new (no stored position at all), or it
    was an ORPHAN last build (zero incident edges in the stored edge list)
    and has gained its first edge this build — its old position was an
    arbitrary ring placement, not a real one, so this is the one case where
    "moved topology" earns a reposition. Every other already-positioned node
    is FIXED, unconditionally, even if its neighbor set changed (gained or
    lost an edge to/from some OTHER node): only the node whose OWN links
    changed should ever move, and even then only out of the orphan ring.
    Tested directly (see the graph_page job log): a fixture page linking to
    an already-connected existing node must NOT nudge that node — first
    draft of this rule also repositioned any node whose neighbor set changed
    at all, which moved training_curves_v1 by a few pixels for gaining one
    new inbound link, failing exactly that check."""
    stored_pos, stored_edges = _load_positions()
    stored_neigh = _neighbor_sets(stored_edges)
    cur_neigh = _neighbor_sets(edges)

    fixed = {}
    movable = []
    for nid in sorted(node_ids):
        was_orphan = not stored_neigh.get(nid)
        newly_connected = was_orphan and bool(cur_neigh.get(nid))
        if nid in stored_pos and not newly_connected:
            fixed[nid] = tuple(stored_pos[nid])
        else:
            movable.append(nid)

    pos = dict(fixed)
    if fixed:
        xs = [p[0] for p in fixed.values()]
        ys = [p[1] for p in fixed.values()]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        max_r = max(math.hypot(x - cx, y - cy) for x, y in fixed.values())
    else:
        cx, cy, max_r = CANVAS_W / 2, CANVAS_H / 2, 0.0

    orphan_movables = []
    for nid in movable:
        neigh_pos = [pos[m] for m in cur_neigh.get(nid, set()) if m in pos]
        if neigh_pos:
            pos[nid] = (sum(p[0] for p in neigh_pos) / len(neigh_pos),
                        sum(p[1] for p in neigh_pos) / len(neigh_pos))
        else:
            orphan_movables.append(nid)

    golden_angle = math.pi * (3 - math.sqrt(5))
    ring_r = max_r + 220.0
    for i, nid in enumerate(sorted(orphan_movables)):
        angle = i * golden_angle
        pos[nid] = (cx + ring_r * math.cos(angle), cy + ring_r * math.sin(angle))

    # deterministic tiny nudge so two movable nodes seeded at the same
    # centroid don't sit exactly on top of each other (0-distance forces)
    for nid in movable:
        x, y = pos[nid]
        a = _det01(nid) * 2 * math.pi
        pos[nid] = (x + math.cos(a) * 0.5, y + math.sin(a) * 0.5)

    if movable:
        movable_set = set(movable)
        # k (the FR spring constant) sets the equilibrium spacing between
        # any two nodes connected by an edge (attraction f(dist)=dist^2/k
        # balances repulsion f(dist)=k^2/dist at dist=k) -- the plain
        # sqrt(area/n) formula gave k=153, which under GRAVITY's pull packed
        # the actually-connected cluster tight enough that constant-screen-
        # size labels collided (found by looking at a screenshot, not by any
        # metric: DOM overlap isn't checked here, only eyeballed). The x2.4
        # factor is tuned empirically for THIS label length/font size, not
        # derived; re-tune by eye if node count or label length changes a lot.
        k = math.sqrt((CANVAS_W * CANVAS_H) / max(len(node_ids), 1)) * 2.4
        iters = 150
        temp0 = 90.0
        GRAVITY = 0.4
        for it in range(iters):
            disp = {nid: [0.0, 0.0] for nid in movable}
            for i, a in enumerate(node_ids):
                ax, ay = pos[a]
                for b in node_ids[i + 1:]:
                    if a not in movable_set and b not in movable_set:
                        continue
                    bx, by = pos[b]
                    dx, dy = ax - bx, ay - by
                    dist = math.hypot(dx, dy) or 0.01
                    force = (k * k) / dist
                    fx, fy = dx / dist * force, dy / dist * force
                    if a in movable_set:
                        disp[a][0] += fx; disp[a][1] += fy
                    if b in movable_set:
                        disp[b][0] -= fx; disp[b][1] -= fy
            for a, b in edges:
                if a not in movable_set and b not in movable_set:
                    continue
                ax, ay = pos[a]; bx, by = pos[b]
                dx, dy = ax - bx, ay - by
                dist = math.hypot(dx, dy) or 0.01
                force = (dist * dist) / k
                fx, fy = dx / dist * force, dy / dist * force
                if a in movable_set:
                    disp[a][0] -= fx; disp[a][1] -= fy
                if b in movable_set:
                    disp[b][0] += fx; disp[b][1] += fy
            # gravity: a weak pull toward the graph's centroid, standard for
            # Fruchterman-Reingold on a SPARSE real-world graph (most nodes
            # have zero or one edge) — without it, poorly-connected nodes
            # have nothing but unopposed repulsion pushing them outward and
            # never stop drifting. Found live: the first full bootstrap
            # build (every node movable at once, no fixed anchors yet)
            # produced a ~13800-unit-wide layout on a nominal 1200x800
            # canvas with this term absent; adding it brought a from-scratch
            # rebuild back to a sane multiple of the canvas size.
            for nid in movable:
                x, y = pos[nid]
                disp[nid][0] += (cx - x) * GRAVITY
                disp[nid][1] += (cy - y) * GRAVITY
            cool = temp0 * (1 - it / iters)
            for nid in movable:
                dx, dy = disp[nid]
                dl = math.hypot(dx, dy) or 0.01
                step = min(dl, max(cool, 0.5))
                x, y = pos[nid]
                pos[nid] = (x + dx / dl * step, y + dy / dl * step)

    return {nid: (round(pos[nid][0], 2), round(pos[nid][1], 2)) for nid in node_ids}


# ------------------------------------------------------------------ build --
def build_graph_data():
    nodes = scan_nodes()
    edges = scan_edges(nodes)
    node_ids = sorted(n["id"] for n in nodes)
    positions = compute_layout(node_ids, edges)
    _save_positions(positions, edges)

    in_deg = {nid: 0 for nid in node_ids}
    out_deg = {nid: 0 for nid in node_ids}
    for a, b in edges:
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
    out_edges = [{"source": a, "target": b, "type": "link"} for a, b in edges]
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
    return f'<div class="graph-legend" id="graph-legend">{items}</div>'


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
      <p class="sub">built {now} &middot; {n_nodes} pages, {n_edges} content links, {n_orphans} not yet connected</p>
      <p>Every published page is a node, positioned once and remembered between rebuilds so the
      map stays navigable by memory. A directed edge means one page's own rendered content links
      the other: page-tree links, outlines, and the theme toggle are excluded, only links inside
      the article body count. Built by <code>tools/build_graph.py</code>, which reads the same page
      inventory as the Pages tab and crawls each page's HTML for its real content links; positions
      persist in <code>.console_build/graph_positions.json</code> so a rebuild never reshuffles pages
      you already know the layout of. Only brand-new pages, or pages whose links changed, move.</p>
      <div class="graph-toolbar">
        <input type="search" id="graph-search" placeholder="Search pages&hellip;" aria-label="Search pages">
        {legend_html()}
      </div>
      <div class="graph-layout">
        <div class="graph-canvas-wrap">
          <svg id="graph-svg" class="graph-svg" role="img" aria-label="Page relationship graph"></svg>
          <div class="graph-hint">scroll to zoom &middot; drag to pan &middot; drag a node to move it</div>
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
