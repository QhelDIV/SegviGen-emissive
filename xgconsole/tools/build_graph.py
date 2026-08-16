#!/usr/bin/env python3
"""build_graph.py — the Lightgen JOB graph (graph.json + graph.html).

An Obsidian-style graph of the project's WORK: nodes are board jobs
(owner-ratified 2026-08-14: causality is a property of work items; pages
are the jobs' artifacts), plus the published pages no job claims (legacy
artifacts, drawn hollow). A job's `page:` field absorbs that page — the
job node carries its URL, the page stops being its own node. Directed
edges: the board's `upstreams:` field (primary; declared at dispatch),
pages.yaml `upstreams:` for legacy pages, crawled content links, and
hand-verified typed relations. Click-through: every node carries a
`detail` payload (motivation/outcome/log tail for jobs, blurb for pages)
that graph_view.js renders as the side card. The point is orientation —
seeing how the project's work connects and grew — for the owner and for
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
      "nodes": [  # kind "job":
                 {"id", "kind", "title", "label", "track", "status", "url",
                  "page_name", "board_url", "created", "modified", "x", "y",
                  "in_degree", "out_degree", "has_upstream",
                  "detail": {"executor", "motivation", "outcome",
                              "log_tail", "n_log"}},
                  # kind "page" (legacy, no board entry):
                 {"id", "kind", "title", "label", "tier", "url", "created",
                  "modified", "x", "y", "in_degree", "out_degree",
                  "has_upstream", "detail": {"blurb"}}, ...],
      "edges": [{"source", "target", "type": "link"}, ...]
    }
"id" is the board slug for a job (already the short unique name), the
inventory name for a legacy page; a job slug may never collide with a
surviving page id (build-failing guard).
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
from xgpage import jobs as xjobs

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import build_console as bc  # noqa: E402
import inventory_pages as ip  # noqa: E402

JOBS_DIR = REPO / "jobs"

PUBLISH_DEST = bc.PUBLISH_DEST
BASE_URL = bc.BASE_URL
SITE_ROOT = bc.SITE_ROOT
SITE_HOST = urlsplit(BASE_URL).netloc

GRAPH_EDGES_YAML = REPO / "web" / "graph_edges.yaml"

GRAPH_CONFIG = xg.GraphConfig(
    site_root=SITE_ROOT,
    site_host=SITE_HOST,
    # Job-graph compaction (2026-08-14): the default spring_k=340 left
    # repulsion (k^2/dist) flinging peripheral degree-1 nodes ~3000 world
    # units out, and the fit-everything overview zoom collapsed to 0.057.
    # 170 shrinks repulsion-dominated spacing (~1/4 of stock) while the declutter
    # pass still enforces min_sep; re-minted once on adoption.
    spring_k=170.0,
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
FIT_ZOOM = 0.106
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


# ----------------------------------------------------------- job node scan --
def _iso(ts):
    """Board timestamps are 'YYYY-MM-DD HH:MM'; the renderer's date parsing
    (and JSON consumers generally) want ISO8601."""
    return ts.replace(" ", "T") if ts else ""


def scan_job_nodes():
    """Every board entry becomes a graph node (the job-graph conversion,
    owner-ratified 2026-08-14: causality is a property of work items; pages
    are the jobs' artifacts). Node id = the board slug, which is already the
    short unique name the owner asked for. A job's `page:` field ABSORBS
    that page: the page stops being its own node and the job node carries
    its URL and name instead (double-click opens the page, the card links
    both page and board). The full job record the card needs (motivation,
    outcome, log tail) rides along in `detail` -- the same parsed fields the
    jobs tab renders, no second bookkeeping surface."""
    out = []
    for p in sorted(JOBS_DIR.glob("*.md")):
        e = xjobs.parse(p)
        pg = (e.get("page") or "").strip()
        page_name = "" if (not pg or pg.lower().startswith("none")) else pg
        ups = [s.strip() for s in (e.get("upstreams") or "").split(",") if s.strip()]
        log = e.get("log", [])
        out.append({
            "id": p.stem, "kind": "job", "label": p.stem,
            "title": e.get("title") or p.stem,
            "track": e.get("track", ""), "status": e.get("status", ""),
            "created": _iso(e.get("started", "")), "modified": _iso(e.get("updated", "")),
            "page_name": page_name,
            "upstreams": ups,
            "detail": {
                "executor": e.get("executor", ""),
                "motivation": e.get("motivation", ""),
                "outcome": e.get("outcome", ""),
                "log_tail": [f"{ts} {txt}" for ts, _a, txt in log[-3:]],
                "n_log": len(log),
            },
        })
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
    """JOB-GRAPH model (2026-08-14, owner-ratified): nodes are board JOBS
    plus the published pages no job claims (legacy artifacts); a job's
    `page:` absorbs that page, so every edge endpoint that referenced the
    page (crawled link, typed relation, pages.yaml upstream) is RE-KEYED to
    the owning job's slug. Job causality (`upstreams:` on the board entry)
    is the primary directed-edge store; pages.yaml `upstreams:` covers only
    the jobless legacy pages."""
    pages = scan_nodes()
    jobs = scan_job_nodes()
    curation = ip.load_curation()

    page_ids = sorted(p["id"] for p in pages)
    claimed = {}  # page name -> owning job slug (first claimant wins)
    for j in jobs:
        pn = j["page_name"]
        if pn and pn in set(page_ids) and pn not in claimed:
            claimed[pn] = j["id"]
        elif pn and pn not in set(page_ids):
            print(f"[jobs] {j['id']}: page {pn!r} not in the inventory, kept as job-only node", file=sys.stderr)
    page_by_id = {p["id"]: p for p in pages}
    for pn, slug in claimed.items():
        # a job absorbing a page inherits its URL; the page node disappears
        j = next(j for j in jobs if j["id"] == slug)
        j["url"] = page_by_id[pn]["url"]
        # a page's publish date can predate board adoption; keep the JOB's
        # own started date for the timeline (work order, not publish order)
    artifact_pages = [p for p in pages if p["id"] not in claimed]

    # id collision guard: a job slug must not equal a surviving page id
    art_ids = {p["id"] for p in artifact_pages}
    for j in jobs:
        if j["id"] in art_ids:
            sys.exit(f"error: job slug {j['id']!r} collides with an unclaimed page id; "
                     f"claim the page (set-page) or rename one")

    # --- edges are DISCOVERED on pages, then re-keyed to job endpoints ---
    rekey = {pn: slug for pn, slug in claimed.items()}
    rk = lambda nid: rekey.get(nid, nid)
    crawled = xg.scan_edges(pages, _read_node_html, GRAPH_CONFIG)  # plain (a, b) page tuples
    typed_edges = xg.load_typed_edges(page_ids, GRAPH_CONFIG)
    page_upstream_edges = load_upstream_edges(pages, curation)
    job_upstream_edges = []
    job_ids = {j["id"] for j in jobs}
    page_id_set = set(page_ids)
    for j in jobs:
        for up in j["upstreams"]:
            # an upstream may be a board slug OR a page name (a job motivated
            # by pre-board work, e.g. render_doc <- workspace/render_sweep);
            # a claimed page folds to its owning job like any other endpoint
            if up in job_ids:
                src = up
            elif up in page_id_set:
                src = rk(up)
            else:
                print(f"[jobs] {j['id']}: dropping upstream {up!r}", file=sys.stderr)
                continue
            if src != j["id"]:
                job_upstream_edges.append({"source": src, "target": j["id"], "type": "upstream"})

    def rekey_edges(es):
        out = []
        for e in es:
            a, b = rk(e["source"]), rk(e["target"])
            if a != b:  # re-keying can fold a page->page edge into a self-loop
                out.append({"source": a, "target": b, "type": e["type"]})
        return out
    typed_edges = rekey_edges(typed_edges)
    # Display transform (owner-ratified 2026-08-14): one direction rule for
    # the whole graph -- every arrow reads earlier/feeding -> later/current,
    # dashed words still read along the arrow as a sentence. Authors keep
    # writing the natural active vocabulary in graph_edges.yaml ("A
    # supersedes B"); the build flips supersedes into "replaced by"
    # (old -> new, along time) and renames the rest to plain reading words.
    TYPED_DISPLAY = {"supersedes": ("replaced by", True),   # (label, flip)
                     "evidence-for": ("supports", False),
                     "part-of": ("part of", False),
                     "same-page": ("same page", False)}
    disp = []
    for e in typed_edges:
        label, flip = TYPED_DISPLAY.get(e["type"], (e["type"], False))
        a, b = (e["target"], e["source"]) if flip else (e["source"], e["target"])
        disp.append({"source": a, "target": b, "type": label})
    typed_edges = disp
    upstream_edges = job_upstream_edges + rekey_edges(page_upstream_edges)
    # Crawled links are FLIPPED to cited -> citing (owner-caught on
    # render_doc/render_sweep, 2026-08-14): a citation means the cited work
    # fed the page citing it, so drawing the arrow from the cited work makes
    # every arrow on the graph read earlier -> later, consistent with the
    # upstream arrows. Citation direction (who links whom) stays recoverable
    # from the page HTML; the graph shows flow, not hyperlink direction.
    crawled = sorted({(rk(b), rk(a)) for a, b in crawled if rk(a) != rk(b)})

    node_ids = sorted(job_ids | art_ids)

    # LAYOUT and degree counting treat a typed OR upstream edge as a real
    # connection too -- combined as plain (a, b) pairs, type-blind, for
    # compute_layout() and the position-persistence neighbor-set diff.
    typed_pairs = [(e["source"], e["target"]) for e in typed_edges]
    upstream_pairs = [(e["source"], e["target"]) for e in upstream_edges]
    all_pairs = sorted(set(crawled) | set(typed_pairs) | set(upstream_pairs))
    in_deg_pre = {nid: 0 for nid in node_ids}
    for _a, b in all_pairs:
        in_deg_pre[b] += 1
    labels = {}
    for j in jobs:
        labels[j["id"]] = j["label"]
    for p in artifact_pages:
        labels[p["id"]] = curation.get(p["id"], {}).get("shortname") or p["id"]
    node_reach = {nid: node_reach_excess(labels[nid], in_deg_pre[nid]) for nid in node_ids}
    positions = xg.compute_layout(node_ids, all_pairs, GRAPH_CONFIG, node_reach=node_reach)

    # Corral the isolated nodes (job-graph round): the FR pass has nothing
    # tethering a degree-0 node, so repulsion flings it thousands of world
    # units out -- measured 7000x7600 bbox for a graph whose connected mass
    # spans a third of that, which drove the fit-everything zoom to 0.057
    # and overlapped the constant-screen-size circles. Isolated nodes park
    # instead on a deterministic shelf (id-sorted grid) under the connected
    # mass: honest ("not yet connected" reads as exactly that), compact,
    # and stable across rebuilds.
    connected = {a for a, _b in all_pairs} | {b for _a, b in all_pairs}
    isolated = sorted(nid for nid in node_ids if nid not in connected)
    if isolated:
        cxs = [positions[nid][0] for nid in connected] or [0.0]
        cys = [positions[nid][1] for nid in connected] or [0.0]
        x0, x1 = min(cxs), max(cxs)
        shelf_y = max(cys) + 300
        cols = max(1, int((x1 - x0) // 220) + 1)
        for i, nid in enumerate(isolated):
            positions[nid] = (x0 + (i % cols) * 220.0, shelf_y + (i // cols) * 240.0)
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
    for j in jobs:
        nid = j["id"]
        x, y = positions[nid]
        out_nodes.append({
            "id": nid, "kind": "job", "title": j["title"], "label": labels[nid],
            "track": j["track"], "status": j["status"],
            "url": j.get("url", ""), "page_name": j["page_name"],
            "board_url": f"{BASE_URL}/jobs.html",
            "created": j["created"], "modified": j["modified"],
            "x": x, "y": y, "in_degree": in_deg[nid], "out_degree": out_deg[nid],
            "has_upstream": has_upstream[nid], "detail": j["detail"]})
    for p in artifact_pages:
        nid = p["id"]
        x, y = positions[nid]
        blurb = curation.get(nid, {}).get("blurb") or ""
        out_nodes.append({
            "id": nid, "kind": "page", "title": p["title"], "label": labels[nid],
            "tier": p["tier"], "url": p["url"],
            "created": p["created"], "modified": p["modified"],
            "x": x, "y": y, "in_degree": in_deg[nid], "out_degree": out_deg[nid],
            "has_upstream": has_upstream[nid],
            "detail": {"blurb": blurb}})
    out_nodes.sort(key=lambda n: n["id"])

    # A pair covered by a curated typed OR upstream edge is strictly more
    # informative than the generic crawled "link" between the same two
    # nodes -- suppress the plain link edge for any ordered pair a typed
    # or upstream edge already covers.
    covered_pairs = set(typed_pairs) | set(upstream_pairs)
    out_edges = ([{"source": a, "target": b, "type": "link"} for a, b in crawled if (a, b) not in covered_pairs]
                 + typed_edges + upstream_edges)
    # dedupe (job upstreams and re-keyed page upstreams can coincide)
    seen = set()
    out_edges = [e for e in out_edges
                 if (k := (e["source"], e["target"], e["type"])) not in seen and not seen.add(k)]
    return {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "nodes": out_nodes, "edges": out_edges}


def write_graph_json(out_dir=None):
    data = build_graph_data()
    dest = pathlib.Path(out_dir or PUBLISH_DEST) / "graph.json"
    dest.write_text(json.dumps(data, indent=1))
    return data


# -------------------------------------------------------------- the page --
# Job nodes color by TRACK (the stable identity of the work); artifact
# pages -- published pages no board entry claims, mostly pre-board history
# -- are one muted hollow style, secondary by design. The old four-tier
# page palette died with the page-graph model.
TRACK_LEGEND = [("research", "research job", "var(--good)"),
                ("tooling", "tooling job", "var(--blue-ink)"),
                ("paper", "paper job", "var(--violet-ink, var(--blue-ink))")]


def legend_html():
    items = "".join(
        f'<span class="gl-item"><span class="gl-dot" style="background:{color}"></span>{html.escape(label)}</span>'
        for _, label, color in TRACK_LEGEND)
    page_item = '<span class="gl-item"><span class="gl-dot gl-dot-page"></span>page (no board entry)</span>'
    haspage_item = ('<span class="gl-item"><span class="gl-dot gl-dot-haspage" '
                    'style="background:var(--good)"></span>job with a page (hollow center)</span>')
    # ONE direction rule for the whole graph (owner-ratified 2026-08-14,
    # after two rounds of legend confusion): every arrow reads earlier or
    # feeding work -> later or current work; dashed words are chosen so the
    # sentence also reads along the arrow ("supersedes" became "replaced
    # by", flipped at build time -- see build_graph_data()'s TYPED_DISPLAY).
    arrow_item = ('<span class="gl-item"><span class="gl-arrow"></span>'
                  'every arrow: earlier work &rarr; what followed from it</span>')
    typed_item = ('<span class="gl-item"><span class="gl-dash"></span>'
                  'dashed: curated relationship, read along the arrow '
                  '(&ldquo;A replaced by&nbsp;&rarr; B&rdquo;)</span>')
    return f'<div class="graph-legend" id="graph-legend">{items}{haspage_item}{page_item}{arrow_item}{typed_item}</div>'


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


def build_graph_tab(out_dir, graph_src=None):
    data = write_graph_json(out_dir)
    n_jobs = sum(1 for n in data["nodes"] if n["kind"] == "job")
    n_pages = len(data["nodes"]) - n_jobs
    n_edges = len(data["edges"])
    n_roots = sum(1 for n in data["nodes"] if n["kind"] == "job" and not n["has_upstream"])
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    base = bc.xc.console_base(bc.CONFIG, out_dir)
    src = graph_src or f"{SITE_ROOT}/graph.json"

    body = f'''
    <section class="graph-page" data-graph-src="{src}">
      <p class="sub">built {now} &middot; {n_jobs} jobs, {n_pages} legacy pages, {n_edges} edges, {n_roots} root jobs</p>
      <p>Every board job is a node; a job that produced a page carries it (double-click opens the
      page), and published pages predating the board appear as smaller hollow nodes. Every edge is
      a directed arrow, tail at the work that motivated, head at the work it motivated: the board's
      own <code>upstreams:</code> field is the primary source (declared at dispatch, when the
      motivation is written), <code>web/pages.yaml</code> covers the legacy pages, plain arrows are
      content links crawled from the pages' rendered HTML (drawn from the cited work to the page
      citing it, so every arrow reads earlier &rarr; later), and dashed labeled arrows are curated
      relationships checked by hand (&ldquo;replaced by&rdquo;, &ldquo;supports&rdquo;,
      &ldquo;part of&rdquo;, &ldquo;same page&rdquo;), worded so the sentence also reads along
      the arrow in the same earlier-first direction. Click a node
      for its full record in the side card &mdash; motivation, status, outcome, latest log &mdash;
      straight from the board entry; click empty space or press Escape to clear. Double-click opens
      the page (or the board for a job without one). Map positions persist across rebuilds;
      Timeline bins by date, busier days finer.</p>
      <div class="graph-toolbar">
        <input type="search" id="graph-search" placeholder="Search pages&hellip;" aria-label="Search pages">
        <div class="graph-modes" role="group" aria-label="Layout mode">
          <button type="button" data-graph-mode="timeline" class="active">Timeline</button>
          <button type="button" data-graph-mode="map">Map</button>
        </div>
        {legend_html()}
      </div>
      <div class="graph-layout">
        <div class="graph-canvas-wrap">
          <svg id="graph-svg" class="graph-svg" role="img" aria-label="Page relationship graph"></svg>
          <div class="graph-hint">scroll to zoom &middot; drag to pan &middot; click selects, double-click opens &middot; Esc clears</div>
        </div>
        <aside class="graph-side">
          <div id="graph-detail" class="graph-detail" hidden></div>
          <div id="graph-roots" class="graph-orphans">
            <h3>No known upstream</h3>
            <p class="sub">Nothing points in with an <code>upstreams:</code> arrow: should only be
            genuine starting points (a fresh owner ask, a first attempt). Any other job here is
            missing its board <code>upstreams:</code>; any other page, its
            <code>web/pages.yaml</code> entry.</p>
            <ul id="graph-orphan-list"></ul>
          </div>
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
    ap.add_argument("--stage", action="store_true",
                    help="build into PUBLISH_DEST/_preview/graph_v2/ instead of the live root "
                         "(live-URL iteration without touching the shipped tab)")
    args = ap.parse_args()
    if args.check:
        data = write_graph_json()
        url = f"{bc.BASE_URL}/graph.json"
    elif args.stage:
        stage = pathlib.Path(PUBLISH_DEST) / "_preview" / "graph_v2"
        stage.mkdir(parents=True, exist_ok=True)
        data = build_graph_tab(stage, graph_src=f"{SITE_ROOT}/_preview/graph_v2/graph.json")
        url = f"{bc.BASE_URL}/_preview/graph_v2/graph.html"
    else:
        data = build_graph_tab(PUBLISH_DEST)
        url = f"{bc.BASE_URL}/graph.html"
    n_no_upstream = sum(1 for n in data["nodes"] if not n["has_upstream"])
    print(f"graph: {len(data['nodes'])} nodes, {len(data['edges'])} edges, {n_no_upstream} with no upstream -> {url}")


if __name__ == "__main__":
    main()
