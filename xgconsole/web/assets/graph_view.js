/* graph_view.js — client renderer for the page-relationship graph.
 * Lightgen-local (not part of the xgpage package). Reads graph.json (the
 * contract is documented in tools/build_graph.py), draws a custom SVG force
 * graph, and re-fetches every 2 minutes to pick up new pages without a full
 * reload. Positions come from the server (tools/build_graph.py's persisted
 * layout); this script never recomputes the base layout, it only offers a
 * session-local drag re-settle via the vendored d3-force physics module
 * (web/assets/vendor/d3-force*.js + its d3-quadtree/d3-dispatch/d3-timer
 * dependencies, loaded before this file) — dragging never writes back to
 * graph.json, so a reload always returns to the server layout.
 */
(function () {
  "use strict";

  var TIER_COLOR = {
    root: "var(--good)",
    preview: "var(--blue-ink)",
    update: "var(--violet-ink, var(--blue-ink))",
    workspace: "var(--accent-ink)"
  };
  var TIER_LABEL = { root: "root", workspace: "workspace", preview: "preview", update: "update" };

  var REFRESH_MS = 120000;
  var SVG_NS = "http://www.w3.org/2000/svg";

  function radiusFor(inDeg) {
    var r = 6 + Math.min(inDeg, 12) * 1.4;
    return Math.max(6, Math.min(22, r));
  }

  function el(tag, attrs) {
    var e = document.createElementNS(SVG_NS, tag);
    for (var k in attrs) if (attrs.hasOwnProperty(k)) e.setAttribute(k, attrs[k]);
    return e;
  }

  function init(root) {
    var svgEl = root.querySelector("#graph-svg");
    var orphanList = root.querySelector("#graph-orphan-list");
    var searchInput = root.querySelector("#graph-search");
    var src = svgEl.getAttribute("data-graph-src") || root.getAttribute("data-graph-src");

    var state = {
      nodes: [],          // last-loaded node records, by id
      nodeById: {},
      edges: [],
      manualPos: {},       // id -> {x,y} session-local drag override (map mode only)
      transform: { x: 0, y: 0, k: 1 },
      viewInited: false,
      mode: "map",         // "map" | "timeline" (round-2 addition)
      timelinePos: {}      // id -> {x,y}, recomputed whenever data loads; see computeTimelineLayout()
    };

    var gViewport = el("g", { "class": "viewport" });
    var gTimeChrome = el("g", { "class": "timeline-chrome" });
    var gEdges = el("g", { "class": "edges" });
    var gNodes = el("g", { "class": "nodes" });
    gViewport.appendChild(gTimeChrome);
    gViewport.appendChild(gEdges);
    gViewport.appendChild(gNodes);
    svgEl.appendChild(gViewport);

    function applyTransform() {
      var t = state.transform;
      gViewport.setAttribute("transform", "translate(" + t.x + "," + t.y + ") scale(" + t.k + ")");
      updateNodeScale();
    }

    function boundsOf(nodes) {
      if (!nodes.length) return { minX: 0, minY: 0, maxX: 1000, maxY: 700 };
      var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      nodes.forEach(function (n) {
        var p = posOf(n);
        minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
        minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
      });
      return { minX: minX, minY: minY, maxX: maxX, maxY: maxY };
    }

    function posOf(n) {
      if (state.mode === "timeline") return state.timelinePos[n.id] || { x: 0, y: 0 };
      return state.manualPos[n.id] || { x: n.x, y: n.y };
    }

    // -------------------------------------------------------- timeline --
    var TIMELINE_LANES = ["root", "workspace", "preview", "update"]; // fixed vertical order, matches the legend
    var TIMELINE_LANE_H = 150;
    var TIMELINE_W = 2800; // world-unit span for the full date range
    var TIMELINE_MIN_GAP = 100; // minimum x gap between two nodes sharing a lane

    function parseCreated(s) {
      var t = Date.parse((s || "").replace(" ", "T"));
      return isNaN(t) ? null : t;
    }

    function laneFor(tier) {
      var i = TIMELINE_LANES.indexOf(tier);
      return i === -1 ? TIMELINE_LANES.length - 1 : i;
    }

    function computeTimelineLayout() {
      // Deterministic positions from date + lane ONLY, no physics (round-2
      // spec: "keep it simple"). x is a linear scale of `created` across
      // the full observed range; within a lane, nodes are sorted by date
      // and pushed apart by a minimum gap where two dates would otherwise
      // collide (a same-day publishing burst is common here) -- a
      // sequential one-pass sweep, not an iterative solver.
      var times = state.nodes.map(function (n) { return parseCreated(n.created); }).filter(function (t) { return t !== null; });
      var minT = times.length ? Math.min.apply(null, times) : 0;
      var maxT = times.length ? Math.max.apply(null, times) : minT + 1;
      var span = Math.max(maxT - minT, 1);
      var byLane = {};
      state.nodes.forEach(function (n) {
        var lane = laneFor(n.tier);
        (byLane[lane] = byLane[lane] || []).push(n);
      });
      var pos = {};
      Object.keys(byLane).forEach(function (laneKey) {
        var laneIdx = +laneKey;
        var nodes = byLane[laneKey].slice().sort(function (a, b) {
          return (parseCreated(a.created) || minT) - (parseCreated(b.created) || minT);
        });
        var lastX = -Infinity;
        nodes.forEach(function (n) {
          var t = parseCreated(n.created);
          var x = t === null ? 0 : ((t - minT) / span) * TIMELINE_W;
          if (x < lastX + TIMELINE_MIN_GAP) x = lastX + TIMELINE_MIN_GAP;
          lastX = x;
          pos[n.id] = { x: x, y: laneIdx * TIMELINE_LANE_H };
        });
      });
      return { pos: pos, minT: minT, maxT: maxT };
    }

    var timelineRange = { minT: 0, maxT: 1 };

    // {g, x, y, kind: "chrome"|"edge"} -- world anchor point; counter-scaled
    // in updateNodeScale(). Two independent owners (renderTimelineChrome()
    // and render()'s typed-edge labels) each rebuild their OWN kind from
    // scratch without clobbering the other's entries.
    var scaledTextEls = [];
    function constScaleText(kind, cls, x, y, text) {
      // Same technique as node markers (see updateNodeScale()): the label's
      // OWN group gets translate(x,y) in world space (so it pans/zooms with
      // the content it annotates) then an inner scale(1/k) that cancels the
      // ambient zoom, so the text renders at a constant SCREEN size
      // regardless of how far out the view is. Timeline chrome and typed-
      // edge labels need this for the identical reason node labels did in
      // map mode (round-1 bug): this graph's natural zoom range goes well
      // under 1.
      var g = el("g", { transform: "translate(" + x + "," + y + ")" });
      var t = el("text", { "class": cls });
      t.textContent = text;
      g.appendChild(t);
      scaledTextEls.push({ g: g, x: x, y: y, kind: kind });
      return g;
    }

    function renderTimelineChrome() {
      gTimeChrome.innerHTML = "";
      scaledTextEls = scaledTextEls.filter(function (s) { return s.kind !== "chrome"; });
      if (state.mode !== "timeline") return;
      var leftX = -160;
      TIMELINE_LANES.forEach(function (tier, i) {
        gTimeChrome.appendChild(constScaleText("chrome", "tl-lane-label", leftX, i * TIMELINE_LANE_H + 4, TIER_LABEL[tier] || tier));
        var line = el("line", {
          "class": "tl-lane-line", x1: leftX + 20, y1: i * TIMELINE_LANE_H, x2: TIMELINE_W + 60, y2: i * TIMELINE_LANE_H
        });
        gTimeChrome.appendChild(line);
      });
      var ticks = 5;
      var bottomY = TIMELINE_LANES.length * TIMELINE_LANE_H - TIMELINE_LANE_H + 60;
      for (var i = 0; i <= ticks; i++) {
        var t = timelineRange.minT + (timelineRange.maxT - timelineRange.minT) * (i / ticks);
        var x = (i / ticks) * TIMELINE_W;
        var vline = el("line", { "class": "tl-tick-line", x1: x, y1: -30, x2: x, y2: bottomY });
        gTimeChrome.appendChild(vline);
        gTimeChrome.appendChild(constScaleText("chrome", "tl-tick-label", x, bottomY + 22, new Date(t).toISOString().slice(0, 10)));
      }
      updateNodeScale();
    }

    function largestComponent() {
      // Union-find over edges (undirected for this purpose): with edges
      // this sparse, the "connected" set is really several SEPARATE small
      // islands (e.g. the ovoxel pair, the paper-skeleton pair) that
      // gravity alone doesn't pull close to each other without also
      // over-compressing the spacing WITHIN each island. Fitting to
      // degree>0 alone still let those far-apart islands force a tiny
      // overall scale (found live: bounding box stayed ~2700-4700 wide
      // even after re-tuning gravity, because the components are just
      // genuinely scattered, not merely orphans). Fitting to the LARGEST
      // component is the throughline of the project's own development --
      // the part this graph exists to show -- everything else (smaller
      // islands, true orphans) stays reachable by panning, zooming out, or
      // search, same as before.
      var parent = {};
      function find(x) { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; }
      function union(a, b) { a = find(a); b = find(b); if (a !== b) parent[a] = b; }
      state.nodes.forEach(function (n) { parent[n.id] = n.id; });
      state.edges.forEach(function (e) {
        if (parent[e.source] !== undefined && parent[e.target] !== undefined) union(e.source, e.target);
      });
      var groups = {};
      state.nodes.forEach(function (n) {
        var r = find(n.id);
        (groups[r] = groups[r] || []).push(n);
      });
      var best = [];
      Object.keys(groups).forEach(function (r) { if (groups[r].length > best.length) best = groups[r]; });
      return best;
    }

    function fitView() {
      var rect = svgEl.getBoundingClientRect();
      var w = rect.width || 900, h = rect.height || 500;
      var b, pad;
      if (state.mode === "timeline") {
        // Timeline has no clutter problem to dodge (deterministic,
        // non-overlapping by construction) -- fit EVERY node, plus the lane
        // labels/date-tick chrome's own extent (fixed offsets, see
        // renderTimelineChrome()).
        b = boundsOf(state.nodes);
        // Extra left margin: lane labels ("workspace") are counter-scaled
        // to a CONSTANT SCREEN width, so how much WORLD space they actually
        // need depends on k, which isn't known until after this division --
        // found live, clipped at the canvas edge with a plain -170 world-
        // unit guess. -320 is a generous fixed margin sized for
        // "workspace" (the longest lane label) at the zoom levels this
        // graph's node count produces; revisit if a much longer label or a
        // much larger point count changes that zoom a lot.
        b.minX = Math.min(b.minX, -320);
        b.minY = Math.min(b.minY, -40);
        b.maxY = Math.max(b.maxY, (TIMELINE_LANES.length - 1) * TIMELINE_LANE_H + 90);
        pad = 40;
      } else {
        var core = largestComponent();
        b = boundsOf(core.length > 1 ? core : state.nodes);
        pad = 60;
      }
      var gw = Math.max(b.maxX - b.minX, 1), gh = Math.max(b.maxY - b.minY, 1);
      var k = Math.min((w - pad * 2) / gw, (h - pad * 2) / gh, 1.4);
      k = Math.max(k, 0.02); // safety floor only (avoid a degenerate k=0), not a "never zoom out further" clamp
      var cx = (b.minX + b.maxX) / 2, cy = (b.minY + b.maxY) / 2;
      state.transform = { x: w / 2 - cx * k, y: h / 2 - cy * k, k: k };
      applyTransform();
    }

    // ---------------------------------------------------------- rendering
    var nodeEls = {}; // id -> {g, circle, label}
    var edgeEls = []; // {a,line,gradient}

    function neighborsOf(id) {
      var s = {};
      state.edges.forEach(function (e) {
        if (e.source === id) s[e.target] = true;
        if (e.target === id) s[e.source] = true;
      });
      return s;
    }

    function render() {
      // edges
      gEdges.innerHTML = "";
      scaledTextEls = scaledTextEls.filter(function (s) { return s.kind !== "edge"; });
      var defs = el("defs", {});
      gEdges.appendChild(defs);
      edgeEls = [];
      state.edges.forEach(function (e, i) {
        var a = state.nodeById[e.source], b = state.nodeById[e.target];
        if (!a || !b) return;
        var pa = posOf(a), pb = posOf(b);
        var gradId = "geg" + i;
        var grad = el("linearGradient", {
          id: gradId, gradientUnits: "userSpaceOnUse",
          x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y
        });
        grad.appendChild(el("stop", { offset: "0%", "stop-color": "var(--muted)", "stop-opacity": "0.28" }));
        grad.appendChild(el("stop", { offset: "100%", "stop-color": "var(--accent-ink)", "stop-opacity": "0.55" }));
        defs.appendChild(grad);
        var mx = (pa.x + pb.x) / 2, my = (pa.y + pb.y) / 2;
        var dx = pb.x - pa.x, dy = pb.y - pa.y;
        var bow = Math.min(24, Math.hypot(dx, dy) * 0.08);
        var nx = -dy, ny = dx;
        var nl = Math.hypot(nx, ny) || 1;
        var cx = mx + (nx / nl) * bow, cy = my + (ny / nl) * bow;
        var isTyped = e.type && e.type !== "link";
        var path = el("path", {
          "class": "gn-edge" + (isTyped ? " typed" : ""),
          d: "M" + pa.x + "," + pa.y + " Q" + cx + "," + cy + " " + pb.x + "," + pb.y,
          stroke: "url(#" + gradId + ")",
          // same reasoning as the node inner-group counter-scale above: an
          // edge's PATH is spatial (should stretch with zoom), but its LINE
          // WEIGHT is a constant visual property. non-scaling-stroke is the
          // purpose-built SVG attribute for exactly this, no extra group
          // needed the way node markers required.
          "vector-effect": "non-scaling-stroke"
        });
        path.dataset.source = e.source; path.dataset.target = e.target;
        gEdges.appendChild(path);
        var labelG = null;
        if (isTyped) {
          // Curated typed edge: dashed line (CSS .typed) + the relationship
          // word AT the edge's own midpoint (bow-adjusted, same point the
          // curve actually passes through), constant screen size like node
          // labels -- see the legend/CSS comment for why a per-edge label
          // beats a per-type legend entry with three types sharing one
          // "dashed" treatment.
          labelG = constScaleText("edge", "gn-typed-label", cx, cy, e.type);
          gEdges.appendChild(labelG);
        }
        edgeEls.push({ source: e.source, target: e.target, path: path, gradient: grad, a: a, b: b, label: labelG });
      });

      // nodes
      gNodes.innerHTML = "";
      nodeEls = {};
      var invK = 1 / (state.transform.k || 1);
      state.nodes.forEach(function (n) {
        var p = posOf(n);
        var isOrphan = !(n.in_degree || n.out_degree);
        var g = el("g", {
          "class": "gn-node" + (isOrphan ? " gn-orphan" : ""),
          transform: "translate(" + p.x + "," + p.y + ")"
        });
        g.dataset.id = n.id;
        // Node MARKERS (circle + label) render at a constant SCREEN size
        // regardless of zoom, via this inner counter-scale group -- the
        // usual node-link-diagram convention (Obsidian included): edges are
        // spatial and stretch with zoom, but a node's own visual size is an
        // intrinsic property (radius encodes in-degree), not a function of
        // how far out the view happens to be fitted. Without this, the
        // initial fit-to-view zoom on this graph's real (sparse, spread-out)
        // layout lands around k=0.16, which shrank the 10.5px label text to
        // under 2 effective screen pixels -- found by actually looking at a
        // screenshot, not by any DOM metric. updateNodeScale() re-sets this
        // group's scale on every pan/zoom.
        var inner = el("g", { "class": "gn-inner", transform: "scale(" + invK + ")" });
        var r = radiusFor(n.in_degree || 0);
        var selRing = el("circle", { "class": "gn-sel-ring", r: r + 4 });
        var circle = el("circle", { "class": "gn-circle", r: r, fill: TIER_COLOR[n.tier] || "var(--muted)" });
        var label = el("text", { "class": "gn-label", x: r + 5, y: 4 });
        label.textContent = n.title && n.title.length <= 46 ? n.title : (n.title || n.id).slice(0, 44) + "…";
        inner.appendChild(selRing);
        inner.appendChild(circle);
        inner.appendChild(label);
        g.appendChild(inner);
        wireNode(g, n);
        gNodes.appendChild(g);
        nodeEls[n.id] = { g: g, inner: inner, circle: circle, label: label, selRing: selRing };
      });
    }

    function updateNodeScale() {
      var invK = 1 / (state.transform.k || 1);
      Object.keys(nodeEls).forEach(function (id) {
        nodeEls[id].inner.setAttribute("transform", "scale(" + invK + ")");
      });
      // Timeline chrome labels (constScaleText) and typed-edge labels use
      // the same constant-screen-size technique; keep them in lockstep with
      // every node-marker rescale.
      scaledTextEls.forEach(function (s) {
        s.g.setAttribute("transform", "translate(" + s.x + "," + s.y + ") scale(" + invK + ")");
      });
    }

    function moveNode(id) {
      var n = state.nodeById[id];
      var p = posOf(n);
      var rec = nodeEls[id];
      if (rec) rec.g.setAttribute("transform", "translate(" + p.x + "," + p.y + ")");
      var invK = 1 / (state.transform.k || 1);
      edgeEls.forEach(function (e) {
        if (e.source !== id && e.target !== id) return;
        var pa = posOf(e.a), pb = posOf(e.b);
        e.gradient.setAttribute("x1", pa.x); e.gradient.setAttribute("y1", pa.y);
        e.gradient.setAttribute("x2", pb.x); e.gradient.setAttribute("y2", pb.y);
        var mx = (pa.x + pb.x) / 2, my = (pa.y + pb.y) / 2;
        var dx = pb.x - pa.x, dy = pb.y - pa.y;
        var bow = Math.min(24, Math.hypot(dx, dy) * 0.08);
        var nx = -dy, ny = dx;
        var nl = Math.hypot(nx, ny) || 1;
        var cx = mx + (nx / nl) * bow, cy = my + (ny / nl) * bow;
        e.path.setAttribute("d", "M" + pa.x + "," + pa.y + " Q" + cx + "," + cy + " " + pb.x + "," + pb.y);
        if (e.label) {
          // Keep the typed-edge label's world anchor (and the parallel
          // bookkeeping entry updateNodeScale() reads) in sync with the
          // edge it annotates during a live drag, same combined
          // translate+scale convention updateNodeScale() applies everywhere
          // else.
          var entry = scaledTextEls.filter(function (s) { return s.g === e.label; })[0];
          if (entry) { entry.x = cx; entry.y = cy; }
          e.label.setAttribute("transform", "translate(" + cx + "," + cy + ") scale(" + invK + ")");
        }
      });
    }

    // ------------------------------------------------- interaction state
    // Explicit state machine (2026-08-10 redesign, owner-reported bug: "if I
    // click one node, things get cluttered, and I have to refresh the page
    // to reset"). THREE independent pieces of UI state, one render function,
    // one reset function that clears all three at once:
    //   ui.hoverId    -- TRANSIENT. Set on pointerenter, cleared on
    //                    pointerleave AND on window blur/visibilitychange
    //                    (see below: opening a new tab mid-hover does not
    //                    reliably fire pointerleave on the page left behind,
    //                    which was the root cause of the reported bug --
    //                    click used to open a new tab immediately, so a
    //                    hover-then-click sequence could leave the ORIGINAL
    //                    tab's dim-and-highlight stuck forever with no event
    //                    left to clear it, and a refresh was the only way
    //                    out. Reproduced live, fixed at the model level, not
    //                    by special-casing that one path).
    //   ui.selectedId -- STICKY. Set by clicking a node (toggles off if you
    //                    click the SAME node again), survives until an
    //                    explicit clear. Click no longer opens the page --
    //                    double-click does (see wireNode) -- so a single
    //                    click can never itself trigger a stuck-looking
    //                    navigation-plus-dim combo.
    //   ui.searchHits -- the current search box matches, independent of the
    //                    above, but cleared by the SAME reset as everything
    //                    else, not only by blanking the search box by hand.
    // The one QUESTION every state must answer per the owner's ask: "what
    // undoes this?" hover -> leave the node (or nothing else is holding
    // it). selected -> click empty background, or Escape. search -> clear
    // the box, or click empty background, or Escape. Escape and a
    // background click/tap both call resetToBaseline(), which clears all
    // three unconditionally -- there is no reachable combination of
    // hover/select/search/drag that resetToBaseline() does not fully undo.
    var ui = { hoverId: null, selectedId: null, searchHits: [] };

    function currentFocusId() {
      // Hover is a temporary "look here" that always wins visually while
      // active; releasing it falls back to the sticky selection (or to
      // baseline if nothing is selected) rather than to some third state.
      return ui.hoverId || ui.selectedId;
    }

    function applyEmphasis() {
      var focus = currentFocusId();
      if (focus && state.nodeById[focus]) {
        var neigh = neighborsOf(focus);
        neigh[focus] = true;
        svgEl.classList.add("dimmed");
        Object.keys(nodeEls).forEach(function (nid) {
          nodeEls[nid].g.classList.toggle("hi", !!neigh[nid]);
        });
        edgeEls.forEach(function (e) {
          e.path.classList.toggle("lit", e.source === focus || e.target === focus);
        });
      } else {
        svgEl.classList.remove("dimmed");
        Object.keys(nodeEls).forEach(function (nid) { nodeEls[nid].g.classList.remove("hi"); });
        edgeEls.forEach(function (e) { e.path.classList.remove("lit"); });
      }
      Object.keys(nodeEls).forEach(function (nid) {
        nodeEls[nid].g.classList.toggle("selected", nid === ui.selectedId);
      });
    }

    function applySearchHighlight() {
      var hits = {};
      ui.searchHits.forEach(function (id) { hits[id] = true; });
      Object.keys(nodeEls).forEach(function (nid) {
        nodeEls[nid].g.classList.toggle("search-hit", !!hits[nid]);
      });
    }

    function resetToBaseline() {
      ui.hoverId = null;
      ui.selectedId = null;
      ui.searchHits = [];
      if (searchInput && searchInput.value) searchInput.value = "";
      applyEmphasis();
      applySearchHighlight();
    }

    function selectNode(id) {
      ui.selectedId = ui.selectedId === id ? null : id;
      applyEmphasis();
    }

    // window/tab losing focus is exactly the failure mode that used to
    // strand a hover -- clear the TRANSIENT piece defensively whenever that
    // can happen, regardless of why. A sticky selection is untouched (it is
    // meant to survive; only Escape/background clears it).
    window.addEventListener("blur", function () { if (ui.hoverId) { ui.hoverId = null; applyEmphasis(); } });
    document.addEventListener("visibilitychange", function () {
      if (document.hidden && ui.hoverId) { ui.hoverId = null; applyEmphasis(); }
    });
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape") resetToBaseline();
    });

    // -------------------------------------------------------------- drag
    var dragState = null;
    function wireNode(g, n) {
      g.addEventListener("pointerenter", function () {
        if (dragState) return;
        ui.hoverId = n.id;
        applyEmphasis();
      });
      g.addEventListener("pointerleave", function () {
        if (dragState) return;
        if (ui.hoverId === n.id) { ui.hoverId = null; applyEmphasis(); }
      });
      g.addEventListener("pointerdown", function (ev) {
        ev.stopPropagation();
        g.setPointerCapture(ev.pointerId);
        var start = svgPoint(ev);
        // click-vs-drag threshold in SCREEN pixels (clientX/clientY), not
        // world space: found live (round-2 QA) that comparing world-space
        // distance made the threshold zoom-dependent -- at a zoomed-OUT
        // view (k well under 1, the common case here given how spread out
        // this graph is), a couple of screen pixels of jitter divides by k
        // into MORE than 3 world units, so a stationary click kept getting
        // misclassified as a drag and selectNode() never ran. Screen-space
        // matches what "did the pointer actually move" means to a user, and
        // matches the background-tap threshold below, which was already
        // correct.
        dragState = { id: n.id, moved: false, startX: start.x, startY: start.y,
                      clientX0: ev.clientX, clientY0: ev.clientY, sim: null };
        startLocalSim(n.id);
      });
      g.addEventListener("pointermove", function (ev) {
        if (!dragState || dragState.id !== n.id) return;
        var p = svgPoint(ev);
        if (Math.hypot(ev.clientX - dragState.clientX0, ev.clientY - dragState.clientY0) > 3) dragState.moved = true;
        state.manualPos[n.id] = { x: p.x, y: p.y };
        if (dragState.sim) {
          // The tick handler deliberately skips the dragged node itself (it
          // must follow the pointer exactly, not the sim's own integration),
          // so nothing else applies this DOM update on the sim path unless
          // done here too.
          var dn = dragState.sim.nodesById[n.id];
          if (dn) { dn.fx = p.x; dn.fy = p.y; }
        }
        moveNode(n.id);
      });
      g.addEventListener("pointerup", function (ev) {
        if (!dragState || dragState.id !== n.id) return;
        var moved = dragState.moved;
        stopLocalSim();
        dragState = null;
        // A drag NEVER latches a mode (round-2 requirement): it only ever
        // moves a node. A plain tap (no movement) SELECTS -- it does not
        // navigate, so a single click can never combine with a stranded
        // hover the way the old click-opens-immediately behavior did.
        if (!moved) {
          selectNode(n.id);
        } else {
          // Third instance of the SAME underlying failure mode as the
          // dblclick fix above, found by the journey harness (J7): pointer
          // CAPTURE (g.setPointerCapture, needed so the drag tracks the
          // pointer outside the node's own hit box) also suppresses the
          // natural pointerleave a real drag-away would otherwise fire, so
          // hover state set by the mouse-move-then-down that STARTED the
          // drag stayed dimmed/highlighted after release with nothing left
          // to clear it -- a drag-flavored version of the exact "stuck"
          // bug the owner reported. Clear it explicitly on drag end.
          if (ui.hoverId === n.id) { ui.hoverId = null; applyEmphasis(); }
        }
      });
      g.addEventListener("dblclick", function (ev) {
        ev.preventDefault();
        // Defensive: opening a new tab can leave the pointer "hovering" an
        // element the mouse never actually left (the tab switch, not a
        // pointerleave, is what changes) -- clear the transient piece
        // before navigating so there is nothing left to strand.
        ui.hoverId = null;
        applyEmphasis();
        window.open(n.url, "_blank", "noopener");
      });
      g.addEventListener("click", function (ev) { ev.preventDefault(); });
    }

    function startLocalSim(id) {
      if (typeof d3 === "undefined" || !d3.forceSimulation) return; // graceful no-op without the vendored lib
      // Timeline positions are deterministic-from-data by design (round-2
      // spec: "no physics in this mode") -- posOf() already ignores
      // manualPos/sim output while in timeline mode, so a drag has no
      // visible effect there; skip spinning up the simulation at all rather
      // than run physics nothing will render.
      if (state.mode === "timeline") return;
      var neigh = neighborsOf(id);
      var localIds = Object.keys(neigh).concat([id]);
      var simNodes = localIds.map(function (nid) {
        var p = posOf(state.nodeById[nid]);
        return { id: nid, x: p.x, y: p.y, fx: nid === id ? p.x : null, fy: nid === id ? p.y : null };
      });
      var byId = {};
      simNodes.forEach(function (sn) { byId[sn.id] = sn; });
      var simLinks = state.edges.filter(function (e) { return byId[e.source] && byId[e.target]; })
        .map(function (e) { return { source: e.source, target: e.target }; });
      var sim = d3.forceSimulation(simNodes)
        .force("link", d3.forceLink(simLinks).id(function (d) { return d.id; }).distance(70).strength(0.5))
        .force("charge", d3.forceManyBody().strength(-120))
        .force("collide", d3.forceCollide(26))
        .alphaTarget(0.35)
        .on("tick", function () {
          simNodes.forEach(function (sn) {
            if (sn.id === id) return; // dragged node follows the pointer, not the sim
            state.manualPos[sn.id] = { x: sn.x, y: sn.y };
            moveNode(sn.id);
          });
        });
      dragState.sim = { sim: sim, nodesById: byId };
    }
    function stopLocalSim() {
      if (dragState && dragState.sim) {
        dragState.sim.sim.alphaTarget(0);
        setTimeout(function () { dragState && dragState.sim && dragState.sim.sim.stop(); }, 400);
      }
    }

    // ---------------------------------------------------- pan / zoom
    function svgPoint(ev) {
      var rect = svgEl.getBoundingClientRect();
      var sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
      var t = state.transform;
      return { x: (sx - t.x) / t.k, y: (sy - t.y) / t.k };
    }
    var panState = null;
    var pointers = {};
    var bgTap = null; // {x0,y0,moved} -- distinguishes a background CLICK (deselect) from a PAN drag
    svgEl.addEventListener("pointerdown", function (ev) {
      if (ev.target !== svgEl) return; // node handlers already stopPropagation
      pointers[ev.pointerId] = { x: ev.clientX, y: ev.clientY };
      var ids = Object.keys(pointers);
      if (ids.length === 1) {
        svgEl.setPointerCapture(ev.pointerId);
        svgEl.classList.add("panning");
        panState = { x0: ev.clientX, y0: ev.clientY, tx0: state.transform.x, ty0: state.transform.y };
        bgTap = { x0: ev.clientX, y0: ev.clientY, moved: false };
      }
    });
    svgEl.addEventListener("pointermove", function (ev) {
      if (!pointers[ev.pointerId]) return;
      pointers[ev.pointerId] = { x: ev.clientX, y: ev.clientY };
      var ids = Object.keys(pointers);
      if (ids.length === 2) {
        var p0 = pointers[ids[0]], p1 = pointers[ids[1]];
        var dist = Math.hypot(p1.x - p0.x, p1.y - p0.y);
        if (panState && panState.pinchDist) {
          var factor = dist / panState.pinchDist;
          zoomAt((p0.x + p1.x) / 2, (p0.y + p1.y) / 2, factor);
        }
        panState = panState || {};
        panState.pinchDist = dist;
      } else if (ids.length === 1 && panState) {
        state.transform.x = panState.tx0 + (ev.clientX - panState.x0);
        state.transform.y = panState.ty0 + (ev.clientY - panState.y0);
        applyTransform();
        if (bgTap && Math.hypot(ev.clientX - bgTap.x0, ev.clientY - bgTap.y0) > 3) bgTap.moved = true;
      }
    });
    function endPointer(ev) {
      delete pointers[ev.pointerId];
      if (Object.keys(pointers).length === 0) {
        panState = null;
        svgEl.classList.remove("panning");
        // A background TAP (no pan movement) is the "click empty background"
        // reset the round-2 model requires: distinguished from a pan drag by
        // the same movement threshold used everywhere else (node drag,
        // click-vs-drag), not a separate ad hoc rule.
        if (bgTap && !bgTap.moved) resetToBaseline();
        bgTap = null;
      }
    }
    svgEl.addEventListener("pointerup", endPointer);
    svgEl.addEventListener("pointercancel", endPointer);
    svgEl.addEventListener("pointerleave", endPointer);

    function zoomAt(clientX, clientY, factor) {
      var rect = svgEl.getBoundingClientRect();
      var sx = clientX - rect.left, sy = clientY - rect.top;
      var t = state.transform;
      var k2 = Math.max(0.12, Math.min(5, t.k * factor));
      var wx = (sx - t.x) / t.k, wy = (sy - t.y) / t.k;
      t.k = k2;
      t.x = sx - wx * k2;
      t.y = sy - wy * k2;
      applyTransform();
    }
    svgEl.addEventListener("wheel", function (ev) {
      ev.preventDefault();
      var factor = Math.exp(-ev.deltaY * 0.0018);
      zoomAt(ev.clientX, ev.clientY, factor);
    }, { passive: false });

    // ------------------------------------------------------------ search
    function doSearch(q) {
      q = (q || "").trim().toLowerCase();
      if (!q) { ui.searchHits = []; applySearchHighlight(); return; }
      var hits = [];
      var best = null;
      state.nodes.forEach(function (n) {
        var hay = (n.title + " " + n.id).toLowerCase();
        if (hay.indexOf(q) === -1) return;
        var score = n.id.toLowerCase() === q ? 3 : (n.title.toLowerCase().indexOf(q) === 0 ? 2 : 1);
        if (!best || score > best.score) best = { n: n, score: score };
        hits.push(n.id);
      });
      ui.searchHits = hits;
      applySearchHighlight();
      if (best) {
        var rect = svgEl.getBoundingClientRect();
        var p = posOf(best.n);
        var k = Math.max(state.transform.k, 1.3);
        state.transform = { k: k, x: rect.width / 2 - p.x * k, y: rect.height / 2 - p.y * k };
        applyTransform();
      }
    }
    if (searchInput) searchInput.addEventListener("input", function () { doSearch(searchInput.value); });

    // ------------------------------------------------------------ orphans
    function renderOrphans() {
      var orphans = state.nodes.filter(function (n) { return (n.in_degree || 0) === 0 && (n.out_degree || 0) === 0; });
      orphanList.innerHTML = "";
      if (!orphans.length) {
        var li = document.createElement("li");
        li.className = "go-empty";
        li.textContent = "None: every page has at least one content link.";
        orphanList.appendChild(li);
        return;
      }
      orphans.sort(function (a, b) { return (a.title || a.id).localeCompare(b.title || b.id); });
      orphans.forEach(function (n) {
        var li = document.createElement("li");
        var a = document.createElement("a");
        a.href = n.url; a.target = "_blank"; a.rel = "noopener";
        a.textContent = n.title || n.id;
        var tier = document.createElement("span");
        tier.className = "go-tier";
        tier.textContent = TIER_LABEL[n.tier] || n.tier;
        a.appendChild(tier);
        li.appendChild(a);
        orphanList.appendChild(li);
      });
    }

    // -------------------------------------------------------------- load
    function applyData(data) {
      var newIds = {};
      data.nodes.forEach(function (n) { newIds[n.id] = true; });
      // drop manual overrides for nodes that no longer exist
      Object.keys(state.manualPos).forEach(function (id) { if (!newIds[id]) delete state.manualPos[id]; });
      if (ui.hoverId && !newIds[ui.hoverId]) ui.hoverId = null;
      if (ui.selectedId && !newIds[ui.selectedId]) ui.selectedId = null;
      ui.searchHits = ui.searchHits.filter(function (id) { return newIds[id]; });
      state.nodes = data.nodes;
      state.nodeById = {};
      data.nodes.forEach(function (n) { state.nodeById[n.id] = n; });
      state.edges = data.edges;
      var tl = computeTimelineLayout();
      state.timelinePos = tl.pos;
      timelineRange = { minT: tl.minT, maxT: tl.maxT };
      render();
      renderTimelineChrome();
      renderOrphans();
      // render() rebuilds every node/edge element from scratch (2-minute
      // background refresh included); reapply whatever UI state was active
      // so a periodic refresh never silently clears a selection/search the
      // user is actively looking at.
      applyEmphasis();
      applySearchHighlight();
      if (!state.viewInited) { fitView(); state.viewInited = true; }
    }

    function load(first) {
      fetch(src, { cache: "no-cache" }).then(function (r) { return r.json(); }).then(function (data) {
        applyData(data);
      }).catch(function (err) {
        if (first) orphanList.innerHTML = '<li class="go-empty">Could not load graph.json.</li>';
      });
    }

    window.addEventListener("resize", function () { if (state.viewInited) applyTransform(); });

    // ------------------------------------------------------------- mode --
    var modeButtons = root.querySelectorAll("[data-graph-mode]");
    function setMode(mode) {
      if (mode === state.mode) return;
      state.mode = mode;
      resetToBaseline(); // a mode switch is a clean re-render, not a state to carry across (round-2 spec)
      modeButtons.forEach(function (b) { b.classList.toggle("active", b.getAttribute("data-graph-mode") === mode); });
      // Timeline packs nodes by date, not by a physics-relaxed spread, so a
      // busy week collides labels the same way round 1's orphans did in map
      // mode (found live: several same-week root pages rendered as one
      // unreadable smear of overlapping text). Same fix as orphans there:
      // dots only by default, label on hover/select/search -- the point of
      // this view is the density/shape of activity over time, not reading
      // every title simultaneously.
      svgEl.classList.toggle("mode-timeline", mode === "timeline");
      render();
      renderTimelineChrome();
      fitView();
    }
    modeButtons.forEach(function (b) {
      b.addEventListener("click", function () { setMode(b.getAttribute("data-graph-mode")); });
    });

    load(true);
    setInterval(function () { load(false); }, REFRESH_MS);
  }

  document.addEventListener("DOMContentLoaded", function () {
    var root = document.querySelector(".graph-page");
    if (root) init(root);
  });
})();
