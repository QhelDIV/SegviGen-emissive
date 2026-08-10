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
  var TIER_ORDER = ["root", "workspace", "preview", "update"];
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
      manualPos: {},       // id -> {x,y} session-local drag override
      transform: { x: 0, y: 0, k: 1 },
      viewInited: false
    };

    var gViewport = el("g", { "class": "viewport" });
    var gEdges = el("g", { "class": "edges" });
    var gNodes = el("g", { "class": "nodes" });
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
      return state.manualPos[n.id] || { x: n.x, y: n.y };
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
      var core = largestComponent();
      var b = boundsOf(core.length > 1 ? core : state.nodes);
      var pad = 60;
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
        var path = el("path", {
          "class": "gn-edge", d: "M" + pa.x + "," + pa.y + " Q" + cx + "," + cy + " " + pb.x + "," + pb.y,
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
        edgeEls.push({ source: e.source, target: e.target, path: path, gradient: grad, a: a, b: b });
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
        var circle = el("circle", { "class": "gn-circle", r: r, fill: TIER_COLOR[n.tier] || "var(--muted)" });
        var label = el("text", { "class": "gn-label", x: r + 5, y: 4 });
        label.textContent = n.title && n.title.length <= 46 ? n.title : (n.title || n.id).slice(0, 44) + "…";
        inner.appendChild(circle);
        inner.appendChild(label);
        g.appendChild(inner);
        wireNode(g, n);
        gNodes.appendChild(g);
        nodeEls[n.id] = { g: g, inner: inner, circle: circle, label: label };
      });
    }

    function updateNodeScale() {
      var invK = 1 / (state.transform.k || 1);
      Object.keys(nodeEls).forEach(function (id) {
        nodeEls[id].inner.setAttribute("transform", "scale(" + invK + ")");
      });
    }

    function moveNode(id) {
      var n = state.nodeById[id];
      var p = posOf(n);
      var rec = nodeEls[id];
      if (rec) rec.g.setAttribute("transform", "translate(" + p.x + "," + p.y + ")");
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
      });
    }

    // ------------------------------------------------------------- hover
    function clearHover() {
      svgEl.classList.remove("dimmed");
      Object.keys(nodeEls).forEach(function (id) { nodeEls[id].g.classList.remove("hi"); nodeEls[id].label.classList.remove("hi"); });
      edgeEls.forEach(function (e) { e.path.classList.remove("lit"); });
    }
    function setHover(id) {
      var neigh = neighborsOf(id);
      neigh[id] = true;
      svgEl.classList.add("dimmed");
      Object.keys(nodeEls).forEach(function (nid) {
        var on = !!neigh[nid];
        nodeEls[nid].g.classList.toggle("hi", on);
      });
      edgeEls.forEach(function (e) {
        e.path.classList.toggle("lit", e.source === id || e.target === id);
      });
    }

    // -------------------------------------------------------------- drag
    var dragState = null;
    function wireNode(g, n) {
      g.addEventListener("pointerenter", function () { if (!dragState) setHover(n.id); });
      g.addEventListener("pointerleave", function () { if (!dragState) clearHover(); });
      g.addEventListener("pointerdown", function (ev) {
        ev.stopPropagation();
        g.setPointerCapture(ev.pointerId);
        var start = svgPoint(ev);
        dragState = { id: n.id, moved: false, startX: start.x, startY: start.y, sim: null };
        startLocalSim(n.id);
      });
      g.addEventListener("pointermove", function (ev) {
        if (!dragState || dragState.id !== n.id) return;
        var p = svgPoint(ev);
        if (Math.hypot(p.x - dragState.startX, p.y - dragState.startY) > 3) dragState.moved = true;
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
        if (!moved) window.open(n.url, "_blank", "noopener");
      });
      g.addEventListener("click", function (ev) { ev.preventDefault(); });
    }

    function startLocalSim(id) {
      if (typeof d3 === "undefined" || !d3.forceSimulation) return; // graceful no-op without the vendored lib
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
    svgEl.addEventListener("pointerdown", function (ev) {
      if (ev.target !== svgEl) return; // node handlers already stopPropagation
      pointers[ev.pointerId] = { x: ev.clientX, y: ev.clientY };
      var ids = Object.keys(pointers);
      if (ids.length === 1) {
        svgEl.setPointerCapture(ev.pointerId);
        svgEl.classList.add("panning");
        panState = { x0: ev.clientX, y0: ev.clientY, tx0: state.transform.x, ty0: state.transform.y };
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
      }
    });
    function endPointer(ev) {
      delete pointers[ev.pointerId];
      if (Object.keys(pointers).length === 0) { panState = null; svgEl.classList.remove("panning"); }
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
      Object.keys(nodeEls).forEach(function (id) { nodeEls[id].g.classList.remove("search-hit"); });
      if (!q) return;
      var best = null;
      state.nodes.forEach(function (n) {
        var hay = (n.title + " " + n.id).toLowerCase();
        if (hay.indexOf(q) === -1) return;
        var score = n.id.toLowerCase() === q ? 3 : (n.title.toLowerCase().indexOf(q) === 0 ? 2 : 1);
        if (!best || score > best.score) best = { n: n, score: score };
        var rec = nodeEls[n.id];
        if (rec) rec.g.classList.add("search-hit");
      });
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
        li.textContent = "None — every page has at least one content link.";
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
      state.nodes = data.nodes;
      state.nodeById = {};
      data.nodes.forEach(function (n) { state.nodeById[n.id] = n; });
      state.edges = data.edges;
      render();
      renderOrphans();
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
    load(true);
    setInterval(function () { load(false); }, REFRESH_MS);
  }

  document.addEventListener("DOMContentLoaded", function () {
    var root = document.querySelector(".graph-page");
    if (root) init(root);
  });
})();
