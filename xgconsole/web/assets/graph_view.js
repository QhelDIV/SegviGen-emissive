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

  // JOB-graph model (2026-08-14): nodes are board jobs (filled, colored by
  // track) plus legacy pages no job claims (smaller, hollow, one muted
  // style -- artifacts, secondary by design). The old four-tier page
  // palette died with the page-graph model.
  var TRACK_COLOR = {
    research: "var(--good)",
    tooling: "var(--blue-ink)",
    paper: "var(--violet-ink, var(--blue-ink))"
  };
  var TIER_LABEL = { root: "root", workspace: "workspace", preview: "preview", update: "update" };

  function colorFor(n) {
    if (n.kind === "job") return TRACK_COLOR[n.track] || "var(--muted)";
    return "var(--muted)";
  }

  var REFRESH_MS = 120000;
  var SVG_NS = "http://www.w3.org/2000/svg";

  function radiusFor(inDeg, kind) {
    var r = 6 + Math.min(inDeg, 12) * 1.4;
    if (kind === "page") r = 4.5 + Math.min(inDeg, 12) * 1.0;  // artifacts read smaller
    return Math.max(4.5, Math.min(22, r));
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
    var detailEl = root.querySelector("#graph-detail");
    var rootsEl = root.querySelector("#graph-roots");
    var src = svgEl.getAttribute("data-graph-src") || root.getAttribute("data-graph-src");

    var state = {
      nodes: [],          // last-loaded node records, by id
      nodeById: {},
      edges: [],
      manualPos: {},       // id -> {x,y} session-local drag override (map mode only)
      transform: { x: 0, y: 0, k: 1 },
      viewInited: false,
      mode: "timeline",    // "map" | "timeline"; timeline is the DEFAULT
                           // (owner-ratified 2026-08-14: "much easier to
                           // navigate"); keep in sync with the baked
                           // .active button in build_graph.py's toolbar
      timelinePos: {}      // id -> {x,y}, recomputed whenever data loads; see computeTimelineBins()
    };

    var gViewport = el("g", { "class": "viewport" });
    var gTimeChrome = el("g", { "class": "timeline-chrome" });
    var gEdges = el("g", { "class": "edges" });
    var gNodes = el("g", { "class": "nodes" });
    gViewport.appendChild(gTimeChrome);
    gViewport.appendChild(gEdges);
    gViewport.appendChild(gNodes);
    svgEl.appendChild(gViewport);
    svgEl.classList.toggle("mode-timeline", state.mode === "timeline");

    // Zoom level past which labels reveal generally instead of needing a
    // per-node hover (round-3): the fit-to-view zoom for a real graph this
    // size lands well under 1 (fitting everything into the viewport at
    // once is exactly the crowded case labels-hidden-by-default exists
    // for), so a threshold a bit above that default -- reached by a couple
    // of scroll-zoom steps -- reads as "the user zoomed in to look at
    // something", not the initial overview.
    var LABEL_REVEAL_K = 0.55;

    function applyTransform() {
      var t = state.transform;
      gViewport.setAttribute("transform", "translate(" + t.x + "," + t.y + ") scale(" + t.k + ")");
      // Timeline reveals names much earlier than Map (its bins and stacks
      // guarantee spacing once zoomed slightly), but NOT at the overview
      // fit: 69 constant-size labels in one viewport overlap by geometry,
      // no tuning escapes that (tried 0.16 = always-on, measured a text
      // smear). 0.32 is where an 18-char label clears the 260-unit bin
      // gap; one-two scroll steps from the fit, and hover still reveals
      // any single label at any zoom.
      var revealK = state.mode === "timeline" ? 0.32 : LABEL_REVEAL_K;
      svgEl.classList.toggle("labels-visible", t.k >= revealK);
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
    // Round-3c rewrite (owner sketch, superseding round-3's per-day-width
    // scheme entirely): NO tier lanes (tier is color + legend only, same
    // as map mode); pages bin by date -- ALGORITHMICALLY, not one slot per
    // calendar day -- and stack VERTICALLY within a bin, a column of
    // circles with short labels under them, exactly the "circles stacked
    // vertically... arcs curving into them" sketch. A year band and a
    // month band sit above the bin row (round-3b, composed with this:
    // "show year, month on the top... only show day at the column").
    var TIMELINE_STACK_GAP = 110;  // vertical gap between two stacked circles in the same bin
    var TIMELINE_BIN_GAP = 260;    // horizontal gap between bin x-positions
    // Bin cap (round-3c, owner: "The bin can be a day, or can be several
    // days, depending smartly (need an algorithm... you decide)"): 7 is
    // the master's pick, tuned by eye against this project's own busiest
    // day (8 pages on one real day) so that day still gets its own tall
    // bin rather than needing to split unnaturally, while a run of quiet
    // 1-2-page days merges into one bin instead of eight nearly-empty
    // columns in a row.
    var TIMELINE_BIN_CAP = 7;
    var DAY_MS = 86400000;
    var MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

    function parseCreated(s) {
      var t = Date.parse((s || "").replace(" ", "T"));
      return isNaN(t) ? null : t;
    }
    function dayKey(t) { return new Date(t).toISOString().slice(0, 10); }

    function computeTimelineBins() {
      var UNKNOWN = "unknown";
      var byDay = {}; // dayKey -> [{n, t}], UNKNOWN holds unparseable dates
      state.nodes.forEach(function (n) {
        var t = parseCreated(n.created);
        var key = t === null ? UNKNOWN : dayKey(t);
        (byDay[key] = byDay[key] || []).push({ n: n, t: t });
      });
      var activeDays = Object.keys(byDay).filter(function (k) { return k !== UNKNOWN; }).sort();

      // ALGORITHMIC, DETERMINISTIC binning: walk days chronologically,
      // merging a day into the current bin as long as doing so keeps the
      // bin's running page count at or under TIMELINE_BIN_CAP; a single
      // day whose OWN count already exceeds the cap still gets its own bin
      // (taller, never split across two bins) rather than being forced to
      // fit. Pure function of the sorted per-day counts already in the
      // data -- no randomness, so a content-unchanged rebuild reproduces
      // the identical bin set (and therefore the identical layout) every
      // time, same invariant Map mode's persisted positions guarantee a
      // different way.
      var bins = [];
      var cur = null;
      activeDays.forEach(function (day) {
        var count = byDay[day].length;
        if (cur && cur.count + count <= TIMELINE_BIN_CAP) {
          cur.days.push(day);
          cur.count += count;
        } else {
          cur = { days: [day], count: count };
          bins.push(cur);
        }
      });

      var pos = {};
      var binMeta = [];
      bins.forEach(function (bin, i) {
        var x = i * TIMELINE_BIN_GAP;
        var recs = [];
        bin.days.forEach(function (day) { recs = recs.concat(byDay[day]); });
        recs.sort(function (a, b) { return (a.t || 0) - (b.t || 0); });
        var n = recs.length;
        var top = -((n - 1) * TIMELINE_STACK_GAP) / 2; // stack centered on y=0
        recs.forEach(function (r, j) { pos[r.n.id] = { x: x, y: top + j * TIMELINE_STACK_GAP }; });
        binMeta.push({ x: x, days: bin.days, count: n, top: top });
      });

      // Unknown-date nodes (should not occur in practice -- every page's
      // `created` comes from a real file mtime -- but stay renderable
      // rather than silently vanishing if it ever does): their own bin,
      // one gap to the left of the first real bin.
      if (byDay[UNKNOWN]) {
        var uRecs = byDay[UNKNOWN].slice().sort(function (a, b) { return (a.t || 0) - (b.t || 0); });
        var ux = -TIMELINE_BIN_GAP;
        var un = uRecs.length;
        var utop = -((un - 1) * TIMELINE_STACK_GAP) / 2;
        uRecs.forEach(function (r, j) { pos[r.n.id] = { x: ux, y: utop + j * TIMELINE_STACK_GAP }; });
      }

      return { pos: pos, bins: binMeta };
    }

    function binDayLabel(bin) {
      // Bare day number(s) only (round-3b: "only show day at the column
      // to avoid cluttering" -- the year/month bands above already say
      // which year/month). A merged bin shows a day-number range; the
      // month-crossing case is rare (would need >TIMELINE_BIN_CAP pages'
      // worth of quiet days spanning a month boundary) but still renders
      // sensibly with an explicit month abbreviation on each side.
      var days = bin.days;
      var first = days[0], last = days[days.length - 1];
      var d0 = +first.slice(8, 10), d1 = +last.slice(8, 10);
      if (days.length === 1) return String(d0);
      if (first.slice(0, 7) === last.slice(0, 7)) return d0 + "–" + d1;
      return MONTH_ABBR[+first.slice(5, 7) - 1] + " " + d0 + " – " + MONTH_ABBR[+last.slice(5, 7) - 1] + " " + d1;
    }

    function computeHeaderGroups(binMeta) {
      // Month/year bands span the x-range of the consecutive bins that
      // fall inside them (bins are already chronological by construction,
      // so a simple run-length grouping is correct, no re-sorting needed).
      var months = [], years = [];
      var curMonth = null, curYear = null;
      var halfGap = TIMELINE_BIN_GAP / 2;
      binMeta.forEach(function (bin) {
        var d0 = bin.days[0];
        var y = +d0.slice(0, 4), m = +d0.slice(5, 7);
        if (!curMonth || curMonth.year !== y || curMonth.month !== m) {
          curMonth = { year: y, month: m, x0: bin.x - halfGap, x1: bin.x + halfGap };
          months.push(curMonth);
        } else {
          curMonth.x1 = bin.x + halfGap;
        }
        if (!curYear || curYear.year !== y) {
          curYear = { year: y, x0: bin.x - halfGap, x1: bin.x + halfGap };
          years.push(curYear);
        } else {
          curYear.x1 = bin.x + halfGap;
        }
      });
      // stackTop: the highest (most negative y) any bin's own stack
      // reaches, across every bin -- the header rows anchor above THIS,
      // not above any one bin, so a tall bin never collides with the
      // header no matter where it sits in the sequence.
      var stackTop = 0;
      binMeta.forEach(function (bin) { stackTop = Math.min(stackTop, bin.top); });
      var dayRowY = stackTop - 56;
      var monthRowY = dayRowY - 34;
      var yearRowY = monthRowY - 30;
      return { months: months, years: years, stackTop: stackTop, dayRowY: dayRowY, monthRowY: monthRowY, yearRowY: yearRowY };
    }

    var timelineRange = { bins: [], months: [], years: [], stackTop: 0, dayRowY: -56, monthRowY: -90, yearRowY: -120 };

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
      var bins = timelineRange.bins;
      if (!bins.length) return;
      var lastX = bins[bins.length - 1].x;
      var halfGap = TIMELINE_BIN_GAP / 2;

      // Year band (top), month band (below it), bin day-label row (just
      // above the tallest stack) -- round-3b's "show year, month on the
      // top... only show day at the column", composed with round-3c's
      // bins replacing individual day columns. Each band is a translucent
      // rect spanning its own x-range plus a centered, constant-screen-size
      // label; consecutive bands alternate a hair of opacity so adjacent
      // months/years are visually separable even when a label is long
      // enough to slightly crowd its neighbor's.
      function band(rowY, rowH, groups, cls, fmt) {
        groups.forEach(function (g, i) {
          var rect = el("rect", {
            "class": cls + (i % 2 ? " alt" : ""), x: g.x0, y: rowY - rowH / 2,
            width: Math.max(g.x1 - g.x0, 1), height: rowH
          });
          gTimeChrome.appendChild(rect);
          gTimeChrome.appendChild(constScaleText("chrome", cls + "-label", (g.x0 + g.x1) / 2, rowY + 4, fmt(g)));
        });
      }
      band(timelineRange.yearRowY, 30, timelineRange.years, "tl-year-band", function (g) { return String(g.year); });
      band(timelineRange.monthRowY, 30, timelineRange.months, "tl-month-band",
        function (g) { return MONTH_ABBR[g.month - 1]; });

      // Bin day-label row: bare day number(s), one per bin, plus a thin
      // guide line from the label down to that bin's own topmost circle
      // (not a shared axis line -- bins don't share a row height the way
      // the old tier lanes did, each bin's stack top is wherever its own
      // page count put it).
      bins.forEach(function (bin) {
        gTimeChrome.appendChild(constScaleText("chrome", "tl-day-label", bin.x, timelineRange.dayRowY + 4, binDayLabel(bin)));
        var vline = el("line", {
          "class": "tl-day-tick", x1: bin.x, y1: timelineRange.dayRowY + 14, x2: bin.x, y2: bin.top - 20
        });
        gTimeChrome.appendChild(vline);
      });
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
        // non-overlapping by construction) -- fit EVERY node, plus the
        // year/month/day header's own extent above the stacks (fixed
        // offsets, see renderTimelineChrome()/computeHeaderGroups()) and a
        // margin below for the shortname labels that now render UNDER each
        // circle (round-3c) rather than beside it.
        b = boundsOf(state.nodes);
        b.minY = Math.min(b.minY, timelineRange.yearRowY - 30);
        b.maxY = Math.max(b.maxY, -timelineRange.stackTop + 40); // symmetric stack + label clearance below
        b.minX -= 60; b.maxX += 60;
        pad = 40;
      } else {
        // Job-graph round: fit ALL nodes, not the largest component. The
        // component trick predates the conversion; on the job graph the
        // work splits into a few real clusters (the June legacy pages, the
        // August job wave), and fitting one of them cropped live jobs
        // clean off the canvas edge (measured: ckpt8_eval rendered at
        // x=926 in a 918px-wide viewport, unclickable). An overview that
        // silently hides nodes is worse than a slightly wider fit.
        b = boundsOf(state.nodes);
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

    // Directed-graph arrowheads (round-3, owner: "we want directed graph,
    // instead of undirected graph. show arrows instead" -- EVERY edge, of
    // every type, in both modes). Same constant-screen-size convention as
    // node markers and labels: an outer group does the WORLD-space
    // translate+rotate (so the arrow pans/zooms with its edge and points
    // the right way regardless of zoom, since uniform scale never changes
    // an angle), an inner group does the pure counter-scale. The tip sits
    // pulled back from the target's CENTER by that node's own screen
    // radius plus a small gap, so it touches the circle's boundary instead
    // of overlapping it -- radius is already a screen-constant number
    // (radiusFor()), so this offset can be applied directly in the same
    // counter-scaled local space with no zoom-dependent conversion needed.
    var ARROW_LEN = 9, ARROW_WID = 3.2, ARROW_GAP = 2;
    function arrowPathD(radius) {
      var tipX = -(radius + ARROW_GAP), baseX = tipX - ARROW_LEN;
      return "M" + tipX + ",0 L" + baseX + "," + (-ARROW_WID) + " L" + baseX + "," + ARROW_WID + " Z";
    }
    function arrowAngleDeg(pb, cx, cy) { return Math.atan2(pb.y - cy, pb.x - cx) * 180 / Math.PI; }
    function buildArrow(container, pb, cx, cy, radius, isTyped) {
      var outer = el("g", { "class": "gn-arrow", transform: "translate(" + pb.x + "," + pb.y + ") rotate(" + arrowAngleDeg(pb, cx, cy) + ")" });
      var inner = el("g", { transform: "scale(" + (1 / (state.transform.k || 1)) + ")" });
      inner.appendChild(el("path", { "class": "gn-arrowhead" + (isTyped ? " typed" : ""), d: arrowPathD(radius) }));
      outer.appendChild(inner);
      container.appendChild(outer);
      return { outer: outer, inner: inner, radius: radius };
    }
    function updateArrow(rec, pb, cx, cy) {
      rec.outer.setAttribute("transform", "translate(" + pb.x + "," + pb.y + ") rotate(" + arrowAngleDeg(pb, cx, cy) + ")");
    }

    // Edge curvature: both modes use a perpendicular offset from the
    // straight line's midpoint, only the magnitude differs. Timeline's
    // bin/stack layout needs a visibly bigger arc to read as "curving
    // into" a column the way the owner's sketch draws it than map mode's
    // subtle declutter-only nudge does.
    function edgeBow(dx, dy) {
      var dist = Math.hypot(dx, dy);
      return state.mode === "timeline" ? Math.min(70, dist * 0.22) : Math.min(24, dist * 0.08);
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
        var bow = edgeBow(dx, dy);
        var nx = -dy, ny = dx;
        var nl = Math.hypot(nx, ny) || 1;
        var cx = mx + (nx / nl) * bow, cy = my + (ny / nl) * bow;
        var isTyped = e.type && e.type !== "link" && e.type !== "upstream";
        var path = el("path", {
          "class": "gn-edge" + (isTyped ? " typed" : "") + (e.type === "upstream" ? " upstream" : ""),
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
        var arrow = buildArrow(gEdges, pb, cx, cy, radiusFor(b.in_degree || 0, b.kind), isTyped);
        edgeEls.push({ source: e.source, target: e.target, path: path, gradient: grad, a: a, b: b, label: labelG, arrow: arrow });
      });

      // nodes
      gNodes.innerHTML = "";
      nodeEls = {};
      var invK = 1 / (state.transform.k || 1);
      var isTimeline = state.mode === "timeline";
      state.nodes.forEach(function (n) {
        var p = posOf(n);
        var isOrphan = !n.has_upstream;
        var g = el("g", {
          "class": "gn-node gn-kind-" + (n.kind || "page") + (isOrphan ? " gn-orphan" : ""),
          transform: "translate(" + p.x + "," + p.y + ")"
        });
        g.dataset.id = n.id;
        // Full title on hover (round-3: node labels switched from title to
        // a short slug/shortname, so the full title needs a home) --
        // native SVG <title> is the plain-HTML-equivalent tooltip, no
        // custom positioning/z-index code to maintain.
        var titleEl = el("title", {});
        titleEl.textContent = n.title || n.id;
        g.appendChild(titleEl);
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
        var r = radiusFor(n.in_degree || 0, n.kind);
        var selRing = el("circle", { "class": "gn-sel-ring", r: r + 4 });
        var circle = el("circle", { "class": "gn-circle", r: r });
        if (n.kind === "page") {
          // hollow artifact: stroke carries the muted color, fill stays
          // the page background so edges read as passing BEHIND it
          circle.setAttribute("fill", "var(--paper, #f4f1ea)");
          circle.setAttribute("stroke", colorFor(n));
          circle.setAttribute("stroke-width", "1.6");
        } else {
          circle.setAttribute("fill", colorFor(n));
        }
        // A job that produced a page renders as a donut: a small hole in
        // the page style's own fill, so "hollow = page" reads as one
        // system (a page sits inside the job). Owner-asked hint,
        // 2026-08-14. pointer-events stay on the group; the hole is
        // decoration.
        var pageHole = null;
        if (n.kind === "job" && n.page_name) {
          pageHole = el("circle", { "class": "gn-page-hole",
                                     r: Math.max(2.6, r * 0.42) });
        }
        // Label text = the page's short id or registered shortname (round-3,
        // owner: "a unique shortname for the job, like an id, not a full
        // name which is very long"); the full title moved to the <title>
        // tooltip above. Position: to the right of the circle in Map mode
        // (unchanged), CENTERED BELOW it in Timeline mode -- the owner's
        // sketch draws "page short names under the circle" for the
        // vertical-stack layout, which only reads correctly with a
        // centered, below-anchored label.
        var labelText = n.label || n.id;
        // Display the basename only: zone prefixes (workspace/, updates/)
        // are chrome that eats the label budget; the full id lives in the
        // tooltip and the detail card. Then truncate: timeline stacks
        // center labels under circles in adjacent fixed-width bins, so
        // anything past ~18 chars overlaps its neighbor at the reveal zoom
        // (measured); map labels extend rightward and afford more.
        var slash = labelText.lastIndexOf("/");
        if (slash !== -1 && slash < labelText.length - 1) labelText = labelText.slice(slash + 1);
        var maxChars = isTimeline ? 18 : 46;
        if (labelText.length > maxChars) labelText = labelText.slice(0, maxChars - 1) + "…";
        var label = el("text", {
          "class": "gn-label",
          x: isTimeline ? 0 : r + 5,
          y: isTimeline ? r + 16 : 4,
          "text-anchor": isTimeline ? "middle" : "start"
        });
        label.textContent = labelText;
        inner.appendChild(selRing);
        inner.appendChild(circle);
        if (pageHole) inner.appendChild(pageHole);
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
      // Arrowheads: same counter-scale convention, but their OWN inner
      // group (built per-edge in buildArrow(), not tracked in
      // scaledTextEls since they also need a rotation the text-label path
      // doesn't).
      edgeEls.forEach(function (e) {
        if (e.arrow) e.arrow.inner.setAttribute("transform", "scale(" + invK + ")");
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
        var bow = edgeBow(dx, dy);
        var nx = -dy, ny = dx;
        var nl = Math.hypot(nx, ny) || 1;
        var cx = mx + (nx / nl) * bow, cy = my + (ny / nl) * bow;
        e.path.setAttribute("d", "M" + pa.x + "," + pa.y + " Q" + cx + "," + cy + " " + pb.x + "," + pb.y);
        if (e.arrow) updateArrow(e.arrow, pb, cx, cy);
        if (e.label) {
          // Keep the typed-edge label's world anchor (and the parallel
          // bookkeeping entry updateNodeScale() reads) in sync with the
          // edge it annotates during a live drag, same combined
          // translate+scale convention updateNodeScale() applies everywhere
          // else.
          var entry = scaledTextEls.filter(function (s) { return s.g === e.label; })[0];
          if (entry) { entry.x = cx; entry.y = cy; }
          e.label.setAttribute("transform", "translate(" + cx + "," + cy + ") scale(" + (1 / (state.transform.k || 1)) + ")");
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
          var lit = e.source === focus || e.target === focus;
          e.path.classList.toggle("lit", lit);
          if (e.arrow) e.arrow.outer.classList.toggle("lit", lit);
          // Typed-edge labels are hidden by default now (round-3, same
          // collision reason node labels are) -- revealing one when ITS
          // OWN edge lights up needs a direct class on the label itself,
          // not a CSS sibling selector: `~` matches every later sibling
          // matching the selector, not just the one paired label, so it
          // would reveal OTHER edges' labels too in a shared <g> (found
          // live trying exactly that before switching to this).
          if (e.label) e.label.classList.toggle("lit", lit);
        });
      } else {
        svgEl.classList.remove("dimmed");
        Object.keys(nodeEls).forEach(function (nid) { nodeEls[nid].g.classList.remove("hi"); });
        edgeEls.forEach(function (e) {
          e.path.classList.remove("lit");
          if (e.arrow) e.arrow.outer.classList.remove("lit");
          if (e.label) e.label.classList.remove("lit");
        });
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
      renderDetail();
    }

    function selectNode(id) {
      ui.selectedId = ui.selectedId === id ? null : id;
      applyEmphasis();
      renderDetail();
    }

    // ------------------------------------------------------- detail card
    // Click a node -> its full record in the side rail, straight from
    // graph.json's `detail` payload (the same parsed board fields the jobs
    // tab renders; no second fetch, no second bookkeeping surface). The
    // card REPLACES the no-upstream list while a selection is active --
    // one rail, two exclusive occupants -- and every path that clears the
    // selection (Escape, background click, toggling the same node) also
    // restores the list, so the rail can never end up empty or doubled.
    function fmtDate(s) { return s ? String(s).slice(0, 10) : ""; }

    function centerOn(id) {
      var n = state.nodeById[id];
      if (!n) return;
      var rect = svgEl.getBoundingClientRect();
      var p = posOf(n);
      var k = Math.max(state.transform.k, 0.8);
      state.transform = { k: k, x: rect.width / 2 - p.x * k, y: rect.height / 2 - p.y * k };
      applyTransform();
    }

    function renderDetail() {
      if (!detailEl || !rootsEl) return;
      var id = ui.selectedId;
      if (!id || !state.nodeById[id]) {
        detailEl.hidden = true;
        detailEl.innerHTML = "";
        rootsEl.hidden = false;
        return;
      }
      var n = state.nodeById[id];
      var d = n.detail || {};
      detailEl.innerHTML = "";
      function div(cls, text) {
        var e = document.createElement("div");
        e.className = cls;
        if (text) e.textContent = text;
        detailEl.appendChild(e);
        return e;
      }
      var kick = div("gd-kicker");
      var slug = document.createElement("span");
      slug.className = "gd-slug";
      slug.textContent = n.label || n.id;
      kick.appendChild(slug);
      function badge(text, extraCls) {
        var b = document.createElement("span");
        b.className = "gd-badge" + (extraCls ? " " + extraCls : "");
        b.textContent = text;
        kick.appendChild(b);
      }
      if (n.kind === "job") {
        if (n.status) badge(n.status, "gd-status-" + n.status);
        if (n.track) badge(n.track);
      } else {
        badge((TIER_LABEL[n.tier] || n.tier || "legacy"));
      }
      // page-or-not, up front (owner-asked 2026-08-14): a linked "page"
      // chip when there is one (hover names it), a muted "no page" chip
      // otherwise -- the answer is visible before any scrolling.
      if (n.kind === "job" && n.page_name) {
        var pa = document.createElement("a");
        pa.className = "gd-badge gd-page-chip";
        pa.href = n.url; pa.target = "_blank"; pa.rel = "noopener";
        pa.textContent = "page ↗";
        pa.title = n.page_name;
        kick.appendChild(pa);
      } else if (n.kind === "job") {
        badge("no page", "gd-nopage");
      } else {
        var pa2 = document.createElement("a");
        pa2.className = "gd-badge gd-page-chip";
        pa2.href = n.url; pa2.target = "_blank"; pa2.rel = "noopener";
        pa2.textContent = "page ↗";
        pa2.title = n.id;
        kick.appendChild(pa2);
      }
      var h = document.createElement("h3");
      h.className = "gd-title";
      h.textContent = n.title || n.id;
      detailEl.appendChild(h);
      div("gd-dates", n.kind === "job"
        ? ("started " + fmtDate(n.created) + (n.modified ? " · updated " + fmtDate(n.modified) : ""))
        : ("published " + fmtDate(n.created)));
      if (n.kind === "job") {
        if (d.motivation) { div("gd-label", "Motivation"); div("gd-text", d.motivation); }
        if (d.outcome) {
          div("gd-label", "Outcome");
          div("gd-text gd-outcome", d.outcome);
        }
      } else if (d.blurb) {
        // curated pages.yaml blurbs are authored HTML from this repo
        div("gd-text gd-blurb").innerHTML = d.blurb;
      }
      var ups = [], downs = [];
      state.edges.forEach(function (e) {
        if (e.type !== "upstream") return;
        if (e.target === id) ups.push(e.source);
        if (e.source === id) downs.push(e.target);
      });
      function chips(labelText, ids) {
        if (!ids.length) return;
        div("gd-label", labelText);
        var wrap = div("gd-chips");
        ids.forEach(function (cid) {
          var c = document.createElement("button");
          c.type = "button";
          c.className = "gd-chip";
          var cn = state.nodeById[cid];
          c.textContent = (cn && cn.label) || cid;
          c.title = (cn && cn.title) || cid;
          c.addEventListener("click", function () {
            ui.selectedId = cid;  // direct set, not toggle: a chip always navigates TO its node
            applyEmphasis();
            renderDetail();
            centerOn(cid);
          });
          wrap.appendChild(c);
        });
      }
      chips("Motivated by", ups);
      chips("Motivated", downs);
      if (n.kind === "job" && n.board_url) {
        var links = div("gd-links");
        var ba = document.createElement("a");
        ba.href = n.board_url; ba.target = "_blank"; ba.rel = "noopener";
        ba.textContent = "Board entry";
        links.appendChild(ba);
      }
      // Latest log LAST (owner-asked 2026-08-14): identity and substance
      // first, the raw trail at the bottom -- and always shown now, not
      // only for outcome-less jobs.
      if (n.kind === "job" && d.log_tail && d.log_tail.length) {
        div("gd-label", "Latest log");
        d.log_tail.forEach(function (l) { div("gd-log", l); });
      }
      detailEl.hidden = false;
      rootsEl.hidden = true;
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
        var hay = (n.title + " " + n.id + " " + (n.label || "")).toLowerCase();
        if (hay.indexOf(q) === -1) return;
        var score = n.id.toLowerCase() === q || (n.label || "").toLowerCase() === q ? 3
          : (n.title.toLowerCase().indexOf(q) === 0 ? 2 : 1);
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

    // ------------------------------------------------------- completeness
    // Round-3: replaces the old zero-DEGREE "orphan" list with a
    // zero-UPSTREAM list (has_upstream, computed server-side from
    // web/pages.yaml's `upstreams:` field) -- a page can have plenty of
    // content-link/typed edges and still be missing the one thing this
    // panel now tracks: a stated motivation. Per the brief, this list
    // "should only be genuine roots"; anything else here is a registered
    // page still missing its upstreams: entry.
    function renderOrphans() {
      var orphans = state.nodes.filter(function (n) { return !n.has_upstream; });
      orphanList.innerHTML = "";
      if (!orphans.length) {
        var li = document.createElement("li");
        li.className = "go-empty";
        li.textContent = "None: every page states an upstream.";
        orphanList.appendChild(li);
        return;
      }
      // jobs first (their gap is actionable on the board today), then
      // legacy pages; alphabetical within each group
      orphans.sort(function (a, b) {
        var ka = a.kind === "job" ? 0 : 1, kb = b.kind === "job" ? 0 : 1;
        if (ka !== kb) return ka - kb;
        return (a.label || a.id).localeCompare(b.label || b.id);
      });
      orphans.forEach(function (n) {
        var li = document.createElement("li");
        var a = document.createElement("a");
        a.href = "#"; a.className = "go-node";
        a.addEventListener("click", function (ev) {
          ev.preventDefault();
          ui.selectedId = n.id;
          applyEmphasis();
          renderDetail();
          centerOn(n.id);
        });
        a.textContent = n.label || n.id;
        var chip = document.createElement("span");
        chip.className = "go-tier";
        chip.textContent = n.kind === "job" ? (n.track || "job") : "page";
        a.appendChild(chip);
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
      var tl = computeTimelineBins();
      state.timelinePos = tl.pos;
      var hdr = computeHeaderGroups(tl.bins);
      timelineRange = { bins: tl.bins, months: hdr.months, years: hdr.years, stackTop: hdr.stackTop,
                         dayRowY: hdr.dayRowY, monthRowY: hdr.monthRowY, yearRowY: hdr.yearRowY };
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
        // Log to the console (found live, round-3: this used to swallow a
        // RENDER-time exception silently, showing the identical "could not
        // load" message a genuine fetch failure would, which sent a real
        // JS bug on a bad trail as a network problem) -- the message below
        // is only the network-level fallback UI, not the whole story.
        console.error("graph.html: load/render failed", err);
        if (first) orphanList.innerHTML = '<li class="go-empty">Could not load or render graph.json (see console).</li>';
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
/* job-graph v2 (2026-08-14) */
