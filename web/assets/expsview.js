/* expsview.js -- the interactive experiments-view component (run overlay,
   EMA smoothing, x-axis switch, log/linear scale + zoom/pan + tooltips,
   media scrubber). See xgpage.expsview's module docstring for the manifest
   contract this reads and the charting-stack rationale (Vega-Lite, already
   vendored for this project's own prior static exps chart).

   Same module.exports guard pattern as xg3.js / xg_codepop.js: the IIFE
   below exports the PURE logic (no DOM, no fetch) first and returns early
   under node:test, so tests/js/expsview_logic.test.js exercises the exact
   bytes this page loads, then the browser wiring runs unchanged in a real
   page. Self-initializing: a page includes this script once, tagged with
   `data-expsview-src` (the manifest URL) and `data-expsview-container`
   (the mount element's id) via xgpage.expsview.expsview_scripts() -- no
   per-page glue JS required. */
(function () {
  "use strict";

  // ---- fixed 8-slot categorical palette (dataviz skill's validated default
  // instance; light/dark steps of the same hues, matching the project's own
  // prior fir/vulcan/solar cluster palette in tools/build_exps.py for slots
  // 1/2/3). Colors are assigned by a run's POSITION IN THE MANIFEST, never
  // by its rank among currently-checked runs, so a run's color is stable
  // regardless of what else is selected. Past 8 runs, no ninth hue is
  // invented -- FALLBACK is used for every additional run (a known cap, not
  // a bug; see the module docstring).
  var PALETTE = [
    { light: "#2a78d6", dark: "#3987e5" }, // blue
    { light: "#eb6834", dark: "#d95926" }, // orange
    { light: "#1baf7a", dark: "#199e70" }, // aqua
    { light: "#eda100", dark: "#c98500" }, // yellow
    { light: "#e87ba4", dark: "#d55181" }, // magenta
    { light: "#008300", dark: "#008300" }, // green
    { light: "#4a3aa7", dark: "#9085e9" }, // violet
    { light: "#e34948", dark: "#e66767" }  // red
  ];
  var FALLBACK = { light: "#8A8578", dark: "#847F70" }; // ink-3, any slot past 8

  var ALLOWED_AXES = ["epoch", "step", "wall_clock", "samples"];

  // ---- pure logic, exported for node:test (no DOM, no fetch) ---------------
  var LOGIC = {
    PALETTE: PALETTE,
    FALLBACK: FALLBACK,
    ALLOWED_AXES: ALLOWED_AXES,

    // Debiased exponential moving average, the convention wandb's own
    // smoothing slider uses (and TensorBoard before it): a plain EMA pulls
    // the first points toward 0 (last starts at 0), so the running value is
    // divided by (1 - weight^n) to remove that warm-start bias. weight=0 is
    // the identity transform (returns `values` unchanged, by construction:
    // debias_weight = 1 - 0^1 = 1, last = 0*0 + 1*x = x), which is exactly
    // the "0 = raw" contract the smoothing slider promises. `values` is a
    // plain array of numbers in their natural (epoch) order -- smoothing
    // always runs in that order regardless of which x-axis is displayed,
    // since epoch order is training order no matter which axis a point's
    // moment in training is labeled with.
    ema: function (values, weight) {
      if (weight <= 0) return values.slice();
      var out = new Array(values.length);
      var last = 0, denom;
      for (var i = 0; i < values.length; i++) {
        last = last * weight + (1 - weight) * values[i];
        denom = 1 - Math.pow(weight, i + 1);
        out[i] = last / denom;
      }
      return out;
    },

    // Which axis modes exist ANYWHERE in the manifest, in ALLOWED_AXES's
    // fixed order -- drives which axis-mode buttons the toolbar shows at
    // all (a mode nobody has, e.g. "samples" today, gets no button rather
    // than a button that always does nothing).
    axesUnion: function (runs) {
      var have = {};
      runs.forEach(function (r) { (r.axes || []).forEach(function (a) { have[a] = true; }); });
      return ALLOWED_AXES.filter(function (a) { return have[a]; });
    },

    // Per-run gate: does THIS run carry real data for `axis`? Reads the
    // run's own declared `axes` list (build-time validated against its
    // series in xgpage.expsview.validate_manifest, so this trusts it
    // without re-scanning every point at render time).
    runHasAxis: function (run, axis) {
      return !!(run.axes && run.axes.indexOf(axis) !== -1);
    },

    // Color for the run at manifest-order index `i` (see PALETTE's
    // docstring above): {light, dark} for i < 8, FALLBACK beyond.
    colorForIndex: function (i) {
      return i < PALETTE.length ? PALETTE[i] : FALLBACK;
    },

    // Minimal structural validation of a fetched manifest -- the build-time
    // Python validator (xgpage.expsview.validate_manifest) is the real
    // gate; this is the runtime half, defending the page against a stale or
    // hand-edited exps.json rather than trusting fetch() blindly. Throws a
    // plain Error with a message naming what's wrong; returns the parsed
    // {metric, axisLabels, runs} unchanged (no reshaping) on success.
    parseManifest: function (raw) {
      if (!raw || typeof raw !== "object") throw new Error("expsview: manifest is not an object");
      if (typeof raw.metric !== "string") throw new Error("expsview: manifest missing 'metric'");
      if (!Array.isArray(raw.runs)) throw new Error("expsview: manifest 'runs' is not a list");
      raw.runs.forEach(function (r) {
        if (!r.slug) throw new Error("expsview: a run is missing 'slug'");
        if (!Array.isArray(r.series)) throw new Error("expsview: run " + r.slug + " 'series' is not a list");
        if (!Array.isArray(r.axes)) throw new Error("expsview: run " + r.slug + " 'axes' is not a list");
      });
      return { metric: raw.metric, axisLabels: raw.axis_labels || {}, runs: raw.runs };
    }
  };

  if (typeof module !== "undefined" && module.exports) { module.exports = LOGIC; return; }

  // ---- browser wiring --------------------------------------------------
  function isDark() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t === "dark") return true;
    if (t === "light") return false;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function cssVar(name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  function init(root, manifest) {
    var metric = manifest.metric;
    var axisLabels = manifest.axisLabels;
    var runs = manifest.runs;
    var axisModes = LOGIC.axesUnion(runs);
    var runColor = {}; // slug -> {light, dark}
    runs.forEach(function (r, i) { runColor[r.slug] = LOGIC.colorForIndex(i); });

    var state = {
      selected: {}, // slug -> bool
      xAxis: axisModes.indexOf("epoch") !== -1 ? "epoch" : (axisModes[0] || "epoch"),
      yScale: "linear",
      smooth: 0,
      mediaSlug: null
    };
    runs.forEach(function (r) { state.selected[r.slug] = r.series.length > 0; });

    var el = {
      runs: root.querySelector('[data-ev="runs"]'),
      smooth: root.querySelector('[data-ev="smooth"]'),
      smoothVal: root.querySelector('[data-ev="smooth-val"]'),
      axisModes: root.querySelector('[data-ev="axis-modes"]'),
      scaleBtns: root.querySelectorAll('[data-ev="scale"]'),
      resetZoom: root.querySelector('[data-ev="reset-zoom"]'),
      chart: root.querySelector('[data-ev="chart"]'),
      axisWarning: root.querySelector('[data-ev="axis-warning"]'),
      mediaPanel: root.querySelector('[data-ev="media-panel"]'),
      mediaRun: root.querySelector('[data-ev="media-run"]'),
      mediaCaption: root.querySelector('[data-ev="media-caption"]'),
      mediaImg: root.querySelector('[data-ev="media-img"]'),
      mediaSlider: root.querySelector('[data-ev="media-slider"]'),
      mediaLabel: root.querySelector('[data-ev="media-label"]')
    };

    // ---- run checkboxes (a colored swatch matches the chart's own color
    // scale so the toolbar doubles as the legend; a legend box would only
    // repeat it).
    runs.forEach(function (r, i) {
      var id = "ev-run-" + r.slug;
      var wrap = document.createElement("label");
      wrap.className = "ev-run";
      wrap.htmlFor = id;
      var swatch = document.createElement("span");
      swatch.className = "ev-swatch";
      swatch.style.background = isDark() ? runColor[r.slug].dark : runColor[r.slug].light;
      var cb = document.createElement("input");
      cb.type = "checkbox"; cb.id = id; cb.checked = !!state.selected[r.slug];
      cb.disabled = r.series.length === 0;
      cb.addEventListener("change", function () { state.selected[r.slug] = cb.checked; render(); });
      var txt = document.createElement("span");
      txt.textContent = r.title + (r.series.length === 0 ? " (no data)" : "");
      wrap.appendChild(cb); wrap.appendChild(swatch); wrap.appendChild(txt);
      el.runs.appendChild(wrap);
    });

    // ---- axis-mode buttons (only modes ANY run in the manifest carries).
    axisModes.forEach(function (mode) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = axisLabels[mode] || mode;
      btn.dataset.axis = mode;
      if (mode === state.xAxis) btn.classList.add("active");
      btn.addEventListener("click", function () {
        state.xAxis = mode;
        el.axisModes.querySelectorAll("button").forEach(function (b) { b.classList.remove("active"); });
        btn.classList.add("active");
        render();
      });
      el.axisModes.appendChild(btn);
    });

    // ---- smoothing slider
    el.smooth.addEventListener("input", function () {
      state.smooth = parseFloat(el.smooth.value);
      el.smoothVal.textContent = state.smooth.toFixed(2);
      render();
    });

    // ---- y-scale toggle
    el.scaleBtns.forEach(function (btn) {
      btn.addEventListener("click", function () {
        state.yScale = btn.dataset.scale;
        el.scaleBtns.forEach(function (b) { b.classList.remove("active"); });
        btn.classList.add("active");
        render();
      });
    });

    // ---- reset zoom: a fresh vegaEmbed call is a fresh view (zoom/pan
    // state lives in that view's own signals), simpler and just as cheap as
    // reaching into vega-lite's auto-generated bind:scales signal names at
    // this dataset size.
    el.resetZoom.addEventListener("click", render);

    // ---- theme reactivity: the existing theme toggle (theme_toggle.js)
    // sets data-theme on <html> but dispatches no event, so this observes
    // the attribute directly and re-renders with the new light/dark palette
    // -- without this, a live toggle click would leave the chart on its
    // initial-load colors until the next full page load.
    new MutationObserver(function () { render(); })
      .observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

    // ---- media panel
    var mediaRuns = runs.filter(function (r) { return r.media && r.media.entries && r.media.entries.length; });
    if (mediaRuns.length) {
      el.mediaPanel.hidden = false;
      mediaRuns.forEach(function (r) {
        var opt = document.createElement("option");
        opt.value = r.slug; opt.textContent = r.title;
        el.mediaRun.appendChild(opt);
      });
      state.mediaSlug = mediaRuns[0].slug;
      el.mediaRun.value = state.mediaSlug;
      el.mediaRun.addEventListener("change", function () {
        state.mediaSlug = el.mediaRun.value;
        renderMedia();
      });
      el.mediaSlider.addEventListener("input", function () { paintMedia(); });
      renderMedia();
    }

    function renderMedia() {
      var run = runs.filter(function (r) { return r.slug === state.mediaSlug; })[0];
      if (!run || !run.media) return;
      var m = run.media;
      el.mediaSlider.max = String(m.entries.length - 1);
      el.mediaSlider.value = String(m.entries.length - 1); // land on the latest round by default
      el.mediaCaption.textContent = m.source === "wandb"
        ? "logged directly to wandb during training. "
        : "from this run's local visualizer log. ";
      el.mediaCaption.textContent += m.rounds_selected + " of " + m.total_rounds_available +
        " rounds shown, cap rule: " + m.cap_rule + ".";
      paintMedia();
    }
    function paintMedia() {
      var run = runs.filter(function (r) { return r.slug === state.mediaSlug; })[0];
      if (!run || !run.media) return;
      var idx = parseInt(el.mediaSlider.value, 10);
      var entry = run.media.entries[idx];
      if (!entry) return;
      el.mediaImg.src = entry.url;
      el.mediaImg.alt = entry.kind === "ground_truth" ? "ground truth" : "epoch " + entry.epoch;
      el.mediaLabel.textContent = entry.kind === "ground_truth" ? "ground truth" : "epoch " + entry.epoch;
    }

    // ---- chart
    function buildSpec() {
      var dark = isDark();
      var ink = cssVar("--ink", dark ? "#EEEBE1" : "#21201C");
      var muted = cssVar("--muted", dark ? "#B3AD9D" : "#5D5A50");
      var line = cssVar("--line", dark ? "#3B382F" : "#E2DECF");

      var domain = [], range = [];
      runs.forEach(function (r, i) {
        domain.push(r.slug);
        var c = LOGIC.colorForIndex(i);
        range.push(dark ? c.dark : c.light);
      });

      var rows = [];
      var skipped = [];
      runs.forEach(function (r) {
        if (!state.selected[r.slug]) return;
        if (!LOGIC.runHasAxis(r, state.xAxis)) { if (r.series.length) skipped.push(r.title); return; }
        var pts = r.series.filter(function (p) { return p[state.xAxis] != null && p[metric] != null; })
          .slice().sort(function (a, b) { return a.epoch - b.epoch; });
        if (state.yScale === "log") pts = pts.filter(function (p) { return p[metric] > 0; });
        if (!pts.length) return;
        var raw = pts.map(function (p) { return p[metric]; });
        var smoothed = LOGIC.ema(raw, state.smooth);
        pts.forEach(function (p, i) {
          rows.push({ x: p[state.xAxis], y: smoothed[i], cluster: p.cluster, run: r.slug, runTitle: r.title, variant: "smoothed" });
          if (state.smooth > 0) {
            rows.push({ x: p[state.xAxis], y: raw[i], cluster: p.cluster, run: r.slug, runTitle: r.title, variant: "raw" });
          }
        });
      });

      el.axisWarning.hidden = skipped.length === 0;
      if (skipped.length) {
        el.axisWarning.textContent = "hidden on this axis (no " + (axisLabels[state.xAxis] || state.xAxis) +
          " data): " + skipped.join(", ") + ".";
      }

      var xTitle = axisLabels[state.xAxis] || state.xAxis;
      var axisEnc = { labelColor: muted, titleColor: ink, gridColor: line, domainColor: line };
      var colorEnc = {
        field: "run", type: "nominal", title: "run",
        scale: { domain: domain, range: range },
        legend: null // the toolbar's own swatches are the legend
      };
      var tooltip = [
        { field: "runTitle", type: "nominal", title: "run" },
        { field: "x", type: "quantitative", title: xTitle },
        { field: "y", type: "quantitative", title: metric, format: ".4f" },
        { field: "cluster", type: "nominal" }
      ];

      var layers = [];
      layers.push({
        transform: [{ filter: "datum.variant === 'raw'" }],
        mark: { type: "line", interpolate: "monotone", strokeWidth: 1.4, opacity: 0.35 },
        encoding: {
          x: { field: "x", type: "quantitative", title: xTitle, axis: axisEnc },
          y: { field: "y", type: "quantitative", title: metric, scale: { type: state.yScale }, axis: axisEnc },
          color: colorEnc, detail: { field: "run", type: "nominal" }
        }
      });
      layers.push({
        // The pan/zoom param lives on THIS layer only, not at the top level
        // of the layered spec: a `bind: scales` interval param declared at
        // the top level of a two-layer spec (found live via the worked
        // example's Playwright pass -- both layers here share identical x/y
        // encoding, which is exactly the case that trips it) compiles to a
        // genuinely duplicated "<name>_x" signal in vega-lite v5's own
        // output and throws at parse time; scoping the param to one layer
        // still binds the WHOLE view's resolved (shared) scale, since
        // layered specs share x/y scales by default, so behavior is
        // unaffected -- only the compiled-signal collision goes away.
        params: [{ name: "ev_zoom", select: "interval", bind: "scales" }],
        transform: [{ filter: "datum.variant === 'smoothed'" }],
        mark: { type: "line", interpolate: "monotone", strokeWidth: 2, point: { filled: true, size: 32 } },
        encoding: {
          x: { field: "x", type: "quantitative", title: xTitle, axis: axisEnc },
          y: { field: "y", type: "quantitative", title: metric, scale: { type: state.yScale }, axis: axisEnc },
          color: colorEnc, detail: { field: "run", type: "nominal" }, tooltip: tooltip
        }
      });

      return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        data: { values: rows },
        width: "container", height: 320, background: "transparent",
        layer: layers
      };
    }

    // A dragged slider (smoothing) or a rapid succession of clicks fires
    // several 'input'/'change' events before the previous vegaEmbed() promise
    // resolves; two overlapping embeds into the SAME container previously
    // crashed with "Duplicate signal name: ev_zoom_x" (found live via the
    // worked example's Playwright pass, not by inspection) because the
    // in-flight view's auto-generated bind:scales signal was never torn
    // down before the next parse. Fixed two ways: (1) rAF-coalesce bursts of
    // render requests into one, and (2) finalize() the previous view and
    // clear the container synchronously before starting the next embed, and
    // drop a completed embed's result if a newer render started meanwhile
    // (the renderToken check) rather than letting a stale promise resolution
    // clobber a fresher one.
    var currentView = null, renderToken = 0, renderRaf = null;

    function doRender() {
      if (!window.vegaEmbed) return;
      var token = ++renderToken;
      if (currentView) { try { currentView.finalize(); } catch (e) { /* already gone */ } currentView = null; }
      el.chart.innerHTML = "";
      window.vegaEmbed(el.chart, buildSpec(), { actions: false, renderer: "svg" }).then(function (result) {
        if (token !== renderToken) { try { result.view.finalize(); } catch (e) { /* stale */ } return; }
        currentView = result.view;
      }).catch(function (e) {
        console.error("expsview chart render failed:", e); // eslint-disable-line no-console
      });
      runs.forEach(function (r) {
        var sw = root.querySelector('label[for="ev-run-' + r.slug + '"] .ev-swatch');
        if (sw) sw.style.background = isDark() ? runColor[r.slug].dark : runColor[r.slug].light;
      });
    }

    function render() {
      if (renderRaf) cancelAnimationFrame(renderRaf);
      renderRaf = requestAnimationFrame(function () { renderRaf = null; doRender(); });
    }

    render();
  }

  document.addEventListener("DOMContentLoaded", function () {
    var scripts = document.querySelectorAll("script[data-expsview-src]");
    scripts.forEach(function (script) {
      var src = script.getAttribute("data-expsview-src");
      var containerId = script.getAttribute("data-expsview-container");
      var root = document.getElementById(containerId);
      if (!root || !src) return;
      fetch(src).then(function (r) { return r.json(); }).then(function (raw) {
        init(root, LOGIC.parseManifest(raw));
      }).catch(function (e) {
        root.innerHTML = '<p class="sub">Could not load the experiments manifest.</p>';
        console.error("expsview:", e); // eslint-disable-line no-console
      });
    });
  });
})();
