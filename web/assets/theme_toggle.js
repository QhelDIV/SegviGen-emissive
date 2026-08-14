/* theme_toggle.js -- xgpage day/night toggle runtime (engine-level, 2026-07-23).
   Loaded by page(theme="v2"/"v3", theme_toggle=True) (the default for both
   languages), after theme2.css so the CSS's :root[data-theme="light"|"dark"]
   hook it drives already exists (that hook predates this file -- it was
   added 2026-07-14 for the v2 palette and was, until today, only exercised
   by a hand-rolled per-page button on the 2026-07-22 meeting page). Cycles
   auto -> light -> dark -> auto on click of #xg-theme-btn; "auto" means no
   override (prefers-color-scheme governs, unchanged). Persists the choice
   in localStorage under ONE site-wide key ("xg-theme") so it follows the
   reader across every xgpage v2/v3 page on the origin (console, workspace,
   standalone report) -- see page()'s inline head anti-flash script for the
   matching read-before-paint half of this mechanism (sets data-theme
   before first paint so navigating between pages never flashes the wrong
   theme).
   v1 ("dark report") pages never load this file: theme.css ships a single
   :root palette with no light counterpart, so page() only wires the
   toggle for theme="v2"/"v3" (see xgpage SKILL.md's design-language
   registry, "v1 stays dark-only"). */
(function () {
  var KEY = "xg-theme";
  var ORDER = ["auto", "light", "dark"];
  var GLYPH = { auto: "◐", light: "☼", dark: "☽" };
  var LABEL = { auto: "Auto", light: "Light", dark: "Dark" };

  // ---- pure decision logic, exported for node:test (same module.exports
  // guard pattern as xg3.js: in node, require() gets these functions and
  // returns before any browser code runs; in the browser, module is
  // undefined and the runtime below executes exactly as before) ----------
  var LOGIC = {
    nextMode: function (mode) { return ORDER[(ORDER.indexOf(mode) + 1) % ORDER.length]; },
    glyphFor: function (mode) { return GLYPH[mode]; },
    labelFor: function (mode) { return LABEL[mode]; },
    // localStorage can hold anything (stale value, another site version,
    // manual tampering); any unrecognized value normalizes to auto rather
    // than the toggle silently getting stuck off-cycle.
    normalize: function (saved) {
      return (saved === "light" || saved === "dark") ? saved : "auto";
    }
  };
  if (typeof module !== "undefined" && module.exports) { module.exports = LOGIC; return; }

  var root = document.documentElement;
  function apply(mode) {
    if (mode === "auto") { root.removeAttribute("data-theme"); }
    else { root.setAttribute("data-theme", mode); }
  }
  function paint(btn, mode) {
    btn.textContent = LOGIC.glyphFor(mode);
    btn.setAttribute("data-mode", mode);
    btn.setAttribute("aria-label", "Theme: " + LOGIC.labelFor(mode) + " (click to change)");
  }
  var btn = document.getElementById("xg-theme-btn");
  if (btn) {
    var stored = null;
    try { stored = localStorage.getItem(KEY); } catch (e) { /* private mode etc. */ }
    var mode = LOGIC.normalize(stored);
    // the head anti-flash script already applied data-theme for light/dark;
    // this just brings the button glyph in sync with that state.
    paint(btn, mode);
    btn.addEventListener("click", function () {
      mode = LOGIC.nextMode(mode);
      apply(mode);
      paint(btn, mode);
      try { localStorage.setItem(KEY, mode); } catch (e) {}
    });
  }
})();
