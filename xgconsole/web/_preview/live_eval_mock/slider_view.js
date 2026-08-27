/* slider_view.js -- the live_eval redesign mockup's checkpoint scrubber.
 *
 * Progressive enhancement over a server-rendered wall: build_mock.py bakes
 * the LATEST checkpoint's images and IoU captions directly into the grid, so
 * a reader with no JS sees a complete, correct wall. This script only swaps
 * <img src> / caption text on the cells that change across checkpoints
 * (per-draw thumbnails, their IoU captions, and the row-header IoU); it
 * never rebuilds the grid.
 *
 * Data contract: ckpts.json (same directory), read via
 * [data-sw-root][data-src]. Shape:
 *   { "run", "draws", "img_dir_tmpl" ("img/step%07d"),
 *     "shapes": [{sid, geom, gt, gt_frac}, ...]   (row order == display order)
 *     "checkpoints": [{step, epoch, iou, n_scored,
 *                      per_shape: {sid: {iou_mean, iou_per_draw:[...]}}}],
 *     "default_idx" }
 *
 * State: the current checkpoint index lives only in the DOM (the range
 * input's value). Nothing is persisted (no localStorage, no URL hash) --
 * a refresh always re-renders the page at default_idx (the newest
 * checkpoint), which is also what the no-JS reader sees. That is the
 * "refresh lands on latest" invariant the simulated-user QA checks.
 */
(function () {
  "use strict";

  function dirFor(tmpl, step) {
    // tmpl is "img/step%07d" -- the only printf-style piece we need.
    var digits = String(step);
    while (digits.length < 7) digits = "0" + digits;
    return tmpl.replace("%07d", digits);
  }

  function imgSrc(dir, sid, draw) {
    return dir + "/" + sid + "_d" + draw + ".png";
  }

  function initOne(root) {
    var src = root.getAttribute("data-src");
    if (!src) return;
    fetch(src)
      .then(function (r) { return r.json(); })
      .then(function (data) { wire(root, data); })
      .catch(function (err) {
        // Data failed to load: leave the server-rendered latest-checkpoint
        // wall exactly as it is (that IS the no-JS fallback) and disable the
        // now-nonfunctional controls rather than leaving them looking live.
        var range = root.querySelector("[data-sw-range]");
        var prev = root.querySelector("[data-sw-prev]");
        var next = root.querySelector("[data-sw-next]");
        [range, prev, next].forEach(function (el) { if (el) el.disabled = true; });
        if (window.console) console.warn("slider_view: could not load " + src, err);
      });
  }

  function wire(root, data) {
    var grid = document.querySelector("[data-sw-grid]");
    var range = root.querySelector("[data-sw-range]");
    var prevBtn = root.querySelector("[data-sw-prev]");
    var nextBtn = root.querySelector("[data-sw-next]");
    var label = root.querySelector("[data-sw-label]");
    if (!grid || !range) return;

    var ckpts = data.checkpoints;
    var n = ckpts.length;
    var cache = {}; // dir -> true once its images have been asked for once

    function preload(idx) {
      if (idx < 0 || idx >= n) return;
      var dir = dirFor(data.img_dir_tmpl, ckpts[idx].step);
      if (cache[dir]) return;
      cache[dir] = true;
      data.shapes.forEach(function (sh) {
        for (var k = 0; k < data.draws; k++) {
          var im = new Image();
          im.src = imgSrc(dir, sh.sid, k);
        }
      });
    }

    function render(idx) {
      idx = Math.max(0, Math.min(n - 1, idx));
      var ck = ckpts[idx];
      var dir = dirFor(data.img_dir_tmpl, ck.step);

      var cells = grid.querySelectorAll("[data-sw-cell]");
      for (var i = 0; i < cells.length; i++) {
        var cell = cells[i];
        var sid = cell.getAttribute("data-sw-sid");
        var draw = parseInt(cell.getAttribute("data-sw-draw"), 10);
        var img = cell.querySelector("img");
        var cap = cell.querySelector("[data-sw-cap]");
        var per = ck.per_shape[sid];
        var iouVal = per ? per.iou_per_draw[draw] : null;

        img.onerror = (function (cellRef, imgRef) {
          return function () {
            imgRef.style.display = "none";
            if (!cellRef.querySelector(".gf-placeholder")) {
              var ph = document.createElement("div");
              ph.className = "gf-placeholder";
              ph.textContent = "no panel";
              cellRef.insertBefore(ph, cellRef.firstChild);
            }
          };
        })(cell, img);
        var existingPh = cell.querySelector(".gf-placeholder");
        if (existingPh) existingPh.remove();
        img.style.display = "";
        img.src = imgSrc(dir, sid, draw);
        if (cap) cap.textContent = iouVal === null || iouVal === undefined
          ? "not scored" : "IoU " + iouVal.toFixed(3);
      }

      var rowlabels = grid.querySelectorAll("[data-sw-rowlabel]");
      for (var j = 0; j < rowlabels.length; j++) {
        var rl = rowlabels[j];
        var rsid = rl.getAttribute("data-sw-rowlabel");
        var rper = ck.per_shape[rsid];
        rl.textContent = rsid.slice(0, 8) + "  IoU "
          + (rper ? rper.iou_mean.toFixed(3) : "—");
      }

      if (label) {
        label.innerHTML = "step " + ck.step + " &middot; epoch " + ck.epoch
          + " &middot; screen IoU " + ck.iou.toFixed(3);
      }
      range.value = String(idx);
      if (prevBtn) prevBtn.disabled = idx === 0;
      if (nextBtn) nextBtn.disabled = idx === n - 1;
      preload(idx - 1);
      preload(idx + 1);
    }

    range.addEventListener("input", function () {
      render(parseInt(range.value, 10));
    });
    if (prevBtn) prevBtn.addEventListener("click", function () {
      render(parseInt(range.value, 10) - 1);
    });
    if (nextBtn) nextBtn.addEventListener("click", function () {
      render(parseInt(range.value, 10) + 1);
    });
    // Convenience: left/right arrows scrub even when focus isn't on the
    // range input itself, as long as the reader isn't typing somewhere else.
    document.addEventListener("keydown", function (e) {
      var tag = (document.activeElement && document.activeElement.tagName) || "";
      if (tag === "INPUT" && document.activeElement !== range) return;
      if (tag === "TEXTAREA" || document.activeElement.isContentEditable) return;
      if (e.key === "ArrowLeft") { render(parseInt(range.value, 10) - 1); e.preventDefault(); }
      else if (e.key === "ArrowRight") { render(parseInt(range.value, 10) + 1); e.preventDefault(); }
    });

    // Exposed for the QA journey script: force a render without touching the
    // real UI, to exercise the missing-image / placeholder path on demand.
    root._swTestRender = render;

    render(data.default_idx);
  }

  function init() {
    var roots = document.querySelectorAll("[data-sw-root]");
    for (var i = 0; i < roots.length; i++) initOne(roots[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
