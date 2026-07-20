/* lightgen shared page interaction layer — design-system source of truth.
 * Extracted 2026-07-06 from build_finetune_page.py's inline PAGE_SCRIPT.
 * Provides: per-section expand/collapse, expand-all/collapse-all, outline
 * click-to-navigate with auto-expand, anchor-jump auto-expand (incl. on first
 * load and on hashchange), and scrollspy highlighting.
 *
 * Guarded throughout: every querySelector(All) call degrades to a no-op if the
 * page has no .outline / .collapsible sections, so plain pages (no outline, no
 * preview/expand) can include this script harmlessly with zero markup changes.
 *
 * Published (merge-copy) to PUBLISH_DEST/assets/ui.js; pages include it via
 * <script src="../assets/ui.js"></script>, no build-time templating needed.
 */
(function() {
  function setExpanded(section, expanded) {
    var btn = section.querySelector('.expand-btn');
    if (expanded) {
      section.classList.add('expanded');
      if (btn) btn.textContent = 'Collapse ▴';
    } else {
      section.classList.remove('expanded');
      if (btn) btn.textContent = 'Expand section ▾';
    }
  }

  // wire each section's own expand/collapse button
  document.querySelectorAll('.collapsible').forEach(function(section) {
    var btn = section.querySelector('.expand-btn');
    if (!btn) return;
    btn.addEventListener('click', function() {
      setExpanded(section, !section.classList.contains('expanded'));
    });
  });

  // expand all / collapse all
  var expandAllBtn = document.getElementById('expand-all-btn');
  if (expandAllBtn) {
    expandAllBtn.addEventListener('click', function() {
      var sections = document.querySelectorAll('.collapsible');
      var anyCollapsed = Array.prototype.some.call(sections, function(s) {
        return !s.classList.contains('expanded');
      });
      sections.forEach(function(s) { setExpanded(s, anyCollapsed); });
      expandAllBtn.textContent = anyCollapsed ? 'Collapse all' : 'Expand all';
    });
  }

  // outline link clicks: expand the target section first (if collapsed), then let the
  // browser's normal anchor-jump handle scrolling
  document.querySelectorAll('.ol-link, .ol-sublink').forEach(function(link) {
    link.addEventListener('click', function() {
      var targetId = link.getAttribute('data-target');
      if (!targetId) return;
      var section = document.getElementById(targetId);
      if (section && section.classList.contains('collapsible') && !section.classList.contains('expanded')) {
        setExpanded(section, true);
      }
    });
  });

  // direct/deep-link load with a hash (e.g. a cross-page anchor link into a
  // collapsed section): expand the containing section before the browser tries
  // to scroll to it
  function expandForHash(hash) {
    if (!hash) return;
    var el;
    try { el = document.querySelector(hash); } catch (e) { return; }
    if (!el) return;
    var section = el.closest ? el.closest('.collapsible') : null;
    if (section && !section.classList.contains('expanded')) {
      setExpanded(section, true);
    }
  }
  expandForHash(window.location.hash);
  if (window.location.hash) {
    // re-scroll after expansion changes layout height
    requestAnimationFrame(function() {
      requestAnimationFrame(function() {
        var el = document.querySelector(window.location.hash);
        if (el) el.scrollIntoView();
      });
    });
  }
  window.addEventListener('hashchange', function() {
    expandForHash(window.location.hash);
    requestAnimationFrame(function() {
      var el = document.querySelector(window.location.hash);
      if (el) el.scrollIntoView();
    });
  });

  // scrollspy: highlight the outline entry for whichever spy target is nearest the top.
  // Pages add data-spy to any element they want trackable (top-level sections and/or
  // notable subsections), and data-spy-link="<that element's id>" on the matching
  // outline <a>. No-ops entirely if a page has no [data-spy] elements.
  var spyTargets = Array.prototype.slice.call(document.querySelectorAll('[data-spy]'));
  var spyLinks = Array.prototype.slice.call(document.querySelectorAll('[data-spy-link]'));
  function updateScrollspy() {
    if (!spyTargets.length) return;
    var current = spyTargets[0];
    for (var i = 0; i < spyTargets.length; i++) {
      if (spyTargets[i].getBoundingClientRect().top - 130 <= 0) current = spyTargets[i];
      else break;
    }
    spyLinks.forEach(function(l) { l.classList.remove('active'); });
    var id = current.id;
    var match = spyLinks.filter(function(l) { return l.getAttribute('data-spy-link') === id; })[0];
    if (match) match.classList.add('active');
  }
  if (spyTargets.length) {
    window.addEventListener('scroll', updateScrollspy, { passive: true });
    window.addEventListener('resize', updateScrollspy);
    updateScrollspy();
  }

  // full-path file mentions (code.fpath, data-path="..."): click copies the full
  // path and flashes "copied" for ~1.1s. Falls back to a hidden-textarea +
  // execCommand('copy') if the async Clipboard API isn't available (e.g. non-secure
  // context) — no-ops (silently) if neither works, since the hover title still
  // shows the path either way.
  document.querySelectorAll('code.fpath').forEach(function(el) {
    var original = el.textContent;
    var flashTimer = null;
    el.addEventListener('click', function() {
      var path = el.getAttribute('data-path') || original;
      function flash() {
        if (flashTimer) clearTimeout(flashTimer);
        el.textContent = 'copied ✓';
        el.classList.add('copied');
        flashTimer = setTimeout(function() {
          el.textContent = original;
          el.classList.remove('copied');
        }, 1100);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(path).then(flash, function() {
          try {
            var ta = document.createElement('textarea');
            ta.value = path; ta.style.position = 'fixed'; ta.style.opacity = '0';
            document.body.appendChild(ta); ta.select();
            document.execCommand('copy'); document.body.removeChild(ta);
            flash();
          } catch (e) { /* clipboard unavailable — hover title still shows the path */ }
        });
      } else {
        try {
          var ta2 = document.createElement('textarea');
          ta2.value = path; ta2.style.position = 'fixed'; ta2.style.opacity = '0';
          document.body.appendChild(ta2); ta2.select();
          document.execCommand('copy'); document.body.removeChild(ta2);
          flash();
        } catch (e) { /* clipboard unavailable — hover title still shows the path */ }
      }
    });
  });

  // client-side filter box (v11, console Pages/Agent notes tabs): an
  // <input data-filter-input="SCOPE"> filters every [data-filter-item="SCOPE"]
  // element by substring match against its data-filter-text (falls back to the
  // item's own textContent). No-op if the page has no matching input. A sibling
  // [data-filter-empty="SCOPE"] element (if present) is shown only when the
  // filter matches nothing, hidden otherwise.
  document.querySelectorAll('[data-filter-input]').forEach(function(input) {
    var scope = input.getAttribute('data-filter-input');
    var items = Array.prototype.slice.call(
      document.querySelectorAll('[data-filter-item="' + scope + '"]'));
    if (!items.length) return;
    var empty = document.querySelector('[data-filter-empty="' + scope + '"]');
    if (empty) empty.style.display = 'none';
    input.addEventListener('input', function() {
      var q = input.value.trim().toLowerCase();
      var shown = 0;
      items.forEach(function(item) {
        var hay = (item.getAttribute('data-filter-text') || item.textContent || '').toLowerCase();
        var match = !q || hay.indexOf(q) !== -1;
        item.style.display = match ? '' : 'none';
        if (match) shown++;
      });
      if (empty) empty.style.display = shown ? 'none' : '';
    });
  });
})();

/* Interactive 3D lightbox: click a .v3d thumbnail -> open its data-glb in the #mv3d
 * model-viewer modal (orbit/zoom). No-ops if the page has no #mv3d modal or no .v3d
 * images, so this is safe to ship in the shared ui.js for every page. The GLB is set
 * on open and cleared on close, so nothing downloads until a thumbnail is clicked and
 * only one WebGL context is live at a time. Requires model_viewer_head() (the
 * self-hosted <model-viewer> element) + model_viewer_modal() on the page (xgpage.py). */
(function() {
  function init() {
    // The #mv3d modal is emitted via extra_body_end, i.e. AFTER this ui.js <script>,
    // so run on DOMContentLoaded (below) to ensure it's in the DOM before we bind.
    var modal = document.getElementById('mv3d');
    if (!modal) return;
    var mv = document.getElementById('mv3d-viewer');
    var titleEl = document.getElementById('mv3d-title');
    var dl = document.getElementById('mv3d-dl');
    function open(glb, ttl) {
      if (!glb) return;
      if (mv) mv.setAttribute('src', glb);
      if (titleEl) titleEl.textContent = ttl || '';
      if (dl) dl.setAttribute('href', glb);
      modal.classList.add('open');
    }
    function close() {
      modal.classList.remove('open');
      if (mv) mv.removeAttribute('src');  // free the GLB / WebGL context
    }
    document.querySelectorAll('.v3d').forEach(function(im) {
      im.addEventListener('click', function() { open(im.dataset.glb, im.dataset.title); });
    });
    modal.addEventListener('click', function(e) { if (e.target === modal) close(); });
    var btn = document.getElementById('mv3d-close');
    if (btn) btn.addEventListener('click', close);
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape' && modal.classList.contains('open')) close();
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
