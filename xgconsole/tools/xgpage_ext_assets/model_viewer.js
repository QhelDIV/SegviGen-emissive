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
