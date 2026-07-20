/* xg3.js — runtime for xgpage design-language v3 ("workspace") pages,
   2026-07-16. Loaded ONLY by pages built with xgpage.page(theme="v3"),
   after ui.js and xg2.js (both of whose handlers keep working; xg2.js
   provides tooltips + compare sliders). Provides:
   (1) mobile drawer: the page-tree rail hides behind the top-bar toggle
       below 1200px (button #v3-menu, scrim #v3-scrim, body class
       v3-tree-open);
   (2) outline scrollspy: highlights the .v3-ol-link whose section is
       nearest the top (same nearest-above-threshold rule ui.js uses;
       self-contained so sections need no data-spy markup);
   (3) hypothes.is collision handling: when the annotation sidebar opens
       at the right edge (where the outline rail lives), body gets
       v3-annot-open and theme3.css fades the outline out — annotation
       mode replaces the outline instead of fighting it for pixels.
   All handlers are defensive no-ops on pages missing the elements. */
(function () {
  // ---- pure decision logic (versioning + tree), exported for node:test -------
  // These are the byte-level rules the browser runtime below uses; extracting
  // them lets tests/js exercise them with no DOM (Layer 0). In node (require),
  // module.exports is set and we RETURN before any browser code runs; in the
  // browser module is undefined, so the runtime executes exactly as before.
  var VLOGIC = {
    // version labels are software-style "X.y" STRINGS; compare as strings
    // (String() tolerates the legacy integer manifests). The parseInt("0.1")->0
    // regression is a named unit test against these.
    vsEq: function (a, b) { return String(a) === String(b); },
    // "latest label" = the LAST manifest entry, by ORDER, never a numeric max.
    currentByOrder: function (versions) {
      return (versions && versions.length)
        ? String(versions[versions.length - 1].v) : null; },
    // living-page-canonical: the not-current banner shows on a SNAPSHOT only
    // (data-living absent); the living page is never stale.
    bannerVisible: function (living) { return !living; },
    // highlight a snapshot's own row (never a row on the living page).
    rowIsCurrent: function (living, rowV, current) {
      return !living && String(rowV) === String(current); },
    // tree leaf active: exact page, or a /v/X.y/ snapshot under a stable leaf.
    treeLeafActive: function (leafHref, here) {
      if (!leafHref) return false;
      var h = leafHref.replace(/index\.html$/, "");
      return h === here || here.indexOf(h + "v/") === 0; }
  };
  if (typeof module !== "undefined" && module.exports) { module.exports = VLOGIC; return; }

  // ---- mobile drawer
  var menuBtn = document.getElementById("v3-menu");
  var scrim = document.getElementById("v3-scrim");
  function setTree(open) {
    document.body.classList.toggle("v3-tree-open", open);
    if (menuBtn) menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
  }
  if (menuBtn) {
    menuBtn.addEventListener("click", function () {
      setTree(!document.body.classList.contains("v3-tree-open"));
    });
    if (scrim) scrim.addEventListener("click", function () { setTree(false); });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") setTree(false);
    });
    document.querySelectorAll(".v3-tree .v3-tree-link").forEach(function (a) {
      a.addEventListener("click", function () { setTree(false); });
    });
  }

  // ---- outline scrollspy (nearest [id] target above a 110px threshold)
  var links = Array.prototype.slice.call(
    document.querySelectorAll(".v3-outline .v3-ol-link[href^='#']"));
  var targets = links
    .map(function (l) { return document.getElementById(l.getAttribute("href").slice(1)); })
    .filter(Boolean);
  function updateSpy() {
    if (!targets.length) return;
    var current = targets[0];
    for (var i = 0; i < targets.length; i++) {
      if (targets[i].getBoundingClientRect().top - 110 <= 0) current = targets[i];
      else break;
    }
    links.forEach(function (l) {
      l.classList.toggle("active", l.getAttribute("href") === "#" + current.id);
    });
  }
  if (targets.length) {
    window.addEventListener("scroll", updateSpy, { passive: true });
    window.addEventListener("resize", updateSpy);
    updateSpy();
  }

  // ---- runtime tree nav (workspace rollout, 2026-07-16): a tree carrying
  // data-tree fetches the zone's tree.json at load and re-renders its list,
  // computing the active leaf from location.pathname. The baked list is the
  // no-JS fallback; frozen snapshot pages stay content-immutable while
  // their sidebar tracks the zone's CURRENT tree. Schema mirrors v3_tree():
  // {"entries": [{label, children:[{label, href, meta?}]} | {label, href}]}.
  //
  // ZONE-STICKY chrome (2026-07-17, user-ratified): an operator who reaches
  // a workspace/daily page from the console keeps CONSOLE context — the
  // sidebar renders the console tree instead of the zone's own (the console
  // tree is a superset, so the current leaf still resolves and highlights).
  // Mechanism: a tree-link click while the console tree is shown sets
  // sessionStorage xg_zone=console; workspace pages check the flag before
  // rendering. sessionStorage ONLY — no URL changes of any kind, so
  // hypothes.is annotation identity is untouched. A quiet exit control at
  // the tree top clears the flag and re-renders the zone's own tree; direct
  // visits (no flag; the advisor path) behave exactly as before. Law
  // amendment recorded in the skill registry: baked CONTENT never links the
  // console (unchanged, build-enforced); runtime CHROME may follow operator
  // context, because only a console arrival can set the flag.
  var treeNav = document.querySelector(".v3-tree[data-tree]");
  if (treeNav && window.fetch) {
    var ownTreeSrc = treeNav.getAttribute("data-tree");
    var isConsolePage = /console_tree\.json(\?|$)/.test(ownTreeSrc);
    var consoleTreeSrc = isConsolePage ? ownTreeSrc :
      ownTreeSrc.replace(/workspace\/tree\.json(\?.*)?$/, "console_tree.json");
    var stickyPossible = !isConsolePage && consoleTreeSrc !== ownTreeSrc;
    var stickyNow = stickyPossible &&
      sessionStorage.getItem("xg_zone") === "console";

    var renderTree = function (src, stickyMode) {
      fetch(src, { cache: "no-cache" })
        .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
        .then(function (data) {
          var oldUl = treeNav.querySelector(".v3-tree-list");
          if (!oldUl || !data || !data.entries) return;
          var here = location.pathname.replace(/index\.html$/, "");
          function leafLi(e) {
            var li = document.createElement("li");
            var a = document.createElement("a");
            // active: exact page, or a version snapshot under a stable leaf
            // (/workspace/<page>/ redirects to /workspace/<page>/v/N/ —
            // found 2026-07-17: exact matching left versioned pages with no
            // highlighted leaf at all)
            var active = VLOGIC.treeLeafActive(e.href, here);
            a.className = "v3-tree-link" + (active ? " active" : "");
            a.href = e.href;
            a.textContent = e.label;
            if (e.meta) {
              var m = document.createElement("span");
              m.className = "v3-leaf-meta";
              m.textContent = e.meta;
              a.appendChild(m);
            }
            a.addEventListener("click", function () { setTree(false); });
            li.appendChild(a);
            return li;
          }
          var ul = document.createElement("ul");
          ul.className = "v3-tree-list";
          data.entries.forEach(function (e) {
            if (e.children) {
              var li = document.createElement("li");
              li.className = "v3-tree-group";
              var lab = document.createElement("div");
              lab.className = "v3-tree-grouplabel";
              lab.textContent = e.label;
              var sub = document.createElement("ul");
              sub.className = "v3-tree-sublist";
              e.children.forEach(function (k) { sub.appendChild(leafLi(k)); });
              li.appendChild(lab);
              li.appendChild(sub);
              ul.appendChild(li);
            } else {
              ul.appendChild(leafLi(e));
            }
          });
          oldUl.replaceWith(ul);
          // keep the brand subtitle honest about WHICH tree is shown
          var sub2 = treeNav.querySelector(".v3-brand-sub");
          if (sub2 && data.subtitle) sub2.textContent = data.subtitle;
          document.body.classList.toggle("v3-zone-sticky", !!stickyMode);
          var oldExit = treeNav.querySelector(".v3-zone-exit");
          if (oldExit) oldExit.remove();
          if (stickyMode) mountExit();
        })
        .catch(function () { /* baked tree remains — the no-JS/offline fallback */ });
    };

    var mountExit = function () {
      var d = document.createElement("div");
      d.className = "v3-zone-switch v3-zone-exit";
      var a = document.createElement("a");
      a.className = "v3-zone-link";
      a.href = "#";
      a.textContent = "switch to workspace view";
      a.addEventListener("click", function (ev) {
        ev.preventDefault();
        sessionStorage.removeItem("xg_zone");
        renderTree(ownTreeSrc, false);
      });
      d.appendChild(a);
      var head = treeNav.querySelector(".v3-tree-head");
      if (head && head.nextSibling) treeNav.insertBefore(d, head.nextSibling);
      else treeNav.appendChild(d);
    };

    renderTree(stickyNow ? consoleTreeSrc : ownTreeSrc, stickyNow);

    // context propagation: a tree-link click while the CONSOLE tree is
    // shown (a real console page, or sticky mode) marks console context
    // for the destination page
    document.addEventListener("click", function (e) {
      var a = e.target.closest(".v3-tree .v3-tree-link");
      if (!a) return;
      if (isConsolePage || document.body.classList.contains("v3-zone-sticky")) {
        sessionStorage.setItem("xg_zone", "console");
      } else {
        sessionStorage.removeItem("xg_zone");
      }
    });
  }

  // ---- version picker + not-current banner (living pages and snapshots).
  // The manifest ([{v, date, note, sha, snapshot?, sha256?}] newest-last) is
  // fetched EAGERLY at load from data-versions — on a living page that is
  // "./versions.json", on an immutable /v/N/ snapshot "../../versions.json"
  // (the LIVING manifest) — so a superseded snapshot detects itself at
  // runtime and grows the "not current" banner while its bytes never
  // change. Menu rows link to their snapshots (baseDir + v/K/); manifest
  // entries with snapshot:false (pre-tooling history) render unlinked.
  document.querySelectorAll("details.v3-vpick[data-versions]").forEach(function (pick) {
    var src = pick.getAttribute("data-versions");
    var baseDir = src.slice(0, src.lastIndexOf("/") + 1); // "" or "../../"
    // LIVING-PAGE-CANONICAL (2026-07-19, arXiv model): the stable URL is the
    // CURRENT living page; /v/X.y/ are immutable bookmarks. data-living marks the
    // living page (no banner there); a snapshot carries its own "X.y" in
    // data-current and always shows the not-current banner. Versions are X.y
    // strings; String() also tolerates the legacy integer manifests in the zone.
    var current = pick.getAttribute("data-current");   // "" on living, "X.y" on a snapshot
    var living = pick.hasAttribute("data-living");
    var menu = pick.querySelector(".v3-vmenu");
    fetch(src, { cache: "no-cache" })
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (versions) {
        var livingHref = baseDir || "./";                // the stable/living URL
        menu.textContent = "";
        // "Current version" row = the living page (always the newest state)
        var cur = document.createElement("li");
        var ca = document.createElement("a");
        ca.href = livingHref;
        ca.textContent = "Current version" + (living ? "  (here)" : "");
        cur.appendChild(ca);
        if (living) cur.className = "current";
        menu.appendChild(cur);
        // labeled snapshots, newest-first
        versions.slice().reverse().forEach(function (v) {
          var vv = String(v.v);
          var li = document.createElement("li");
          if (VLOGIC.rowIsCurrent(living, vv, current)) li.className = "current";
          var label = "Version " + vv + " · " + v.date + (v.note ? " · " + v.note : "");
          if (v.snapshot === false) {
            // a label with no preserved snapshot: inert row, quiet suffix.
            li.className += " nolink";
            li.textContent = label + " · label only";
          } else {
            var a = document.createElement("a");
            a.href = baseDir + "v/" + vv + "/";
            a.textContent = label;
            li.appendChild(a);
          }
          menu.appendChild(li);
        });
        // not-current banner: on a SNAPSHOT only (the living page is never stale)
        if (VLOGIC.bannerVisible(living)) {
          var b = document.createElement("div");
          b.className = "v3-vbanner";
          var s = document.createElement("span");
          s.textContent = "Version " + current + " · not current · ";
          var a2 = document.createElement("a");
          a2.href = livingHref;
          a2.textContent = "view current version";
          b.appendChild(s);
          b.appendChild(a2);
          var pg = document.querySelector(".v3-main .page");
          if (pg) pg.insertBefore(b, pg.firstChild);
        }
      })
      .catch(function () {
        menu.textContent = "";
        var li = document.createElement("li");
        li.className = "v3-verr";
        li.textContent = "history unavailable";
        menu.appendChild(li);
      });
    document.addEventListener("click", function (e) {
      if (pick.open && !pick.contains(e.target)) pick.open = false;
    });
  });

  // ---- hypothes.is sidebar collision watch
  // The modern client's <hypothesis-sidebar> host is a ZERO-SIZE element;
  // the visual sidebar is div.sidebar-container inside its (open) shadow
  // root — position:fixed, 428px wide, left == viewport width when
  // collapsed (only an 18px toggle strip shows at vw-22), sliding left by
  // its width when opened. "Open" test: the container reaches more than
  // 60px into the viewport. Fallback for the legacy client: the
  // .annotator-frame element in the light DOM, same rect test. Polled
  // (500ms) rather than observed: the open/close is a transform on nodes
  // we don't own, so a rect poll is the only contract-free signal. No-op
  // when the embed is absent.
  function annotHost() {
    var el = document.querySelector("hypothesis-sidebar");
    if (el && el.shadowRoot) {
      var c = el.shadowRoot.querySelector(".sidebar-container");
      if (c) return c;
    }
    return el || document.querySelector(".annotator-frame");
  }
  var polling = false;
  function pollAnnot() {
    var host = annotHost();
    if (!host) return;
    var r = host.getBoundingClientRect();
    var open = r.width > 0 && (window.innerWidth - r.left) > 60;
    document.body.classList.toggle("v3-annot-open", open);
  }
  function startPollingIfEmbedded() {
    if (polling) return;
    if (annotHost() || document.querySelector('script[src*="hypothes.is"]')) {
      polling = true;
      setInterval(pollAnnot, 500);
    }
  }
  startPollingIfEmbedded();
  // the embed script is async; re-check a few times after load
  var retries = 0;
  var t = setInterval(function () {
    startPollingIfEmbedded();
    if (polling || ++retries > 20) clearInterval(t);
  }, 500);
})();
