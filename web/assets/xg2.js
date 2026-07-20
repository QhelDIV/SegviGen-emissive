/* xg2.js — runtime for xgpage design-language v2 pages (2026-07-14).
   Loaded ONLY by pages built with xgpage.page(theme="v2"), after ui.js
   (which stays for outline/collapse/fpath handlers and no-ops otherwise).
   Provides: (1) a shared hover tooltip for any [data-tip] element (chart
   bars, annotation dots), (2) compare-slider wiring for every .compare
   block (class-based; no per-instance ids needed). All handlers are
   defensive no-ops on pages without the elements. */
(function () {
  // ---- shared tooltip for [data-tip]
  var tipped = document.querySelectorAll("[data-tip]");
  if (tipped.length) {
    var tip = document.createElement("div");
    tip.id = "xg2-tooltip";
    tip.setAttribute("role", "tooltip");
    document.body.appendChild(tip);
    tipped.forEach(function (el) {
      el.addEventListener("mousemove", function (e) {
        tip.textContent = el.getAttribute("data-tip");
        tip.style.opacity = "1";
        var w = tip.offsetWidth, h = tip.offsetHeight;
        var x = Math.min(e.clientX + 14, window.innerWidth - w - 10);
        var y = e.clientY - h - 12 < 8 ? e.clientY + 16 : e.clientY - h - 12;
        tip.style.left = x + "px";
        tip.style.top = y + "px";
      });
      el.addEventListener("mouseleave", function () { tip.style.opacity = "0"; });
    });
  }

  // ---- compare sliders
  document.querySelectorAll(".compare").forEach(function (cmp) {
    var top = cmp.querySelector(".top");
    var div = cmp.querySelector(".divider");
    var inp = cmp.querySelector("input[type=range]");
    if (!top || !div || !inp) return;
    inp.addEventListener("input", function () {
      var v = +inp.value;
      top.style.clipPath = "inset(0 " + (100 - v) + "% 0 0)";
      div.style.left = v + "%";
    });
  });
})();
