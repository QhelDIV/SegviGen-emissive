/* xg_codepop.js — runtime for the inline code-reference popover component
   (2026-07-24; generalized to any trigger element 2026-07-24). Loaded ONLY
   by pages built with page(theme="v2"|"v3", needs_codepop=True). ANY
   element carrying `data-codepop="<anchor>"` is a trigger -- a `.code-chip`
   button (from lp.CodeBook.chip()) is one presentation of that trigger, not
   the only one; a page can wire a custom element (an SVG <g>, say) onto the
   same anchor and it opens the same popover. The anchor must match a
   `<template id="cp-tpl-<anchor>">` (from lp.CodeBook.mount() /
   lp.CodeBook.register()) holding the syntax-highlighted snippet, baked at
   BUILD TIME from a pinned commit, plus a footer path/sha/permalink line.

   CLICK is the primary interaction (hover does not exist on phones, and
   this component must pass qa_widths at 390px); no hover affordance is
   implemented. Dismissible via outside click, Escape, or re-clicking the
   open trigger. Does not trap page scroll: the panel is position:fixed and
   repositions on scroll/resize instead of locking the body, so the reader
   can keep scrolling with a popover open.

   Non-native triggers (anything but a real <button> or an <a href>) get
   role="button", tabindex="0", and a keydown handler translating Enter/
   Space to a click, so an SVG diagram node is as keyboard-operable as a
   chip -- a page author only has to set data-codepop (and ideally
   aria-label) on the element; the rest of the affordance is automatic.
   Buttons keep native Enter/Space handling, so they are excluded from the
   synthetic keydown listener to avoid a double-fire. */
(function () {
  // ---- pure positioning logic, exported for node:test (no DOM) -------------
  var POS = {
    // Place a floating panel of size (w, h) relative to a trigger rect
    // (chip.getBoundingClientRect()-shaped: {top, bottom, left, right}),
    // clamped inside a (vw, vh) viewport with `margin` px of breathing
    // room on every edge. Prefers BELOW the chip; flips ABOVE when there
    // is not enough room below but there is more room above. Horizontal:
    // starts left-aligned to the chip, then slides left just enough to
    // keep the panel's right edge inside the viewport (never past the
    // left margin) — the case this exists for is a chip near the right
    // edge of a narrow (390px) viewport.
    place: function (chip, w, h, vw, vh, margin) {
      margin = margin == null ? 8 : margin;
      var spaceBelow = vh - chip.bottom;
      var spaceAbove = chip.top;
      var placement = (spaceBelow >= h + margin || spaceBelow >= spaceAbove) ? "below" : "above";
      var top = placement === "below" ? chip.bottom + 6 : chip.top - h - 6;
      top = Math.max(margin, Math.min(top, vh - h - margin));
      var left = Math.min(chip.left, vw - w - margin);
      left = Math.max(left, margin);
      return { top: top, left: left, placement: placement };
    },
    // A native <button> or an <a href> already fires click on Enter/Space
    // and already carries an implicit interactive role -- a trigger of
    // either shape is excluded from the synthetic keydown listener (else
    // Enter would open-then-close a button in one keystroke) and from the
    // auto-set role/tabindex (else a real anchor loses its href semantics).
    // Takes primitives, not a DOM element, so this stays testable under
    // node:test with no DOM.
    isNativeActivatable: function (tagNameLower, hasHref) {
      return tagNameLower === "button" || (tagNameLower === "a" && !!hasHref);
    }
  };
  if (typeof module !== "undefined" && module.exports) { module.exports = POS; return; }

  // ---- browser wiring
  var openChip = null, panel = null;

  function closePopover() {
    if (panel) { panel.remove(); panel = null; }
    if (openChip) { openChip.setAttribute("aria-expanded", "false"); openChip = null; }
  }

  function reposition() {
    if (!panel || !openChip) return;
    var r = openChip.getBoundingClientRect();
    var pos = POS.place(r, panel.offsetWidth, panel.offsetHeight,
      window.innerWidth, window.innerHeight);
    panel.style.top = pos.top + "px";
    panel.style.left = pos.left + "px";
    panel.classList.toggle("cp-above", pos.placement === "above");
  }

  function openPopover(chip) {
    var id = chip.getAttribute("data-codepop");
    var tpl = document.getElementById("cp-tpl-" + id);
    if (!tpl) return;
    panel = document.createElement("div");
    panel.className = "code-popover";
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", chip.getAttribute("aria-label") || chip.textContent);
    panel.appendChild(tpl.content.cloneNode(true));
    document.body.appendChild(panel);
    openChip = chip;
    chip.setAttribute("aria-expanded", "true");
    reposition();
    var closeBtn = panel.querySelector(".cp-close");
    if (closeBtn) closeBtn.addEventListener("click", closePopover);
  }

  document.querySelectorAll("[data-codepop]").forEach(function (trigger) {
    // Auto-affordance: only fill in what the markup left unset, so a
    // <button> chip (which already has role/tabindex/aria-* implicitly or
    // explicitly) is untouched, and a bare SVG <g> gets exactly what it's
    // missing.
    if (!trigger.hasAttribute("role")) trigger.setAttribute("role", "button");
    if (!trigger.hasAttribute("tabindex")) trigger.setAttribute("tabindex", "0");
    if (!trigger.hasAttribute("aria-haspopup")) trigger.setAttribute("aria-haspopup", "dialog");
    if (!trigger.hasAttribute("aria-expanded")) trigger.setAttribute("aria-expanded", "false");
    trigger.addEventListener("click", function (e) {
      e.stopPropagation();
      var wasOpen = openChip === trigger;
      closePopover();
      if (!wasOpen) openPopover(trigger);
    });
    if (!POS.isNativeActivatable(trigger.tagName.toLowerCase(), trigger.hasAttribute("href"))) {
      trigger.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " " || e.key === "Spacebar") {
          e.preventDefault();
          // trigger.click() -- not trigger.dispatchEvent(new MouseEvent(...))
          // -- would be the natural call, but SVGElement.click() is not
          // implemented in every engine (an SVG <g> trigger throws
          // "trigger.click is not a function" where an HTML element would
          // not). Dispatching the event directly works identically for
          // both HTML and SVG elements and is what the click listener
          // above is already keyed to.
          trigger.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
        }
      });
    }
  });

  document.addEventListener("click", function (e) {
    if (panel && !panel.contains(e.target)) closePopover();
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closePopover();
  });
  window.addEventListener("scroll", reposition, { passive: true });
  window.addEventListener("resize", reposition);
})();
