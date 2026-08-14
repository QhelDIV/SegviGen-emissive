"""xgpage_ext.py — lightgen-local xgpage extensions NOT in the standalone
`xgpage` package (~/studio/xgpage).

Migration note (2026-07-22): lightgen migrated off its vendored `tools/xgpage.py`
fork onto the standalone package (`uv pip install -e ~/studio/xgpage`). The
package covers every function lightgen's builders call EXCEPT the click-to-load
3D (model-viewer) lightbox used by `results_2k_v1`'s "paper-style mesh view"
(`viewer_img`, `model_viewer_modal`, `model_viewer_head`) — a lightgen-only
addition that was never upstreamed. This module preserves them.

RECOVERY NOTE (important — read before trusting this file blindly): the
in-repo `tools/xgpage.py` fork these were extracted from had ALREADY LOST these
three function bodies before this migration started (an earlier same-project
sync of `tools/xgpage.py` onto the v3-capable canonical copy overwrote them
without preserving this fork-local addition — a real regression, caught only
now because `segvigen_emissive/build_results_2k_page.py` still calls them and
would have failed loudly on its next rebuild). They were NOT copied from a
working source file; they were RECONSTRUCTED from three independent pieces of
evidence, cross-checked against each other:
  1. the exact call sites in `segvigen_emissive/build_results_2k_page.py`
     (signature: `lp.viewer_img(img, glb, cap=..., title=...)`,
     `lp.model_viewer_head()`, `lp.model_viewer_modal()`),
  2. the exact markup already present in the LIVE published
     `results_2k_v1/index.html` (byte-for-byte: the `<img class="v3d"
     data-glb=... data-title=...><div class="cap">...<span
     class="v3d-badge">...</span></div>` shape, the `#mv3d` modal's full
     attribute list, and the `<script type="module" src="{assets_rel}/
     model-viewer.min.js">` head tag),
  3. the lightgen memory note `reference_model_viewer_lightbox.md`
     (documents the intended API shape: `viewer_img(img_src, glb_src, *, cap,
     alt, title, badge)`, the click→`#mv3d`→`ui.js` wiring, the
     `model-viewer.min.js@3.5.0` self-host requirement).
The CSS (`.v3d`, `.v3d-badge`, `.mv3d-*`) and JS (`ui.js`'s guarded
`.v3d`-click IIFE) were NOT lost — they survived independently in
`web/assets/theme.css`/`ui.js` (verified present, unchanged) since assets are
edited separately from xgpage.py. Re-verify against a fresh build of
`results_2k_v1` (this migration's QA gate) before trusting this file further;
if the rebuilt HTML differs byte-for-byte from the already-published live
page in the `.v3d`/`#mv3d`-related markup, something here is still wrong.

Usage: `import xgpage as lp; import xgpage_ext as lpx` — pages that use the
3D lightbox call `lpx.viewer_img(...)`, `lpx.model_viewer_head()`,
`lpx.model_viewer_modal()` alongside the normal `lp.*` package calls.

Do NOT modify the shared `~/studio/xgpage` package from this file. If these
three functions ever earn a place in every project (not just lightgen),
upstreaming them is a coordinated follow-up (the package also serves
somages) — not decided here.
"""
import html as _html_mod


def _esc(s):
    return _html_mod.escape(str(s))


def _esc_attr(s):
    return _html_mod.escape(str(s), quote=True)


def model_viewer_head(assets_rel="../assets"):
    """The self-hosted <model-viewer> custom-element module script — feed to
    xgpage's `page(extra_head=...)`. Pinned, self-hosted
    `web/assets/model-viewer.min.js` (@google/model-viewer 3.5.0 — see the
    lightgen memory note for why 3.5.0 specifically, not `latest`)."""
    return f'<script type="module" src="{_esc_attr(assets_rel)}/model-viewer.min.js"></script>'


def model_viewer_modal():
    """The #mv3d lightbox shell — feed to `page(extra_body_end=...)` ONCE per
    page. `web/assets/ui.js` binds `.v3d` thumbnail clicks to open it (see
    that file's guarded IIFE); this function only emits the modal markup."""
    return (
        '<div id="mv3d" class="mv3d-modal">'
        '<div class="mv3d-bar">'
        '<span id="mv3d-title"></span>'
        '<a id="mv3d-dl" class="mv3d-dl" href="#" download>download GLB</a>'
        '<button id="mv3d-close" class="mv3d-close" type="button">&#10005; close</button>'
        '</div>'
        '<model-viewer id="mv3d-viewer" camera-controls auto-rotate '
        'rotation-per-second="18deg" interaction-prompt="none" exposure="1.15" '
        'tone-mapping="neutral" shadow-intensity="0.35" shadow-softness="1">'
        '</model-viewer>'
        '</div>'
    )


def viewer_img(img_src, glb_src, *, cap="", alt="", title="", badge="&#128269; 3D"):
    """A clickable thumbnail that opens `glb_src` in the `#mv3d` orbit/zoom
    modal on click (wired by `web/assets/ui.js`); renders a plain `<img>` (no
    lightbox, no badge) if `glb_src` is falsy, so a page can call this
    uniformly even for shapes with no preview GLB yet.

    - img_src / glb_src: image and GLB paths, site- or page-relative per the
      caller's own convention (this function does not rewrite them).
    - cap: raw HTML caption placed under the image (matches xgpage's other
      figure captions — entities/markup allowed, NOT escaped).
    - alt: img alt text (escaped).
    - title: the modal's title-bar text on open — pass a LITERAL "·"
      character if you want one, never the `&middot;` entity (`ui.js` sets
      this via `el.textContent`, so an HTML entity would render as literal
      escaped text, not the character — see the memory note this was
      recovered from).
    - badge: raw HTML for the small corner badge (default the magnifier +
      "3D" glyph); only shown when glb_src is present.
    """
    if not glb_src:
        return f'<img src="{_esc_attr(img_src)}" alt="{_esc_attr(alt)}">' + (
            f'<div class="cap">{cap}</div>' if cap else "")
    return (
        f'<img class="v3d" src="{_esc_attr(img_src)}" alt="{_esc_attr(alt)}" '
        f'data-glb="{_esc_attr(glb_src)}" data-title="{_esc_attr(title)}">'
        f'<div class="cap">{cap} <span class="v3d-badge">{badge}</span></div>'
    )
