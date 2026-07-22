#!/usr/bin/env python3
"""sync_xgpage_assets.py — refresh web/assets/ from the installed xgpage package.

Migration note (2026-07-22): web/assets/ is now a GENERATED published copy of
the standalone `xgpage` package's bundled assets (theme.css, ui.js, theme2.css,
xg2.js, theme3.css, xg3.js, katex/) — never hand-edit these files directly,
edit the package (~/studio/xgpage) and rerun this script.

LIGHTGEN-LOCAL PATCH (recovered after a real regression, 2026-07-22): the
model-viewer 3D lightbox needs THREE pieces (see tools/xgpage_ext.py's module
docstring for the Python side) — a separate FILE (model-viewer.min.js, not
shipped by the package, survives untouched as a sibling `publish_assets`
never touches) AND two FRAGMENTS inside files the package DOES own and fully
overwrite: the `.v3d`/`.mv3d-*` CSS block in theme.css and the `.v3d`-click
IIFE in ui.js. The first `publish_assets()` call in THIS migration silently
stripped both fragments (they lived only in lightgen's old fork, never
upstreamed) — caught only because results_2k_v1's lightbox was checked with
real eyes (Playwright, not just curl 200s) per the migration's own QA gate.
Recovered verbatim from the lightgen-ops git repo's initial commit
(`git show cce8c8d:web/assets/theme.css` / `ui.js`) and vendored here as
`tools/xgpage_ext_assets/model_viewer.{css,js}` — this script APPENDS them
after every `publish_assets()` call, idempotently (checked by a marker
string, so re-running this doesn't accumulate duplicate copies). ANY
future package-asset sync for this project MUST go through this script,
never a bare `publish_assets()` call, or the lightbox silently breaks again.

This is the first hop of a two-hop COPY-variant sync (the repo lives on
local-scratch, unservable — see project-console skill's "Publishing model"):
    package assets -> web/assets/ (this script, incl. the model-viewer patch)
    web/assets/ -> PUBLISH_DEST/assets/ (tools/build_console.py's sync_assets(),
        run on every console build)
Run this script whenever the package changes, independent of a console
rebuild (e.g. after editing ~/studio/xgpage and before rebuilding just one
report page).

Usage:
    .venv_console/bin/python tools/sync_xgpage_assets.py
"""
import pathlib

from xgpage.publish import publish_assets

REPO = pathlib.Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO / "web" / "assets"
EXT_ASSETS = pathlib.Path(__file__).resolve().parent / "xgpage_ext_assets"

_MARKER = "interactive 3D lightbox"  # first line of the CSS comment; also checked in ui.js's comment


def _append_once(dest_file, patch_text, marker=_MARKER):
    """Append patch_text to dest_file unless a line containing `marker` is
    already present (idempotent across repeated syncs)."""
    text = dest_file.read_text()
    if marker in text:
        return False
    dest_file.write_text(text.rstrip("\n") + "\n\n" + patch_text)
    return True


def sync():
    dest = publish_assets(ASSETS_DIR)
    css_patched = _append_once(ASSETS_DIR / "theme.css", (EXT_ASSETS / "model_viewer.css").read_text())
    js_patched = _append_once(ASSETS_DIR / "ui.js", (EXT_ASSETS / "model_viewer.js").read_text())
    return dest, css_patched, js_patched


def main():
    dest, css_patched, js_patched = sync()
    print(f"synced xgpage package assets -> {dest}")
    print(f"model-viewer CSS patch {'applied' if css_patched else 'already present'}")
    print(f"model-viewer JS patch {'applied' if js_patched else 'already present'}")
    survivors = [p.name for p in ASSETS_DIR.iterdir()
                 if p.name not in {"theme.css", "ui.js", "theme2.css", "xg2.js",
                                    "theme3.css", "xg3.js", "katex"}]
    if survivors:
        print(f"lightgen-only sibling files preserved: {survivors}")


if __name__ == "__main__":
    main()
