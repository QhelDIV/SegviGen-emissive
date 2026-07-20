#!/usr/bin/env python3
"""build_workspace.py — the lightgen research WORKSPACE zone's landing page.

LITE scope (owner decision, 2026-07-19 — see tools/workspace_zone.py's module
docstring for the full rationale): builds exactly ONE page,
workspace/index.html, carrying the workspace zone's own v3 tree (Overview +
Results & proposals) and linking OUT to the project's existing, UNTOUCHED
collaborator-facing report pages. No living-page system, no versioning
(publish_version.py), no hypothes.is — none of that is imported or called
anywhere in this file.

ZONE-LINK GUARD: build() renders the page HTML first, checks it for any
console-zone href via workspace_zone.console_links_in(), and refuses to
write the file if the guard fires (see main() — `--check-guard` runs a
self-test that PROVES this actually catches a planted violation, not just
that the real page happens to pass).

Usage:
    .venv_console/bin/python tools/build_workspace.py             # stage only (not servable, see below)
    .venv_console/bin/python tools/build_workspace.py --publish   # publish (the only servable target)
    .venv_console/bin/python tools/build_workspace.py --check-guard  # prove the zone-link guard fires
"""
import argparse, pathlib, sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import xgpage as xg
import workspace_zone as wz

SITE_ROOT = wz.SITE_ROOT
BASE_URL = wz.BASE_URL
PUBLISH_DEST = wz.PUBLISH_DEST
ASSETS_DIR = REPO / "web/assets"
ASSETS_REL = f"{SITE_ROOT}/assets"
FAVICON = f"{SITE_ROOT}/assets/images/favicon.png"

LANDING_HREF = f"{wz.WORKSPACE_URL}/index.html"

DEK_HTML = (
    "Lightgen fine-tunes SegviGen to produce binary emissive-region masks for textured "
    "3D assets. This page collects the current results and proposals for collaborators — "
    "the console (internal notes, to-dos, experiment logs) lives elsewhere."
)


def collab_pages_html():
    out = ['<div class="clist-group">']
    for slug, label, meta in wz.COLLAB_PAGES:
        out.append(f'<a class="clist-item" href="{SITE_ROOT}/{slug}/index.html">'
                    f'<div class="ci-title">{xg._esc(label)}<span class="ci-meta">{xg._esc(meta)}</span></div>'
                    f'</a>')
    out.append('</div>')
    return "\n".join(out)


def build_html():
    header = (f'<header><div class="eyebrow">lightgen &middot; research workspace</div>'
              f'<h1>Lightgen research workspace</h1>'
              f'<p class="dek">{DEK_HTML}</p></header>')
    body = f'''
    <section>
      <h2>Results &amp; proposals</h2>
      {collab_pages_html()}
    </section>
    '''
    return xg.page(
        title="Lightgen research workspace",
        header_html=header,
        body_sections=[body],
        theme="v3",
        tree_html=wz.tree_html(active_href=LANDING_HREF),
        nav_title="Workspace",
        assets_rel=ASSETS_REL,
        assets_dir=ASSETS_DIR,
        extra_head=f'<link rel="icon" href="{FAVICON}">',
    )


def build(out_dir):
    html = build_html()
    violations = wz.console_links_in(html)
    if violations:
        sys.exit("ZONE-LINK GUARD FAILED: workspace/index.html illegally links to the "
                 f"console: {violations}. Fix the page before publishing.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(html)
    for p in [out_dir, *out_dir.rglob("*")]:
        try:
            p.chmod(p.stat().st_mode | (0o005 if p.is_dir() else 0o004))
        except OSError:
            pass
    return out_dir / "index.html"


def check_guard():
    """Self-test proving the guard actually fires: plant a real console href
    into a throwaway copy of the page and confirm console_links_in() catches
    it, THEN confirm the real built page is clean. Never writes anything."""
    real_html = build_html()
    real_violations = wz.console_links_in(real_html)
    print(f"real workspace page: {len(real_violations)} console links found "
          f"(expect 0) -> {'PASS' if not real_violations else 'FAIL'}")

    planted = real_html.replace(
        "</section>",
        f'</section><a href="{wz.SITE_ROOT}/index.html">back to console</a>', 1)
    planted_violations = wz.console_links_in(planted)
    print(f"planted-violation copy: {len(planted_violations)} console links found "
          f"(expect >=1) -> {'PASS (guard fires)' if planted_violations else 'FAIL (guard is broken)'}")
    if planted_violations:
        print(f"  caught: {planted_violations}")

    ok = (not real_violations) and bool(planted_violations)
    print("GUARD SELF-TEST:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--check-guard", action="store_true",
                     help="prove the zone-link guard fires on a planted violation; writes nothing")
    args = ap.parse_args()

    if args.check_guard:
        ok = check_guard()
        sys.exit(0 if ok else 1)

    if args.publish:
        out_dir = PUBLISH_DEST / "workspace"
        idx = build(out_dir)
        wz.write_tree_json()
        print(f"published: {idx}")
        print(f"URL: {BASE_URL}/workspace/index.html")
    else:
        out_dir = REPO / "web/_preview" / "workspace_landing"
        idx = build(out_dir)
        print(f"staged locally (not servable from local-scratch): {idx}")
        print("run with --publish to write directly to the servable NFS PUBLISH_DEST")


if __name__ == "__main__":
    main()
