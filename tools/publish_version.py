#!/usr/bin/env python3
"""publish_version.py — lightgen's thin driver over xgpage.versioning
(extracted 2026-08-10, mirroring the console/jobs/graph extractions).

WHAT LIVES HERE (project policy, not shared mechanism -- see
xgpage.versioning's module docstring for what moved into the package):

WHICH PAGES VERSION (policy, owner-set 2026-08-09): version the LIVING
workspace pages, the ones listed in workspace_zone.LIVING_PAGES: they carry
a stable URL a collaborator may cite or annotate, they change over time,
and a reader needs to be able to go back to the state they were shown.
Everything else does not version -- date-stamped pages (daily reports and
anything whose identity already IS a date) are immutable by construction,
so a snapshot would duplicate what the date already fixes; web/_preview/
pages are build outputs and staging copies, not addresses anyone cites.
The practical consequence: a page gets a v/ directory only after it has
been added to workspace_zone.LIVING_PAGES, never before (see
register_in_tree()'s DURABILITY note for the mechanism that makes the two
lists agree).

THE ZONE-LINK GUARD: a workspace page must never link the console (a zone
separation unique to this project) -- wired in as the package's
config.extra_guard, alongside the package's own universal canonical-tag
rule.

TREE REGISTRATION: register_in_tree() merges a versioned page into
workspace/tree.json without disturbing the rest, wired in as
config.register_hook. This registration alone is NOT durable --
workspace_zone.write_tree_json() rewrites tree.json wholesale from
tree_entries(), so a page registered only here survives until the next
`build_workspace.py --publish` and then silently vanishes from the rail. A
versioned page must therefore ALSO be listed in workspace_zone.LIVING_PAGES
(fixed 2026-08-06).

Usage:
    .venv_console/bin/python tools/publish_version.py <page> --note "..." \\
        [--from SRC_DIR] [--version X.y] [--date YYYY-MM-DD] [--dry-run]
    .venv_console/bin/python tools/publish_version.py --check
    .venv_console/bin/python tools/publish_version.py --list <page>
    .venv_console/bin/python tools/publish_version.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from xgpage import versioning as xv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import workspace_zone as wz  # noqa: E402  (local zone module, NOT xgpage)

TREE_JSON = wz.TREE_JSON
REPO_MIRROR = REPO / "web/workspace"           # --commit target (durability only)


def register_in_tree(page: str, label: str, meta: str) -> None:
    """Merge the page into workspace/tree.json without disturbing the rest.
    See the module docstring's TREE REGISTRATION section for the durability
    caveat (a page must ALSO be in workspace_zone.LIVING_PAGES). The group
    heading comes from the shared constant workspace_zone.LIVING_GROUP_LABEL
    so the two writers cannot disagree."""
    if page not in {slug for slug, _, _ in wz.LIVING_PAGES}:
        print(f"warning: {page!r} is not in workspace_zone.LIVING_PAGES -- this "
              "tree registration will be dropped by the next "
              "`build_workspace.py --publish`. Add it there.")
    href = f"{wz.WORKSPACE_URL}/{page}/index.html"
    data = (json.loads(TREE_JSON.read_text()) if TREE_JSON.exists()
            else {"title": "Lightgen", "subtitle": "research workspace", "entries": []})
    for group in data.get("entries", []):
        for leaf in group.get("children", []):
            if leaf.get("href") == href:
                leaf["label"], leaf["meta"] = label, meta
                TREE_JSON.write_text(json.dumps(data, indent=1))
                return
    data.setdefault("entries", []).append(
        {"label": wz.LIVING_GROUP_LABEL,
         "children": [{"label": label, "href": href, "meta": meta}]})
    TREE_JSON.write_text(json.dumps(data, indent=1))


CONFIG = xv.VersioningConfig(
    pages_root=wz.WORKSPACE_DIR,
    assets_dir=wz.PUBLISH_DEST / "assets",
    assets_rel=f"{wz.SITE_ROOT}/assets",
    base_url=f"{wz.BASE_URL}/workspace",
    repo=REPO,
    repo_mirror=REPO_MIRROR,
    extra_guard=wz.console_links_in,
    extra_guard_sweep_name="zone-link",
    extra_guard_sweep_desc="file(s) linking the console",
    self_test_extra_guard_plant=f'<a href="{wz.SITE_ROOT}/roadmap.html">c</a>',
    register_hook=register_in_tree,
    extra_commit_files=((TREE_JSON, REPO_MIRROR / "tree.json"),),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("page", nargs="?", help="workspace page slug")
    ap.add_argument("--note", default="", help="one-line note for the manifest entry")
    ap.add_argument("--from", dest="source", help="source dir holding index.html + assets "
                    "(default: the page's own published dir)")
    ap.add_argument("--version", help="explicit label (default: next integer)")
    ap.add_argument("--date", help="YYYY-MM-DD (default: today)")
    ap.add_argument("--label", help="tree label for the page")
    ap.add_argument("--meta", default="", help="tree meta one-liner")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--redirect-living", action="store_true",
                    help="make the living index.html a redirect at the current "
                         "snapshot (contradicts the shipped xg3.js contract)")
    ap.add_argument("--commit", action="store_true",
                    help="mirror snapshot HTML + versions.json + tree.json into "
                         "the repo and git-commit them (default OFF)")
    ap.add_argument("--check", action="store_true", help="re-hash every snapshot "
                    "and sweep the zone for canonical links")
    ap.add_argument("--list", action="store_true", help="print a page's manifest")
    ap.add_argument("--self-test", action="store_true",
                    help="prove the guards fire on planted violations")
    ap.add_argument("--retire", metavar="V", help="replace snapshot V's bytes with "
                    "a redirect to the living page and mark it label-only")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if xv.self_test(CONFIG) else 1)
    if args.check:
        sys.exit(0 if xv.check(CONFIG, [args.page] if args.page else None) else 1)
    if args.list:
        if not args.page:
            ap.error("--list needs a page")
        print(json.dumps(xv.load_manifest(CONFIG.pages_root / args.page), indent=1))
        return
    if args.retire:
        if not args.page:
            ap.error("--retire needs a page")
        xv.retire_snapshot(CONFIG, args.page, args.retire)
        return
    if not args.page:
        ap.error("a page slug is required")
    if not args.note:
        ap.error("--note is required when minting")
    xv.mint(CONFIG, args.page, args.note, source=Path(args.source) if args.source else None,
           version=args.version, date=args.date, label=args.label, meta=args.meta,
           dry_run=args.dry_run, redirect_living=args.redirect_living,
           commit=args.commit)


if __name__ == "__main__":
    main()
