"""workspace_zone.py — the lightgen research WORKSPACE zone definition.

Ported from somages `tools/workspace_zone.py` (read in full before adapting —
see its own module docstring for the design rationale this mirrors). LITE
scope (owner decision, 2026-07-19): zone separation + tree + one-way switcher
+ the zone-link enforcement guard ONLY. Explicitly OMITTED, do not build:
per-page X.y VERSIONING (somages' publish_version.py / immutable /v/N/
snapshots — no `v3_version_slot()` call anywhere in this zone) and
hypothes.is ANNOTATION (no `<script src="https://hypothes.is/embed.js">`
anywhere in this zone). somages' build_workspace.py hard-couples its "living
pages" to both (`version_slot()` calls `xg.v3_version_slot(...)`, reading a
per-page `versions.json` that only exists because publish_version.py mints
it) — lightgen's build_workspace.py (see that file) does not import or call
either mechanism at all, so there is nothing to "stub out"; it was simply
never wired in.

STRUCTURAL DIFFERENCE FROM SOMAGES (LITE scope, deliberate): somages' zone
holds "living pages" (web/workspace/<slug>/, each its OWN v3 page carrying
the persistent left tree — see build_workspace.py there) PLUS daily reports.
lightgen's zone holds exactly ONE page it builds itself — the workspace
Overview/landing page (web/workspace/index.html, built by
tools/build_workspace.py) — and links OUT to the project's EXISTING
collaborator-facing report pages (results_2k_v1/, pipeline_glb_direct/,
glb_direct_pilot_v1/, ...), which are untouched xgpage v2 pages with no
persistent rail of their own ("no report-page content touched" — 2026-07-19
brief). The tree therefore does NOT persist once you click through to one of
those pages; it persists only on the workspace landing page itself. This is
the concrete cost of "lite" — flagged explicitly in the build report, not
silently accepted.

ZONE BOUNDARY LAW (one-way rule, unchanged from somages): the console
(operator zone) may link INTO this workspace; NOTHING in the workspace zone
may link to the console. This zone definition therefore contains no console
URLs, by construction, and `console_links_in()` below is the mechanical
enforcement — every workspace-zone page build calls it and FAILS LOUDLY on a
hit (see build_workspace.py's build()).

COPY-variant paths (unchanged from lightgen's other tools/*.py): WORKSPACE_DIR
points at PUBLISH_DEST, not a repo-local web/workspace/ (the repo is on
local-scratch, unservable — see project-console skill's "Publishing model").
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SITE_ROOT = "/projects/omages/yanxg/lightgen"
BASE_URL = f"https://aspis.cmpt.sfu.ca{SITE_ROOT}"
WORKSPACE_URL = f"{SITE_ROOT}/workspace"

REPO = Path(__file__).resolve().parent.parent
PUBLISH_DEST = Path("/project/3dlg-hcvc/omages/www/yanxg/lightgen")  # COPY variant
WORKSPACE_DIR = PUBLISH_DEST / "workspace"
TREE_JSON = WORKSPACE_DIR / "tree.json"
TREE_JSON_URL = f"{WORKSPACE_URL}/tree.json"

# The collaborator-facing pages this zone links to (pre-existing, untouched
# report pages — see module docstring). slug -> (label, meta one-liner).
# Extend this dict + COLLAB_PAGES as new collaborator pages are published;
# the emissive-GT page master is hand-building will land here once its slug
# is known (2026-07-19 roadmap: "hand-building the crystal-clear collaborator
# page — emissive-GT story"). Do NOT add a placeholder href that 404s.
COLLAB_PAGES = [
    ("results_2k_v1", "2k fine-tune results", "clean negative result"),
    ("pipeline_glb_direct", "Direct-GLB proposal", "the original proposal"),
    ("glb_direct_pilot_v1", "Direct-GLB pilot verdict",
     "verdict: the o-voxel emissive attr is broken"),
]

# LIVING (versioned) pages: workspace/<slug>/, each a stable URL backed by
# immutable /v/X.y/ snapshots minted by tools/publish_version.py (2026-08-06).
# These live INSIDE the zone, unlike COLLAB_PAGES above, which link out to
# untouched report pages. They belong here rather than only in
# publish_version.py's tree registration because write_tree_json() rewrites
# tree.json WHOLESALE from tree_entries(): a page registered only at mint time
# survived until the next `build_workspace.py --publish` and then silently
# vanished from the rail. slug -> (label, meta).
LIVING_PAGES = [
    ("paper_skeleton", "Paper skeleton", "the claim chain"),
    ("render_sweep", "Render sweep", "why Filmic +1.5 for the box figures"),
    ("diagnostics", "Diagnostics", "why every emissive model sits near 0.1 IoU"),
    ("rendering", "Rendering setups", "the five named lighting setups"),
]
# The rail heading for LIVING_PAGES. publish_version.register_in_tree() reads
# this same constant, so the minting tool and this module cannot disagree
# about the group's name.
LIVING_GROUP_LABEL = "Paper"

# console-zone page URLs, for enforcing the one-way rule mechanically.
CONSOLE_HREFS = [f"{SITE_ROOT}/{p}" for p in (
    "index.html", "pages.html", "roadmap.html", "state.html", "experiments.html",
    "worklog.html", "todo.html", "diffusionnet.html", "clarifications.html",
    "notes/", "console/", "console_tree.json")]


def console_links_in(html_text: str) -> list[str]:
    """Return the console hrefs a workspace-zone page illegally contains."""
    return [h for h in CONSOLE_HREFS if h in html_text]


def tree_entries():
    """The zone tree as v3_tree()/tree.json entries (hrefs site-absolute)."""
    return [
        {"label": "Overview", "children": [
            {"label": "Workspace", "href": f"{WORKSPACE_URL}/index.html"},
        ]},
        {"label": LIVING_GROUP_LABEL, "children": [
            {"label": label, "href": f"{WORKSPACE_URL}/{slug}/index.html", "meta": meta}
            for slug, label, meta in LIVING_PAGES
        ]},
        {"label": "Results & proposals", "children": [
            {"label": label, "href": f"{SITE_ROOT}/{slug}/index.html", "meta": meta}
            for slug, label, meta in COLLAB_PAGES
        ]},
    ]


def tree_html(active_href=None):
    """Baked tree markup (the no-JS fallback; runtime refresh via tree.json)."""
    import xgpage as xg  # the installed package (uv pip install -e ~/studio/xgpage); migrated 2026-07-22
    entries = tree_entries()
    for g in entries:
        for leaf in g.get("children", []):
            if leaf["href"] == active_href:
                leaf["active"] = True
    return xg.v3_tree(entries, title="Lightgen", subtitle="research workspace",
                      tree_src=TREE_JSON_URL)


def console_workspace_group():
    """The console tree's "Workspace" group: the console sidebar is a
    SUPERSET of the workspace (the one-way law is untouched — the workspace
    still never links back)."""
    children = [{"label": "Workspace overview", "href": f"{WORKSPACE_URL}/index.html"}]
    children += [{"label": label, "href": f"{WORKSPACE_URL}/{slug}/index.html", "meta": meta}
                 for slug, label, meta in LIVING_PAGES]
    children += [{"label": label, "href": f"{SITE_ROOT}/{slug}/index.html", "meta": meta}
                 for slug, label, meta in COLLAB_PAGES]
    return {"label": "Workspace", "children": children}


CONSOLE_TREE_JSON = PUBLISH_DEST / "console_tree.json"


def write_tree_json():
    """Rewrite the zone's runtime nav manifest, and refresh the CONSOLE
    tree's Workspace group in place (the console sidebar is a superset of
    this zone)."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    TREE_JSON.write_text(json.dumps({
        "title": "Lightgen", "subtitle": "research workspace",
        "entries": tree_entries()}, indent=1))
    if CONSOLE_TREE_JSON.exists():
        try:
            data = json.loads(CONSOLE_TREE_JSON.read_text())
            for g in data.get("entries", []):
                if g.get("label") == "Workspace":
                    g["children"] = console_workspace_group()["children"]
            CONSOLE_TREE_JSON.write_text(json.dumps(data, indent=1))
        except (ValueError, OSError) as e:
            print(f"warning: console tree refresh skipped ({e})")
    return TREE_JSON
