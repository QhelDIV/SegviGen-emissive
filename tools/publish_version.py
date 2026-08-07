#!/usr/bin/env python3
"""publish_version.py — per-page X.y versioning for the lightgen research
WORKSPACE zone. Ported from somages (2026-08-06), reconstructed from the
published somages artifacts under
/project/3dlg-hcvc/omages/www/yanxg/somages/workspace/ plus the contract
documented in xgpage's v3_version_slot() and assets/xg3.js.

WHAT THIS OWNS
--------------
A versioned page is a LIVING URL backed by immutable numbered snapshots:

    workspace/<page>/index.html      the living page (always the newest state)
    workspace/<page>/v/<V>/index.html   an immutable snapshot, own img/
    workspace/<page>/versions.json   the manifest, newest LAST
    workspace/tree.json              the zone's runtime nav manifest

Once a page has a v/ directory this tool is its ONLY writer.

LIVING-PAGE-CANONICAL (this is the contract the SHIPPED RUNTIME implements)
--------------------------------------------------------------------------
assets/xg3.js (VLOGIC.bannerVisible / rowIsCurrent, and the picker block that
distinguishes "./versions.json" on a living page from "../../versions.json" on
a snapshot) implements the arXiv model, ratified 2026-07-19 and recorded in
xgpage/core.py::v3_version_slot: the stable URL SERVES the living page; /v/X.y/
are immutable bookmarks. The living page never shows a "not current" banner;
every snapshot always does.

The observed somages artifacts agree: workspace/goal/index.html is a full v3
page carrying data-living="1", and the meta-refresh + location.replace stub is
used in the OTHER direction — v/2/index.html (a retired legacy integer
snapshot) redirects BACK to "../../". That stub is implemented here as
`retire_snapshot()`; it is not what the living page becomes.

Flipping the living index.html into a redirect at the current snapshot would
break the shipped runtime two ways: the redirect target is a snapshot, so
xg3.js would stamp "Version N · not current · view current version" on the
version that IS current, and that banner's link points back at the living URL,
which redirects to the snapshot again. `--redirect-living` implements the
redirect variant anyway, for a caller who wants it, and prints that warning.

HARD RULE — NO rel="canonical", ANYWHERE IN THIS SYSTEM
-------------------------------------------------------
Annotation tooling resolves targets through canonical URLs. One canonical tag
plus one annotation merges two versions' identities server-side: annotations
made on v1 surface on v2 and vice versa, destroying the per-version record
this system exists to preserve, irreversibly. Version isolation depends on
every snapshot keeping its own URL identity. `canonical_violations()` is the
mechanical enforcement; it runs at mint time (on the bytes about to be
written) and over the whole zone in --check. --self-test PROVES it fires.

SNAPSHOT INTEGRITY
------------------
`snapshot_sha256()` is the exact algorithm recovered from the published
somages manifests (verified byte-identical against goal/, contact-analysis/
and experiment-design/ v/0.2): sha256 over, for every file in the snapshot
sorted by POSIX relative path, the relative path's utf-8 bytes followed by the
file's bytes. --check re-hashes every snapshot:true entry and fails on drift.

LIGHTGEN ADAPTATION — COPY VARIANT, NEVER rmtree
------------------------------------------------
somages publishes by symlink; lightgen cannot (the repo's web/ is on local
scratch, which the web server cannot traverse), so every tool here writes
directly into PUBLISH_DEST and MERGE-COPIES. A wipe-then-copy once destroyed
every published page on this project. There is no rmtree in this file.

Usage:
    .venv_console/bin/python tools/publish_version.py <page> --note "..." \
        [--from SRC_DIR] [--version X.y] [--date YYYY-MM-DD] [--dry-run]
    .venv_console/bin/python tools/publish_version.py --check
    .venv_console/bin/python tools/publish_version.py --list <page>
    .venv_console/bin/python tools/publish_version.py --self-test
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import workspace_zone as wz  # noqa: E402  (local zone module, NOT xgpage)

PAGES_ROOT = wz.WORKSPACE_DIR                 # /project/.../lightgen/workspace
TREE_JSON = wz.TREE_JSON
ASSETS_DIR = wz.PUBLISH_DEST / "assets"
BASE_URL = wz.BASE_URL
WORKSPACE_URL = wz.WORKSPACE_URL
REPO_MIRROR = REPO / "web/workspace"           # --commit target (durability only)

# Entries never carried into a snapshot: build inputs, caches, and the
# versioning system's own bookkeeping (a snapshot must not contain a manifest
# that would shadow the living one at ../../versions.json).
SKIP_NAMES = {"index.html", "versions.json", "tree.json", "v",
              "__pycache__", ".DS_Store"}
SKIP_SUFFIXES = {".py", ".pyc"}

CANONICAL_RE = re.compile(r'<link[^>]*rel\s*=\s*["\']?canonical', re.I)
VSLOT_RE = re.compile(r'<details class="v3-vslot v3-vpick".*?</details>\s*', re.S)
PAGE_OPEN_RE = re.compile(r'<div class="page[^"]*">')
V3MAIN_RE = re.compile(r'<main class="v3-main">')
BODY_XG3_RE = re.compile(r'<body[^>]*class="[^"]*\bxg3\b')
HEAD_CLOSE_RE = re.compile(r'</head>', re.I)
BODY_CLOSE_RE = re.compile(r'</body>', re.I)
LIVING_SLOT_RE = re.compile(r'<details class="v3-vslot v3-vpick"[^>]*\bdata-living\b')


# --------------------------------------------------------------------------
# integrity
# --------------------------------------------------------------------------
def snapshot_sha256(snap_dir: Path) -> str:
    """The somages snapshot hash, recovered from published manifests: for every
    file under snap_dir sorted by POSIX relative path, feed the relative path's
    utf-8 bytes then the file's bytes into one sha256."""
    h = hashlib.sha256()
    files = [p for p in snap_dir.rglob("*") if p.is_file()]
    for p in sorted(files, key=lambda p: p.relative_to(snap_dir).as_posix()):
        h.update(p.relative_to(snap_dir).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def canonical_violations(html_text: str) -> list[str]:
    """Every <link rel=canonical> in the text. NON-EMPTY MUST FAIL THE BUILD —
    see the module docstring for why this is irreversible, not cosmetic."""
    return CANONICAL_RE.findall(html_text)


def _guard_html(label: str, html_text: str) -> None:
    """Both zone laws, on bytes about to be written. Exits on either."""
    if canonical_violations(html_text):
        sys.exit(f"CANONICAL GUARD FAILED: {label} contains <link rel=\"canonical\">. "
                 "A canonical tag merges two versions' annotation identity "
                 "server-side and is not recoverable. Remove it and re-mint.")
    console = wz.console_links_in(html_text)
    if console:
        sys.exit(f"ZONE-LINK GUARD FAILED: {label} illegally links to the "
                 f"console: {console}.")


# --------------------------------------------------------------------------
# the version slot (shell markup only; xg3.js fills the menu at runtime)
# --------------------------------------------------------------------------
def _vslot_html(version: str, date: str, manifest: str, living: bool) -> str:
    import xgpage as xg
    return xg.v3_version_slot(version=version, date=date, manifest=manifest,
                              living=living)


def picker_css(assets_dir: Path = ASSETS_DIR) -> str:
    """Compatibility shim for a source page built on xgpage v2 (body class
    "xg2", no v3 shell). theme3.css scopes ALL picker styling under `.xg3`,
    and putting xg3 on the body would drag 69 unrelated v3 typographic rules
    onto a page whose historical appearance the snapshot exists to preserve.
    So: lift ONLY the v3-v* rules out of the live theme3.css and drop the
    `.xg3 ` scope. Extracted at mint time, so it cannot drift from the
    stylesheet by hand. Returns "" if theme3.css is unreadable."""
    css = assets_dir / "theme3.css"
    if not css.exists():
        return ""
    out = []
    for sel, body in re.findall(r'^(\.xg3 [^{@]*?)\{([^}]*)\}', css.read_text(), re.M):
        if not re.search(r'v3-(vslot|vpick|vmenu|vbanner|vnote|verr)', sel):
            continue
        out.append(re.sub(r'\.xg3\s+', '', sel).strip() + " {" + body.strip() + "}")
    if not out:
        return ""
    return ("<style>\n/* version-picker shim: v3-v* rules lifted from "
            "theme3.css at mint time (source page is xgpage v2, so the .xg3 "
            "body scope is deliberately NOT applied). */\n"
            + "\n".join(out) + "\n</style>\n")


def strip_vslot(html_text: str) -> str:
    """Remove any version slot, so re-minting never stacks two pickers."""
    return VSLOT_RE.sub("", html_text)


def inject_vslot(html_text: str, *, version: str, date: str, living: bool,
                 assets_rel: str, assets_dir: Path = ASSETS_DIR) -> str:
    """Return html_text carrying exactly one version slot in the mode asked
    for. On a v2-theme page also adds the mount point xg3.js needs for the
    not-current banner (`.v3-main .page`), the picker shim CSS, and xg3.js
    itself — whose handlers the file's own header documents as defensive
    no-ops on pages missing the v3 elements."""
    html_text = strip_vslot(html_text)
    manifest = "versions.json" if living else "../../versions.json"
    slot = _vslot_html(version, date, manifest, living)

    m = PAGE_OPEN_RE.search(html_text)
    if not m:
        sys.exit('page shell not recognised: no <div class="page..."> found. '
                 "This tool versions xgpage v2/v3 pages only.")

    is_v3 = bool(BODY_XG3_RE.search(html_text))
    if not is_v3:
        if not V3MAIN_RE.search(html_text):
            # wrap: <main> opens before div.page and closes before </body>.
            html_text = (html_text[:m.start()] + '<main class="v3-main">\n'
                         + html_text[m.start():])
            html_text = BODY_CLOSE_RE.sub(lambda _m: "</main>\n</body>", html_text, count=1)
            m = PAGE_OPEN_RE.search(html_text)
        shim = picker_css(assets_dir)
        if shim and "version-picker shim" not in html_text:
            html_text = HEAD_CLOSE_RE.sub(lambda _m: shim + "</head>", html_text, count=1)
        xg3 = f'{assets_rel}/xg3.js'
        if "xg3.js" not in html_text:
            html_text = BODY_CLOSE_RE.sub(
                lambda _m: f'<script src="{xg3}"></script>\n</body>', html_text, count=1)
        m = PAGE_OPEN_RE.search(html_text)

    return html_text[:m.end()] + "\n" + slot + html_text[m.end():]


# --------------------------------------------------------------------------
# manifest + tree
# --------------------------------------------------------------------------
def load_manifest(page_dir: Path) -> list[dict]:
    f = page_dir / "versions.json"
    return json.loads(f.read_text()) if f.exists() else []


def write_manifest(page_dir: Path, versions: list[dict]) -> None:
    (page_dir / "versions.json").write_text(json.dumps(versions, indent=1))


def next_version(versions: list[dict]) -> str:
    """Labels are STRINGS ("0.1", "2"). Auto-increment only from an integer
    label; anything else must be named explicitly with --version."""
    if not versions:
        return "1"
    last = str(versions[-1]["v"])
    if last.isdigit():
        return str(int(last) + 1)
    sys.exit(f'last version is "{last}", which this tool will not '
             "auto-increment. Pass --version explicitly.")


def git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return ""


def register_in_tree(page: str, label: str, meta: str) -> None:
    """Merge the page into workspace/tree.json without disturbing the rest.

    DURABILITY: this registration alone is NOT durable --
    workspace_zone.write_tree_json() rewrites tree.json wholesale from
    tree_entries(), so a page registered only here survived until the next
    `build_workspace.py --publish` and then silently vanished from the rail.
    A versioned page must therefore ALSO be listed in
    workspace_zone.LIVING_PAGES, which is where tree_entries() reads it from
    (fixed 2026-08-06). The group heading comes from the shared constant
    workspace_zone.LIVING_GROUP_LABEL so the two writers cannot disagree.
    Warns when a page is registered here but absent from LIVING_PAGES."""
    if page not in {slug for slug, _, _ in wz.LIVING_PAGES}:
        print(f"warning: {page!r} is not in workspace_zone.LIVING_PAGES -- this "
              "tree registration will be dropped by the next "
              "`build_workspace.py --publish`. Add it there.")
    href = f"{WORKSPACE_URL}/{page}/index.html"
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


def _publish_perms(root: Path) -> None:
    for p in [root, *root.rglob("*")]:
        try:
            p.chmod(p.stat().st_mode | (0o005 if p.is_dir() else 0o004))
        except OSError:
            pass


def merge_copy_assets(src: Path, dest: Path) -> list[str]:
    """Copy every asset entry from src into dest, MERGING. Never removes an
    existing tree (see the module docstring)."""
    copied = []
    dest.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src.iterdir()):
        if entry.name in SKIP_NAMES or entry.suffix in SKIP_SUFFIXES:
            continue
        target = dest / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, target)
        copied.append(entry.name)
    return copied


# --------------------------------------------------------------------------
# mint
# --------------------------------------------------------------------------
def mint(page: str, note: str, *, source: Path | None = None, version: str | None = None,
         date: str | None = None, label: str | None = None, meta: str = "",
         dry_run: bool = False, redirect_living: bool = False,
         commit: bool = False) -> str:
    page_dir = PAGES_ROOT / page
    src = source or page_dir
    src_index = src / "index.html"
    if not src_index.exists():
        sys.exit(f"no source page: {src_index}")

    versions = load_manifest(page_dir)
    V = version or next_version(versions)
    if any(str(e["v"]) == V for e in versions):
        sys.exit(f"version {V} is already in {page}/versions.json. "
                 "Snapshots are never overwritten.")
    snap_dir = page_dir / "v" / V
    if snap_dir.exists():
        sys.exit(f"REFUSING TO CLOBBER: {snap_dir} already exists. "
                 "Snapshots are immutable; pick a different --version.")

    date = date or _dt.date.today().isoformat()
    base = strip_vslot(src_index.read_text())
    assets_rel = f"{wz.SITE_ROOT}/assets"

    snap_html = inject_vslot(base, version=V, date=date, living=False,
                             assets_rel=assets_rel)
    live_html = inject_vslot(base, version=V, date=date, living=True,
                             assets_rel=assets_rel)
    _guard_html(f"{page} v/{V}/index.html", snap_html)
    _guard_html(f"{page} index.html (living)", live_html)

    if dry_run:
        print(f"[dry-run] would mint {page} v{V} ({date}) from {src}")
        print(f"[dry-run] snapshot -> {snap_dir}")
        print(f"[dry-run] living   -> {page_dir/'index.html'}")
        return V

    snap_dir.mkdir(parents=True)
    copied = merge_copy_assets(src, snap_dir)
    for f in snap_dir.rglob("*.html"):
        _guard_html(str(f), f.read_text(errors="ignore"))
    (snap_dir / "index.html").write_text(snap_html)

    page_dir.mkdir(parents=True, exist_ok=True)
    merge_copy_assets(src, page_dir)
    if redirect_living:
        print("WARNING: --redirect-living contradicts the shipped xg3.js "
              "contract; the current snapshot will claim to be 'not current' "
              "and its banner link loops back to itself. See the module docstring.")
        (page_dir / "index.html").write_text(_redirect_stub(f"v/{V}/", page))
    else:
        (page_dir / "index.html").write_text(live_html)

    entry = {"v": V, "date": date, "note": note, "sha": git_sha(),
             "snapshot": True, "sha256": snapshot_sha256(snap_dir)}
    versions.append(entry)
    write_manifest(page_dir, versions)
    register_in_tree(page, label or page.replace("_", " ").capitalize(), meta)
    _publish_perms(page_dir)

    print(f"minted {page} v{V} ({date})  note: {note}")
    print(f"  snapshot : {BASE_URL}/workspace/{page}/v/{V}/")
    print(f"  living   : {BASE_URL}/workspace/{page}/")
    print(f"  sha256   : {entry['sha256']}")
    print(f"  assets   : {copied or 'none'}")
    if commit:
        _commit(page, V)
    return V


def _redirect_stub(target: str, page: str) -> str:
    """meta-refresh + location.replace, for static hosting. Used by
    retire_snapshot() (a retired snapshot pointing at the living page) and by
    the discouraged --redirect-living mode."""
    return (f'<!doctype html><meta charset="utf-8">\n'
            f'<meta http-equiv="refresh" content="0; url={target}">\n'
            f'<title>{page} (moved to the current version)</title>\n'
            f'<script>location.replace("{target}");</script>\n'
            f'<a href="{target}">current version</a>\n')


def retire_snapshot(page: str, version: str) -> None:
    """Replace a snapshot's bytes with a stub redirecting to the living page —
    the one place somages uses meta-refresh + location.replace (observed at
    workspace/goal/v/2/). Marks the manifest entry snapshot:false so the
    picker renders it as a label with no preserved bytes."""
    page_dir = PAGES_ROOT / page
    snap = page_dir / "v" / version
    if not snap.exists():
        sys.exit(f"no such snapshot: {snap}")
    (snap / "index.html").write_text(_redirect_stub("../../", page))
    versions = load_manifest(page_dir)
    for e in versions:
        if str(e["v"]) == version:
            e["snapshot"] = False
            e.pop("sha256", None)
    write_manifest(page_dir, versions)
    print(f"retired {page} v{version} -> redirect to living page")


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------
def living_slot_missing(page_dir: Path) -> str | None:
    """ERASURE GUARD. Once a page has a v/ directory this tool is its only
    writer, but nothing at the filesystem level enforces that: a page's own
    builder republishing over <page>/index.html silently strips the injected
    version slot. The snapshots survive intact, so every hash still verifies,
    while the LIVING url quietly loses its entire history -- invisible and
    lossy, the worst shape this failure can take. --check deliberately does
    not hash the living page (it is mutable by design, that is the point of
    living-page-canonical), so this presence assertion is the only thing that
    can catch it. Returns a reason string, or None when the page is healthy."""
    idx = page_dir / "index.html"
    if not idx.exists():
        return f"living page missing entirely: {idx}"
    t = idx.read_text(errors="ignore")
    if LIVING_SLOT_RE.search(t):
        return None
    if VSLOT_RE.search(t):
        return ("living index.html carries a SNAPSHOT slot (no data-living) -- "
                "snapshot bytes were copied over the living page")
    return ("living index.html has NO version slot -- a rebuild almost "
            "certainly republished over it and stripped the picker; "
            "re-mint, or restore from the newest snapshot")


def check(pages: list[str] | None = None) -> bool:
    ok = True
    # A page is versioned if it has EITHER a manifest or a v/ dir; a page with
    # one and not the other is itself a defect worth surfacing.
    targets = ([PAGES_ROOT / p for p in pages] if pages
               else sorted(d for d in PAGES_ROOT.iterdir()
                           if d.is_dir() and ((d / "versions.json").exists()
                                              or (d / "v").is_dir())))
    if not targets:
        print("no versioned pages found under", PAGES_ROOT)
    for page_dir in targets:
        versions = load_manifest(page_dir)
        print(f"[{page_dir.name}] {len(versions)} manifest entries")
        for e in versions:
            V = str(e["v"])
            if not e.get("snapshot", True):
                print(f"  v{V}: label only (no snapshot bytes) -- skipped")
                continue
            snap = page_dir / "v" / V
            if not snap.exists():
                print(f"  v{V}: FAIL missing snapshot dir {snap}")
                ok = False
                continue
            got = snapshot_sha256(snap)
            if got != e.get("sha256"):
                print(f"  v{V}: FAIL sha256 drift\n"
                      f"        recorded {e.get('sha256')}\n"
                      f"        actual   {got}")
                ok = False
            else:
                print(f"  v{V}: ok  {got[:16]}...")
        if (page_dir / "v").is_dir():
            reason = living_slot_missing(page_dir)
            if reason:
                print(f"  living page: FAIL {reason}")
                ok = False
            else:
                print("  living page: ok  carries a data-living version slot")

    # zone-wide sweeps: no canonical tag, no console link, anywhere.
    can, zone = [], []
    for f in sorted(PAGES_ROOT.rglob("*.html")):
        t = f.read_text(errors="ignore")
        if canonical_violations(t):
            can.append(str(f))
        if wz.console_links_in(t):
            zone.append(str(f))
    print(f"canonical sweep: {len(can)} file(s) with rel=\"canonical\" "
          f"across {PAGES_ROOT} -> {'PASS' if not can else 'FAIL'}")
    for f in can:
        print("  ", f)
    print(f"zone-link sweep: {len(zone)} file(s) linking the console -> "
          f"{'PASS' if not zone else 'FAIL'}")
    for f in zone:
        print("  ", f)
    ok = ok and not can and not zone
    print("CHECK:", "PASS" if ok else "FAIL")
    return ok


def self_test() -> bool:
    """Prove the guards fire, on planted violations, writing nothing."""
    results = []
    planted = '<head></head><body class="xg2"><div class="page">x</div></body>'
    results.append(("clean page has no canonical",
                    not canonical_violations(planted)))
    bad = planted.replace("</head>", '<link rel="canonical" href="/x/"></head>')
    results.append(("planted canonical is caught",
                    bool(canonical_violations(bad))))
    bad2 = planted.replace("x", f'<a href="{wz.SITE_ROOT}/roadmap.html">c</a>')
    results.append(("planted console link is caught",
                    bool(wz.console_links_in(bad2))))
    # hash sensitivity: one byte must change the digest
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "snap"
        (d / "img").mkdir(parents=True)
        (d / "index.html").write_text("hello")
        (d / "img" / "a.png").write_bytes(b"\x00\x01")
        h1 = snapshot_sha256(d)
        (d / "img" / "a.png").write_bytes(b"\x00\x02")
        results.append(("one flipped byte changes sha256", snapshot_sha256(d) != h1))
        # renaming a file must also change it (path is hashed, not just bytes)
        (d / "img" / "a.png").write_bytes(b"\x00\x01")
        (d / "img" / "a.png").rename(d / "img" / "b.png")
        results.append(("a renamed file changes sha256", snapshot_sha256(d) != h1))
    # erasure guard: a living page that lost (or never had) its slot must fail
    with tempfile.TemporaryDirectory() as td:
        pd = Path(td) / "page"
        (pd / "v" / "1").mkdir(parents=True)
        living = ('<details class="v3-vslot v3-vpick" data-versions="versions.json" '
                  'data-current="" data-living="1"><summary>Version 1</summary></details>')
        snap = ('<details class="v3-vslot v3-vpick" data-versions="../../versions.json" '
                'data-current="1"><summary>Version 1</summary></details>')
        (pd / "index.html").write_text(f"<body>{living}</body>")
        results.append(("healthy living page passes", living_slot_missing(pd) is None))
        (pd / "index.html").write_text("<body>rebuilt by the page's own builder</body>")
        results.append(("living page with the slot STRIPPED is caught",
                        bool(living_slot_missing(pd))))
        (pd / "index.html").write_text(f"<body>{snap}</body>")
        results.append(("snapshot bytes copied over the living page are caught",
                        bool(living_slot_missing(pd))))
        (pd / "index.html").unlink()
        results.append(("missing living page is caught", bool(living_slot_missing(pd))))
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    ok = all(p for _, p in results)
    print("SELF-TEST:", "PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------
# git (default OFF; the published tree is NOT in the repo -- see report)
# --------------------------------------------------------------------------
def _commit(page: str, V: str) -> None:
    """Mirror the version BYTES into the repo and commit them. lightgen
    publishes by COPY, so the published tree lives outside git entirely; the
    only way to give snapshots durability is a repo-local mirror. Snapshot
    HTML + the two nav manifests ONLY, never images."""
    dest = REPO_MIRROR / page / "v" / V
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PAGES_ROOT / page / "v" / V / "index.html", dest / "index.html")
    shutil.copy2(PAGES_ROOT / page / "versions.json",
                 REPO_MIRROR / page / "versions.json")
    shutil.copy2(TREE_JSON, REPO_MIRROR / "tree.json")
    paths = [str(dest / "index.html"), str(REPO_MIRROR / page / "versions.json"),
             str(REPO_MIRROR / "tree.json")]
    subprocess.run(["git", "-C", str(REPO), "add", *paths], check=True)
    subprocess.run(["git", "-C", str(REPO), "commit", "-m",
                    f"workspace/{page}: mint v{V}"], check=True)
    print(f"committed {len(paths)} file(s) for {page} v{V}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("page", nargs="?", help="workspace page slug")
    ap.add_argument("--note", default="", help="one-line note for the manifest entry")
    ap.add_argument("--from", dest="source", help="source dir holding index.html + img/ "
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
                    "and sweep the zone for canonical/console links")
    ap.add_argument("--list", action="store_true", help="print a page's manifest")
    ap.add_argument("--self-test", action="store_true",
                    help="prove the guards fire on planted violations")
    ap.add_argument("--retire", metavar="V", help="replace snapshot V's bytes with "
                    "a redirect to the living page and mark it label-only")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if args.check:
        sys.exit(0 if check([args.page] if args.page else None) else 1)
    if args.list:
        if not args.page:
            ap.error("--list needs a page")
        print(json.dumps(load_manifest(PAGES_ROOT / args.page), indent=1))
        return
    if args.retire:
        if not args.page:
            ap.error("--retire needs a page")
        retire_snapshot(args.page, args.retire)
        return
    if not args.page:
        ap.error("a page slug is required")
    if not args.note:
        ap.error("--note is required when minting")
    mint(args.page, args.note, source=Path(args.source) if args.source else None,
         version=args.version, date=args.date, label=args.label, meta=args.meta,
         dry_run=args.dry_run, redirect_living=args.redirect_living,
         commit=args.commit)


if __name__ == "__main__":
    main()
