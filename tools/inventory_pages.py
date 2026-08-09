#!/usr/bin/env python3
"""inventory_pages.py — the console Pages database: scan, manifest, fragment.

Ported from the somages console (tools/inventory_pages.py, v13-era) — see that
project's module docstring for the full design rationale (runtime-data model,
the three manifest writers, the offline itables bundle constraints). This copy
adapts two things for lightgen's COPY-variant publishing (the repo lives on
local-scratch; the aspis web server cannot traverse a symlink to it — see
project-console skill "Storage constraint"):
  - WEBDIR/PREVIEW_DIR/MANIFEST point at PUBLISH_DEST (the real NFS www dir),
    not REPO/"web" (which holds only pages.yaml + README.md here).
  - An extra scan tier, "update" (PUBLISH_DEST/updates/<name>/index.html, one
    level deeper than a normal top-level page) — lightgen's daily-report genre
    keeps one standalone page per day (updates/<date>/) rather than somages'
    single continuous journal page, so those entries need their own scan pass
    to appear in the database at all.

RUNTIME DATA: the table's rows live in web/pages.json (served from
PUBLISH_DEST), fetched at page load — pages.html carries only the widget
skeleton + an init script. Two writers keep the manifest fresh here (lightgen
has no workspace-zone publish hook to also refresh it from, unlike somages):
  1. every console build (build_console.py),
  2. a user-crontab sweep — not yet installed for lightgen; run manually with
     `.venv_console/bin/python tools/inventory_pages.py --refresh` after
     creating or publishing any page, per the xgpage skill's standing rule.

Modes:
    --manifest            scan (cached) + curation join -> PUBLISH_DEST/pages.json
    --out <fragment.html> the Pages-tab HTML fragment (skeleton + init;
                          NEEDS .venv_itables for the offline DT bundle)
The manifest path needs only stdlib + PyYAML (runs under .venv_console).

Curation layer: web/pages.yaml ({name, tags, important, blurb} — the same
v11 per-page convention lightgen's console already used) joins onto the scan.
Pages absent render honestly (no tags, no star).

OFFLINE BUNDLE CONSTRAINTS: identical to the somages original — see that
file's docstring if modifying render_table_fragment(); unchanged here.
"""
import argparse
import datetime
import html
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SITE_ROOT = "/projects/omages/yanxg/lightgen"
BASE_URL = f"https://aspis.cmpt.sfu.ca{SITE_ROOT}"

# COPY variant: the servable NFS dir, not REPO/"web" (source-only here).
PUBLISH_DEST = pathlib.Path("/project/3dlg-hcvc/omages/www/yanxg/lightgen")
WEBDIR = PUBLISH_DEST
PREVIEW_DIR = WEBDIR / "_preview"
UPDATES_DIR = WEBDIR / "updates"
NOTES_DIR = REPO / "notes"
PAGES_YAML = REPO / "web" / "pages.yaml"
MANIFEST = WEBDIR / "pages.json"

CACHE_FILE = pathlib.Path.home() / ".cache/lightgen_pages_scan_cache.json"
SWEEP_LOG = pathlib.Path.home() / ".cache/lightgen_pages_sweep.log"

# Top-level dirs in PUBLISH_DEST that are infrastructure, not pages.
ROOT_EXCLUDE = {"_preview", "assets", "notes", "search", "stylesheets",
                "javascripts", "img", "fonts", "overrides", "console",
                "roadmap", "updates"}
# The console's own staging build dir — not a "page", it's this console.
CONSOLE_STAGING_NAME = "console_v11"

COLUMNS = ["name", "tier", "title", "tags", "created", "modified", "size_mb",
           "n_html", "important"]
# per-column <th> class, same order as COLUMNS (see render_table_fragment();
# the matching <td> classes are applied via columnDefs' className below).
# "notes" folds into the "title" cell as a .db-subline instead of its own
# column; size_mb/n_html stay in the data (searchable) but hidden from view
# -- fewer, denser visible columns beat a wide table that needs to scroll
# (2026-08-09 redesign, see theme3.css's .dbwrap history comment).
HEAD_CLASSES = ["", "dt-nowrap", "", "dt-hide-narrow", "dt-nowrap dt-right dt-hide-narrow",
                "dt-nowrap dt-right dt-hide-narrow", "", "", ""]

# When (frozen) a page was first observed by the sweep; survives republishes,
# which rewrite every file's mtime and would otherwise reset any filesystem-
# derived "created". Backfilled once from the earliest current file mtime.
CREATED_FILE = pathlib.Path.home() / ".cache/lightgen_pages_created.json"
TIME_FMT = "%Y-%m-%d %H:%M"


# ------------------------------------------------------------------ scan ----
def _title_from_index(idx: pathlib.Path, fallback: str) -> str:
    head = idx.read_text(errors="ignore")[:4000]
    if m := re.search(r"<title>(.*?)</title>", head, re.S):
        t = html.unescape(re.sub(r"\s+", " ", m.group(1))).strip()
        if t:
            return t
    if m := re.search(r"<h1[^>]*>(.*?)</h1>", head, re.S):
        t = html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
        t = re.sub(r"\s+", " ", t)
        if t:
            return t
    return fallback


def _is_redirect_stub(idx: pathlib.Path) -> bool:
    head = idx.read_text(errors="ignore")[:2000]
    return "http-equiv=\"refresh\"" in head or "http-equiv='refresh'" in head


def _dir_stats(d: pathlib.Path):
    files = [p for p in d.rglob("*") if p.is_file()]
    size_mb = round(sum(p.stat().st_size for p in files) / 1e6, 2) if files else 0.0
    mtimes = [p.stat().st_mtime for p in files] or [d.stat().st_mtime]
    modified = datetime.datetime.fromtimestamp(max(mtimes)).strftime(TIME_FMT)
    # created ESTIMATE = earliest surviving file mtime; frozen on first sight
    # (see _frozen_created). Approximate for pages that predate tracking, exact
    # and stable for anything created from now on.
    created_est = datetime.datetime.fromtimestamp(min(mtimes)).strftime(TIME_FMT)
    n_html = len(list(d.glob("*.html")))
    return size_mb, created_est, modified, n_html


def _load_created() -> dict:
    try:
        return json.loads(CREATED_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _save_created(reg: dict):
    try:
        CREATED_FILE.parent.mkdir(parents=True, exist_ok=True)
        CREATED_FILE.write_text(json.dumps(reg))
    except OSError:
        pass


def _plain(s: str) -> str:
    """Strip inline markdown for table-cell display."""
    s = re.sub(r"\[\[([^\]]+)\]\]", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]*)\)", r"\1", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    return s.replace("**", "")


_NOTE_STEM_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_(.+)$")


def _notes_index():
    out = {}
    if not NOTES_DIR.exists():
        return out
    for p in NOTES_DIR.glob("*.md"):
        m = _NOTE_STEM_RE.match(p.stem)
        if not m:
            continue
        lines = p.read_text(errors="ignore").splitlines()
        status = tldr = ""
        for ln in lines[:8]:
            if sm := re.match(r"\*{0,2}Status\*{0,2}:\s*(.+)", ln.strip()):
                status = sm.group(1).strip()
                break
        for ln in lines:
            if tm := re.match(r"\*{0,2}TL;?DR\*{0,2}:\s*(.+)", ln.strip(), re.I):
                tldr = tm.group(1).strip()
                break
        out[m.group(1)] = (status, tldr)
    return out


def _load_cache():
    try:
        return json.loads(CACHE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _cache_key(d: pathlib.Path) -> str:
    """Per-directory freshness key (see somages original for the full
    rationale) — dir mtime for create/delete/rename, index.html mtime for
    an in-place republish."""
    idx = d / "index.html"
    try:
        i = idx.stat().st_mtime_ns
    except OSError:
        i = 0
    return f"{d.stat().st_mtime_ns}:{i}"


def scan_rows(use_cache: bool = False):
    notes_idx = _notes_index()
    cache = _load_cache() if use_cache else {}
    created_reg = _load_created()
    new_cache = {}
    rows = []
    found_root = set()

    def one(d: pathlib.Path, tier: str, url: str, name: str = None):
        idx = d / "index.html"
        if not idx.exists() or _is_redirect_stub(idx):
            return
        name = name or d.name
        key = _cache_key(d)
        ck = f"{tier}:{name}"
        cached = cache.get(ck)
        if cached and cached.get("key") == key:
            row = dict(cached["row"])
            if not row.get("created"):  # cache predates the created column
                row["created"] = created_reg.get(name) or _dir_stats(d)[1]
        else:
            size_mb, created_est, modified, n_html = _dir_stats(d)
            status, tldr = notes_idx.get(name, ("", ""))
            # freeze created on first observation; never let a republish move it
            created = created_reg.get(name) or created_est
            row = {"name": name, "tier": tier,
                   "title": _title_from_index(idx, name),
                   "created": created, "modified": modified,
                   "size_mb": size_mb, "n_html": n_html, "url": url,
                   "notes": (f"{status} — {tldr}" if status and tldr
                             else (status or tldr))}
        created_reg[name] = row["created"]  # persist the frozen value
        new_cache[ck] = {"key": key, "row": row}
        rows.append(row)

    for d in sorted(WEBDIR.iterdir()):
        if not d.is_dir() or d.name in ROOT_EXCLUDE or d.name.startswith("."):
            continue
        idx = d / "index.html"
        if idx.exists() and not _is_redirect_stub(idx):
            found_root.add(d.name)
        one(d, "root", f"{BASE_URL}/{d.name}/index.html")

    if PREVIEW_DIR.exists():
        for d in sorted(PREVIEW_DIR.iterdir()):
            if (not d.is_dir() or d.name.startswith(".")
                    or d.name == CONSOLE_STAGING_NAME
                    or d.name in found_root):  # stale copy of a promoted page
                continue
            one(d, "preview", f"{BASE_URL}/_preview/{d.name}/index.html")

    # lightgen-specific tier: daily-report / team-update pages, one level
    # deeper on disk (PUBLISH_DEST/updates/<date>/index.html) — see module
    # docstring. name carries the "updates/" prefix so it matches the SAME
    # names used in web/pages.yaml's curation entries (name: updates/2026-07-19).
    if UPDATES_DIR.exists():
        for d in sorted(UPDATES_DIR.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            one(d, "update", f"{BASE_URL}/updates/{d.name}/index.html",
                name=f"updates/{d.name}")

    if use_cache:
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(json.dumps(new_cache))
        except OSError:
            pass
    _save_created(created_reg)
    return rows


# ------------------------------------------------------------- curation ----
def load_curation():
    """web/pages.yaml curation layer: name -> {tags, important, blurb}."""
    try:
        import yaml
        data = yaml.safe_load(PAGES_YAML.read_text()) or {}
    except (OSError, ValueError, ImportError):
        return {}
    out = {}
    for p in data.get("pages", []):
        meta = {"tags": p.get("tags") or [],
                "important": bool(p.get("important")),
                "blurb": p.get("blurb") or ""}
        name = p["name"]
        out[name] = meta
        # Preview pages are keyed path-like in pages.yaml ("_preview/foo",
        # matching the `updates/<date>` convention) but scan_rows names a
        # preview row by its bare dir name, so the join silently missed and
        # tags/star/blurb never rendered for ANY preview page. Alias both
        # spellings rather than renaming the rows (the tier badge already
        # says "preview"; a prefixed name column would just duplicate it).
        # Safe against collision: scan_rows skips a preview dir whose name
        # also exists at root, so bare names stay unique across tiers.
        if name.startswith("_preview/"):
            out.setdefault(name[len("_preview/"):], meta)
    return out


# ------------------------------------------------------------- rendering ----
IMP_ATTR = ' class="db-imp" title="important page"'


def render_data_rows(rows):
    """Pre-rendered cell values (HTML strings + raw numerics) in COLUMNS
    order, newest-modified first. Server-side rendering keeps the client
    init dumb and the delegated chrome handlers unchanged."""
    cur = load_curation()
    out = []
    for r in sorted(rows, key=lambda r: r["modified"], reverse=True):
        meta = cur.get(r["name"], {})
        tags = meta.get("tags", [])
        imp = meta.get("important", False)
        blurb = meta.get("blurb", "")
        name_html = (f'<a href="{r["url"]}"{IMP_ATTR if imp else ""}'
                     f' target="_blank" rel="noopener">{html.escape(r["name"])}</a>')
        tier_html = f'<span class="badge tier-{r["tier"]}">{r["tier"]}</span>'
        title_html = (f'<span title="{html.escape(blurb, quote=True)}">'
                      f'{html.escape(r["title"])}</span>'
                      if blurb else html.escape(r["title"]))
        n = _plain(r["notes"] or "")
        if n:
            title_html += f'<span class="db-subline">{html.escape(n)}</span>'
        tags_html = " ".join(
            f'<span class="db-tag" data-tag="{html.escape(t, quote=True)}">'
            f'{html.escape(t)}</span>' for t in tags)
        out.append([name_html, tier_html, title_html, tags_html,
                    r["created"], r["modified"], r["size_mb"], r["n_html"],
                    "pinned-important" if imp else ""])
    return out


def write_manifest(quiet: bool = True) -> bool:
    """Scan (cached) + curation join -> PUBLISH_DEST/pages.json. Returns True
    if the manifest changed; appends a one-line sweep-log entry when it does."""
    rows = scan_rows(use_cache=True)
    data = render_data_rows(rows)
    new = {"columns": COLUMNS, "data": data}
    try:
        old = json.loads(MANIFEST.read_text())
        unchanged = (old.get("columns") == new["columns"]
                     and old.get("data") == new["data"])
    except (OSError, ValueError):
        unchanged = False
    if unchanged:
        return False
    new["generated"] = datetime.datetime.now().isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps(new))
    try:
        MANIFEST.chmod(MANIFEST.stat().st_mode | 0o004)
        SWEEP_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SWEEP_LOG.open("a") as f:
            f.write(f"{new['generated']} pages.json updated: {len(data)} pages\n")
    except OSError:
        pass
    if not quiet:
        print(f"pages.json updated ({len(data)} pages)", file=sys.stderr)
    return True


# ------------------------------------------------------------- fragment ----
# scrollX OFF (2026-08-09 redesign): DataTables' split-scroll header/body
# containers could desync (drag one, the header doesn't follow -- caught
# live on the jobs table, a column's label landed over a different column's
# data), and paired with the "nowrap" class it clipped genuinely long
# free-text fields mid-glyph instead of wrapping them. A single flowing
# table (scrollX off, no "nowrap" class, wrap-by-default CSS in theme3.css)
# can't desync against itself. size_mb/n_html stay in the data (so search
# still reaches them) but are hidden from view -- see COLUMNS/HEAD_CLASSES.
DT_ARGS = {
    "classes": ["display", "compact", "hover", "results", "dbtable"],
    "layout": {"topStart": "pageLength", "topEnd": "search",
               "bottomStart": "info", "bottomEnd": "paging"},
    "pageLength": 25,
    "autoWidth": False,
    "scrollX": False,
    # pinned-first (hidden importance sentinel), then newest-modified
    "order": [[8, "desc"], [5, "desc"]],
    "columnDefs": [
        {"targets": 1, "className": "dt-nowrap"},
        {"targets": 3, "className": "dt-hide-narrow"},
        {"targets": 4, "className": "dt-nowrap dt-right dt-hide-narrow"},
        {"targets": 5, "className": "dt-nowrap dt-right dt-hide-narrow"},
        {"targets": 6, "visible": False},  # size_mb (searchable, not shown)
        {"targets": 7, "visible": False},  # n_html (searchable, not shown)
        {"targets": 8, "visible": False},  # importance sentinel (searchable)
    ],
    "text_in_header_can_be_selected": True,
    "style": {"caption-side": "bottom", "margin": "auto",
              "table-layout": "auto", "width": "100%"},
}

CHROME_JS = """<script>
document.addEventListener('click', function (e) {
  var th = e.target.closest('th.dt-orderable-asc, th.dt-orderable-desc');
  if (!th || e.target.closest('.dt-column-order') ||
      e.target.closest('a, button, input')) return;
  var ind = th.querySelector('.dt-column-order');
  if (ind) ind.dispatchEvent(new MouseEvent('click',
      {bubbles: true, shiftKey: e.shiftKey}));
});
function dtSearch(v) {
  var inp = document.querySelector('.dt-search input');
  if (!inp) return;
  inp.value = v;
  inp.dispatchEvent(new Event('input', {bubbles: true}));
}
document.addEventListener('click', function (e) {
  var chip = e.target.closest('.db-tag');
  if (chip) dtSearch(chip.getAttribute('data-tag'));
});
(function () {
  var tries = 0;
  var timer = setInterval(function () {
  var search = document.querySelector('.dt-search');
  if (!search) { if (++tries > 60) clearInterval(timer); return; }
  clearInterval(timer);
  var btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'db-imp-toggle';
  btn.innerHTML = '&#9733; important only';
  var saved = '';
  btn.addEventListener('click', function () {
    var inp = document.querySelector('.dt-search input');
    var on = btn.classList.toggle('on');
    if (on) { saved = inp.value === 'pinned-important' ? '' : inp.value; dtSearch('pinned-important'); }
    else { dtSearch(saved); }
  });
  search.appendChild(btn);
  }, 150);
})();
</script>"""


def render_table_fragment():
    """The Pages-tab widget: offline DT bundle bootstrap + an EMPTY skeleton
    table + an init module that fetches /pages.json at load and constructs
    the ITable from it. No data is baked into the page; the manifest is the
    single live source. Requires .venv_itables (for the bundle only)."""
    import itables.javascript as ij

    dt_bundle = ij.opt.dt_bundle
    init_datatables = ij.read_package_file("html/init_datatables.html")
    connected_import = ("import { set_or_remove_dark_class } from '"
                        + ij.UNPKG_DT_BUNDLE_URL_NO_VERSION + "';")
    local_import = ("const { set_or_remove_dark_class } = await window."
                    + ij._ITABLES_UNDERSCORE_VERSION + ";")
    init_datatables = ij.replace_value(init_datatables, connected_import, local_import)
    offline_bundle_html = ij.generate_init_offline_itables_html(dt_bundle)

    version_var = ij._ITABLES_UNDERSCORE_VERSION           # e.g. _itables_2_8_1
    ready_event = ("itables-" + version_var.replace("_itables_", "")
                   .replace("_", ".") + "-ready")
    thead = "".join(f'<th class="{cls}">{html.escape(c)}</th>' if cls else f"<th>{html.escape(c)}</th>"
                    for c, cls in zip(COLUMNS, HEAD_CLASSES))
    dt_args = dict(DT_ARGS)
    dt_args["table_html"] = f"<table><thead><tr>{thead}</tr></thead></table>"

    skeleton = ('<table id="pagesdb"><tbody><tr><td class="db-loading">'
                'Loading the live page inventory&hellip;</td></tr></tbody></table>'
                '<noscript><p>JavaScript is required for the table; the raw '
                f'inventory is at <a href="{SITE_ROOT}/pages.json">pages.json'
                '</a>.</p></noscript>')

    init_script = f"""<script type="module">
    (async () => {{
        async function init() {{
            const {{ ITable }} = await window.{version_var};
            const table = document.querySelector("#pagesdb:not(.dataTable)");
            if (!table) return;
            const resp = await fetch("{SITE_ROOT}/pages.json", {{ cache: "no-cache" }});
            const manifest = await resp.json();
            let dt_args = {json.dumps(dt_args)};
            dt_args["data_json"] = JSON.stringify(manifest.data);
            new ITable(table, dt_args);
        }}
        if (window.{version_var}) {{
            init();
        }} else {{
            window.addEventListener("{ready_event}", () => {{ init(); }});
        }}
    }})();
</script>"""

    return init_datatables + offline_bundle_html + skeleton + init_script + CHROME_JS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write the Pages-tab HTML fragment here")
    ap.add_argument("--manifest", "--refresh", dest="manifest", action="store_true",
                    help="refresh PUBLISH_DEST/pages.json (cached scan; logs on change)")
    args = ap.parse_args()
    if not args.out and not args.manifest:
        ap.error("need --out and/or --manifest")

    if args.manifest:
        changed = write_manifest(quiet=False)
        if not changed:
            print("pages.json unchanged", file=sys.stderr)

    if args.out:
        rows = scan_rows(use_cache=True)
        print(f"inventory_pages: {len(rows)} pages "
              f"({sum(1 for r in rows if r['tier']=='root')} root, "
              f"{sum(1 for r in rows if r['tier']=='preview')} preview, "
              f"{sum(1 for r in rows if r['tier']=='update')} update)",
              file=sys.stderr)
        fragment = render_table_fragment()
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(fragment)
        print(f"wrote {out_path} ({len(fragment)} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
