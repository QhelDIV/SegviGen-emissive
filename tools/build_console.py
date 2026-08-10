#!/usr/bin/env python3
"""build_console.py — the Lightgen Console, a thin driver over xgpage.console.

Aggregates the project's living documents, experiments, and visuals into one
navigable console site, published to aspis. One URL to see and think from:

    https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/index.html

Usage:
    .venv_console/bin/python tools/build_console.py             # build -> staging (PUBLISH_DEST/_preview/console_v11/)
    .venv_console/bin/python tools/build_console.py --publish   # build -> live root (PUBLISH_DEST)
    .venv_console/bin/python tools/build_console.py --watch     # poll + rebuild every ~2s (staging)
    .venv_console/bin/python tools/build_console.py --watch --publish   # poll + rebuild live

Needs `markdown` + `pymdown-extensions` (the `xgpage[console]` extra) in
`.venv_console`, and `itables==2.8.1` + `PyYAML` in the separate `.venv_itables`
(Pages-tab database fragment only — see inventory_pages.py's module docstring).
Recreate with:
    python3.11 -m venv .venv_console && .venv_console/bin/pip install -e '/localhome/xya120/studio/xgpage[console]'
    python3.11 -m venv .venv_itables && .venv_itables/bin/pip install itables==2.8.1 PyYAML

Architecture (v14, 2026-07-23): the console GENRE — the v3 shell + page tree,
the living-document parsers, the KPI/notes/doc renderers, and the CLI harness —
now lives in the standalone xgpage package as `xgpage.console` (mirrors somages'
`tools/build_console.py`, the reference thin driver, read in full before this
rewrite). This file is the lightgen DRIVER: it fills a ConsoleConfig with the
project's COPY-variant paths and identity, defines the page tree (Console /
Workspace / Updates / Project docs — injecting the workspace-zone group AND the
per-day Updates group, neither of which the generic module knows about), keeps
the project-specific scanners (the itables pages database, the hand-synced
experiments table, the Team-updates/Current-highlights pages.yaml renderers),
and composes the tabs from the generic primitives. Migration history (v11-v13.2)
lives in the project-console skill's version log; what remains here is
lightgen-specific content and composition, not engine mechanics.

Preserved DELIBERATELY DIFFERENT from the somages driver (unchanged since v13):
  - BRIEF.md stays the Overview body (authored, milestone-cadence) rather than
    a live LOG.md headline — lightgen's BRIEF.md is actively maintained (see
    AGENTS.md's "ratified" markers) and there is no LOG.md here. Re-evaluate if
    BRIEF.md ever goes stale the way somages' did.
  - ROADMAP.md (not LOG.md) sources the Roadmap tab — it is ALREADY the
    actively-maintained fast-update file, not a rotted hand-written doc.
  - The Workspace group links directly to lightgen's own workspace_zone.py
    (LITE scope: zone + switcher + zone-link guard, no versioning/annotation
    — see that module's docstring); Updates links to per-day report pages
    (lightgen keeps one standalone page per day, no journal aggregator).

COPY-variant publishing (unchanged since v10): this repo lives on
local-scratch, which the aspis web server cannot traverse — no
symlink-from-www is possible. PUBLISH_DEST is therefore the real NFS www
directory (not `web/` in the repo); every write (staging AND publish) lands
directly under PUBLISH_DEST. Staging is `PUBLISH_DEST/_preview/console_v11/`
(servable, since a repo-local `web/_preview/` would NOT be); publish is
PUBLISH_DEST itself. `web/` in the repo holds only pages.yaml + README.md +
the generated (gitignored) assets/ staging copy — see sync_assets() below for
why this project needs an asset-sync step somages' driver does not.

Publish safety (unchanged): writes are file-by-file; NEVER rmtree the target
(a publish step once wiped every page, 2026-07-04).
"""
import datetime, html, pathlib, re, shutil, subprocess, sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))  # local modules below (workspace_zone, sync_xgpage_assets) — NOT xgpage
import xgpage as xg
from xgpage import console as xc
import workspace_zone as wz  # the research workspace zone (LITE scope, see its own docstring)

TEMPLATE_VERSION = 14
# v11-v13.2: see the project-console skill's version log for the MkDocs
#   retirement, the v3-shell re-port, and the workspace-zone addition.
# v14 (2026-07-23, this rewrite): the 707-line hand-rolled fork replaced by a
#   thin driver over xgpage.console (mirrors somages' v13.2 migration) — the
#   generic engine (v3 shell, tree rendering, living-document parsers, notes
#   list, CLI harness) now lives in the package; this file keeps only
#   lightgen-specific content, tree structure, and the two extra tab-adjacent
#   groups (Workspace, Updates) the generic module has no concept of.

SITE_ROOT = "/projects/omages/yanxg/lightgen"
BASE_URL = f"https://aspis.cmpt.sfu.ca{SITE_ROOT}"
CONSOLE_URL = f"{BASE_URL}/index.html"

# COPY variant (see module docstring): PUBLISH_DEST is the real NFS www dir,
# not a repo-local web/ folder. NEVER rmtree this path; every write is
# file-by-file merge (xc.write_page / shutil.copytree(..., dirs_exist_ok=True)).
PUBLISH_DEST = pathlib.Path("/project/3dlg-hcvc/omages/www/yanxg/lightgen")
STAGE_SUBDIR = "_preview/console_v11"
ASSETS_DIR = REPO / "web/assets"                            # repo-local sync target (hop 1, see sync_assets())
ASSETS_REL = f"{SITE_ROOT}/assets"                           # ...served from PUBLISH_DEST/assets (hop 2)
FAVICON = f"{SITE_ROOT}/assets/images/favicon.png"           # inherited from the old MkDocs build; harmless to keep reusing

NOTES_DIR = REPO / "notes"
UPDATES_DIR = PUBLISH_DEST / "updates"

MD_EXT = ["tables", "sane_lists"]

CONFIG = xc.ConsoleConfig(
    site_root=SITE_ROOT,
    base_url=BASE_URL,
    publish_dest=PUBLISH_DEST,
    stage_preview=PUBLISH_DEST / STAGE_SUBDIR,
    stage_subdir=STAGE_SUBDIR,
    assets_dir=ASSETS_DIR,
    assets_rel=ASSETS_REL,
    favicon=FAVICON,
    title="Lightgen",
    subtitle="console",
    watched_paths=[REPO / "BRIEF.md", REPO / "ROADMAP.md", REPO / "AGENTS.md",
                   REPO / "web/pages.yaml", NOTES_DIR, UPDATES_DIR],
    md_extensions=MD_EXT,
)

# Root reference docs shown as Project docs sub-tabs (skipped if missing).
REFERENCE_DOCS = [
    ("segvigen_emissive/WORKLOG.md", "worklog", "Worklog (autonomous run)"),
    ("todo.md", "todo", "Todo"),
    ("diffusionnet_project.md", "diffusionnet", "DiffusionNet project"),
    ("clarifications.md", "clarifications", "Clarifications"),
]


def md(text):
    """Project-bound markdown render (site-root absolutize)."""
    return xc.md(text, SITE_ROOT, MD_EXT)


def open_decisions(agents_text):
    """xc.parse_open_decisions() plus a lightgen-only filter: AGENTS.md marks
    a decision resolved IN PLACE with `~~...RESOLVED~~` rather than deleting
    the numbered item (keeps the resolution's reasoning on record), and the
    Overview/Roadmap tabs should show only genuinely open items. somages'
    AGENTS.md has no such convention (checked: zero RESOLVED markers there),
    so this filter has no equivalent in the generic package — it stays here,
    not upstreamed."""
    items = xc.parse_open_decisions(agents_text)
    return [x for x in items if not ("~~" in x and "RESOLVED" in x.upper())]


# ------------------------------------------------------------------ page tree ----
def _updates_tree_children(base):
    """Tree leaves for the console's "Updates" group: one leaf PER DAY
    (lightgen keeps a standalone page per day, unlike somages' single
    continuous journal), newest first. `key` = the date string (unique across
    the whole tree, so build_daily_report.py can mark its own leaf active by
    passing active_key=DATE — the generic xc.render_tree only matches by key,
    not by href). meta = the short tag from the page's own eyebrow line
    ("daily report · DATE · TAG" -> TAG), read directly off the built HTML so
    it never drifts from what the page itself says."""
    if not UPDATES_DIR.exists():
        return []
    rows = []  # (label, dirname, meta)
    for d in sorted(UPDATES_DIR.iterdir()):
        idx = d / "index.html"
        if not d.is_dir() or not idx.exists():
            continue
        head = idx.read_text(errors="ignore")[:3000]
        if "http-equiv=\"refresh\"" in head or "http-equiv='refresh'" in head:
            continue
        m = re.search(r'<div class="eyebrow">([^<]*)</div>', head)
        meta = ""
        if m:
            parts = [p.strip() for p in m.group(1).split("·")]
            if len(parts) >= 3:
                meta = parts[-1]
        date_m = re.search(r"(\d{4}-\d{2}-\d{2})", d.name)
        label = date_m.group(1) if date_m else d.name
        rows.append((label, d.name, meta))
    rows.sort(reverse=True)
    return [{"key": label, "label": label, "href": f"{SITE_ROOT}/updates/{dirname}/index.html",
             "meta": meta} for label, dirname, meta in rows]


def console_tree_entries(base):
    """The console's page tree. "Workspace" is the one-way switcher into the
    research workspace zone (wz.console_workspace_group() — the console
    sidebar is a SUPERSET of that zone; the zone itself never links back,
    enforced by workspace_zone.console_links_in() in build_workspace.py).
    "Updates" links directly to each day's report page (no journal
    aggregator here, unlike somages' daily zone)."""
    return [
        {"label": "Console", "children": [
            {"key": "overview", "label": "Overview", "href": f"{base}/index.html"},
            {"key": "roadmap", "label": "Roadmap", "href": f"{base}/roadmap.html"},
            {"key": "jobs", "label": "Jobs", "href": f"{base}/jobs.html"},
            {"key": "pages", "label": "Pages", "href": f"{base}/pages.html"},
            {"key": "graph", "label": "Graph", "href": f"{base}/graph.html"},
            {"key": "notes", "label": "Agent notes", "href": f"{base}/notes/index.html"},
        ]},
        wz.console_workspace_group(),
        {"label": "Updates", "children": _updates_tree_children(base)},
        {"label": "Project docs", "children": [
            {"key": "state", "label": "Agent state", "href": f"{base}/state.html"},
            {"key": "experiments", "label": "Experiments", "href": f"{base}/experiments.html"},
            {"key": "worklog", "label": "Worklog", "href": f"{base}/worklog.html"},
            {"key": "todo", "label": "Todo", "href": f"{base}/todo.html"},
            {"key": "diffusionnet", "label": "DiffusionNet project", "href": f"{base}/diffusionnet.html"},
            {"key": "clarifications", "label": "Clarifications", "href": f"{base}/clarifications.html"},
        ]},
    ]


def page(out_dir, title, active_key, body_html, wide=False, nav_title=None):
    """Compose one console page with the lightgen tree for this build target."""
    base = xc.console_base(CONFIG, out_dir)
    return xc.console_page(CONFIG, title, active_key, body_html,
                           console_tree_entries(base), base, wide=wide, nav_title=nav_title)


# ------------------------------------------------------------------ scanners ----
def scan_experiments():
    """Key-results table. Lightgen has no checkpoint-dir scan target (runs live
    on the cluster) — hand-embedded from WORKLOG.md's SUMMARY block. Sync
    manually from `segvigen_emissive/WORKLOG.md` SUMMARY when new runs land."""
    head = ("| run | data | epoch | train IoU@0.2 | val IoU@0.2 | notes |\n"
            "|---|---|---|---|---|---|\n")
    rows = [
        ("pilot (full FT)", "232", "ep25", "—", "0.203", "best zero-cond on 232 data"),
        ("pilot", "232", "ep50", "—", "0.042", "collapsed (paints nothing)"),
        ("v2 (oversample)", "232", "ep10", "—", "0.095", "rescues emissive-heavy at thr 0.2"),
        ("v2 (oversample)", "232", "ep30", "—", "0.119", "noisy 16-sample val"),
        ("**v3 (oversample)**", "**512**", "**ep4**", "**0.179**", "**0.176**",
         "**train ≈ val → zero-cond ceiling**"),
        ("v3 (oversample)", "512", "ep8", "0.145", "0.063", "majority-class collapse"),
        ("real-cond baseline", "232", "ep2", "—", "0.230", "job 226802; collapsed after"),
        ("zero-shot oracle", "—", "—", "—", "≈0.235", "frozen pretrained full_seg; the bar to beat"),
    ]
    body = "".join(f"| {' | '.join(r)} |\n" for r in rows)
    note = ("\n_Hand-synced from `segvigen_emissive/WORKLOG.md` SUMMARY (as of 2026-07-02). "
            "Update this table manually when Phase 4 results land._\n")
    return head + body + note


ITABLES_PYTHON = REPO / ".venv_itables/bin/python"
INVENTORY_SCRIPT = REPO / "tools/inventory_pages.py"


def scan_pages_table():
    """Database-view Pages tab: runs tools/inventory_pages.py under the
    dedicated `.venv_itables` venv as a subprocess and embeds its
    self-contained sortable/searchable HTML table fragment (offline
    DataTables bundle, no CDN — see inventory_pages.py for the constraints)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = pathlib.Path(tmpdir) / "pages_inventory_fragment.html"
        result = subprocess.run(
            [str(ITABLES_PYTHON), str(INVENTORY_SCRIPT), "--out", str(out_path),
             "--manifest"],  # console builds also refresh PUBLISH_DEST/pages.json
            cwd=str(REPO), capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"inventory_pages.py failed (exit {result.returncode}):\n{result.stderr}")
        print(f"[pages tab] {result.stderr.strip()}")
        return out_path.read_text()


# ------------------------------------------------------------------- tabs ----
def updates_list_html():
    """Overview's Team-updates list, newest first: reads the same
    updates/<date>/ directories the tree scan and inventory_pages both scan,
    with the blurb from web/pages.yaml when a curated entry exists. No
    xgpage.console equivalent — pages.yaml-driven, lightgen-specific."""
    import yaml
    reg = {"pages": []}
    reg_path = REPO / "web" / "pages.yaml"
    if reg_path.exists():
        reg = yaml.safe_load(reg_path.read_text()) or reg
    by_name = {e["name"]: e for e in reg.get("pages", []) if e.get("name")}

    if not UPDATES_DIR.exists():
        return '<p class="sub">(no updates yet — see <code>tools/build_daily_report.py</code>)</p>'
    rows = []  # (date_key, name, title, blurb)
    for d in sorted(UPDATES_DIR.iterdir()):
        idx = d / "index.html"
        if not d.is_dir() or not idx.exists():
            continue
        head = idx.read_text(errors="ignore")[:2000]
        if "http-equiv=\"refresh\"" in head or "http-equiv='refresh'" in head:
            continue
        m = re.search(r"<title>(.*?)</title>", head, re.S)
        title = (m.group(1).strip() if m else d.name)
        name = f"updates/{d.name}"
        blurb = by_name.get(name, {}).get("blurb", "")
        date_m = re.search(r"(\d{4}-\d{2}-\d{2})", d.name)
        rows.append((date_m.group(1) if date_m else d.name, name, title, blurb))
    if not rows:
        return '<p class="sub">(no updates yet — see <code>tools/build_daily_report.py</code>)</p>'
    rows.sort(reverse=True)
    out = ['<div class="clist-group">']
    for date, name, title, blurb in rows:
        out.append(f'<a class="clist-item" href="{SITE_ROOT}/{name}/index.html">'
                    f'<div class="ci-title">{html.escape(title)}<span class="ci-meta">{date}</span></div>'
                    f'<div class="ci-blurb">{html.escape(blurb)}</div></a>')
    out.append('</div>')
    return "\n".join(out)


def current_highlights_html():
    """Overview's curated highlights: pages.yaml entries with important: true,
    looked up against a live scan of PUBLISH_DEST so a stale/removed page
    never dangles. No xgpage.console equivalent — pages.yaml-driven."""
    import yaml
    reg_path = REPO / "web" / "pages.yaml"
    reg = yaml.safe_load(reg_path.read_text()) if reg_path.exists() else {"pages": []}
    important = [e for e in (reg or {}).get("pages", []) if e.get("important")]
    if not important:
        return '<p class="sub">(no pages marked <code>important: true</code> in web/pages.yaml)</p>'
    out = ['<div class="clist-group">']
    for e in important:
        name = e["name"]
        idx = PUBLISH_DEST / name / "index.html"
        if not idx.exists():
            continue
        head = idx.read_text(errors="ignore")[:2000]
        m = re.search(r"<title>(.*?)</title>", head, re.S)
        title = (m.group(1).strip() if m else name)
        mtime = datetime.datetime.fromtimestamp(idx.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        out.append(f'<a class="clist-item" href="{SITE_ROOT}/{name}/index.html">'
                    f'<div class="ci-title">{html.escape(title)}<span class="ci-meta">updated {mtime}</span></div>'
                    f'<div class="ci-blurb">{html.escape(e.get("blurb", ""))}</div></a>')
    out.append('</div>')
    return "\n".join(out)


def build_overview(out_dir):
    agents = (REPO / "AGENTS.md").read_text()
    deadlines = xc.parse_deadlines(agents)
    decisions = open_decisions(agents)
    dec_html = "".join(
        xg.callout(f'<strong>Open decision {i}:</strong> {html.escape(d)}', warn=True)
        for i, d in enumerate(decisions, 1))
    brief_path = REPO / "BRIEF.md"
    brief_md_src = brief_path.read_text() if brief_path.exists() else "# Overview\n\n_(BRIEF.md missing — write it)_"
    brief_md_src = re.sub(r"^#\s+", "## ", brief_md_src, count=1)
    brief_html = md(brief_md_src)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    body = f'''
    <section class="console-header">
      <h1>Lightgen Console</h1>
      <p class="sub">built {now} · <a href="{CONSOLE_URL}">permalink</a></p>
      {xg.callout('<strong>&#128308; <a href="' + SITE_ROOT + '/roadmap.html">LIVE ROADMAP &rarr;</a></strong><br>'
                   'Real-time now / next / waiting-on-you.')}
      {xc.kpi_row(deadlines)}
      {dec_html}
    </section>
    <section>
      <h2>Team updates</h2>
      {updates_list_html()}
    </section>
    <section>{brief_html}</section>
    <section>
      <h2>Current highlights</h2>
      {current_highlights_html()}
    </section>
    <footer>Rebuild: <code>.venv_console/bin/python tools/build_console.py --publish</code></footer>
    '''
    xc.write_page(out_dir, "index.html", page(out_dir, "Lightgen Console", "overview", body))


def build_roadmap_tab(out_dir):
    """Kept as a live-source page sourced from ROADMAP.md — lightgen's
    ROADMAP.md is the ACTIVELY MAINTAINED fast-update file already, not a
    rotted hand-written doc; there is no LOG.md here to switch to."""
    from build_roadmap import pipeline_html as roadmap_pipeline_html
    md_text = (REPO / "ROADMAP.md").read_text()
    strip, md_clean = roadmap_pipeline_html(md_text)
    body_html = md(md_clean)
    if strip and "</h1>" in body_html:
        body_html = body_html.replace("</h1>", "</h1>\n" + strip, 1)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    body = f'''
    <section class="console-header">
      <p class="sub">Rendered at build time from <code>ROADMAP.md</code> · built {now} ·
      phone-bookmarkable auto-refreshing standalone version (same content, its own live page):
      <a href="{SITE_ROOT}/roadmap/index.html">roadmap/</a></p>
    </section>
    <section>{body_html}</section>
    '''
    xc.write_page(out_dir, "roadmap.html",
                  page(out_dir, "Roadmap — Lightgen Console", "roadmap", body, nav_title="Roadmap"))


def build_pages_tab(out_dir):
    table_fragment = scan_pages_table()
    body = f'''
    <section>
      <p class="sub">Every published page, root and web/_preview/ staging alike, plus
      every daily-report update. Click a name to open it; click any column header to
      sort; search covers every column. Newest-modified first by default.</p>
      <div class="dbwrap">{table_fragment}</div>
    </section>
    '''
    xc.write_page(out_dir, "pages.html",
                  page(out_dir, "Pages — Lightgen Console", "pages", body, wide=True))


JOBS_INVENTORY_SCRIPT = REPO / "tools/inventory_jobs.py"


def scan_jobs_table():
    """Database-view Jobs tab: same pattern as scan_pages_table() — runs
    tools/inventory_jobs.py under .venv_itables as a subprocess and embeds
    its self-contained sortable/searchable HTML table fragment (also
    refreshes PUBLISH_DEST/jobs.json via --manifest)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = pathlib.Path(tmpdir) / "jobs_inventory_fragment.html"
        result = subprocess.run(
            [str(ITABLES_PYTHON), str(JOBS_INVENTORY_SCRIPT), "--out", str(out_path),
             "--manifest"],
            cwd=str(REPO), capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"inventory_jobs.py failed (exit {result.returncode}):\n{result.stderr}")
        print(f"[jobs tab] {result.stderr.strip()}")
        return out_path.read_text()


def build_jobs_tab(out_dir):
    table_fragment = scan_jobs_table()
    body = f'''
    <section>
      <p class="sub">Every non-trivial job, one row: ongoing, frozen, or done. One file
      per job under <code>jobs/</code> in the ops repo, one writer per file; ongoing
      entries silent for more than 3 hours are flagged stale. Rendered from
      <code>jobs/</code> by <code>tools/build_jobs.py</code>.</p>
      <div class="dbwrap">{table_fragment}</div>
    </section>
    <script>setTimeout(()=>location.reload(),120000)</script>
    '''
    xc.write_page(out_dir, "jobs.html",
                  page(out_dir, "Jobs — Lightgen Console", "jobs", body, wide=True))


def build_notes_tabs(out_dir):
    base = xc.console_base(CONFIG, out_dir)
    note_paths = sorted(NOTES_DIR.glob("*.md"), reverse=True)
    notes = []  # (path, title, status, tldr, stale)
    for p in note_paths:
        txt = p.read_text()
        notes.append((p, xc.note_title(txt, fallback=p.stem), *xc.note_meta(txt)))
    html_notes = sorted(NOTES_DIR.glob("*.html"), reverse=True)

    items = [(f"{base}/notes/{p.stem}.html", title, status, tldr, stale)
             for p, title, status, tldr, stale in notes]
    notes_html = xc.notes_list(items)
    if html_notes:
        extra = ['<div class="clist-group"><h2>Interactive pages</h2>']
        for p in html_notes:
            extra.append(f'<a class="clist-item" href="{base}/notes/{p.name}" '
                         f'data-filter-item="notes" data-filter-text="{html.escape(p.stem.lower())}">'
                         f'<div class="ci-title">{html.escape(p.stem)}'
                         f'<span class="ci-meta">standalone HTML ↗</span></div></a>')
        extra.append('</div>')
        notes_html += "".join(extra)

    body = f'''
    <section>
      <p class="sub">Working notes, newest first. Superseded / archived notes are dimmed.</p>
      {notes_html}
    </section>
    '''
    xc.write_page(out_dir, "notes/index.html",
                  page(out_dir, "Agent notes — Lightgen Console", "notes", body))

    for p, title, status, tldr, stale in notes:
        # em-dash separator: the console is the operator zone (exempt from the
        # em-dash guard) and this keeps note pages byte-identical to the
        # pre-migration baseline. The package default is a colon.
        note_body = xc.transform_note(p.read_text(), md, status_sep=" — ")
        body = f'<p><a href="{base}/notes/index.html">&larr; all notes</a></p>{note_body}'
        xc.write_page(out_dir, f"notes/{p.stem}.html", page(out_dir, title, "notes", body))

    for p in html_notes:
        (pathlib.Path(out_dir) / "notes").mkdir(parents=True, exist_ok=True)
        shutil.copy(p, pathlib.Path(out_dir) / "notes" / p.name)


def build_doc_pages(out_dir):
    current_state = (xc.extract_section((REPO / "AGENTS.md").read_text(),
                                        "Current State", end=r"\n---")
                     or "(parse failed — read AGENTS.md)")
    state_md = ("# Agent state (Current State)\n\nAgent-orientation snapshot parsed from "
                "AGENTS.md — dense by design; the human overview is the console Overview tab.\n\n"
                + current_state)
    xc.write_page(out_dir, "state.html",
                  page(out_dir, "Agent state — Lightgen Console", "state",
                       f'<section>{md(state_md)}</section>'))

    exp_md = ("# Experiments & results\n\nHand-synced from `segvigen_emissive/WORKLOG.md`. "
              "Update when new runs land.\n\n" + scan_experiments())
    xc.write_page(out_dir, "experiments.html",
                  page(out_dir, "Experiments — Lightgen Console", "experiments",
                       f'<section>{md(exp_md)}</section>'))

    for src, dst, title in REFERENCE_DOCS:
        src_path = REPO / src
        if not src_path.exists():
            continue
        doc_html = md(src_path.read_text())
        xc.write_page(out_dir, f"{dst}.html",
                      page(out_dir, f"{title} — Lightgen Console", dst,
                           f'<section>{doc_html}</section>'))


def build_jobs_redirect(out_dir):
    """The Jobs tab used to live at jobs/index.html (a standalone xgpage v2
    page); now jobs.html at the console root, console-genre (see
    build_jobs_tab). Old links keep working via this stub. Only written
    against the real publish dest (mirrors build_console_redirect); never
    touch any OTHER file under jobs/ (never rmtree)."""
    if pathlib.Path(out_dir) != PUBLISH_DEST:
        return
    stub = '''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url=../jobs.html">
<script>location.replace("../jobs.html");</script>
<title>Jobs (moved)</title>
</head>
<body>
<p>The Jobs board has moved. <a href="../jobs.html">Click here</a> if you are not redirected automatically.</p>
</body>
</html>'''
    xc.write_page(out_dir, "jobs/index.html", stub)


def build_console_redirect(out_dir):
    """The console used to live at /console/index.html (pre-v10); rewritten
    idempotently on every build — never touch any OTHER file under console/."""
    if pathlib.Path(out_dir) != PUBLISH_DEST:
        return
    stub = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url=../index.html">
<title>Lightgen Console (moved)</title>
</head>
<body>
<p>The Lightgen Console has moved to the project root. <a href="../index.html">Click here</a> if you are not redirected automatically.</p>
</body>
</html>'''
    xc.write_page(out_dir, "console/index.html", stub)


# -------------------------------------------------------------------- build ----
def sync_assets():
    """Two-hop COPY-variant asset sync — somages' driver does NOT need this
    (its folder-model publish_dest IS the real web root), but lightgen's repo
    lives on unservable local-scratch storage, so PUBLISH_DEST is a separate
    NFS directory:
      1. xgpage package -> web/assets/ (tools/sync_xgpage_assets.sync(), which
         ALSO patches back the lightgen-local model-viewer CSS/JS fragments —
         NEVER call xgpage.publish.publish_assets() directly here, or the
         patch gets skipped and the 3D lightbox silently breaks, exactly as
         happened once already during the package migration).
      2. web/assets/ -> PUBLISH_DEST/assets/ (merge-copy; this hop is what
         makes the COPY variant work at all).
    NEVER rmtree either destination."""
    import sync_xgpage_assets
    sync_xgpage_assets.sync()
    dest = PUBLISH_DEST / "assets"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ASSETS_DIR, dest, dirs_exist_ok=True)


def build(out_dir):
    sync_assets()
    base = xc.console_base(CONFIG, out_dir)
    xc.write_tree_json(out_dir, console_tree_entries(base), CONFIG.title, CONFIG.subtitle)
    build_overview(out_dir)
    build_roadmap_tab(out_dir)
    # Jobs before Pages (2026-08-10 ordering fix): inventory_pages.py's
    # job-page join (_load_jobs_index()) reads PUBLISH_DEST/jobs.json back
    # to chip a page's row with the job that produced it, so a full build
    # must refresh jobs.json before scanning pages, or the pages tab reads
    # the PREVIOUS build's jobs data. build_jobs_tab() is the only writer
    # of jobs.json here and has no dependency on anything build_pages_tab()
    # produces, so a straight reorder is the whole fix.
    build_jobs_tab(out_dir)
    build_jobs_redirect(out_dir)
    build_pages_tab(out_dir)
    # Lazy import (not module-level): tools/build_graph.py imports THIS module
    # (`import build_console as bc`) to reuse console_page/console_tree_entries/
    # CONFIG, so a top-level import here would be circular. By the time build()
    # runs, this module has already finished loading, so the nested import
    # resolves cleanly. See build_graph.py's module docstring for the rest.
    #
    # Fail-soft (2026-08-10, owner-requested): build() is a SHARED path every
    # agent's console rebuild goes through; an in-development bug in the
    # graph tab must never take down everyone else's Overview/Roadmap/Jobs/
    # Pages publish. The graph tab's own errors are caught and logged here;
    # every OTHER tab still built above/below is unaffected either way.
    try:
        import build_graph as bg
        bg.build_graph_tab(out_dir)
    except Exception as e:
        print(f"[graph tab] build FAILED, continuing without it: {e}", file=sys.stderr)
    build_notes_tabs(out_dir)
    build_doc_pages(out_dir)
    build_console_redirect(out_dir)


if __name__ == "__main__":
    xc.run_cli(CONFIG, build)
