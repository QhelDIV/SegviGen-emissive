#!/usr/bin/env python3
"""build_console.py — the Lightgen Console (xgpage engine, v13 / v3-shell).

Aggregates the project's living documents, experiments, and visuals into one
navigable console site, published to aspis. One URL to see and think from:

    https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/index.html

Usage:
    .venv_console/bin/python tools/build_console.py             # build -> staging (PUBLISH_DEST/_preview/console_v11/)
    .venv_console/bin/python tools/build_console.py --publish   # build -> live root (PUBLISH_DEST)
    .venv_console/bin/python tools/build_console.py --watch     # poll + rebuild every ~2s (staging)
    .venv_console/bin/python tools/build_console.py --watch --publish   # poll + rebuild live

Needs `markdown` + `PyYAML` in `.venv_console`, and `itables==2.8.1` + `PyYAML`
in the separate `.venv_itables` (Pages-tab database fragment only — see
inventory_pages.py's module docstring for why it's a dedicated venv, kept out
of .venv_console per project env rules). Recreate with:
    python3.11 -m venv .venv_console && .venv_console/bin/pip install markdown PyYAML
    python3.11 -m venv .venv_itables && .venv_itables/bin/pip install itables==2.8.1 PyYAML

Migration history:
  v11 (2026-07-19, this session): MkDocs retired; rebuilt on xgpage's v1/v2
    component library, nav_tabs()/nav_subtabs() shell (see project-console
    skill's v11 template — now KNOWN STALE, see v13 note below).
  v13 (2026-07-19, SAME SESSION, course-corrected): re-ported onto the
    xgpage v3 "workspace" shell (left page tree, centered content column,
    right per-page outline — see xgpage SKILL.md's v3 registry entry),
    matching what the somages console ACTUALLY runs (verified live +
    against its tools/build_console.py, v13/v13.1). The v11 template this
    was first built from turned out to be stale; see the upstreamed
    template/SKILL.md fix for the correction on record. Adopted from
    somages: the v3_tree() left-nav shell, the itables-backed Pages-tab
    DATABASE view (tools/inventory_pages.py, ported — see its own module
    docstring for the COPY-variant path adaptations), theme3.css/xg3.js
    (synced verbatim — xgpage.py itself had NO lightgen-specific fork
    beyond the .v3d model-viewer additions, which live only in theme.css/
    ui.js, not xgpage.py, so the whole v3 API came across as a clean sync).
  Explicitly OMITTED (lightgen has no advisor workspace zone — the console
    is the ONLY zone, so there is nothing to switch between and no one-way
    zone-boundary law to enforce): the workspace-zone tree group + its
    switcher (somages' workspace_zone.py has no lightgen analog — lightgen's
    "Updates" tree group links directly to each updates/<date>/ page, not
    to a separate zone), publish_version.py's version-minting/snapshot
    machinery (no /v/N/ immutable versions here), hypothes.is annotation.
  Deliberately KEPT DIFFERENT from the somages v13.1 pattern: BRIEF.md
    stays the Overview body (authored, milestone-cadence) rather than being
    retired in favor of a live LOG.md headline — v13.1 retired somages'
    BRIEF because it had ROTTED (unmaintained, silently stale); lightgen's
    BRIEF.md is actively maintained (see AGENTS.md's "ratified" markers) and
    lightgen has no LOG.md to switch to. Re-evaluate this call if BRIEF.md
    ever goes stale the way somages' did.

COPY-variant publishing (unchanged since v10/v11 — see the project-console
skill's "Publishing" section, "Storage constraint" paragraph): this repo
lives on local-scratch, which the aspis web server cannot traverse — no
symlink-from-www is possible. PUBLISH_DEST is therefore the real NFS www
directory (not `web/` in the repo); every write (staging AND publish) lands
directly under PUBLISH_DEST. Staging is `PUBLISH_DEST/_preview/console_v11/`
(servable, since a repo-local `web/_preview/` would NOT be); publish is
PUBLISH_DEST itself. `web/` in the repo holds only pages.yaml + README.md.

Publish safety (unchanged): writes are file-by-file; NEVER rmtree the target
(a publish step once wiped every page, 2026-07-04). Shared assets (theme.css,
theme2.css, theme3.css, ui.js, xg2.js, xg3.js, katex/, model-viewer.min.js)
are the repo's source of truth but SERVED from PUBLISH_DEST/assets — every
build() call merge-copies web/assets/ into PUBLISH_DEST/assets/ so an asset
edit goes live regardless of which target (staging or publish) was built.
"""
import argparse, datetime, html, json, pathlib, re, shutil, subprocess, sys, time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))  # xgpage.py's home in this project
import xgpage as xg
from build_roadmap import pipeline_html as roadmap_pipeline_html  # reuse the `## Pipeline` -> strip parser verbatim
import workspace_zone as wz  # the research workspace zone (added 2026-07-19, LITE scope)

TEMPLATE_VERSION = 13
# v13.2 (2026-07-19): added the research WORKSPACE zone (workspace_zone.py +
# tools/build_workspace.py) — a second, advisor-facing v3-tree zone, one-way
# switcher (wz.console_workspace_group()) into it from here. LITE scope, owner
# decision: zone + tree + switcher + the zone-link guard only; no per-page
# versioning, no hypothes.is — see workspace_zone.py's module docstring.

SITE_ROOT = "/projects/omages/yanxg/lightgen"
BASE_URL = f"https://aspis.cmpt.sfu.ca{SITE_ROOT}"
CONSOLE_URL = f"{BASE_URL}/index.html"

# COPY variant (see module docstring): PUBLISH_DEST is the real NFS www dir,
# not a repo-local web/ folder. NEVER rmtree this path; every write is
# file-by-file merge (write_page / shutil.copytree(..., dirs_exist_ok=True)).
PUBLISH_DEST = pathlib.Path("/project/3dlg-hcvc/omages/www/yanxg/lightgen")
STAGE_PREVIEW = PUBLISH_DEST / "_preview" / "console_v11"  # servable staging (see docstring)
ASSETS_DIR = REPO / "web/assets"                            # repo source of truth
ASSETS_REL = f"{SITE_ROOT}/assets"                           # ...served from PUBLISH_DEST/assets (synced in build())
FAVICON = f"{SITE_ROOT}/assets/images/favicon.png"           # inherited from the old MkDocs build; harmless to keep reusing

NOTES_DIR = REPO / "notes"
UPDATES_DIR = PUBLISH_DEST / "updates"

# Root reference docs shown as Project docs sub-tabs (skipped if missing).
REFERENCE_DOCS = [
    ("segvigen_emissive/WORKLOG.md", "worklog", "Worklog (autonomous run)"),
    ("todo.md", "todo", "Todo"),
    ("diffusionnet_project.md", "diffusionnet", "DiffusionNet project"),
    ("clarifications.md", "clarifications", "Clarifications"),
]

MD_EXT = ["tables", "sane_lists"]


_BARE_HREF_RE = re.compile(r'href="(?!https?:|#|mailto:|/)([^"]+)"')


def md(text):
    import markdown  # deferred: only present in .venv_console, see module docstring
    out = markdown.markdown(text, extensions=MD_EXT)
    return _BARE_HREF_RE.sub(lambda m: f'href="{SITE_ROOT}/{m.group(1)}"', out)


def _hash8(path):
    import hashlib
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()[:8]
    except OSError:
        return "0"


# ------------------------------------------------------------------ nav shell ----
def console_base(out_dir):
    return SITE_ROOT if out_dir == PUBLISH_DEST else f"{SITE_ROOT}/_preview/console_v11"


def _updates_tree_children(base):
    """Tree leaves for the console's "Updates" group: one leaf PER DAY
    (lightgen keeps a standalone page per day, unlike somages' single
    continuous journal — see module docstring), newest first. Label = date
    (parsed from the dirname), meta = the short tag from the page's own
    eyebrow line ("daily report · DATE · TAG" -> TAG), read directly off the
    built HTML so it never drifts from what the page itself says."""
    if not UPDATES_DIR.exists():
        return []
    rows = []  # (date_sort_key, dirname, label, meta)
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
    return [{"label": label, "href": f"{SITE_ROOT}/updates/{dirname}/index.html",
             "meta": meta} for label, dirname, meta in rows]


def console_tree_entries(base):
    """The console's page tree. "Workspace" (added 2026-07-19) is the
    one-way switcher into the research workspace zone (wz.console_workspace_
    group() — the console sidebar is a SUPERSET of that zone, same pattern
    as somages; the zone itself never links back, enforced by
    workspace_zone.console_links_in() in build_workspace.py). "Updates"
    links directly to each day's report page (lightgen has no journal
    aggregator — see module docstring), not to a zone."""
    return [
        {"label": "Console", "children": [
            {"key": "overview", "label": "Overview", "href": f"{base}/index.html"},
            {"key": "roadmap", "label": "Roadmap", "href": f"{base}/roadmap.html"},
            {"key": "pages", "label": "Pages", "href": f"{base}/pages.html"},
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


def console_tree_html(base, active_key=None, active_href=None):
    """active_key marks a Console/Project-docs leaf by its `key`; active_href
    marks an Updates leaf by its exact href (dated leaves have no stable key
    across rebuilds, so build_daily_report.py passes its own URL instead)."""
    entries = console_tree_entries(base)
    for g in entries:
        for leaf in g["children"]:
            k = leaf.pop("key", None)
            if (active_key and k == active_key) or (active_href and leaf.get("href") == active_href):
                leaf["active"] = True
    return xg.v3_tree(entries, title="Lightgen", subtitle="console",
                      tree_src=f"{base}/console_tree.json")


def write_console_tree_json(out_dir):
    entries = console_tree_entries(console_base(out_dir))
    for g in entries:
        for leaf in g["children"]:
            leaf.pop("key", None)
    (out_dir / "console_tree.json").write_text(json.dumps({
        "title": "Lightgen", "subtitle": "console", "entries": entries}, indent=1))


def page_shell(title, active_top, body_html, base, active_sub=None, extra_head="",
               nav_title=None, wide=False):
    active_key = active_sub if active_top == "docs" else active_top
    head = f'<link rel="icon" href="{FAVICON}">' + extra_head
    return xg.page(
        title=title,
        body_sections=[body_html],
        theme="v3",
        wide=wide,
        tree_html=console_tree_html(base, active_key=active_key),
        nav_title=nav_title or title.split(" — ")[0],
        assets_rel=ASSETS_REL,
        assets_dir=ASSETS_DIR,
        extra_head=head,
    )


def write_page(out_dir, relpath, content):
    p = out_dir / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


# ------------------------------------------------------------------ parsers ----
def parse_deadlines():
    """`## Deadlines` list in AGENTS.md: `- YYYY-MM-DD — label`. Future ones only."""
    txt = (REPO / "AGENTS.md").read_text()
    m = re.search(r"## Deadlines(.*?)(\n## |\Z)", txt, re.S)
    out = []
    if m:
        for d, label in re.findall(r"-\s*(\d{4}-\d{2}-\d{2})\s*[—–-]+\s*(.+)", m.group(1)):
            days = (datetime.date.fromisoformat(d) - datetime.date.today()).days
            if days >= 0:
                out.append((days, label.strip()))
    return sorted(out)


def parse_open_decisions():
    """Numbered list under '**Open decisions...**' in Current State. Items
    already marked ~~...RESOLVED~~ are dropped (resolved-in-place markers)."""
    txt = (REPO / "AGENTS.md").read_text()
    m = re.search(r"\*\*Open decisions[^\n]*\*\*:?\s*\n(.*?)(\n\s*\n|\n---)", txt, re.S)
    if not m:
        return []
    items = re.findall(r"^\d+\.\s+(.+?)(?=^\d+\.|\Z)", m.group(1), re.S | re.M)
    items = [re.sub(r"\s+", " ", x).strip() for x in items]
    return [x for x in items if not ("~~" in x and "RESOLVED" in x.upper())]


def note_meta(md_path):
    lines = md_path.read_text().splitlines()
    status, tldr = "", ""
    for ln in lines[:8]:
        if m := re.match(r"\*{0,2}Status\*{0,2}:\s*(.+)", ln.strip()):
            status = m.group(1).strip()
            break
    for ln in lines:
        if m := re.match(r"\*{0,2}TL;?DR\*{0,2}:\s*(.+)", ln.strip(), re.I):
            tldr = m.group(1).strip()
            break
    if not tldr:
        for ln in lines[1:]:
            s = ln.strip()
            if s and not s.startswith(("#", "-", "*", "|", ">", "```")):
                tldr = s
                break
    if len(tldr) > 160:
        tldr = tldr[:157] + "…"
    stale = any(w in status.lower() for w in ("supersed", "archiv", "obsolete"))
    return status, tldr, stale


def note_title(md_path):
    for ln in md_path.read_text().splitlines():
        if m := re.match(r"#\s+(.+)", ln.strip()):
            return re.sub(r"\s*\(\d{4}-\d{2}-\d{2}\)\s*$", "", m.group(1)).strip()
    return md_path.stem


def section_current_state():
    txt = (REPO / "AGENTS.md").read_text()
    m = re.search(r"## Current State(.*?)\n---", txt, re.S)
    body = m.group(1) if m else "\n(parse failed — read AGENTS.md)"
    return body.strip()


def scan_experiments():
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


# --------------------------------------------------------- Pages tab (database) ----
ITABLES_PYTHON = REPO / ".venv_itables/bin/python"
INVENTORY_SCRIPT = REPO / "tools/inventory_pages.py"


def scan_pages_table():
    """Database-view Pages tab, ported from the somages v12+ design (user
    request there: "I want a database view — list all pages, sortable, incl.
    preview pages"). Runs tools/inventory_pages.py under the dedicated
    `.venv_itables` venv as a subprocess and embeds its self-contained
    sortable/searchable HTML table fragment (offline DataTables bundle, no
    CDN — see inventory_pages.py for the full constraint list)."""
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


# ------------------------------------------------------------- note transform ----
def transform_note_html(md_path):
    lines = md_path.read_text().splitlines()
    status = tldr = ""
    kept = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if i < 12 and (m := re.match(r"\*{0,2}Status\*{0,2}:\s*(.+)", s)):
            status = m.group(1).strip()
            continue
        if i < 12 and (m := re.match(r"\*{0,2}TL;?DR\*{0,2}:\s*(.+)", s, re.I)):
            tldr = m.group(1).strip()
            continue
        kept.append(re.sub(r"\[\[([^\]]+)\]\]", r"*\1*", ln))
    body_html = md("\n".join(kept))
    if status:
        stale = any(w in status.lower() for w in ("supersed", "archiv", "obsolete"))
        inner = f'<strong>Status — {html.escape(status)}</strong>'
        inner += f'<br>{html.escape(tldr)}' if tldr else '<br><em>(no TL;DR)</em>'
        badge = xg.callout(inner, warn=stale)
        body_html = body_html.replace("</h1>", "</h1>\n" + badge, 1) if "</h1>" in body_html else badge + body_html
    return body_html


# ------------------------------------------------------------------- tab builders ----
def kpi_html(deadlines):
    if not deadlines:
        return ""
    tiles = "".join(
        f'<div class="stat{" soon" if d <= 21 else ""}"><b>{d}</b>'
        f'<span>days · {html.escape(l)}</span></div>' for d, l in deadlines)
    return f'<div class="stat-row">{tiles}</div>'


def updates_list_html():
    """Overview's Team-updates list, newest first: reads the same
    updates/<date>/ directories the tree scan and inventory_pages both scan,
    with the blurb from web/pages.yaml when a curated entry exists."""
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
    never dangles."""
    import yaml
    reg_path = REPO / "web" / "pages.yaml"
    reg = yaml.safe_load(reg_path.read_text()) if reg_path.exists() else {"pages": []}
    important = [e for e in (reg or {}).get("pages", []) if e.get("important")]
    if not important:
        return '<p class="sub">(no pages marked <code>important: true</code> in web/pages.yaml)</p>'
    out = ['<div class="clist-group">']
    for e in important:
        name = e["name"]
        d = PUBLISH_DEST / name
        idx = d / "index.html"
        if not idx.exists():
            continue
        head = idx.read_text(errors="ignore")[:2000]
        m = re.search(r"<title>(.*?)</title>", head, re.S)
        title = (m.group(1).strip() if m else name)
        mtime = datetime.date.fromtimestamp(idx.stat().st_mtime)
        out.append(f'<a class="clist-item" href="{SITE_ROOT}/{name}/index.html">'
                    f'<div class="ci-title">{html.escape(title)}<span class="ci-meta">updated {mtime}</span></div>'
                    f'<div class="ci-blurb">{html.escape(e.get("blurb", ""))}</div></a>')
    out.append('</div>')
    return "\n".join(out)


def build_overview(out_dir):
    base = console_base(out_dir)
    deadlines = parse_deadlines()
    decisions = parse_open_decisions()
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
      {kpi_html(deadlines)}
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
    write_page(out_dir, "index.html",
               page_shell("Lightgen Console", "overview", body, base))


def build_roadmap_tab(out_dir):
    """Kept as a live-source page (unchanged in spirit from the somages
    v13.1 "situation board" idea, but sourced from ROADMAP.md — lightgen's
    ROADMAP.md is the ACTIVELY MAINTAINED fast-update file already, not a
    rotted hand-written doc; there is no LOG.md here to switch to)."""
    base = console_base(out_dir)
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
    write_page(out_dir, "roadmap.html",
               page_shell("Roadmap — Lightgen Console", "roadmap", body, base,
                          nav_title="Roadmap"))


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
    write_page(out_dir, "pages.html",
               page_shell("Pages — Lightgen Console", "pages", body,
                          console_base(out_dir), wide=True))


def build_notes_tabs(out_dir):
    base = console_base(out_dir)
    note_paths = sorted(NOTES_DIR.glob("*.md"), reverse=True)
    items = [(p, note_title(p), *note_meta(p)) for p in note_paths]
    html_notes = sorted(NOTES_DIR.glob("*.html"), reverse=True)

    out = ['<input class="filter-box" type="text" placeholder="Filter notes…" '
           'data-filter-input="notes" aria-label="Filter notes">',
           '<p class="clist-empty" data-filter-empty="notes">No notes match.</p>',
           '<div class="clist-group">']
    for p, title, status, tldr, stale in items:
        href = f"{base}/notes/{p.stem}.html"
        cls = "clist-item stale" if stale else "clist-item"
        status_line = f'<div class="ci-status"><span class="badge">{html.escape(status)}</span></div>' if status else ""
        text = f"{title} {status} {tldr}".lower()
        out.append(f'<a class="{cls}" href="{href}" data-filter-item="notes" '
                    f'data-filter-text="{html.escape(text)}">'
                    f'<div class="ci-title">{html.escape(title)}</div>{status_line}'
                    f'<div class="ci-blurb">{html.escape(tldr)}</div></a>')
    out.append('</div>')
    if html_notes:
        out.append('<div class="clist-group"><h2>Interactive pages</h2>')
        for p in html_notes:
            out.append(f'<a class="clist-item" href="{base}/notes/{p.name}" '
                        f'data-filter-item="notes" data-filter-text="{html.escape(p.stem.lower())}">'
                        f'<div class="ci-title">{html.escape(p.stem)}'
                        f'<span class="ci-meta">standalone HTML ↗</span></div></a>')
        out.append('</div>')

    body = f'''
    <section>
      <p class="sub">Working notes, newest first. Superseded / archived notes are dimmed.</p>
      {"".join(out)}
    </section>
    '''
    write_page(out_dir, "notes/index.html",
               page_shell("Agent notes — Lightgen Console", "notes", body, base))

    for p, title, status, tldr, stale in items:
        note_body = transform_note_html(p)
        body = f'<p><a href="{base}/notes/index.html">&larr; all notes</a></p>{note_body}'
        write_page(out_dir, f"notes/{p.stem}.html", page_shell(title, "notes", body, base))

    for p in html_notes:
        (out_dir / "notes").mkdir(parents=True, exist_ok=True)
        shutil.copy(p, out_dir / "notes" / p.name)


def build_doc_pages(out_dir):
    base = console_base(out_dir)

    state_md = ("# Agent state (Current State)\n\nAgent-orientation snapshot parsed from "
                "AGENTS.md — dense by design; the human overview is the console Overview tab.\n\n"
                + section_current_state())
    write_page(out_dir, "state.html",
               page_shell("Agent state — Lightgen Console", "docs", f'<section>{md(state_md)}</section>',
                           base, active_sub="state"))

    exp_md = ("# Experiments & results\n\nHand-synced from `segvigen_emissive/WORKLOG.md`. "
              "Update when new runs land.\n\n" + scan_experiments())
    write_page(out_dir, "experiments.html",
               page_shell("Experiments — Lightgen Console", "docs", f'<section>{md(exp_md)}</section>',
                           base, active_sub="experiments"))

    for src, dst, title in REFERENCE_DOCS:
        src_path = REPO / src
        if not src_path.exists():
            continue
        doc_html = md(src_path.read_text())
        write_page(out_dir, f"{dst}.html",
                   page_shell(f"{title} — Lightgen Console", "docs", f'<section>{doc_html}</section>',
                               base, active_sub=dst))


def build_console_redirect(out_dir):
    """The console used to live at /console/index.html (pre-v10); rewritten
    idempotently on every build — never touch any OTHER file under console/."""
    if out_dir != PUBLISH_DEST:
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
    write_page(out_dir, "console/index.html", stub)


# -------------------------------------------------------------------- build ----
def sync_assets():
    """Merge-copy the repo's asset source of truth into the servable NFS copy.
    NEVER rmtree."""
    dest = PUBLISH_DEST / "assets"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ASSETS_DIR, dest, dirs_exist_ok=True)


def build(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    sync_assets()
    write_console_tree_json(out_dir)
    build_overview(out_dir)
    build_roadmap_tab(out_dir)
    build_pages_tab(out_dir)
    build_notes_tabs(out_dir)
    build_doc_pages(out_dir)
    build_console_redirect(out_dir)
    for p in [out_dir, *out_dir.rglob("*")]:
        try:
            p.chmod(p.stat().st_mode | (0o005 if p.is_dir() else 0o004))
        except OSError:
            pass


WATCHED_DOCS = ["BRIEF.md", "ROADMAP.md", "AGENTS.md", "web/pages.yaml"]


def _snapshot():
    mtimes = {}
    for rel in WATCHED_DOCS:
        p = REPO / rel
        if p.exists():
            mtimes[rel] = p.stat().st_mtime
    mtimes["notes"] = max((p.stat().st_mtime for p in NOTES_DIR.glob("*")), default=0)
    mtimes["updates"] = (max((p.stat().st_mtime for p in UPDATES_DIR.rglob("*")), default=0)
                         if UPDATES_DIR.exists() else 0)
    return mtimes


def watch(out_dir):
    print(f"watching (2s poll) — rebuilding into {out_dir} on change to "
          f"BRIEF.md / ROADMAP.md / AGENTS.md / web/pages.yaml / notes/ / updates/. Ctrl-C to stop.")
    last = None
    while True:
        cur = _snapshot()
        if cur != last:
            t0 = time.time()
            build(out_dir)
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] rebuilt in {time.time() - t0:.2f}s")
            last = cur
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true",
                     help="build directly into the live PUBLISH_DEST root instead of the staging preview")
    ap.add_argument("--watch", action="store_true",
                     help="poll watched sources every ~2s and rebuild on change (Ctrl-C to stop)")
    args = ap.parse_args()
    out_dir = PUBLISH_DEST if args.publish else STAGE_PREVIEW

    if args.watch:
        build(out_dir)
        watch(out_dir)
        return

    build(out_dir)
    if args.publish:
        print(f"published: {out_dir}")
        print(f"URL: {CONSOLE_URL}")
    else:
        print(f"staged: {out_dir}")
        print(f"URL: {BASE_URL}/_preview/console_v11/index.html")


if __name__ == "__main__":
    main()
