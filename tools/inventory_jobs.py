#!/usr/bin/env python3
"""inventory_jobs.py — the Jobs board database: manifest + DataTables fragment.

Same architecture as inventory_pages.py (the console Pages tab), applied to the
jobs registry: jobs/*.md (one file per job, one writer per file) -> jobs.json
manifest served from PUBLISH_DEST, plus a sortable/searchable offline-DataTables
widget fragment that fetches it at load.

DATA MODEL (2026-08-09 log redesign, owner-directed): each job carries a
`motivation:` (one sentence, written at registration, shown under the title)
and an append-only `log:` block (one `- YYYY-MM-DD HH:MM <sentence>` line per
update, newest last -- see tools/build_jobs.py's docstring for the entry-format
contract agents write to). The table's "updated" column and staleness check are
DERIVED from the last log line's timestamp, not the file's `updated:` field
(kept in the file for a human skimming the raw text, but the log is now the
authoritative continuously-appended record -- an agent could forget to bump
`updated:` when appending a log line, but the line's own timestamp can't drift).
`slurm:` stays in the file as reference detail (job ids belong inside log
sentences, e.g. "training started as job 242211") but no longer gets a table
column. `outcome:` stays for done jobs: it is the final log entry's summary,
rendered with distinct styling both in the "latest" cell and the expanded
timeline (see render_timeline_html()).

Each row's full log renders as a DataTables CHILD ROW (native row.child() API,
reached via the jQuery instance the offline bundle exports alongside ITable --
see render_fragment()'s init script), toggled by clicking anywhere on the row
(except real links). The hidden `log_full` column carries the pre-rendered
timeline HTML AND makes the full log text searchable (DataTables searches
column data regardless of `visible`, the same mechanism the "important" and
"state-order" sentinel columns already relied on).

Modes:
    --manifest            jobs/*.md -> PUBLISH_DEST/jobs.json   (stdlib only)
    --out <fragment.html> the widget fragment (NEEDS .venv_itables)
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
JOBS = REPO / "jobs"
PUBLISH_DEST = pathlib.Path("/project/3dlg-hcvc/omages/www/yanxg/lightgen")
MANIFEST = PUBLISH_DEST / "jobs.json"
STALE_H = 3.0
ORDER = {"ongoing": 0, "frozen": 1, "done": 2}
BADGE_BG = {"ongoing": "#1d7a46", "frozen": "#8a6d1a", "done": "#5a6472"}

COLUMNS = ["job", "status", "executor", "latest", "started", "updated", "log_full", "state-order"]
# per-column <th> class, same order as COLUMNS (see render_fragment(); the
# matching <td> classes are applied via columnDefs' className below). Narrow
# width (see theme3.css's mobile block) keeps job/status/latest visible --
# the log-first point of this board -- and defers executor/started/updated.
HEAD_CLASSES = ["", "dt-nowrap", "dt-nowrap dt-hide-narrow", "",
                "dt-nowrap dt-right dt-hide-narrow",
                "dt-nowrap dt-right dt-hide-narrow", "", ""]

LOG_LINE_RE = re.compile(r"^-\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s+(.*)$")


def parse(path):
    """Plain "key: value" lines, plus a multi-line `log:` block: every line
    immediately after `log:` matching `- YYYY-MM-DD HH:MM <text>` is consumed
    into e["log"] (a list of (timestamp, text) tuples, file order == append
    order == chronological, newest last) until a line that doesn't match --
    normally the next `key:` field (`outcome:`)."""
    e = {"slug": path.stem, "log": []}
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^(\w+):\s*(.*)$", lines[i])
        if m and m.group(1).lower() == "log":
            i += 1
            while i < len(lines):
                lm = LOG_LINE_RE.match(lines[i])
                if not lm:
                    break
                e["log"].append((lm.group(1), lm.group(2).strip()))
                i += 1
            continue
        if m:
            e[m.group(1).lower()] = m.group(2).strip()
        i += 1
    return e


def age_h(ts):
    try:
        t = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M")
        return (datetime.datetime.now() - t).total_seconds() / 3600.0
    except Exception:
        return None


def render_timeline_html(log, has_outcome):
    """The child-row expansion: a formatted timeline, timestamps in their own
    column (CSS grid, see theme3.css .log-entry) so they align regardless of
    sentence length. The final entry of a job with a recorded outcome gets a
    distinct tag + accent (has_outcome is a status/outcome-field flag, not a
    text match, so it can't drift from what render_fragment() considers
    "done")."""
    n = len(log)
    parts = ['<div class="log-timeline">']
    for i, (ts, text) in enumerate(log):
        is_outcome = has_outcome and i == n - 1
        cls = "log-entry log-entry-outcome" if is_outcome else "log-entry"
        tag = '<span class="log-outcome-tag">outcome</span>' if is_outcome else ""
        parts.append(f'<div class="{cls}"><span class="log-ts">{html.escape(ts)}</span>'
                     f'<span class="log-text">{tag}{html.escape(text)}</span></div>')
    parts.append("</div>")
    return "".join(parts)


def rows():
    out = []
    for p in sorted(JOBS.glob("*.md")):
        e = parse(p)
        st = e.get("status", "ongoing")
        log = e["log"] or [(e.get("updated", ""), "")]  # defensive: pre-migration file
        has_outcome = bool(e.get("outcome", "").strip())
        last_ts, last_text = log[-1]
        a = age_h(last_ts)
        stale = st == "ongoing" and a is not None and a > STALE_H

        title = html.escape(e.get("title", e["slug"]))
        if e.get("link"):
            title = f'<a href="{html.escape(e["link"])}">{title}</a>'
        if e.get("motivation"):
            title += f'<span class="db-subline">{html.escape(e["motivation"])}</span>'

        badge = ('<span class="db-badge" style="background:%s;color:#fff">%s</span>'
                 % (BADGE_BG.get(st, "#5a6472"), st))
        if stale:
            badge += ' <span class="db-badge" style="background:#8f2f2f;color:#fff">stale</span>'

        n = len(log)
        is_outcome_last = has_outcome
        tag = '<span class="log-outcome-tag">outcome</span> ' if is_outcome_last else ""
        toggle = (f'<button type="button" class="log-toggle" aria-expanded="false">'
                  f'{n} update{"s" if n != 1 else ""}<span class="log-chevron">&#9662;</span></button>')
        latest = (f'<div class="log-latest{" log-latest-outcome" if is_outcome_last else ""}">'
                  f'<span class="log-ts">{html.escape(last_ts)}</span>'
                  f'<span class="log-text">{tag}{html.escape(last_text)}</span>{toggle}</div>')

        updated = last_ts
        if a is not None:
            # a can go slightly negative when a log line's timestamp lands
            # after render time (clock skew, or the file was written
            # mid-scan) -- clamp rather than show a nonsensical "-0.3h ago".
            age = "just now" if a < 0 else (f"{a:.1f}h ago" if a < 48 else f"{a/24:.0f}d ago")
            updated += f'<span class="db-subline">{age}</span>'

        log_full_html = render_timeline_html(log, has_outcome)

        out.append([title, badge, html.escape(e.get("executor", e.get("owner", ""))),
                    latest, e.get("started", ""), updated, log_full_html,
                    ORDER.get(st, 3)])
    return out


def write_manifest(quiet=False):
    data = rows()
    payload = {"data": data,
               "generated": datetime.datetime.now().isoformat(timespec="seconds")}
    MANIFEST.write_text(json.dumps(payload))
    try:
        MANIFEST.chmod(MANIFEST.stat().st_mode | 0o004)
    except OSError:
        pass
    if not quiet:
        print(f"jobs.json updated ({len(data)} jobs)", file=sys.stderr)


# scrollX OFF (2026-08-09 redesign): DataTables' split-scroll header/body
# containers could desync (drag one, the header doesn't follow -- caught
# live, a column's label landed over a different column's data), and paired
# with the "nowrap" class it clipped genuinely long free-text fields
# mid-glyph instead of wrapping them. A single flowing table (scrollX off,
# no "nowrap" class, wrap-by-default CSS in theme3.css) can't desync against
# itself. See theme3.css's .dbwrap history comment for the full account.
DT_ARGS = {
    "classes": ["display", "compact", "hover", "results", "dbtable"],
    "layout": {"topStart": "pageLength", "topEnd": "search",
               "bottomStart": "info", "bottomEnd": "paging"},
    "pageLength": 25,
    "autoWidth": False,
    "scrollX": False,
    # ongoing first (state sentinel), then most recently updated
    "order": [[7, "asc"], [5, "desc"]],
    "columnDefs": [
        {"targets": 1, "className": "dt-nowrap"},
        {"targets": 2, "className": "dt-nowrap dt-hide-narrow"},
        {"targets": 4, "className": "dt-nowrap dt-right dt-hide-narrow"},
        {"targets": 5, "className": "dt-nowrap dt-right dt-hide-narrow"},
        {"targets": 6, "visible": False},  # log_full (searchable, drives the child row)
        {"targets": 7, "visible": False},  # state-order sentinel (searchable)
    ],
    "text_in_header_can_be_selected": True,
    "style": {"caption-side": "bottom", "margin": "auto",
              "table-layout": "auto", "width": "100%"},
}

# click anywhere on a sortable header (not just the tiny order-indicator
# glyph) to sort -- same convenience delegate as inventory_pages.py's
# CHROME_JS, duplicated rather than shared since this is the only piece of
# that script jobs also needs. The row-expand delegate lives in
# render_fragment()'s init script, not here, since it needs the DataTable API.
SORT_CLICK_JS = """<script>
document.addEventListener('click', function (e) {
  var th = e.target.closest('th.dt-orderable-asc, th.dt-orderable-desc');
  if (!th || e.target.closest('.dt-column-order') ||
      e.target.closest('a, button, input')) return;
  var ind = th.querySelector('.dt-column-order');
  if (ind) ind.dispatchEvent(new MouseEvent('click',
      {bubbles: true, shiftKey: e.shiftKey}));
});
</script>"""


def render_fragment():
    import itables.javascript as ij
    dt_bundle = ij.opt.dt_bundle
    init_datatables = ij.read_package_file("html/init_datatables.html")
    connected_import = ("import { set_or_remove_dark_class } from '"
                        + ij.UNPKG_DT_BUNDLE_URL_NO_VERSION + "';")
    local_import = ("const { set_or_remove_dark_class } = await window."
                    + ij._ITABLES_UNDERSCORE_VERSION + ";")
    init_datatables = ij.replace_value(init_datatables, connected_import, local_import)
    offline = ij.generate_init_offline_itables_html(dt_bundle)
    version_var = ij._ITABLES_UNDERSCORE_VERSION
    thead = "".join(f'<th class="{cls}">{html.escape(c)}</th>' if cls else f"<th>{html.escape(c)}</th>"
                    for c, cls in zip(COLUMNS, HEAD_CLASSES))
    dt_args = dict(DT_ARGS)
    dt_args["table_html"] = f"<table><thead><tr>{thead}</tr></thead></table>"
    skeleton = ('<table id="jobsdb"><tbody><tr><td class="db-loading">'
                'Loading the live jobs board&hellip;</td></tr></tbody></table>'
                '<noscript><p>JavaScript required; raw data at '
                f'<a href="{SITE_ROOT}/jobs.json">jobs.json</a>.</p></noscript>')
    # Row-click-to-expand: reads the DataTable API off the SAME table element
    # via the jQuery the offline bundle exports (`$(table).DataTable()`
    # returns the ALREADY-initialized instance, it does not re-init --
    # verified live before relying on it), so row.child() -- DataTables'
    # native per-row detail-panel API -- is available without needing to
    # dig into ITable's own internals. log_full (column index 6, hidden)
    # carries the pre-rendered timeline HTML; row.data()[6] reads it
    # regardless of the column's visible/hidden CSS state.
    init_script = f"""<script type="module">
    (async () => {{
        async function init() {{
            const {{ ITable, jQuery: $ }} = await window.{version_var};
            const table = document.querySelector("#jobsdb:not(.dataTable)");
            if (!table) return;
            const resp = await fetch("{SITE_ROOT}/jobs.json", {{ cache: "no-cache" }});
            const manifest = await resp.json();
            let dt_args = {json.dumps(dt_args)};
            dt_args["data_json"] = JSON.stringify(manifest.data);
            new ITable(table, dt_args);
            const dt = $(table).DataTable();
            // jQuery delegation on "tbody tr" also matches the child row's
            // OWN <tr> (the one DataTables inserts to hold the expanded
            // timeline) -- dt.row() on that node is not reliably an empty
            // selection, so guard on CONTENT (.log-timeline ancestor) rather
            // than trust row.length; a click inside the expanded panel must
            // never toggle it shut or touch an unrelated row.
            $(table).on("click", "tbody tr", function (e) {{
                if (e.target.closest("a") || e.target.closest(".log-timeline")) return;
                const row = dt.row(this);
                if (!row.length) return;
                if (row.child.isShown()) {{
                    row.child.hide();
                    this.classList.remove("log-expanded");
                }} else {{
                    row.child(row.data()[6]).show();
                    this.classList.add("log-expanded");
                }}
            }});
        }}
        if (window.{version_var}) {{ await init(); }}
        else {{ document.addEventListener("DOMContentLoaded", init); }}
    }})();
    </script>"""
    return offline + init_datatables + skeleton + init_script + SORT_CLICK_JS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.manifest or not args.out:
        write_manifest()
    if args.out:
        pathlib.Path(args.out).write_text(render_fragment())
        print(f"fragment -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
