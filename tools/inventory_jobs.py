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

JOB-PAGE JOIN (2026-08-09, owner-approved): each job carries `page:`, either
the canonical name pages.json uses for its results page (e.g. "fullseg_19" or
"workspace/rendering"), or `page: none (<reason>)` for a job that legitimately
produces no page. `link:` keeps working as a fallback arbitrary URL -- when
`page:` resolves against pages.json that wins; otherwise `link:` (if set)
still links the title. A DONE job with `page:` never set at all (not even to
"none (...)") gets an amber "no page" badge next to its status -- the prompt
that it probably should have said something. An explicit "none (reason)" is a
documented, intentional state: it renders as a quiet note in the expanded
timeline instead, never a badge. See _resolve_page(). Two hidden columns
(page_name, slug) carry the plain resolved page name and this job's file stem
so inventory_pages.py can join back the other direction (which page(s) does a
job produce, and inherit its motivation as a blurb fallback) by reading
jobs.json -- see that module's _load_jobs_index(). Every row also gets a
stable DOM id (`job-<slug>`, assigned in render_fragment()'s init script) so
a page's "job" chip can deep-link straight to the row, not just to jobs.html.

ATTENTION BANDS + RECENCY HEAT (2026-08-09, owner-approved): `needs:
evaluation` on a job means a deliverable passed the MASTER's own review and
is waiting on the owner's eyes -- set and cleared only by the master (see
tools/build_jobs.py's docstring). It does two things: a violet "for your
review" chip (deliberately a color no other badge on this board uses --
amber is "no page", green/gray/red are status/stale) that never fades, and
pins the row into its own band ABOVE every ongoing and done/frozen row
regardless of the job's own status (the `state-order` sentinel, column 7, now
encodes THREE bands: 0 = needs evaluation, 1 = ongoing, 2 = done/frozen --
see the `band` computation in rows()). Any OTHER `needs:` value is reserved
for future use and renders as plain quiet text next to status, never a chip,
never a pin -- an unrecognized value should never look like it's demanding
attention it wasn't asked to demand.

Recency heat (ongoing jobs only, latest-cell text + timestamp only, never
the whole row, never a flagged row) is computed CLIENT-SIDE in
render_fragment()'s init script from the hidden `last_ts` column, not
server-side, specifically so it stays true while the page sits open across
its existing 120s auto-reload -- see decorateRows()'s heatColorMix()/
heatOpacity(). It reads --accent/--ink-3 via getComputedStyle at apply time
(not hardcoded hex) so it's correct in both themes automatically, and
re-applies on a MutationObserver watching <html data-theme> so a live theme
toggle updates it immediately rather than waiting for the next redraw.

LOG-LINE AUTHORSHIP (2026-08-09, owner-directed: a first-person log line
once read as if it might be the owner's own words, when it was the
executing agent's self-assessment): a log line may carry an optional author
token right after its timestamp -- `- YYYY-MM-DD HH:MM [executor] sentence`
-- with conventional values [executor], [master], [owner], or an agent's
actual name (e.g. [jobs-redesign]; rendered exactly as given, no lookup or
validation). A line WITHOUT the token is legacy: it still parses (see
LOG_LINE_RE's optional group) and renders as "executor" by default -- the
FILE is never rewritten to insert a label into an old line, only the
DISPLAY defaults it. See _author_label(): a small quiet label (same size
class as the timestamp, --ink-3) immediately before the sentence, in both
the expanded timeline and the "latest" cell; [owner] alone gets a touch
more visual weight (medium font-weight, full --ink) since owner words are
the record of verdicts, not just progress notes. tools/xgjobs stamps this
automatically on every verb that appends a log line -- see that script's
docstring for --as / XGJOBS_ACTOR.

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
PAGES_MANIFEST = PUBLISH_DEST / "pages.json"
STALE_H = 3.0
BADGE_BG = {"ongoing": "#1d7a46", "frozen": "#8a6d1a", "done": "#5a6472"}
NO_PAGE_BADGE_BG = "#a67c00"  # amber -- a prompt to add page:, not an error color
EVAL_BADGE_BG = "var(--violet-ink)"  # theme-aware (unlike the other fixed-hex badges here
# on purpose: it's a CSS custom property, correct in both themes automatically); distinct
# hue from amber (no-page) and the green/gray/red status+stale badges -- "for your review"
NEEDS_EVAL = "evaluation"

# page_name, slug, last_ts, status_raw: hidden (see module docstring's
# JOB-PAGE JOIN and ATTENTION BANDS sections). last_ts/status_raw are plain
# (unescaped-for-display) values purely for the client-side heat computation
# -- parsing them back out of rendered HTML would be needless fragility when
# Python already has them as plain strings.
COLUMNS = ["job", "status", "executor", "latest", "started", "updated",
           "log_full", "state-order", "page_name", "slug", "last_ts", "status_raw"]
# per-column <th> class, same order as COLUMNS (see render_fragment(); the
# matching <td> classes are applied via columnDefs' className below). Narrow
# width (see theme3.css's mobile block) keeps job/status/latest visible --
# the log-first point of this board -- and defers executor/started/updated.
HEAD_CLASSES = ["", "dt-nowrap", "dt-nowrap dt-hide-narrow", "",
                "dt-nowrap dt-right dt-hide-narrow",
                "dt-nowrap dt-right dt-hide-narrow", "", "", "", "", "", ""]

LOG_LINE_RE = re.compile(r"^-\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s+(?:\[([^\]]+)\]\s+)?(.*)$")
NONE_PAGE_RE = re.compile(r"(?i)^none\b\s*(?:\(([^)]*)\))?")


def parse(path):
    """Plain "key: value" lines, plus a multi-line `log:` block: every line
    immediately after `log:` matching `- YYYY-MM-DD HH:MM [author] <text>`
    (the `[author]` token is OPTIONAL, see LOG_LINE_RE and the module
    docstring's LOG-LINE AUTHORSHIP section) is consumed into e["log"] (a
    list of (timestamp, author_or_None, text) tuples, file order == append
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
                e["log"].append((lm.group(1), lm.group(2), lm.group(3).strip()))
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


def _load_pages_index():
    """name -> live URL, parsed out of pages.json's already-rendered <a
    href> (no raw url survives into that manifest -- see inventory_pages.py's
    render_data_rows()). Missing/unreadable manifest -> empty index; a job's
    page: then just fails to resolve, same as any not-yet-published page."""
    try:
        data = json.loads(PAGES_MANIFEST.read_text())
    except (OSError, ValueError):
        return {}
    idx = {}
    for row in data.get("data", []):
        m_href = re.search(r'href="([^"]*)"', row[0])
        m_name = re.search(r">([^<]*)</a>", row[0])
        if m_href and m_name:
            idx[html.unescape(m_name.group(1))] = m_href.group(1)
    return idx


def _resolve_page(e, pages_index):
    """Returns (href_or_None, none_reason_or_None, page_name_for_join).
    href priority: `page:` resolved against pages.json, else `link:` as the
    fallback arbitrary URL (its original role, kept working). page_name is
    the raw resolved name, for the hidden join column (empty when page: is
    unset or "none (...)")."""
    raw = (e.get("page") or "").strip()
    href = None
    none_reason = None
    page_name = ""
    if raw:
        m = NONE_PAGE_RE.match(raw)
        if m:
            none_reason = (m.group(1) or "").strip()
        elif raw in pages_index:
            href = pages_index[raw]
            page_name = raw
        # else: page: names a page not (yet) in pages.json -- pending work
        # elsewhere; left unlinked rather than guessing a URL that could be
        # wrong once that page actually publishes.
    if not href and e.get("link"):
        href = e.get("link")
    return href, none_reason, page_name


def _author_label(author):
    """A log line's small quiet attribution label (see module docstring's
    LOG-LINE AUTHORSHIP section): author is the raw bracket content LOG_LINE_RE
    captured, or None for a legacy untagged line, which defaults to
    "executor" for DISPLAY ONLY -- the file itself is never rewritten to add
    a label to an old line. Rendered exactly as given (no lookup/validation);
    "owner" alone gets a touch more visual weight since owner words are the
    record of verdicts, not just progress notes."""
    label = (author or "executor").strip()
    cls = " log-author-owner" if label.lower() == "owner" else ""
    return f'<span class="log-author{cls}">{html.escape(label)}</span>'


def render_timeline_html(log, has_outcome, none_reason=None):
    """The child-row expansion: a formatted timeline, timestamps in their own
    column (CSS grid, see theme3.css .log-entry) so they align regardless of
    sentence length. The final entry of a job with a recorded outcome gets a
    distinct tag + accent (has_outcome is a status/outcome-field flag, not a
    text match, so it can't drift from what render_fragment() considers
    "done"). A job with an explicit "page: none (reason)" leads with a quiet
    note (none_reason is None when page: didn't say "none" at all, so this
    never fires for a job that simply hasn't recorded a page yet)."""
    n = len(log)
    parts = ['<div class="log-timeline">']
    if none_reason is not None:
        note = html.escape(none_reason) if none_reason else "no results page recorded"
        parts.append(f'<div class="log-page-note">no results page: {note}</div>'
                     if none_reason else f'<div class="log-page-note">{note}</div>')
    # Rendered NEWEST FIRST (owner-ratified 2026-08-09): the reader arrives
    # from the collapsed row's latest entry and scans backward in time. The
    # FILE stays append-only newest-last; only the rendering reverses.
    for i, (ts, author, text) in sorted(enumerate(log), key=lambda p: p[0], reverse=True):
        is_outcome = has_outcome and i == n - 1
        cls = "log-entry log-entry-outcome" if is_outcome else "log-entry"
        tag = '<span class="log-outcome-tag">outcome</span>' if is_outcome else ""
        parts.append(f'<div class="{cls}"><span class="log-ts">{html.escape(ts)}</span>'
                     f'<span class="log-text">{_author_label(author)}{tag}{html.escape(text)}</span></div>')
    parts.append("</div>")
    return "".join(parts)


def rows():
    pages_index = _load_pages_index()
    out = []
    for p in sorted(JOBS.glob("*.md")):
        e = parse(p)
        st = e.get("status", "ongoing")
        log = e["log"] or [(e.get("updated", ""), None, "")]  # defensive: pre-migration file
        has_outcome = bool(e.get("outcome", "").strip())
        last_ts, last_author, last_text = log[-1]
        a = age_h(last_ts)
        stale = st == "ongoing" and a is not None and a > STALE_H

        href, none_reason, page_name = _resolve_page(e, pages_index)
        title = html.escape(e.get("title", e["slug"]))
        if href:
            title = f'<a href="{html.escape(href)}">{title}</a>'
        if e.get("motivation"):
            title += f'<span class="db-subline">{html.escape(e["motivation"])}</span>'

        needs_raw = (e.get("needs") or "").strip()
        # `needs: evaluation` may carry the review ask after a colon or pipe:
        #   needs: evaluation: Open the board and judge the pinned rows.
        # The ask is REQUIRED in practice (owner feedback 2026-08-09: a bare
        # flag gives no hint what to review); a bare flag still pins but
        # renders a generic instruction.
        m_eval = re.match(r"(?i)^evaluation\s*(?:[:|]\s*(.*))?$", needs_raw)
        needs_eval = bool(m_eval)
        review_ask = ((m_eval.group(1) or "").strip() if m_eval else "")

        badge = ('<span class="db-badge" style="background:%s;color:#fff">%s</span>'
                 % (BADGE_BG.get(st, "#5a6472"), st))
        if stale:
            badge += ' <span class="db-badge" style="background:#8f2f2f;color:#fff">stale</span>'
        page_field_set = bool((e.get("page") or "").strip())
        if st == "done" and not page_field_set:
            badge += (' <span class="db-badge" style="background:%s;color:#fff" '
                      'title="done with no page: recorded -- add one, or page: none (reason)">'
                      'no page</span>' % NO_PAGE_BADGE_BG)
        if needs_eval:
            badge = ('<span class="db-badge db-badge-eval" style="background:%s;color:#fff">'
                      'for your review</span> ' % EVAL_BADGE_BG) + badge
        elif needs_raw:
            # reserved for future needs: values -- plain quiet text, never a
            # chip and never a pin (see module docstring).
            badge += f' <span class="db-note-inline">needs: {html.escape(needs_raw)}</span>'

        n = len(log)
        is_outcome_last = has_outcome
        tag = '<span class="log-outcome-tag">outcome</span> ' if is_outcome_last else ""
        toggle = (f'<button type="button" class="log-toggle" aria-expanded="false">'
                  f'{n} update{"s" if n != 1 else ""}<span class="log-chevron">&#9662;</span></button>')
        latest = (f'<div class="log-latest{" log-latest-outcome" if is_outcome_last else ""}">'
                  f'<span class="log-ts">{html.escape(last_ts)}</span>'
                  f'<span class="log-text">{_author_label(last_author)}{tag}{html.escape(last_text)}</span>{toggle}</div>')
        if needs_eval:
            ask_text = review_ask or ("Open the linked page, judge it with your own eyes, "
                                      "and give a verdict in the CLI.")
            latest = (f'<div class="log-review-ask">Review: {html.escape(ask_text)}</div>'
                      + latest)

        updated = last_ts
        if a is not None:
            # a can go slightly negative when a log line's timestamp lands
            # after render time (clock skew, or the file was written
            # mid-scan) -- clamp rather than show a nonsensical "-0.3h ago".
            age = "just now" if a < 0 else (f"{a:.1f}h ago" if a < 48 else f"{a/24:.0f}d ago")
            updated += f'<span class="db-subline">{age}</span>'

        log_full_html = render_timeline_html(log, has_outcome, none_reason)

        # three attention bands (see module docstring): needs-evaluation
        # rows pin above everything regardless of their own status; within
        # a band, the secondary "updated" DESC sort (DT_ARGS) still applies.
        band = 0 if needs_eval else (1 if st == "ongoing" else 2)

        out.append([title, badge, html.escape(e.get("executor", e.get("owner", ""))),
                    latest, e.get("started", ""), updated, log_full_html,
                    band, page_name, e["slug"], last_ts, st])
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
        {"targets": 7, "visible": False},  # state-order sentinel: 3-band pin/ongoing/done
        {"targets": 8, "visible": False},  # page_name (join key for inventory_pages.py)
        {"targets": 9, "visible": False},  # slug (row-id anchor target)
        {"targets": 10, "visible": False},  # last_ts (client-side heat computation)
        {"targets": 11, "visible": False},  # status_raw (client-side heat: ongoing only)
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
    #
    # Row ids (job-<slug>, column index 9) are assigned manually rather than
    # via DataTables' own `createdRow`/`rowId` options passed through
    # dt_args, to not depend on whether ITable forwards those particular
    # keys -- assignRowIds() runs once right after init (covers the first,
    # synchronous draw that happens inside `new ITable(...)`, before any
    # listener could have been attached) and again on every future "draw"
    # event (sort/search/page-length changes). A page's "job" chip links to
    # "jobs.html#job-<slug>"; on load, a matching hash scrolls to and
    # expands that row so the reader lands on the actual update, not just
    # somewhere on the page.
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
            // Piecewise-linear, continuous decay (age in minutes -> [0,1]).
            // colorMix: 1 = full warm accent, 0 = normal ink; opacity fades
            // ONLY past the "normal" point, so a job updated within the
            // last 6h never looks faded, just gradually less warm.
            function heatColorMix(ageMin) {{
                if (ageMin <= 15) return 1;
                if (ageMin >= 360) return 0;
                if (ageMin <= 60) return 1 - 0.5 * (ageMin - 15) / 45;
                return 0.5 - 0.5 * (ageMin - 60) / 300;
            }}
            function heatOpacity(ageMin) {{
                if (ageMin <= 360) return 1;
                if (ageMin >= 2880) return 0.4;
                if (ageMin <= 1440) return 1 - 0.35 * (ageMin - 360) / 1080;
                return 0.65 - 0.25 * (ageMin - 1440) / 1440;
            }}
            function hexToRgb(hex) {{
                hex = (hex || "").trim().replace("#", "");
                if (hex.length === 3) hex = hex.split("").map(c => c + c).join("");
                const n = parseInt(hex, 16) || 0;
                return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
            }}
            // Row ids (job-<slug>, column index 9) are assigned manually
            // rather than via DataTables' own createdRow/rowId passthrough,
            // to not depend on whether ITable forwards those particular
            // keys. The needs-evaluation row class and recency heat are
            // computed in the SAME per-row pass for efficiency (~10 rows,
            // still one loop, not three). decorateRows() runs once right
            // after init (covers the first, synchronous draw inside `new
            // ITable(...)`, before any listener could have been attached),
            // again on every future "draw" event (sort/search/page-length
            // changes), and again on a live theme toggle (see the
            // MutationObserver below) so heat colors -- read from
            // --accent/--ink-3 via getComputedStyle, not hardcoded hex --
            // are correct in whichever theme is showing right now.
            function decorateRows() {{
                const accentRgb = hexToRgb(getComputedStyle(document.documentElement)
                    .getPropertyValue("--accent"));
                const inkRgb = hexToRgb(getComputedStyle(document.documentElement)
                    .getPropertyValue("--ink-3"));
                const now = Date.now();
                dt.rows().every(function () {{
                    const node = this.node();
                    const d = this.data();
                    if (!node || !d) return;
                    if (d[9]) node.id = "job-" + d[9];
                    const flagged = !!node.querySelector(".db-badge-eval");
                    node.classList.toggle("needs-eval-row", flagged);
                    const latestEl = node.querySelector(".log-latest");
                    if (!latestEl) return;
                    const tsEl = latestEl.querySelector(".log-ts");
                    const textEl = latestEl.querySelector(".log-text");
                    // heat applies to ongoing, unflagged jobs only -- a
                    // flagged row "never fades regardless of age", and
                    // done/frozen rows are never warm to begin with.
                    if (flagged || d[11] !== "ongoing" || !d[10]) {{
                        if (tsEl) {{ tsEl.style.color = ""; tsEl.style.opacity = ""; }}
                        if (textEl) {{ textEl.style.color = ""; textEl.style.opacity = ""; }}
                        return;
                    }}
                    const ageMin = Math.max(0, (now - Date.parse(d[10].replace(" ", "T"))) / 60000);
                    const mix = heatColorMix(ageMin);
                    const op = heatOpacity(ageMin);
                    const rgb = [0, 1, 2].map(i => Math.round(accentRgb[i] * mix + inkRgb[i] * (1 - mix)));
                    const colorStr = "rgb(" + rgb.join(",") + ")";
                    if (tsEl) {{ tsEl.style.color = colorStr; tsEl.style.opacity = op; }}
                    if (textEl) {{ textEl.style.color = colorStr; textEl.style.opacity = op; }}
                }});
            }}
            decorateRows();
            dt.on("draw", decorateRows);
            new MutationObserver(decorateRows).observe(document.documentElement,
                {{ attributes: true, attributeFilter: ["data-theme"] }});
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
            if (location.hash.indexOf("#job-") === 0) {{
                const target = document.getElementById(location.hash.slice(1));
                if (target) {{
                    target.scrollIntoView({{ behavior: "smooth", block: "center" }});
                    const row = dt.row(target);
                    if (row.length && !row.child.isShown()) {{
                        row.child(row.data()[6]).show();
                        target.classList.add("log-expanded");
                    }}
                }}
            }}
        }}
        if (window.{version_var}) {{ await init(); }}
        else {{ document.addEventListener("DOMContentLoaded", init); }}
    }})();
    </script>"""
    legend = ('<p class="sub">A violet "for your review" chip means a deliverable passed '
              'the master\'s review and is waiting on your look; those rows always stay on '
              'top and never fade. An ongoing job\'s latest-update line warms up right after '
              'it is touched and cools toward gray the longer it sits untouched.</p>')
    return offline + init_datatables + skeleton + init_script + SORT_CLICK_JS + legend


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
