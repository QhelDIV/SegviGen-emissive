#!/usr/bin/env python3
"""inventory_jobs.py — the Jobs board database: manifest + DataTables fragment.

Same architecture as inventory_pages.py (the console Pages tab), applied to the
jobs registry: jobs/*.md (one file per job, one writer per file) -> jobs.json
manifest served from PUBLISH_DEST, plus a sortable/searchable offline-DataTables
widget fragment that fetches it at load. Columns include started/updated and a
computed freshness cell so staleness is sortable, which the owner asked for.

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

COLUMNS = ["job", "status", "executor", "slurm", "started", "updated", "state-order"]
# per-column <th> class, same order as COLUMNS (see render_fragment(); the
# matching <td> classes are applied via columnDefs' className below).
# "now / outcome" and "age" fold into the "job" / "updated" cells as a
# .db-subline instead of getting their own column -- fewer, denser columns
# beat a wide table that needs to scroll (2026-08-09 redesign, see
# theme3.css's .dbwrap history comment for why scrollX + nowrap broke).
HEAD_CLASSES = ["", "dt-nowrap", "dt-nowrap dt-hide-narrow", "dt-hide-narrow",
                "dt-nowrap dt-right dt-hide-narrow", "dt-nowrap dt-right", ""]


def parse(path):
    e = {"slug": path.stem}
    for line in path.read_text().splitlines():
        m = re.match(r"^(\w+):\s*(.*)$", line)
        if m:
            e[m.group(1).lower()] = m.group(2).strip()
    return e


def age_h(ts):
    try:
        t = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M")
        return (datetime.datetime.now() - t).total_seconds() / 3600.0
    except Exception:
        return None


def rows():
    out = []
    for p in sorted(JOBS.glob("*.md")):
        e = parse(p)
        st = e.get("status", "ongoing")
        a = age_h(e.get("updated", ""))
        stale = st == "ongoing" and a is not None and a > STALE_H
        title = html.escape(e.get("title", e["slug"]))
        if e.get("link"):
            title = f'<a href="{html.escape(e["link"])}">{title}</a>'
        line = e.get("outcome") if st != "ongoing" and e.get("outcome") else e.get("now", "")
        if line:
            title += f'<span class="db-subline">{html.escape(line)}</span>'
        badge = ('<span class="db-badge" style="background:%s;color:#fff">%s</span>'
                 % (BADGE_BG.get(st, "#5a6472"), st))
        if stale:
            badge += ' <span class="db-badge" style="background:#8f2f2f;color:#fff">stale</span>'
        updated = e.get("updated", "")
        if a is not None:
            # a can go slightly negative when an entry's "updated" timestamp
            # lands after render time (clock skew, or the file was written
            # mid-scan) -- clamp rather than show a nonsensical "-0.3h ago".
            age = "just now" if a < 0 else (f"{a:.1f}h ago" if a < 48 else f"{a/24:.0f}d ago")
            updated += f'<span class="db-subline">{age}</span>'
        out.append([title, badge, html.escape(e.get("executor", e.get("owner", ""))),
                    html.escape(e.get("slurm", "")), e.get("started", ""),
                    updated, ORDER.get(st, 3)])
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
    "order": [[6, "asc"], [5, "desc"]],
    "columnDefs": [
        {"targets": 1, "className": "dt-nowrap"},
        {"targets": 2, "className": "dt-nowrap dt-hide-narrow"},
        {"targets": 3, "className": "dt-hide-narrow"},
        {"targets": 4, "className": "dt-nowrap dt-right dt-hide-narrow"},
        {"targets": 5, "className": "dt-nowrap dt-right"},
        {"targets": 6, "visible": False},
    ],
    "text_in_header_can_be_selected": True,
    "style": {"caption-side": "bottom", "margin": "auto",
              "table-layout": "auto", "width": "100%"},
}

# click anywhere on a sortable header (not just the tiny order-indicator
# glyph) to sort -- same convenience delegate as inventory_pages.py's
# CHROME_JS, duplicated rather than shared since this is the only piece of
# that script jobs also needs.
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
    init_script = f"""<script type="module">
    (async () => {{
        async function init() {{
            const {{ ITable }} = await window.{version_var};
            const table = document.querySelector("#jobsdb:not(.dataTable)");
            if (!table) return;
            const resp = await fetch("{SITE_ROOT}/jobs.json", {{ cache: "no-cache" }});
            const manifest = await resp.json();
            let dt_args = {json.dumps(dt_args)};
            dt_args["data_json"] = JSON.stringify(manifest.data);
            new ITable(table, dt_args);
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
