#!/usr/bin/env python3
"""Render jobs/*.md (one file per job, one writer per file) into the live jobs
page. The one-file-per-job layout is what makes concurrent multi-agent updates
safe: no shared file, no conflicts; this renderer is read-only over the dir.
Modeled on tools/build_roadmap.py (the sanctioned live-ops page genre).

Entry format (plain "key: value" lines, then optional free-text body):
  title/owner/status(ongoing|done|frozen)/started/updated/slurm/link/now/outcome
Rebuild + publish:  python3 tools/build_jobs.py
"""
import datetime
import html
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = os.path.join(REPO, "jobs")
DEST = "/project/3dlg-hcvc/omages/www/yanxg/lightgen/jobs"
URL = "https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/jobs/"
STALE_H = 3.0
ORDER = {"ongoing": 0, "frozen": 1, "done": 2}
COLOR = {"ongoing": "#4cc38a", "frozen": "#e5c07b", "done": "#7f8ea3"}

def parse(path):
    e = {"slug": os.path.splitext(os.path.basename(path))[0]}
    for line in open(path):
        m = re.match(r"^(\w+):\s*(.*)$", line.rstrip())
        if m:
            e[m.group(1).lower()] = m.group(2)
    return e

def age_h(ts):
    try:
        t = datetime.datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M")
        return (datetime.datetime.now() - t).total_seconds() / 3600.0
    except Exception:
        return None

def main():
    entries = [parse(os.path.join(JOBS, f)) for f in sorted(os.listdir(JOBS))
               if f.endswith(".md")]
    entries.sort(key=lambda e: (ORDER.get(e.get("status", "ongoing"), 0),
                                e.get("updated", ""), e.get("slug", "")))
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = []
    for e in entries:
        st = e.get("status", "ongoing")
        a = age_h(e.get("updated", ""))
        stale = st == "ongoing" and a is not None and a > STALE_H
        age = "&#8212;" if a is None else (f"{a:.1f}h ago" if a < 48 else f"{a/24:.0f}d ago")
        link = e.get("link", "")
        title = html.escape(e.get("title", e["slug"]))
        if link:
            title = f'<a href="{html.escape(link)}">{title}</a>'
        badge = (f'<span style="color:{COLOR.get(st, "#ccc")};font-weight:600">{st}</span>'
                 + (' <span style="color:#e06c75;font-weight:600">&#9888; stale</span>' if stale else ""))
        line = e.get("outcome", "") if st != "ongoing" and e.get("outcome") else e.get("now", "")
        rows.append(
            "<tr><td>%s</td><td>%s</td><td class=m>%s</td><td class=m>%s</td>"
            "<td class=m>%s</td><td>%s</td></tr>"
            % (title, badge, html.escape(e.get("owner", "")),
               html.escape(e.get("slurm", "")), age, html.escape(line)))
    page = """<!doctype html><meta charset="utf-8">
<meta http-equiv="refresh" content="120">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>lightgen jobs</title>
<style>
 body{background:#14161a;color:#d7dae0;font:15px/1.5 -apple-system,system-ui,sans-serif;
      margin:2rem auto;max-width:1080px;padding:0 1rem}
 h1{font-size:1.25rem} a{color:#61afef;text-decoration:none}
 table{border-collapse:collapse;width:100%%} 
 td,th{padding:.45rem .6rem;border-bottom:1px solid #262a31;text-align:left;vertical-align:top}
 th{color:#8b93a3;font-weight:600;font-size:.8rem;text-transform:uppercase;letter-spacing:.04em}
 .m{font-family:ui-monospace,monospace;font-size:.85rem;white-space:nowrap}
 .foot{color:#5c6470;font-size:.8rem;margin-top:1rem}
 @media(max-width:600px){td,th{padding:.35rem .3rem}.m{white-space:normal}}
</style>
<h1>lightgen &#8212; jobs</h1>
<table><tr><th>job</th><th>status</th><th>owner</th><th>slurm</th><th>updated</th><th>now / outcome</th></tr>
%s</table>
<div class=foot>rendered %s &#183; one file per job under jobs/ &#183; ongoing entries silent &gt;%sh are flagged stale</div>
""" % ("\n".join(rows), now, STALE_H)
    os.makedirs(DEST, exist_ok=True)
    with open(os.path.join(DEST, "index.html"), "w") as f:
        f.write(page)
    print(f"{len(entries)} entries -> {URL}")

if __name__ == "__main__":
    main()
