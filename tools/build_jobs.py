#!/usr/bin/env python3
"""Render jobs/*.md into the live jobs board, composed with the xgpage system
(same engine as the console/pages page; hand-rolled HTML is banned by project
convention). One file per job, one writer per file: that layout is what makes
concurrent multi-agent updates safe; this renderer is read-only over jobs/.

Entry format (plain "key: value" lines):
  title / executor / status(ongoing|done|frozen) / started / updated / slurm /
  link / now / outcome
Rebuild + publish:  .venv_console/bin/python tools/build_jobs.py
"""
import datetime
import html
import os
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import xgpage as xg  # noqa: E402

JOBS = REPO / "jobs"
DEST = pathlib.Path("/project/3dlg-hcvc/omages/www/yanxg/lightgen/jobs")
URL = "https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/jobs/"
STALE_H = 3.0
ORDER = {"ongoing": 0, "frozen": 1, "done": 2}
BADGE = {"ongoing": ("#1d7a46", "#eafff3"), "frozen": ("#8a6d1a", "#fff8e0"),
         "done": ("#5a6472", "#f2f4f7")}


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


def main():
    entries = sorted((parse(p) for p in JOBS.glob("*.md")),
                     key=lambda e: (ORDER.get(e.get("status", "ongoing"), 0),
                                    e.get("updated", ""), e["slug"]))
    n_on = sum(1 for e in entries if e.get("status") == "ongoing")
    n_stale = sum(1 for e in entries
                  if e.get("status") == "ongoing"
                  and (lambda a: a is not None and a > STALE_H)(age_h(e.get("updated", ""))))

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    band = xg.statband([(str(len(entries)), "jobs tracked"),
                        (str(n_on), "ongoing"),
                        (str(n_stale), "stale"),
                        (stamp.split()[1], "rendered")])
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        frag = pathlib.Path(td) / "jobs_fragment.html"
        r = subprocess.run([str(REPO / ".venv_itables/bin/python"),
                            str(REPO / "tools/inventory_jobs.py"),
                            "--manifest", "--out", str(frag)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("inventory_jobs failed:\n" + r.stderr)
        table = frag.read_text()
    intro = xg.prose(
        "<p>One entry per non-trivial job, one file per entry under "
        "<code>jobs/</code> in the ops repo, one writer per file (the owning "
        "executing agent or the master), so concurrent updates never conflict. Ongoing "
        "entries silent for more than %d hours are flagged stale. Statuses: "
        "ongoing, frozen, done (with a one-line outcome).</p>" % int(STALE_H))
    body = xg.section_v2("board", "01", "Every non-trivial job, one row, freshness visible",
                         intro + band + table)
    html_out = xg.page(
        title="lightgen jobs",
        body_sections=[body],
        header_html=xg.hero_header(
            "lightgen operations",
            "Jobs board",
            dek_html="Live registry of running, frozen, and finished "
                     "workstreams. Auto-refreshes; rendered from jobs/ by "
                     "tools/build_jobs.py."),
        theme="v2",
        assets_rel="../assets",
        extra_head='<meta http-equiv="refresh" content="120">')
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "index.html").write_text(html_out)
    print(f"{len(entries)} entries -> {URL}")


if __name__ == "__main__":
    main()
