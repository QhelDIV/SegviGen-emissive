#!/usr/bin/env python3
"""Thin driver: rebuild only the console's Jobs tab (jobs.html), the cheap
per-job-update rebuild command for agents. Full genre/composition lives in
build_console.py's build_jobs_tab() (console shell, no hero — see that
module); this file just calls it against the live publish dest, no full
console rebuild and no asset resync.

Entry format for jobs/*.md (2026-08-09 log redesign, owner-directed):
  title / executor / status(ongoing|done|frozen) / started / updated / slurm /
  link / motivation / log: / outcome

  title, executor, status, started, updated, slurm, link: plain "key: value"
  lines, unchanged.

  motivation: one sentence, written ONCE at registration -- why this job
  exists. Shown under the title on the board; it does not change as the job
  runs.

  log: an APPEND-ONLY block. Every line under it that matches
  "- YYYY-MM-DD HH:MM <sentence>" is one update; write a new line, NEVER edit
  or delete an old one (the board renders every line you have ever written as
  a timeline). Newest line goes LAST. The board's "updated" column and its
  staleness check (ongoing + no update in 3h = flagged stale) are computed
  from your last log line's timestamp, not from the `updated:` field -- keep
  bumping `updated:` too for anyone reading the raw file, but the log line's
  own timestamp is what the board actually trusts.

  outcome: for a DONE job only, the final one-sentence-or-few summary --
  also append it as your last log line (it renders distinctly there, tagged
  "outcome"). Leave blank while the job is ongoing or frozen.

  slurm: stays in the file as reference detail (grep-able job ids) but does
  NOT get its own column on the board -- if a job id matters to the reader,
  say it inside a log sentence ("training started as job 242211"), don't
  make them go find the slurm field.

  WRITING STANDARD for motivation and log lines: complete plain sentences,
  written for the owner reading the board, not for another agent. No
  internal artifact names, no task ids, no compressed jargon -- say what
  happened and why it matters, the way you'd say it out loud.

Rebuild + publish:  .venv_console/bin/python tools/build_jobs.py
"""
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import build_console as bc  # noqa: E402


def main():
    bc.build_jobs_tab(bc.PUBLISH_DEST)
    bc.build_jobs_redirect(bc.PUBLISH_DEST)
    print(f"jobs.html -> {bc.BASE_URL}/jobs.html")


if __name__ == "__main__":
    main()
