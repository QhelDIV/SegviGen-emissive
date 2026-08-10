#!/usr/bin/env python3
"""Thin driver: rebuild only the console's Jobs tab (jobs.html), the cheap
per-job-update rebuild command. Full genre/composition lives in
build_console.py's build_jobs_tab() (console shell, no hero — see that
module); this file just calls it against the live publish dest, no full
console rebuild and no asset resync.

THE INTERFACE IS tools/xgjobs (2026-08-09, owner-directed): agents update
the board by running the CLI, never by hand-editing jobs/*.md. It owns every
mechanical detail this file used to just document and hope for -- local-
clock timestamps (no more NFS-skew or hand-typed-timestamp bugs), an flock
per job file (safe under concurrent writers), format validation against
inventory_jobs.py's own parser before anything is written, the writing-
standard lint (no em dash, no banned words, no #<task-id> references), and
an inline rebuild (this file) after every call, so publishing is never a
separate step. `tools/xgjobs --help` documents every verb with an example;
one line per common case:

    tools/xgjobs start <slug> --title T --motivation M --executor E \\
        (--page NAME | --no-page REASON)
    tools/xgjobs log <slug> "what happened, one plain sentence"
    tools/xgjobs done <slug> --outcome "the final one-sentence summary"
    tools/xgjobs frozen <slug> [--reason "..."]
    tools/xgjobs reopen <slug> "why it's active again"
    tools/xgjobs flag <slug>     # MASTER-ONLY by convention: pins the row
    tools/xgjobs unflag <slug>   # for the owner's review until cleared

Hand-editing jobs/*.md directly is DEPRECATED for agents; it remains a
documented EMERGENCY path only (e.g. recovering a file the CLI refuses to
touch). If you ever do need to hand-edit, the field order inventory_jobs.py
expects is: title / executor / status(ongoing|done|frozen) / started /
updated / slurm / link / page / needs / motivation / log: / outcome. `page:`
is the canonical pages.json name for this job's results page (not a URL;
`page: none (<reason>)` for a job that legitimately produces none -- a DONE
job with `page:` unset entirely gets an amber "no page" flag as a prompt).
`needs: evaluation` pins the row with a violet "for your review" chip.
`log:` is APPEND-ONLY, one `- YYYY-MM-DD HH:MM <sentence>` line per update,
newest last -- the board's "updated" column and staleness check read the
LAST log line's timestamp, not the `updated:` field. `outcome:` is the
final summary for a done job, also appended as the log's last line. Writing
standard: complete plain sentences for the owner, no internal artifact
names, no task ids, no compressed jargon -- exactly what the CLI's lint
enforces mechanically, so hand-edits meet the same bar.

Rebuild + publish (after a hand-edit, or on its own):
    .venv_console/bin/python tools/build_jobs.py
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
