#!/usr/bin/env python3
"""Thin driver: rebuild only the console's Jobs tab (jobs.html), the cheap
per-job-update rebuild command for agents. Full genre/composition lives in
build_console.py's build_jobs_tab() (console shell, no hero — see that
module); this file just calls it against the live publish dest, no full
console rebuild and no asset resync.

Entry format for jobs/*.md (plain "key: value" lines):
  title / executor / status(ongoing|done|frozen) / started / updated / slurm /
  link / now / outcome

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
