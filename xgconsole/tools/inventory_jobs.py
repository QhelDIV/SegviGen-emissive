#!/usr/bin/env python3
"""inventory_jobs.py — lightgen's thin driver over xgpage.jobs (extracted
2026-08-10, mirroring the console precedent: build_console.py stays the
project-owned page composer, this file supplies only lightgen's own paths
and calls the package for everything generic -- parsing, rendering,
manifest architecture). See xgpage.jobs's module docstring for the field
contract, the job-page join, attention bands/recency heat, the track
feature, and log-line authorship; this file carries lightgen's CONFIG only.

The CLI contract below (--manifest, --out FRAGMENT_PATH) is unchanged from
before the extraction -- build_console.py's scan_jobs_table() subprocesses
this exact script with both flags and reads its stdout/stderr, so it stays
a runnable script with this exact interface even though the logic itself
now lives in the package.

Modes:
    --manifest            jobs/*.md -> PUBLISH_DEST/jobs.json   (stdlib only)
    --out <fragment.html> the widget fragment (NEEDS .venv_itables)
"""
import argparse
import pathlib
import sys

from xgpage import jobs as xj

REPO = pathlib.Path(__file__).resolve().parent.parent
SITE_ROOT = "/projects/omages/yanxg/lightgen"
PUBLISH_DEST = pathlib.Path("/project/3dlg-hcvc/omages/www/yanxg/lightgen")

CONFIG = xj.JobsConfig(
    jobs_dir=REPO / "jobs",
    publish_dest=PUBLISH_DEST,
    site_root=SITE_ROOT,
    pages_manifest=PUBLISH_DEST / "pages.json",
    track_values=("research", "tooling", "paper"),
    # pre-extraction wording, kept verbatim for byte parity: lightgen's
    # reviewer role is specifically named "master" on this board.
    legend_html=('<p class="sub">A violet "for your review" chip means a deliverable passed '
                'the master\'s review and is waiting on your look; those rows always stay on '
                'top and never fade. An ongoing job\'s latest-update line warms up right after '
                'it is touched and cools toward gray the longer it sits untouched.</p>'),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.manifest or not args.out:
        xj.write_manifest(CONFIG)
    if args.out:
        pathlib.Path(args.out).write_text(xj.render_fragment(CONFIG))
        print(f"fragment -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
