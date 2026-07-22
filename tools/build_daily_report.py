#!/usr/bin/env python3
"""build_daily_report.py — a date-stamped daily-report page (jday genre).

Genre studied directly from the somages console's ACTUAL current
implementation (course-corrected 2026-07-19 — an earlier version of this
script mirrored an older, since-superseded somages format; see
tools/build_console.py's module docstring for the correction history):

    https://aspis.cmpt.sfu.ca/projects/omages/yanxg/somages/daily/index.html
    (source: somages tools/build_daily.py; CSS: page-inlined <style> there)

somages renders EVERY day as one <section class="jday"> on a single
continuous journal page (kicker = day-of-week, h2 = date, then "Todos"
(checkbox-style), "Done", and terse recap "jbullets" blocks, a quieter
"jfine" footnote line, with an Archive footer for old frozen pages) — that
journal/aggregator shape is v3 WORKSPACE-ZONE machinery lightgen explicitly
omits (the console is lightgen's only zone; there is no separate research
workspace to hold a journal). What IS mirrored: the exact jday content-block
VOCABULARY (kicker/date/Todos/jbullets/jfine — ported verbatim into
web/assets/theme3.css, see that file's "jday" comment block for the promotion
rationale) and its visual grammar (checkbox todos, no results statband per
xgpage SKILL.md's D5 genre note — a plan/to-do memo has no headline result to
band). Adapted structurally to lightgen's ORIGINAL assignment: ONE STANDALONE
PAGE PER DAY (updates/<date>/index.html), not appended to a shared journal —
each page is wired into the console's OWN v3 tree (page_shell/console_tree_html
from build_console.py) under its "Updates" group, since there's no separate
workspace zone to hold it.

Each day's report is its own small script (copy this file, change DATE/
CONTENT below).

Usage:
    .venv_console/bin/python tools/build_daily_report.py            # stage only (not servable, see below)
    .venv_console/bin/python tools/build_daily_report.py --publish  # publish (the only servable target)
"""
import argparse, datetime, pathlib, sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))  # local modules below (build_console) — NOT xgpage
import xgpage as xg  # the installed package (uv pip install -e ~/studio/xgpage); migrated 2026-07-22
from build_console import (SITE_ROOT, BASE_URL, PUBLISH_DEST, ASSETS_DIR, ASSETS_REL,
                            CONSOLE_URL, FAVICON, console_tree_html)

# ---- per-day content (copy this file for the next day's report) ----
DATE = "2026-07-19"
DIRNAME = DATE  # PUBLISH_DEST/updates/<DIRNAME>/index.html — no slug, per this day's brief
TAG = "direct-GLB pivot"
PROJECT_TITLE = "Lightgen: binary emissive-region segmentation"

PIPELINE_EXPLAINER_URL = f"{SITE_ROOT}/pipeline_glb_direct/index.html"
UPDATE_HREF = f"{SITE_ROOT}/updates/{DIRNAME}/index.html"

DEK_HTML = (
    "Today's pipeline work pivots off the current somage-bake data chain toward voxelizing "
    "the original GLB directly."
)

# "→ " prefix marks an in-progress item (the somages convention, e.g. its
# 2026-07-18 entry: "→ Get the pipeline ready..." — plain-text marker inside
# the Todos list, no separate CSS state). Items 1-2 updated 2026-07-19 per
# owner correction: the 50-shape pilot has been KICKED OFF (an Opus worker is
# running it), so these read in-progress, not "to start".
TODO_ITEMS = [
    ("→ Write <code>build_dataset_direct.py</code> — stage 1 of the new pipeline: original "
     "TexVerse .glb → glb_to_vxz keeping the emissive attribute → ONE voxelization → input "
     "side (emissive channel zeroed — no leakage) + target side (threshold emissive → binary "
     "per-voxel GT) + slat encoding with shared coordinates. <b>In progress — worker running.</b>"),
    ("→ 50-shape stratified pilot (~10–15 shapes per glow-size bucket) through the new "
     "builder; A/B the new per-voxel GT against the current somage-derived GT; produce a "
     "visual comparison page (disagreement examples, stratified by glow size). "
     "<b>In progress — worker running.</b>"),
    ("Go/no-go checkpoint on the pilot: bake sane on real TexVerse materials AND GT "
     "meaningfully different on tiny-glow → scale to the full 2k rebuild + retrain; "
     "GT ≈ old → stop, pivot to model-side levers."),
]

CARRYOVER_ITEMS = [
    "<code>--image</code> real-cond smoke test for <code>predict_emissive.py</code> (zero-cond only so far).",
    "Owner to share the pipeline explainer with the team.",
    "Submodule pointer-bump PR to <code>dongchen-yang/lightgen</code> (<code>f3443da</code> → current).",
]

FACTS_ITEMS = [
    "SegviGen-emissive fork reorganized + merged (main @ <code>9b71cf8</code>) with "
    "<code>predict_emissive.py</code> (glb → voxel mask + per-face mesh masks, GPU-validated).",
    "Console migrated to the v3 workspace shell (v13) — this very page's infrastructure.",
]


def weekday_name(date_str):
    y, m, d = (int(x) for x in date_str.split("-"))
    return datetime.date(y, m, d).strftime("%A")


def jday_section_html():
    todos = "".join(f"<li>{item}</li>" for item in TODO_ITEMS)
    carry = "".join(f"<li>{item}</li>" for item in CARRYOVER_ITEMS)
    fine = "".join(f"<p>{item}</p>" for item in FACTS_ITEMS)
    return (
        f'<section id="day-{DATE}" class="jday">'
        f'<div class="jday-kicker">{weekday_name(DATE)}</div>'
        f'<h2 class="jday-date">{DATE} — {TAG}</h2>'
        f'<div class="prose" style="margin-bottom:20px">{DEK_HTML} Full argument + evidence: '
        f'<a href="{PIPELINE_EXPLAINER_URL}">the pipeline explainer</a>.</div>'
        f'<div class="jblock"><div class="jlabel">Todos</div><ul class="jtodos">{todos}</ul></div>'
        f'<div class="jblock"><div class="jlabel">Carry-overs</div><ul class="jbullets">{carry}</ul></div>'
        f'<div class="jfine">{fine}</div>'
        f'</section>'
    )


def build():
    header = (f'<header><div class="eyebrow">lightgen &middot; daily report</div>'
              f'<h1>{PROJECT_TITLE}</h1></header>')
    body = jday_section_html()
    footer = (f'<footer style="margin-top:2.5rem;padding-top:1.2rem;border-top:1px solid var(--line);'
              f'color:var(--ink-3);font-size:.85rem"><a href="{CONSOLE_URL}">&larr; Lightgen console</a></footer>')

    return xg.page(
        title=f"Daily report {DATE} — Lightgen",
        header_html=header,
        body_sections=[body, footer],
        theme="v3",
        tree_html=console_tree_html(SITE_ROOT, active_href=UPDATE_HREF),
        nav_title=f"Update {DATE}",
        assets_rel=ASSETS_REL,
        assets_dir=ASSETS_DIR,
        extra_head=f'<link rel="icon" href="{FAVICON}">',
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()
    html = build()

    if args.publish:
        out_dir = PUBLISH_DEST / "updates" / DIRNAME
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "index.html").write_text(html)
        for p in [out_dir, *out_dir.rglob("*")]:
            try:
                p.chmod(p.stat().st_mode | (0o005 if p.is_dir() else 0o004))
            except OSError:
                pass
        print(f"published: {out_dir / 'index.html'}")
        print(f"URL: {BASE_URL}/updates/{DIRNAME}/index.html")
    else:
        out_dir = REPO / "web/_preview" / f"daily_{DIRNAME}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "index.html").write_text(html)
        print(f"staged locally (not servable from local-scratch): {out_dir / 'index.html'}")
        print("run with --publish to write directly to the servable NFS PUBLISH_DEST")


if __name__ == "__main__":
    main()
