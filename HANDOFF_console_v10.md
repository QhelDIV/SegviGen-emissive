# Handoff — Console v5 → v10 (2026-07-04)

Written by a cross-project dispatch agent (somages session). The console sync
was initiated from the somages session with the user's approval.

---

## What was done and why

The project-console skill was rewritten to v10 on 2026-07-04 (rewritten as one
coherent spec; the skill file is at `~/.claude/skills/project-console/SKILL.md`).
The lightgen console fork (`tools/build_console.py`) was on v5 and needed to be
brought up to the new design.

**v10 key changes vs v5:**
- Console lives at the **project root URL** (`…/lightgen/index.html`) instead of
  `…/lightgen/console/index.html`. The old URL is now a redirect stub.
- **BRIEF.md** (new file, see below) is the Overview body — an authored,
  plain-language document rewritten at milestones, not aggregated agent output.
- Nav tabs renamed to match v10 spec: **Overview / Pages / Agent notes / Project docs**.
  Old "Situation" and "Reference" sections moved accordingly.
- **Pages tab** now uses a curated `web/pages.yaml` registry merged with an
  auto-scan of the published directory. Four groups: Fine-tuning results / Data /
  Baselines / Archive.
- **`web/` directory** created in the repo (`web/pages.yaml`, `web/README.md`,
  `web/_preview/`). It is the source-of-truth for the curated registry.
- Publish step uses **merge-copy** (copytree dirs_exist_ok=True, never rmtree)
  directly into the NFS www directory.

**One structural deviation from the somages reference:** The skill's folder model
calls for `www/yanxg/lightgen → repo/web/` as a symlink, but lightgen's repo is
on `/local-scratch2/` (local scratch on cs-3dlg-25) which the aspis web server
cannot follow. Fix: `PUBLISH_DEST = /project/3dlg-hcvc/omages/www/yanxg/lightgen`
(real NFS directory). `scan_pages()` scans PUBLISH_DEST instead of REPO/web.
Everything else matches the v10 design. If the repo is ever moved to NFS, the
symlink model can be adopted; swap PUBLISH_DEST back to `REPO / "web"` and create
the symlink from www.

## Files changed

| File | Change |
|------|--------|
| `tools/build_console.py` | Synced to TEMPLATE_VERSION 10 (all v10 changes above) |
| `BRIEF.md` | New — authored draft (see ratification note below) |
| `web/README.md` | New |
| `web/pages.yaml` | New — 4 groups, 8 pages |
| `AGENTS.md` | Quick-reference line updated: old console URL → new root URL + rebuild note |
| `HANDOFF_console_v10.md` | This file |

Published to `/project/3dlg-hcvc/omages/www/yanxg/lightgen/`:
- `index.html`, `pages.html`, `state.html`, `experiments.html`, `worklog.html`
- `notes/index.html`, `notes/*.html` (3 notes + metrics_explainer.html)
- `console/index.html` replaced with redirect stub → `../index.html`
- All pre-existing page dirs (`dataset_gallery_v1`, `finetune_binary_v1`, etc.) untouched

No commits were made — **lightgen is not a git repo**. Recommendation from the
somages session's experience: `git init` + a first commit early — an
uncommitted tree means no revert points when agent edits go wrong, and it
blocks worktree-isolated agents entirely.

Old www directory was backed up as
`/project/3dlg-hcvc/omages/www/yanxg/lightgen_bak_20260704` before any changes.
Verified working → **deleted 2026-07-04**.

## Needs your ratification

Three things were written by inference from project docs and need a human check
before treating them as canonical:

### (a) BRIEF.md content

File: `/local-scratch2/xya120/studio/misc/lightgen/BRIEF.md`

Drafted from AGENTS.md + WORKLOG.md by an agent with no direct experiment access.
The three-point evidence chain, job numbers, and IoU figures were all read from
AGENTS.md — not independently verified against the actual WORKLOG. Possible issues:

- The framing of "the bar is IoU ≈ 0.235" may understate or misstate nuance in the
  actual results (WORKLOG has more detail than AGENTS.md).
- The "Now — what's in flight" section describes jobs 231171/231172 as if they just
  launched (2026-07-02) — you may have results by now that should supersede this.
- The "Next" section is generic; replace with your actual next step once Phase 4
  results are in hand.

**Action:** Open BRIEF.md in the lightgen session, read it against the actual
WORKLOG, and rewrite any section that doesn't match reality. Mark the file header
"Status: ratified" when done.

### (b) CURRENT_VISUALS selection

The Overview shows `published_visuals_md()` — a list of all pages from
`segvigen_emissive/web_index.json`, linked (not embedded as images). The selection
of which pages to list is determined by that JSON file. Pages are ordered by JSON
entry order (not recency). The "superseded" and "deprecated" entries
(`fullseg_canon10`, `fullseg_overfit10_adhoc`) are included with their status badges.

**If you want to change this:** edit `web_index.json` entry order or status fields,
or add a CURRENT_VISUALS list in `build_console.py` with specific (path, caption,
page_href) tuples for image-file-based embedding (like somages does). If there are
new pages not yet in web_index.json, add them there and rebuild.

### (c) pages.yaml blurbs and grouping

File: `/local-scratch2/xya120/studio/misc/lightgen/web/pages.yaml`

Blurbs were copied from `segvigen_emissive/web_index.json` (each entry's `brief`
field). Groups were assigned by inference:

| Group | Pages |
|-------|-------|
| Fine-tuning results | finetune_binary_v1, training_curves_v1 |
| Data | dataset_gallery_v1 |
| Baselines | official_repro, gt_vs_pred_canon10, fullseg_canon10_mesh |
| Archive | fullseg_canon10, fullseg_overfit10_adhoc |

**If the grouping is wrong** (e.g., training_curves_v1 belongs under Baselines,
or you want a "Phase 4" group), edit `web/pages.yaml` and run
`python tools/build_console.py --publish` to republish.

---

## Rebuild command

```bash
python tools/build_console.py --publish
```

Published URL: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/index.html
