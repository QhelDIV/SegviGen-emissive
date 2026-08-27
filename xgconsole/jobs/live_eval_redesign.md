title: live_eval page redesign: concise, curve-first, epoch slider
executor: liveeval-designer
track: tooling
status: ongoing
started: 2026-08-26 20:46
updated: 2026-08-26 20:59
slurm: 
link: 
page: live_eval
upstreams: live_eval_loop
motivation: Owner: no one reads the full text; the curve and the visual wall are the page. Redesign for conciseness, and replace the single-epoch wall with an epoch slider so the same shapes' predictions can be scrubbed across training. Designer produces a mockup on real data first; liveeval-builder incorporates when idle.
log:
- 2026-08-26 20:46 [master] Job started.
- 2026-08-26 20:59 [liveeval-designer] Mockup built and published: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/_preview/live_eval_mock/index.html. Concise layout (statband + curve as the hero, rest folded into one collapsed zone) plus an epoch slider over one fixed 10-shape grid, scrubbing draw thumbnails and their IoU across all 5 checkpoints currently on disk; no-JS fallback shows the newest checkpoint statically, same as the live page today. Verified by DOM measurement at all 8 standard widths (2560 down to 390, zero issues) and a 6-check simulated-user journey script (default-latest, prev/next, keyboard arrows, drag-to-boundary with button disabling, missing-image placeholder fallback, refresh-resets-to-latest) -- all pass. Does not touch live_eval/ or its build.py. Incorporation spec for build.py (HTML structure, the ckpts.json contract, where the JS lives, and one flagged data-retention question: the page's own img/ mirror currently prunes to the last 4 checkpoints, which undercuts the slider's value) is in INCORPORATION.md next to the mockup.
- 2026-08-26 20:59 [liveeval-designer] Note for liveeval-builder: while building this, outputs/live_eval/ was mid-migration (records_legacy_quick/, bridged/, shards/ appeared within minutes; records/ transiently held one file then began rewriting older checkpoints to failed re-sharding stubs). Read-only from here, so no action needed, but the mockup's data was frozen from a records_legacy_quick/ snapshot taken at 20:54 rather than racing the live store; noted in INCORPORATION.md.
outcome: 
