# Lightgen — live roadmap

_The real-time "where are we / what's next / what needs you" view. Master edits this file
as state changes and republishes; auto-refreshes every 2 min. Rewritten 2026-07-23 as a
clean handoff (prior 2k-training arc archived in git history)._

**Jobs board:** https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/jobs/ (one entry per non-trivial job, ongoing/done/frozen, stale-flagged; entries live in `jobs/`, rendered by `tools/build_jobs.py`).

## Pipeline
- done: direct-GLB emissive investigation (found the o_voxel bug)
- done: bug already fixed by Dongchen (June 4), corrected data produced
- done: nonzero-emissive threshold decided (>1/255)
- done: xgpage standalone package migration (core + console)
- doing: data-understanding pages (report + gallery) — built, awaiting promote
- todo: SegviGen fine-tune restart on Dongchen's corrected 74k data
- wait: your calls (promote pages · filtering-gallery framing) ←

## ▶ Now (active)

- **Nothing running.** All workers idle (pipelineworker, consoleworker). Nothing blocked
  except the owner decisions below.

## 🎯 The situation (post-compaction orientation — read this first)

The emissive project pivoted this week. We set out to switch data generation to
direct-GLB→o-voxel; investigating it, we found o_voxel's per-voxel emissive was broken
(a one-line mipmap copy-paste bug). **Dongchen had already found and fixed that exact bug
on 2026-06-04** (TRELLIS.2-lightning commit `eec7840b`; not ours to fix or PR). His
**`uv_voxel_pipeline`** has since processed the corpus into corrected data:

- **Data: `/cs/3dlg-jupiter-project/lightgen/uv_voxel_pipeline/out_uv_voxel_74k/`** —
  72,374 shapes, each: `atlas.npz` (512² UV, for TEXGen) + `emission_voxels_256/*.vxz` +
  `pbr_voxels_256/*.vxz` (256³, for us/SegviGen) + `coords.npz`. SHA == Sketchfab UID.
  This is the GT the emissive fine-tune should now train on. Validated: teddy 0, glowers real.
- **Threshold decision:** binarize emissive as **any authored emission (value > 1/255)**,
  NOT lum>0.04 — the data showed the (0,0.04] dim band is large (43–60% of glowing shapes)
  and real, and 0.04 was starving the tiny-glow regime. (Pure `>0` is a 3-min re-run option.)

## → Next (the actual research path)

1. **256³ vs 512³ reconciliation** — Dongchen's voxels are 256³; our SegviGen fine-tune ran
   at 512³ (glb_to_vxz grid 512 → 32³ latent). Confirm SegviGen can train on 256³ emission
   voxels directly, or derive a 512³ target. THIS is the first concrete fine-tune step.
2. Restart the emissive fine-tune on the corrected 74k data + nonzero target.

## ⏸ Waiting on you

1. **Promote** `uvvox_report` + `uvvox_gallery` (in `_preview`, correct at nonzero now) into
   the workspace zone → permanent tracking.
2. **Filtering-recap page framing** — the "unlit vs PBR" gallery: render from original GLBs?
   how many per class (dozens or hundreds)? (Recap of the filtering is done, in chat + notes.)
3. **emissive_gt / pipeline_glb_direct pages** — they frame the bug as OUR discovery; Dongchen
   found it first. Reframe as team-attributed validation, or shelve. (Both are drafts/unshared.)
4. **Upstream Microsoft PR** — moot (Dongchen fixed it in the team fork); drop unless you want
   it sent to microsoft/TRELLIS.2 upstream (still buggy there).

## 🔗 Key locations (durable)
- Corrected data: `/cs/3dlg-jupiter-project/lightgen/uv_voxel_pipeline/out_uv_voxel_74k/`
- Pages (all `_preview`): uvvox_report, uvvox_gallery (700 ex.), emissive_gt, pipeline_glb_direct
- Console: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/ (thin driver on xgpage.console)
- Ops repo (console/xgpage/notes/page-builders): `QhelDIV/lightgen-ops` (local root = this dir)
- xgpage package: `~/studio/xgpage` (lightgen consumes it; `import xgpage`); model-viewer kept
  lightgen-local (`tools/xgpage_ext.py` + `tools/sync_xgpage_assets.py`)
- Fine-tune fork: `QhelDIV/SegviGen-emissive` (main @ 9b71cf8; predict_emissive.py, EXPERIMENTS.md)

## 📋 Recent (newest first)

- **Nonzero threshold adopted** — re-rendered the 700 gallery + 5 report shapes at >1/255;
  sign 16%→60%, creature 29%→77% (dim gradient now counted), teddy/zeros stay 0. Both pages
  updated + QA-clean. Gallery strata now glow 371 / tiny 235 / zero 94.
- **Console → thin driver on `xgpage.console`** (lightgen-ops @39bfd28), byte-parity verified.
  Lightgen fully on the standalone xgpage package (core + console). Pages auto-track cron added.
- **700-example emission gallery** built (`_preview/uvvox_gallery`) — grey/orange voxels,
  sort-by-glow filter, Sketchfab click-through.
- **uv_voxel data report** built (`_preview/uvvox_report`) — what the corrected data is, emission
  validated correct, vs the old somage→glb→ovoxel path.
- **lightgen-ops repo** created (QhelDIV/lightgen-ops, private) — the local console/xgpage/notes
  tooling, finally versioned. Dev code stays in dongchen-yang/lightgen + the SegviGen-emissive submodule.
- **Fork reorg + predict_emissive.py + per-face mesh masks** merged to SegviGen-emissive main (9b71cf8).
- Earlier arc (archived): the 2k emission-filtered fine-tune was a clean negative result (no
  model beat the 0.219 zero-shot oracle; tiny-glow is the wall) — which motivated the data pivot.

<!-- Workers idle. No git pushes to team repos / no team-facing sends without owner go. -->
