# Lightgen — live roadmap

_The real-time "where are we / what's next / what needs you" view. Master edits this file
as state changes and republishes; auto-refreshes every 2 min. Rewritten 2026-08-10 (the
2026-07 data-pivot arc is archived in git history)._

**Jobs board:** https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/jobs.html (every
workstream, one log per job, review flags pin to the top; written via `tools/xgjobs`).
**Page graph:** https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/graph.html

## Pipeline
- done: v2 split adopted (Dongchen's newbake_vae, 71,646/387/388) + cond backfill
- done: overfit diagnostic, 10 shapes: neither pos_weight memorizes (0.28 / 0.42 vs 0.96 ceiling)
- done: single-shape control on today's code memorizes (0.997 by ep70) — no code regression
- doing: 72k image-conditioned training, capped epochs (first checkpoint due overnight)
- doing: 10-shape pos_weight-1 run extended to 400 epochs (epoch-matched comparison)
- todo: locate the multi-shape blocker (capacity / schedule / interference)
- wait: your verdicts on flagged deliverables ←

## ▶ Now (active)

- **Training, Solar:** the capped 72k conditioned run (epoch 1 done, loss normal, watcher on
  the first checkpoint) and the 400-epoch 10-shape parity run. Watchers + agent heartbeats on both.
- **Team:** rendering-page upgrade (five-setup teaser; emission-sweep video encoding), graph
  round 2 (interaction state model + simulated-user QA + timeline mode), board track filter
  (research / tooling / paper separation).

## 🎯 The situation (read this first)

The overfit mystery narrowed decisively tonight. The pipeline cannot memorize 10 training
shapes at ANY pos_weight (best IoU 0.28 at 1.0, 0.42 at 5.0, against a 0.96 ceiling), but
today's code memorizes a single emissive shape to 0.997, better than July's anchor. So:
no code regression, and the failure is specific to multi-shape training. A separate finding
points at the backbone: zero-shot reconstruction butchers the emissive asset family (pumpkin
to smooth ball, candles to one disc) while reporting plausible part counts, invisible to
metrics, visible only in renders. The 400-epoch parity run decides whether 10 shapes
saturate or just need far longer.

Rendering setups are now named and documented (box render is the project default for
emission figures): RENDERING.md + the workspace page.

## → Next

1. Read the 400-epoch parity curve when it lands; if it saturates, probe interference
   (fewer shapes, higher capacity, schedule variants) before any more 72k spend.
2. First 72k checkpoint: eval + rendered examples on the standard shapes, K-draw protocol.
3. Keep thickening the page graph (cross-links + curated relationship edges).

## ⏸ Waiting on you

1. Review flags on the jobs board (violet rows, each states its ask).
2. Rendering-page v1 version mint happens after the sweep video lands; the flagged page
   will be your review surface.

## 🔗 Key locations (durable)
- Corrected data: `/cs/3dlg-jupiter-project/lightgen/uv_voxel_pipeline/out_uv_voxel_74k/`
- Console: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/ (Jobs / Pages / Graph tabs)
- Rendering setups: RENDERING.md + https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/workspace/rendering/
- Overfit diagnostic page: .../lightgen/_preview/overfit_condtest/
- Ops repo: `QhelDIV/lightgen-ops` (this dir) · xgpage package: `~/studio/xgpage`
- Fine-tune fork: `QhelDIV/SegviGen-emissive`

## 📋 Recent (newest first)

- **Ops overhaul (2026-08-09 night):** jobs board rebuilt as a log-first database with
  review flags, recency heat, authored lines (owner words verbatim in accent chips), and
  the `xgjobs` CLI enforcing the writing standard; page-relationship Graph tab shipped;
  figure numbering + thumbnail strip landed in the xgpage engine; rendering setups named
  and documented with a five-setup teaser and sweep video in progress.
- **Overfit verdicts:** pos_weight ruled out as sole cause; no code regression
  (single-shape 0.997); emissive family is out of distribution for the pretrained
  backbone's reconstruction; multi-shape blocker is the open question.
- **19-shape part-segmentation gallery** (fullseg_19) published; orientation mismatch
  disclosed; one input render fixed (stray icosphere inflated the auto-frame).
- **Old showcase numbers closed:** exact replay proved unseeded single-draw luck
  (per-shape IoU swung 0.984 to 0.180); only K-draw means count from now on.
- **v2 split + cond backfill done;** the uncapped 24h runs timed out saving nothing
  (600GB/epoch cond reads); the capped relaunch is the active 72k run.
