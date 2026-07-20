# lightgen — Orientation

Project: SegviGen/Trellis.2-based emissive-region segmentation for textured 3D
assets. (An earlier DiffusionNet geometry-only baseline is ABANDONED as of
2026-07-02 — no further comparisons to it; SegviGen is the sole focus.)

## Current State

> **Console v10 handoff RATIFIED 2026-07-04** (lightgen session): BRIEF.md
> rewritten against actual Phase-4 results (marked ratified), pages.yaml groups
> and the web_index-driven visuals list accepted as-is. `HANDOFF_console_v10.md`
> kept for history. Console root URL: see Quick reference.


**Mission (set 2026-06-25, team discussion; supersedes the autonomous run below):**
fine-tune the SegviGen **full-segmentation** model for **BINARY segmentation** —
prepare our data into SegviGen's format with only two colors: **white =
emissive region, black = non-emissive**. This refocuses (does not discard) the
earlier autonomous emissive fine-tune experiments — those results are the
evidence base motivating the binary-target plan.

**Status: Phases 1–3 of the battle plan DONE, Phase 4 (the run) LAUNCHED** —
`notes/2026-07-02_battle_plan.md`. Trainer upgrades landed (weighted flow
loss, EMA, best-ckpt tracking, explicit `--cond`/`--init_ckpt`); `train_1k`
(1123) + `val_96` (111) built (a `--pbr_only` filter bug that silently
no-op'd in all prior runs was found and fixed along the way); the overfit
gate passed conditionally — per-sample memorization is bimodal, split by GT
emissive-region size (≥0.36 frac memorizes, ≤0.074 gets stuck), not PBR type
as first suspected. Jobs **231171** (pos_weight 5) and **231172** (control,
W=1) launched 2026-07-02 ~01:40 — 18 epochs, real-cond, quick-val on `val_96`
every 2 epochs. Headline numbers to beat (SegviGen-internal): real-cond
autonomous run hit **val IoU 0.230@ep2** (job 226802) before collapsing;
zero-shot segment-then-label **oracle ≈ 0.235** (frozen pretrained SegviGen —
the bar fine-tuning must clear to be worth doing). Full detail:
`segvigen_emissive/WORKLOG.md`, `notes/2026-07-02_battle_plan.md`.

### Evidence base — autonomous emissive FT run (2026-05-29), from WORKLOG SUMMARY

Standalone fine-tune of SegviGen's `slat_flow_imgshape2tex` flow (init from
`full_seg.ckpt`), conditioned on shape latent + material/PBR latent, zero
image-cond (DINOv3 was gated at the time). Val IoU on a 16-sample val set:

| run | data | epoch | train IoU@0.2 | val IoU@0.2 | notes |
|---|---|---|---|---|---|
| pilot (full FT) | 232 | ep25 | — | 0.203 | best zero-cond on 232 data |
| pilot | 232 | ep50 | — | 0.042 | collapsed (paints nothing) |
| v2 (oversample) | 232 | ep10 | — | 0.095 | rescues emissive-heavy at thr 0.2 |
| v2 (oversample) | 232 | ep30 | — | 0.119 | noisy 16-sample val |
| **v3 (oversample)** | **512** | **ep4** | **0.179** | **0.176** | **train ≈ val → not overfit; the model's zero-cond ceiling** |
| v3 (oversample) | 512 | ep8 | 0.145 | 0.063 | train AND val both dropped → majority-class collapse |

Key diagnosis: the ep4→ep8 drop is **not classical overfitting** — train also
fell (0.179→0.145). The model collapses toward the trivial "predict
non-emissive" solution as training continues; this, not data scale, is the
binding constraint at 512 samples. Class imbalance (~11% emissive) is the
underlying driver (visually confirmed: on a 55%-emissive sample the model
predicts ~0% emissive). Zero-cond ceiling sits at ~0.18–0.20. Full log:
`segvigen_emissive/WORKLOG.md`.

### Published visual pages (pretrained full_seg zero-shot + eval renders)

Live index: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/index.html

- [Official example reproduction (sanity check)](https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/official_repro/index.html) — live
- [GT parts vs predicted full-seg (canonical overfit_split_10)](https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/gt_vs_pred_canon10/index.html) — live
- [Predicted full-seg on the mesh surface (canonical overfit_split_10)](https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/fullseg_canon10_mesh/index.html) — live
- [Predicted full-seg as voxels (intermediate)](https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/fullseg_canon10/index.html) — superseded by the mesh view
- [Ad-hoc 10-shape set (NOT canonical)](https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/fullseg_overfit10_adhoc/index.html) — deprecated, kept for reference only

**Open decisions:**
1. ~~Data-prep format for binary black/white targets~~ — **RESOLVED.** Confirmed
   the official recipe (`example_full_seg.py`) puts colors in
   `baseColorFactor`/`emissiveFactor`, and our data already matches that
   format — no dedicated channel/task-ID needed.
2. ~~Zero-cond vs real-cond baseline / official trainer vs standalone~~ —
   **RESOLVED.** DINOv3 licenses accepted 2026-05-29 (job 226785) and real-cond
   already run (0.230@ep2, job 226802) — real-cond is the baseline going
   forward. No official trainer exists for Gen3DSeg's dual-conditioning scheme
   (TRELLIS.2 repo has zero seg configs) → the standalone `train_emissive.py`
   loop here is plan of record, not a stopgap.
3. **Open:** data scale beyond the planned 1k (train_1k + val_96, Phase 2 of
   the battle plan) — decide after Phase 3/4 results whether more data moves
   the ceiling once the class-imbalance fix is in, per the autonomous run's
   train≈val finding at 512 samples.
4. ~~`full_seg` vs `interactive_seg` warm-start~~ — **RESOLVED.** `full_seg`
   was always the actual warm start in practice; the "discrepancy" was a
   stale docstring, not a real code-path bug. `interactive_seg` remains
   available as an optional ablation later, not a blocker now.

## Deadlines

<!-- none known yet — add as `- YYYY-MM-DD — label` when scheduled -->

---

## Quick reference

- Console: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/index.html
- Rebuild console: `python tools/build_console.py --publish`  (console v11, xgpage engine, migrated 2026-07-19; PUBLISH_DEST = /project/.../www/yanxg/lightgen; merge-only, never rmtree)
- Cluster: solar.cs.sfu.ca — see `cluster_skill/` for access/job helpers.
- Data subsets registry (which datasets exist, provenance): `notes/2026-07-03_data_subsets.md`
- Autonomous worklog (full history): `segvigen_emissive/WORKLOG.md`
- Other reference docs: `todo.md`, `diffusionnet_project.md`, `clarifications.md`, `gpt.md`
- Team updates: draft in chat → `updates/<date>_<slug>/` → `.venv_console/bin/python3 tools/build_update.py updates/<date>_<slug> --publish --pdf`
