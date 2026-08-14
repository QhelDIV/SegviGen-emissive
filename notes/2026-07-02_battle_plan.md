# Binary emissive fine-tune — battle plan (2026-07-02)

Status: active
TL;DR: Fix the class-imbalance collapse in the loss, scale data to 1k real-cond, gate on overfit sanity, then run — must beat the zero-shot SegviGen oracle (0.235). (DiffusionNet comparison ABANDONED 2026-07-02 per owner — SegviGen-only focus.)

**Progress (evening of 2026-07-02):** Phases 1–3 all DONE (Phase 3 conditional pass, see
below); Phase 4 LAUNCHED ~01:40 — jobs 231171 (pos_weight 5) + 231172 (control, W=1) running.

## Evidence base (why this plan)

- Real-cond (DINOv3) run reached **val IoU 0.230@ep2** (job 226802), the best result to date —
  then collapsed by ep6 to 0.035. **Train dropped too** (ep2→ep6), not just val — this is
  majority-class collapse, not overfitting. Conditioning raises the ceiling; it doesn't fix
  the collapse.
- Zero-shot segment-then-label **oracle IoU ≈ 0.235** (label each predicted part emissive iff
  >50% GT-emissive, union) — the bar any fine-tune must clear, since it upper-bounds what
  correct-boundaries-wrong-labels can score.
- **Overfit-1 sanity: IoU 0.968** on a single sample trained to convergence — the training
  loop, latent encode/decode round-trip, and loss are all sound. The plateau on real data is a
  genuine objective/class-imbalance ceiling, not a pipeline bug.
- Data format already matches the official `example_full_seg.py` recipe (verbatim toolkit
  calls) — no data-prep detour needed.
- No official trainer exists for Gen3DSeg's dual-conditioning scheme (shape latent + material/
  PBR latent) — the TRELLIS.2 repo has zero seg configs. → the standalone fine-tune loop here
  is the correct vehicle, not a workaround.

Full history: `segvigen_emissive/WORKLOG.md` (SUMMARY block).

## Phase 1 — trainer upgrades — DONE (agent: coder)

- Per-voxel class-weighted flow MSE: `w = 1 + (W-1)·emis_frac`, mask pooled from
  `output.vxz` 512³ → 32³ latent coords (backfillable onto existing latents).
- EMA of model weights.
- Best-checkpoint-on-val tracking (not just last-epoch).
- Explicit `--cond {real,zero}` flag — kill the silent zero-cond fallback that made the
  autonomous run's early results ambiguous about what was actually conditioning the model.
- `--init_ckpt` choice — verify the `full_seg` vs `interactive_seg` discrepancy before
  committing to one warm-start.
- Sharper oversampling exponent (current 0.1-floor weighting by `emissive_frac` is mild).
- **Landed:** weighted loss + EMA + cond-gating all in. Overfit-1 controls at matched
  training steps: **W5 (pos_weight 5) = 0.885 IoU, W1 (unweighted) = 0.918 IoU** — the
  weighting doesn't break single-sample convergence (see Phase 3 for the fuller finding).
- **`--init_ckpt` resolved:** `full_seg` was always the warm start in practice — the
  "discrepancy" was a stale docstring, not an actual code path using `interactive_seg`.
  No config bug; nothing to fix.

## Phase 2 — data — DONE (agent: data-builder)

- Build `train_1k` (1000 samples, pbr_only, real-cond) + `val_96`.
- Reuse existing 512-set samples that pass the pbr filter where possible.
- ~15% sid overshoot in the build to absorb attrition (failed encodes, filter misses).
- **Landed:** `train_1k` = 1123 samples, `val_96` = 111 samples, attrition only 0.1%.
  `emis_mask` backfilled onto existing latents; mask-vs-metadata correlation r=0.83/0.92
  (train/val) — sanity-consistent.
- **Bug found and fixed:** `--pbr_only` had a latent `pd.NA` bug that meant the filter
  **never actually worked** in any prior run (zero-cond and real-cond alike) — all earlier
  "pbr_only" data was in fact unfiltered. Fixed for `train_1k`/`val_96`; earlier WORKLOG
  results should be read as unfiltered-by-PBR, not as evidence pbr-filtering doesn't matter.

## Phase 3 — gate — DONE, conditional pass

- Ran overfit-1 (and overfit-10) with the new weighted loss.
- **Key finding:** per-sample overfit outcome is **bimodal**, and the separator is
  **emissive-region size, not PBR type** as originally suspected. All shapes that memorize
  have GT emissive_frac **≥0.36**; all shapes that get stuck have emissive_frac **≤0.074**.
  No sample lands in between in this run.
- VAE round-trip re-verified **NOT** the bottleneck even at the extreme (0.03% emissive
  sample still round-trips correctly) — confirms the ceiling is objective/loss-shaped, not
  a decode artifact, even for near-empty masks.
- Conditional pass: the loss lets large-emissive-region samples memorize fine (0.885–0.918
  IoU); the open question is whether it's *sufficient* for small-emissive-region samples at
  1k scale, which Phase 4's stratified eval (below) is designed to answer.

## Phase 4 — the run — LAUNCHED 2026-07-02 ~01:40

- Job **231171** (pos_weight 5) + job **231172** (control, pos_weight/W=1) — head-to-head
  ablation on identical data/schedule.
- 18 epochs, real-cond (DINOv3), 1k train data.
- Quick-val: 16-sample subset of `val_96` evaluated every 2 epochs (fast feedback loop);
  full `val_96` (111) reserved for the final read.
- Threshold sweep + Otsu readout (colors are bimodal but muted, per WORKLOG honest-read).
- **Reference to beat (SegviGen-internal):** 0.230@ep2 (232-data real-cond, prior
  autonomous run) and the zero-shot oracle 0.235.
- training curves (loss + quick-val IoU, both arms, plus the overfit-1/overfit-10 gate curves
  that preceded them): https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/training_curves_v1/index.html

**Paper-vs-code finding (2026-07-03, verified against arXiv 2603.16869 §3.3.2 + shipped code):**
the SegviGen PAPER never mentions an RGB image input for full segmentation — its condition
injection lists only click point tokens (zero-padded in full mode), a task-ID embedding, and
(2D-guided mode only) the guidance-map tokens. But the shipped CODE's full-seg pipeline feeds a
self-rendered photo of the input asset through RMBG+DINOv3 into the flow's cross-attention —
inherited TRELLIS.2 plumbing the authors kept rather than retrained away. It is architecturally
optional (zeros = the CFG "unconditional" branch) but practically load-bearing: the pretrained
weights expect it, and our own ablation measured zero-cond 0.176 vs real-cond 0.230. Our
fine-tune keeps it; dropping it cleanly would itself require a fine-tune.

**Known risk:** median train `emissive_frac` is **1.4%** — most samples are far below the
Phase 3 memorization threshold (0.36) — and tiny-emissive shapes resisted even single-sample
overfitting. If W5 underperforms, the planned next levers are (1) a **stratified eval** by
GT emissive_frac bucket (`0`, `(0, 0.05]`, `(0.05, 0.3]`, `>0.3`) to see whether the
aggregate IoU is being dragged down by the tiny-emissive tail specifically, and (2) a
**balanced per-sample weighting** variant (weight by inverse frequency of a sample's
emissive-frac bucket, not just per-voxel pos_weight) as the next lever.

**PBR-filter finding (2026-07-07 investigation):** `pbrType` is a TOOLING signature
(15× substancepainter-tag gap), not a quality signal; the scans/baked hypothesis is
debunked at aggregate; and non-PBR shapes are MORE label-emissive (86% vs 76% nonzero) —
the filter discards ~25k usable train shapes to keep ~18k. BUT its real justification
(baked-lit albedo = confusing INPUT signal) has never been cleanly tested (the pd.NA bug
means no true filtered-vs-unfiltered run exists). → Named next experiment: **PBR-filter
ablation** (train an arm on a non-PBR-inclusive slice, same size/protocol), and evaluate
`#n_pbr_materials`/`#is_all_pbr` (df_SomgProc_final) as a better-grounded filter.

## Success criteria

Val IoU **> 0.235** (zero-shot SegviGen oracle) — fine-tuning must beat frozen SegviGen +
post-hoc part labeling, or it isn't earning its keep. Beyond that: qualitatively clean
white/black maps on shapes that glow (stratified eval, not just the flat mean — the flat
mean rewards timid all-black predictions via the zero-emissive val shapes).

**Risk:** weighted loss operates in latent space (32³ pooled mask) and may not map cleanly onto
pixel-space class balance after decode — the overfit gate (Phase 3) and the quick eval-every-2-ep
curve in Phase 4 should surface this early rather than after the full 1k run completes.
