# Lightgen, right now

> **Status: ratified 2026-07-04** (rewritten in the lightgen session against actual results;
> supersedes the 2026-07-04 draft written at Phase-4 launch time).

## What the project is

Lightgen fine-tunes SegviGen (a Trellis.2-based 3D part-segmentation model) to produce
**binary emissive-region masks** for textured 3D assets: given a shape with PBR materials
(and optionally one self-rendered photo), predict which surface voxels glow — a white/black
segmentation. Warm start is SegviGen's `full_seg` checkpoint; the corpus is TexVerse somages
(~80k shapes, Dongchen's processing). An earlier DiffusionNet baseline is **abandoned** —
success is measured SegviGen-internally.

**The bar:** zero-shot SegviGen with post-hoc part labeling scores **oracle IoU ≈ 0.235**
(measured on the canonical 10; not yet re-measured on our val set — a known comparability
gap). Fine-tuning must clear that to be worth doing.

## Latest (2k run, 2026-07-07) — a clean negative result

We built a bigger, cleaner dataset (**train_2k_ef**: 2,000 emission-filtered PBR shapes, zero-glow
mostly removed, Dongchen's held-outs excluded) and trained two models — weighted (W5) and
**balanced per-shape weighting** (the loss fix the arithmetic demanded). The honest full-val,
4-samples-averaged result: **neither beats the zero-shot oracle (0.219 nonzero-IoU).** All four
checkpoints sit at 0.10–0.11, statistically tied with the old 1k model (0.117). A 16-sample
quick-val briefly showed 0.179 — but that was small-sample noise (the same checkpoint scores
0.103 averaged; the multi-sample eval we built caught the false +53% before it reached anyone).

**The takeaway:** neither more clean data nor smarter weighting moved the ceiling. Every model
segments large glow well (>30%-glow bucket: 0.32–0.43) and fails on tiny glow (0–5%: 0.04–0.06) —
and tiny-glow shapes dominate. **Tiny-region segmentation is the wall**, and it's untouched by
data volume or loss engineering. Next levers point at the representation/objective (target
resolution, tiny-region-weighted readout) or reframing the metric, not more of the same. Details:
[training curves](https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/training_curves_v1/index.html)
· the [live roadmap](https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/roadmap/index.html).

## Where we are — Phase 4 (the 1k run) taught us four things

Two 18-epoch arms trained 2026-07-02 on `train_1k` (1123 PBR-filtered shapes): **W5**
(emissive-voxel mistakes weighted 5×) vs **W1** (plain loss control). Full results on all
111 val shapes:

**1. The collapse is fixed.** The old failure (peak at epoch 2, all-black by epoch 6) is
gone — both arms train stably for 18 epochs. Cleaner data + oversampling did real work.

**2. But nobody beats the oracle yet.** Flat mean IoU: W5 0.117 / W1 0.161 / W5-EMA 0.096 /
W1-EMA 0.172 — all below 0.230/0.235. And the flat mean is **gamed by timidity**: 25 of 111
val shapes have zero glow and score a free 1.0 for predicting nothing. W1 "wins" the average
by often painting nothing; **W5-EMA is the best actual segmenter** — 0.52 IoU on big-glow
shapes, ~2× everyone else — while over-painting tiny-glow shapes. See the
[metrics explainer](notes/metrics_explainer.html) ("zero-glow scoring trap") and the
[per-shape predictions](https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/finetune_binary_v1/index.html).

**3. Glow size is the real difficulty axis.** Overfit tests: shapes with ≥36% emissive
surface memorize to IoU ≈ 1.0; shapes ≤7% stay near 0 even with 600 steps/sample. Median
training shape has **1.4%** emissive — most of the dataset sits in the hard regime. The VAE
round-trip was checked and is NOT the bottleneck.

**4. Evaluation noise is large enough to flip conclusions.** Generation is stochastic:
identical checkpoints re-evaluated moved 0.096→0.128 and 0.172→0.193. Single-draw
comparisons near these margins are unreliable.

Curves, run configs (node, GPU, batch size 1, no LR schedule), and gate-run context:
[training_curves_v1](https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/training_curves_v1/index.html).
Data statistics + 48 random examples:
[dataset_gallery_v1](https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/dataset_gallery_v1/index.html).

## Next — the three levers, in order

1. **Multi-draw eval averaging** (cheapest): average several sampled generations per shape
   at eval; also re-measure the 0.235 oracle on val_96 itself so the bar and the model are
   scored on the same shapes and the same zero-glow convention.
2. **Checkpoint selection that can't be won by timidity**: select on glowing shapes only
   (or report stratified by glow bucket, which is now standard in our evals).
3. **Loss/recipe iteration**: per-shape *balanced* weighting (positives get a fixed loss
   share, capped — 5× is simultaneously too weak for 1%-glow shapes and over-eager on them)
   and revisit the training recipe (batch size 1, constant lr 1e-5, no scheduler; curves
   show no improvement after ~epoch 10).

## Reference

- Full experimental log: `segvigen_emissive/WORKLOG.md`
- Battle plan (Phases 1–4): `notes/2026-07-02_battle_plan.md`
- Data subsets registry (incl. Dongchen's 1099-shape list vs our train_1k, and the full
  80,735 → 74,503 → 59,602 → 26,264 → 1123 filtering chain): `notes/2026-07-03_data_subsets.md`
- Metrics explainer: [what every number means](notes/metrics_explainer.html)
