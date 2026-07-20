---
title: "Emissive fine-tune: bigger clean data + balanced weighting — a clean negative result"
date: 2026-07-07
tldr: "Neither more clean data nor balanced per-shape weighting beats zero-shot SegviGen on the honest metric. Tiny-glow regions are the wall. (DRAFT — pending Xingguang's review, not yet shared.)"
template: false
---

> **DRAFT for internal review — not shared to the team yet.** Numbers are final; framing is for Xingguang to check first.

**Full visual writeup (charts + rendered predictions):
[results_2k_v1 →](../../results_2k_v1/index.html)** — the comparison and stratified-by-glow-size
charts below, an honest-eval noise callout, and 8 real val_96 shapes with rendered voxel
predictions (appearance | GT target | 2k+W5 pred | 2k+balanced pred) making the tiny-glow wall
visible, not just charted. This update is the text summary; that page is the primary artifact.

## What we tried

Two levers on top of the earlier 1k emissive fine-tune, both aimed at the majority-class collapse
(most surface voxels are non-emissive, so the model drifts toward "paint nothing"):

1. **Bigger, cleaner data** — `train_2k_ef`: 2,000 shapes, emission-filtered (declared-but-dark
   shapes removed), PBR-only, Dongchen's val/test held-outs excluded, ~6% deliberate negatives.
2. **Balanced per-shape loss weighting** — weight each shape's emissive voxels by `(1−p)/p`
   (capped 50), so a 1%-glow shape's tiny region carries real loss mass. (vs. the old flat 5×,
   which the arithmetic showed gives a median shape's emissive region only ~7% of its loss.)

## The result — nobody beats zero-shot

Honest protocol: full 111-shape val set, **4 generations per shape averaged** (single runs vary
±0.09 — this matters below), voxel-based IoU, reported on the shapes that actually glow.

| model | IoU (nonzero-glow shapes) |
|---|---|
| **zero-shot SegviGen oracle** (frozen + label parts) | **0.219** |
| old 1k fine-tune (best) | 0.117 |
| new 2k + W5 weighting | 0.103 – 0.107 |
| new 2k + balanced weighting | 0.112 – 0.114 |

**Neither lever moved the ceiling.** The 2k clean data did not improve on the old 1k; balanced
weighting ≈ plain weighting. Everything sits ~0.10–0.12, well under the 0.219 zero-shot bar.

A cautionary note worth flagging: a 16-sample quick-val briefly showed the 2k model at **0.179**
(a +53% jump). It was small-sample noise — the same checkpoint averages **0.103** on the full set.
The multi-sample eval caught it before it became a claim. Single-draw comparisons at this scale
are not trustworthy.

## Why — the tiny-glow wall

Breaking IoU down by how much of each shape glows makes the problem obvious:

- **Large glow (>30% of surface): every model does fine** — 0.32 to 0.43 IoU.
- **Tiny glow (0–5%): every model fails** — 0.04 to 0.06 IoU.

And tiny-glow shapes dominate the distribution (median shape is ~1.4% emissive). So the aggregate
is pinned down by a failure mode that **neither data volume nor loss weighting addresses**. The
model can find a big glowing panel; it cannot reliably find a small emissive strip.

## Where this points

The wall isn't the data pipeline or the loss recipe — those are now clean and well-instrumented.
It's the **sparse-tiny-region regime**. Candidate next directions (for discussion):

1. **Representation/readout** — the target lives at 512³ voxels but the flow operates at 32³
   latent; tiny regions may be lost in that 16× compression. Higher-res or a tiny-region-aware
   readout is worth testing.
2. **Metric reframe** — is voxel-IoU even the right score for a 0.03%-of-surface emitter? A
   detection-style or boundary-aware metric might reflect usefulness better.
3. **Accept + report honestly** — zero-shot SegviGen (0.219) is a strong, training-free baseline;
   the fine-tune's value proposition needs rethinking if it can't clear it.

_Data provenance, training curves, and per-shape prediction galleries are all on the project
console; happy to walk through any of it._
