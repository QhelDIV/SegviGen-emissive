# LightGen paper skeleton: content of record
# Written 2026-08-06 by the master session, from the owner's own framing.
# THIS IS A SKELETON. Claims are stated as the TARGET the paper argues for.
# Results sections are explicitly marked as not yet backed. That distinction is
# the point of the document: it is a plan, so it must show what is written and
# what still has to be earned.
#
# The page builder composes this. Do not soften the claims, do not add hedges
# that are not here, and do not invent numbers. Every number below is verified;
# anything absent is absent deliberately.

## STATUS VOCABULARY (use it visibly on the page, per section)

  ESTABLISHED  measured, verified, safe to put in the paper today
  TARGET       the claim we intend to support; not yet backed by results
  BLOCKED      cannot be written until a specific thing exists (name it)

---

# 1. Introduction

## 1.1 The gap  [ESTABLISHED]

TRELLIS.2 and other current 3D generation models produce geometry and PBR
material, albedo, metallic, roughness. None of them produce emission. A
generated lamp has a shade that reflects light correctly and emits nothing.

## 1.2 Why emission matters  [TARGET]

Emission is what makes a generated object participate in a scene's lighting
rather than only receive it. Without it, an asset can be lit but cannot light.
Emissive assets are the ones that carry a scene: lamps, screens, signage,
glowing weapons and effects.

## 1.3 Why it is hard  [three difficulties, evidence status differs]

**(a) The data is scarce and noisy.**  [ESTABLISHED]
Emissive assets are rare relative to general 3D data, and the labels that do
exist are unreliable. In our dataset a large group carries an emissive texture
identical to its base color, so the entire object is labeled emissive while
looking perfectly ordinary. Concretely: sampled over 1,998 training shapes,
median emissive coverage is 0.025 while the mean is 0.244, and 22.9 percent of
shapes are more than half emissive. The distribution is bimodal, and the gap
between median and mean IS the noise.

**(b) Emission is not only a texture, it has strength.**  [TARGET, and see the caveat]
Reflectance is bounded; emitted radiance is not. Predicting which surface emits
and what color is insufficient, because how brightly it emits changes the
result entirely.
CAVEAT THE PAGE MUST STATE: our current bake does not preserve strength.
Emission is stored as 8-bit color, and the `KHR_materials_emissive_strength`
extension is dropped. We measured it: 3 of 60 sampled source GLBs carry that
extension. So this difficulty is named as part of the problem, and addressing it
requires a change to the data pipeline that has not been made.

**(c) Evaluation cannot judge the object alone.**  [TARGET]
An emissive object must be placed in an environment to be judged. A texture map
compared against ground truth in isolation does not tell you whether the object
lights a room correctly. Evaluation therefore has to render the asset as a light
source and compare what it illuminates.

## 1.4 What we do  [TARGET]

We solve these problems and achieve emission generation on top of TRELLIS.2, in
two usable modes:
  - given an image, generate a 3D shape WITH emission
  - given an existing 3D shape, generate emission for it

The second mode is what distinguishes this from reconstruction-style approaches:
no reference image of the object glowing is required, because in a generation
pipeline no such image exists.

## 1.5 Positioning against concurrent work  [ESTABLISHED, and REQUIRED]

EmissionGen (arXiv 2604.11006, April 2026) addresses emission texture
generation, so the introduction cannot claim the problem is untouched. It builds
on Hunyuan3D-2.1 Paint, works multi-view then fuses to UV, and contributes
Objaverse-Emission at 40k assets.

Three differences that are ours to claim:
  - **Their input reference image already shows the glow.** Their task is stated
    as "given a reference image and an untextured mesh," with references sampled
    across viewpoints, lighting conditions, and emission intensities. Ours
    predicts emission from geometry and material, with no reference showing it
    lit.
  - **We are 3D-native.** They are multi-view to UV. We operate in TRELLIS.2's
    sparse 3D latent, which composes directly with its geometry and PBR stages.
  - **We compose into generation.** They texture an existing mesh from a photo.
    We add a stage to an image-to-3D pipeline, where no photo of the finished
    object can exist.

Also worth stating plainly, as fact rather than criticism: their evaluation uses
500 assets drawn from the same curated 40k with no separately defined held-out
test set, their Dice is computed on rendered 2D images rather than on the
surface, and their release promises the dataset, not code or weights.

---

# 2. Dataset curation  [ESTABLISHED]

Source: TexVerse. Splits: 74,503 shapes (train 59,602, val 7,450, test 7,451).
Built and usable: 72,546 (train 57,968, val 7,290, test 7,288).

Of the 1,957 not built: 1,036 never had source data, 584 were in a rebake that
completed without producing output, 337 are buildable and were not rebuilt
because they are 0.5 percent of the dataset and cannot move a result.
NOTE FOR THE PAGE: "1,957 missing" and "about 921 missing" are both correct
under different denominators, against the full split versus against what we
expected to be buildable. State both or neither, never one alone.

Class balance, from a 1,998-shape sample: 90.8 percent of shapes have nonzero
emission; above 0.001, 79.7 percent; above 0.01, 59.3 percent; above 0.1,
36.9 percent; above 0.5, 22.9 percent. Median 0.025, mean 0.244.

A filter exists and is deliberately not applied to the current run:
32,121 sids survive it, of which 32,050 are built (train 25,547, val 3,262,
test 3,241). Training unfiltered first is a baseline-over-everything choice,
with filtering held as an ablation.

Geometry validation against an independently produced 256 bake: 72,092,657
vertex pairs over 400 shapes, mean disagreement 0.119 voxel widths, p90 0.29,
worst 1.54, mean signed per-axis offset about +0.001. Occupancy ratio 512 to 256
has median 4.10x, which is surface scaling, as expected.

---

# 3. Method: fine-tuned SegviGen  [mixed status, be precise]

## 3.1 The formulation  [TARGET]

We predict a per-voxel binary emissive mask, then take the emissive texture to
be the input albedo restricted to that mask.

  emission = mask * albedo

This is the key simplification. The model never has to generate emission color,
because the color is already present in the PBR input. It only has to decide
WHERE the object emits.

STATUS: the mask * albedo premise is NOT YET VALIDATED. The measurement that
would validate it is cheap and has not been run: take ground-truth mask, take
base color from the input voxels, form mask * albedo, and compare against
ground-truth emission. Until that number exists, this section states a design,
not a result. Say so on the page.

## 3.2 Why this design avoids the failure everyone else hits  [ESTABLISHED as motivation]

Generating emission as a continuous texture in the TRELLIS.2 latent fails in a
specific, documented way. Fine-tuned emission VAEs reconstruct sparse emissive
regions to solid black, collapse colors toward orange, and leak nonzero emission
across surfaces that should be exactly zero. That last failure persists when a
single sample is memorized for 500 steps, which makes it representational rather
than an optimization shortfall.
Predicting a mask and reusing the input albedo sidesteps the entire problem.

Cross-reference the emission VAE page for the evidence.

## 3.3 Backbone and initialization  [ESTABLISHED]

We fine-tune TRELLIS.2's `slat_flow_imgshape2tex`, the 1.3B sparse DiT, warm
started from SegviGen's `full_seg.ckpt`, a checkpoint already fine-tuned for 3D
part segmentation in the same latent space.

The hypothesis, stated so it can be falsified: a segmentation prior transfers to
emissive-region identification better than a texture-generation prior.

Be honest about the near-neighbor: a TRELLIS.2 variant that replaces all PBR
channels with a single emission channel, then thresholds, is also a mask
predictor. The difference is initialization and loss, not formulation.

## 3.4 The channel hijack  [ESTABLISHED]

The binary target is written into the pretrained encoder's base color slot, with
the other material slots pinned to constants (metallic 0, roughness 255, alpha
255 in 8-bit). Input and output shapes match the pretrained model exactly, so
the encoder is reused byte-identically. No new encoder, no new VAE, no
architecture change.

## 3.5 Three stages, and why not two  [TARGET]

Emission is a third stage after geometry and PBR. The reviewer question to
pre-empt is why not fold emission into the PBR stage. Three arguments:
  - Emissive assets are far rarer than general 3D data, so retraining the PBR
    stage on them would degrade it.
  - Emission has a different codomain: it must be exactly zero almost
    everywhere, and it is unbounded above. Albedo, metallic and roughness are
    bounded, dense, and correlated.
  - A separate stage drops into an existing pipeline without regenerating
    geometry or PBR.
STATUS: the ablation that would settle this, a true joint PBR-plus-emission
model, has not been run. Name it as future work rather than implying it exists.

## 3.6 Conditioning  [ESTABLISHED limitation]

Current runs use zero conditioning. The DiT expects a DINOv3 image embedding and
receives zeros, so the model sees geometry and PBR only. The weights are
available; the conditioning path is unbuilt. This is an ablation, not a
licensing block, and the paper should say which.

Note the asymmetry that must be disclosed: the TEXGen and TRELLIS.2 baselines
ARE conditioned on a thumbnail, and on genuine emitters that thumbnail blows the
emissive region out to white, which makes the region visible in their input and
not in ours.

---

# 4. Results  [BLOCKED]

## 4.1 Baselines

TEXGen (UV space), TRELLIS.2 with albedo replaced by emission, TRELLIS.2 with
all PBR replaced by a single emission channel. DiffusionNet is dropped: it is
too restricted by mesh topology to belong in the final paper.

## 4.2 Metrics  [TARGET]

  - IoU for the mask, reported as the mean over K independent draws with the
    draw standard deviation. Single-draw IoU is unreliable: our own per-epoch
    validation swung between 0.0008 and 0.1499 on adjacent epochs.
  - Best-of-K may appear as an explicitly labeled oracle upper bound, never as
    the headline, because without a selector it is not obtainable at inference
    and it rewards variance.
  - Threshold fixed on validation, never swept on test.
  - Everything stratified by emissive coverage, because a flat mean over a
    bimodal distribution is dominated by degenerate cases.
  - Rendered illumination comparison as the headline for emission quality:
    render the asset as the only light source and compare what it illuminates.

## 4.3 Quantitative table  [BLOCKED]

Cannot be written. Current numbers, all near 0.1 IoU across four unrelated
architectures, indicate a common cause upstream of any model rather than a model
comparison. Our own 24-hour run reached best validation IoU 0.1499 and last
0.0811 across 21 epochs, with training loss moving only 0.2873 to 0.2610 and
validation IoU showing no trend.
DO NOT put 0.1499 in a table. It is a noise peak on a 16-shape quick validation
at one draw per shape.

## 4.4 Qualitative figure  [BLOCKED on results, but the FORM is decided]

Object thumbnail, ground-truth emission, prediction, difference, shown across
the emissive-coverage range rather than hand-picked, with the object always
visible so the reader can tell what is glowing.

---

# 5. What this skeleton needs next, in order

1. The mask * albedo validation (section 3.1). Cheap, no training, and the
   method section rests on it.
2. The diagnostics currently running: VAE round-trip ceiling stratified by
   coverage, trivial baselines, draw variance. These explain the 0.1.
3. A decision on emission strength: restore it in the bake, or remove
   difficulty (b) from the introduction's claims.
4. The DINOv3 conditioning, so our numbers are comparable to baselines that get
   a thumbnail.
5. The joint PBR-plus-emission ablation (section 3.5).
