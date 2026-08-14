# Fact sheet: why every emissive model sits near 0.1 IoU
# Source: outputs/emis_72k_unfilt/run1/diagnostics.json (job 239795, N=300, 39m15s)
# Read directly by the master session 2026-08-07. Do NOT invent or re-round.

## 0. Config, so the page can state scope honestly

dataset dataset_direct, split val_72k, ckpt run1/best.ckpt (-> epoch_0012),
n=300, draws=3, steps=12, seed=0, thresholds [0.2,0.3,0.4,0.5].
Composition: n_ok 300, n_zero_gt 28, n_nonzero_gt 272.

IoU CONVENTION (state this prominently): empty prediction AND empty GT gives
IoU=1.0, matching eval_sample's own convention. This trivially inflates the
ceiling and the all-zero baseline on the 28 exact-zero-GT shapes, which is why
every aggregate is reported both all-shapes and nonzero-GT-only.

## 1. THE VAE ROUND-TRIP CEILING IS NOT THE BOTTLENECK

Decoding the GT latent and thresholding, scored against the true mask.
Nonzero-GT only (n=272):

  thr 0.2  mean 0.9594  median 0.9852  p10 0.9049
  thr 0.3  mean 0.9594  median 0.9847  p10 0.9024
  thr 0.4  mean 0.9585  median 0.9848  p10 0.9006
  thr 0.5  mean 0.9566  median 0.9848  p10 0.8925

Flat across thresholds, and 0.94-0.99 in EVERY stratification bucket (see 4).

This kills a hypothesis the master session had been running on for days: that
the metric scored every model against a reference the VAE had already
corrupted. It does not. The reference is clean. Say so plainly, and say it was
a hypothesis that measurement rejected.

## 2. Coordinate alignment is also not the bottleneck

  mean match frac of decoded GT      0.9673
  min  match frac                    0.8295
  same raw coords input/output       1.0
  emis_mask agreement w/ true voxel  0.9679

## 3. The model, and the aggregate comparison

Model, all 300 shapes, by threshold. Note it barely moves with threshold:

  thr 0.2  mean 0.0890  median 0.0070  p90 0.3333
  thr 0.3  mean 0.0880  median 0.0068  p90 0.3333
  thr 0.4  mean 0.0877  median 0.0070  p90 0.3333
  thr 0.5  mean 0.0885  median 0.0072  p90 0.3333

Model, nonzero-GT only (n=272), computed by the master session from the
per-shape records: mean 0.0719, median 0.0067.

**The median is the headline.** At 0.0067, the typical shape gets essentially
nothing. The mean is carried by a small minority.

Baselines at thr 0.5, nonzero-GT only (n=272), mean IoU:
  all_zero          0.000   (0 by construction on nonzero-GT)
  pbr_heuristic     0.180   (albedo-brightness threshold, best global pct=50)
  random_matched    0.235   (Bernoulli at each shape's own GT density)
  all_one           0.276
  MODEL             0.072

## 4. STRATIFIED, AND THIS CORRECTS THE AGGREGATE READING

By GT emissive_frac bucket, IoU at thr 0.5:

  bucket        n    ceiling  all_zero  all_one  random  MODEL
  [0,0.01)     121   0.952    0.231     0.002    0.233   0.079
  [0.01,0.05)   40   0.953    0.000     0.024    0.012   0.045
  [0.05,0.2)    40   0.938    0.000     0.101    0.054   0.063
  [0.2,0.5)     29   0.975    0.000     0.328    0.201   0.056
  [0.5,0.8)     23   0.973    0.000     0.644    0.482   0.126
  [0.8,1.0]     47   0.995    0.000     0.965    0.939   0.173

**The aggregate "the model loses to random" is TRUE overall and FALSE in the
sparse regime.** In [0.01,0.05) the model scores 0.045 against random's 0.012,
beating it by 3.7x. In [0.05,0.2) it leads random 0.063 to 0.054. Those two
buckets are 80 of the 300 shapes.

In the dense buckets random and all-one win trivially, because when most of a
shape emits, predicting everything is nearly correct for free: all_one reaches
0.965 in [0.8,1.0].

So the honest statement is: **the model has learned something in the sparse
regime and is buried in the aggregate by degenerate buckets where guessing
wins.** This is the stratification argument cutting in the model's favour, and
the page must not present the flat mean as the whole story.

It remains true that a naive albedo-brightness threshold (0.180 overall) beats
the model overall (0.072), and that is the comparison that should worry us.

## 5. Draw variance, which justifies the reporting protocol

Per-shape std across 3 draws, thr 0.5: mean 0.0643, median 0.0056, p90 0.2778.
The p90 exceeds the model's own mean IoU. A single draw is not a measurement.
This is the empirical case for reporting mean over K draws with the std.

## 6. What this rules in and out

RULED OUT as the cause of ~0.1 across four architectures:
  - VAE round-trip corruption of the reference (ceiling 0.96)
  - coordinate misalignment (0.97 agreement)
  - threshold choice (model flat 0.0877-0.0890 across all four)

STILL LIVE, in the order the master session would test them:
  - zero conditioning: the model sees no image, while TEXGen and TRELLIS.2
    baselines receive a thumbnail that blows the emissive region out to white
  - unfiltered training data: ~23% of shapes are the fullbright group where the
    emissive texture equals the base colour and the label is arguably unlearnable
  - the possibility that emission is not inferable from geometry and PBR alone
    at this capacity

## 7. What must NOT be claimed

  - Do not present 0.1499 from the training curve as a result; it is a noise
    peak on a 16-shape quick-val at one draw.
  - Do not say the model beats or loses to baselines without saying which
    regime, given section 4.
  - Do not compare any number here to EmissionGen's 0.98856 Dice: theirs is
    computed per-pixel on rendered 2D views over a curated 40k, with a
    reference image that already shows the glow.
