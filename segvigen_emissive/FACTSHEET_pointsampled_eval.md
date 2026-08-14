# Fact sheet: SegviGen scored through Dongchen's point-sampled evaluation harness

Assembled 2026-08-07 by the pointsampled-eval session. Every number here was measured
this session on the frozen 381-shape `usable` eval set (evaluation/newdata_eval/eval_sets/usable.txt
in the lightgen_repo clone), through evaluation/evaluate_pointsampled.py, UNMODIFIED.
The page/report builder MUST NOT invent, round differently, or extrapolate any number.
If a number you want is not here, say so, do not estimate it.

All artifacts (job scripts, per-shape parquets, converted npz, logs) live under
`/project/3dlg-hcvc/omages/yanxg_scratch/pointsampled/` on solar/NFS, not in this repo.

## 0. What was scored, and what it means

SegviGen predicts a per-voxel BINARY emissive mask at 512^3 resolution. Its stated method
is `emission = mask ⊙ albedo`. Dongchen's harness does not accept a mask; it reads one npz
per shape with `coords` + `pred` (an RGB emission field) and scores it, at his own 256^3
grid, against mesh-surface-sampled ground truth. So every row below required a conversion:
predict the mask (512 grid) → nearest-neighbor downsample onto Dongchen's own 256 coords →
multiply by his own baked albedo at those coords → write `pred` = that RGB field. Full
detail in `/project/.../pointsampled/scripts/convert_to_voxel_pred.py`.

Five checkpoints scored, all under `segvigen_emissive/outputs/`:

| tag | data | config |
|---|---|---|
| `emis_1k_w1` | 1k data | pos_weight 1 |
| `emis_1k_w5` | 1k data | pos_weight 5 |
| `emis_2k_bal` | 2k data | balanced sampling (pos_weight 1, balanced_pos_weight 50) |
| `emis_2k_w5` | 2k data | pos_weight 5 |
| `emis_72k_unfilt` | 72k unfiltered | the 24h run, 21 epochs |

Every model was sampled 3 times per shape (seed 0/1/2, 12 diffusion steps) because the
model is stochastic: measured per-shape draw std on IoU is 0.05-0.09 (mean_draw_std, see
§4), comparable to or larger than most models' own mean IoU, so a single draw is not a
measurement.

## 1. Colour space: verified, not assumed

Read directly from `data_processing/uv_voxel_pipeline/voxelize.py` (the code that wrote
`pbr_voxels_256`, the same GT bake this harness's own ceilings read):

- `voxelize.py:17-21`: "o_voxel's trimesh path does `baseColorFactor=uint8/255` and samples
  textures as raw bytes/255 with **no** sRGB→linear... We pre-bake LINEAR `factor × texture`
  into uint8 textures with white factors so stored voxels match the atlas bake."
- `voxelize.py:141-159` (`_linearize_materials`): base color texture is
  `srgb_to_linear(tex/255) * factor`, clipped to [0,1], written back as uint8 `*255`.

So `base_color` in `pbr_voxels_256` is ALREADY linear, canonical-space uint8: exactly the
space `evaluation/pointsampled/metrics.baked_to_canonical_linear` expects (`clip(x,0,1)`
after `/255`). Albedo = `base_color.astype(float64) / 255.0`, no further sRGB decode. Had
this been wrong (albedo actually sRGB-encoded), multiplying without linearizing would have
shifted every predicted value by roughly a factor of two and looked like a model result.

## 2. Gate check: the number I was originally told to reproduce was the wrong target

The brief asked me to reproduce `iou_trellis` = 0.24497 from
`evaluation/eval_pointsampled/eval_headtohead/summary_eval.json` (n=323) by regenerating a
`gt_emission` label. That target was mis-specified (caught and corrected mid-session): 0.24497
is TRELLIS.2's own DiT MODEL prediction from that run (`pred_voxels` of
`emission_dit_twostream_74k`), not a ground-truth ceiling, scored at `n_points=30000` on a
different, smaller aligned subset (n=323, `n_gt_on` mean 7740). A `gt_emission` regeneration
(pred = the actual baked value) has no reason to reproduce a model's own IoU, and the
populations differ, so this specific number cannot anchor the check.

`evaluation/newdata_eval/out/eval/gt_emission/` (the labelled score dir `score_texgen_v2.sh`
references) does not exist in this clone, so there is no published `gt_emission` summary
JSON to diff against directly. Anchored instead against **EXECUTION_LOG.md's documented
o-voxel bake ceiling** (2026-08-03 section, `_v2` head-to-head table), computed by the same
harness/conventions on n=371:

```
EXECUTION_LOG.md:  IoU med/mean 0.8297 / 0.7177   MAE med/mean 0.00179 / 0.02582
my regeneration:   IoU med/mean 0.8297 / 0.7177   MAE med/mean 0.00179 / 0.02582   (n=371)
```

Exact match to 4 significant figures on both IoU and MAE, same n=371 after the align gate.
This is the strongest verification available in this clone and it passes. Treat 0.7177 as
verified against a real prior measurement, not merely internally self-consistent.

## 3. The two ceiling rows (settles an open item in the paper skeleton)

Both scored on the SAME 371 shapes, same harness call, `align_fp` mean 0.9998:

| label | IoU med/mean | MAE med/mean | PSNR(emission) med/mean |
|---|---|---|---|
| GT voxel emission (pred = the actual baked RGB value) | 0.8297 / 0.7177 | 0.00179 / 0.02582 | 17.79 / 21.08 dB |
| GT mask × albedo (our formulation's ceiling: GT mask, GT albedo) | 0.8052 / 0.6807 | 0.00517 / 0.06237 | 10.92 / 12.18 dB |

Two findings, and both matter, not just the first:

1. **Nearly lossless for localization.** Multiplying by albedo instead of using the true
   baked value costs only 0.037 IoU (mean 0.7177 → 0.6807, about 5%). WHERE the mask says a
   shape emits is almost the whole IoU story regardless of what colour it emits.
2. **Substantially lossy for radiance.** The same substitution costs about 9 dB of emission
   PSNR (21.08 → 12.18 dB) and more than doubles MAE (0.0258 → 0.0624). HOW BRIGHT and WHAT
   COLOUR the formulation can produce is bounded well below the true bake, because albedo is
   frequently dark or grey where the true emissive colour is saturated: at the voxel level,
   of the cells the ORACLE mask turns on, a mean of 7.5% (median 0%, concentrated in a
   minority of shapes) drop back below the 1/255 threshold once multiplied by albedo,
   because the underlying surface colour there is too dark to carry emission (332/381 raw
   shapes have any GT-mask-on cell at all; the 7.5%/0% figures are over those 332). See §5
   for the per-model version of this effect.

**Consequence for reading the model rows' PSNR (§4):** the models score 9-12 dB, i.e. AT or
NEAR the formulation's own 12.18 dB ceiling. A hypothetically perfect mask predictor would
buy almost no radiance improvement over what the models already show. PSNR is not the axis
where these models are failing; IoU is.

**The ceiling itself is not ~1.0, and part of the reason is not a bake defect.** GT voxel
emission scores 0.7177 mean against the SAME mesh-sampled ground truth the harness uses for
every row: a roughly 28% gap between the voxel bake and the render-sampled truth, before
any model is involved. The manager separately confirmed (isolation render, a jack-o'-lantern
fixture outside this 381-shape set) that at least part of this gap is REAL interior geometry
invisible to every camera angle, not bake error: the emissive geometry read 254/255 once
unoccluded, 24/255 in the full scene. Checked whether this shows up inside the 381-shape set
itself: of the 51 shapes with zero mesh-sampled GT-emission points, 3 (5.9%) have
substantial voxel-side emissive fraction (0.86, 0.29, 0.036) despite reading as fully dark
on the mesh-sampled side. So the 0.7177 ceiling is not just a floor imposed by the bake or
the metric; a real fraction of it is geometry that no render-based ground truth, ours or
Dongchen's, can see. A model can be penalised by this harness for correctly predicting
emission that is genuinely there but invisible to the metric's own sampling.

## 4. The five model rows

Pooled across all 3 draws (n=1113 shape-draws = 371 aligned shapes × 3 seeds; align_fp
gate >=0.5, same gate and same shape population as the ceilings above):

| model | IoU med/mean | MAE med/mean | PSNR(emission) med/mean | n_psnr |
|---|---|---|---|---|
| `emis_1k_w1` | 0.0000 / 0.1419 | 0.02866 / 0.10724 | 10.01 / 11.78 dB | 1041 |
| `emis_1k_w5` | 0.0002 / 0.1185 | 0.03337 / 0.08647 | 10.63 / 12.21 dB | 1068 |
| `emis_2k_bal` | 0.0055 / 0.1170 | 0.04932 / 0.11935 | 10.27 / 11.34 dB | 1086 |
| `emis_2k_w5` | 0.0003 / 0.1197 | 0.04207 / 0.13168 |  8.96 / 10.91 dB | 1069 |
| `emis_72k_unfilt` | 0.0000 / 0.1000 | 0.01462 / 0.09475 |  8.71 / 11.16 dB | 1059 |

**The headline is that median IoU is 0.0000 for four of five models** (2k_bal reaches
0.0055, still near zero). On more than half the aligned shapes, every model except 2k_bal
produces a prediction with NO overlap at all with ground truth. The mean (0.10-0.14) is
carried by a minority of shapes with real overlap; do not read the mean as typical
performance; read the median.

Per-shape mean-across-draws (n=371, i.e. average the 3 seeds per shape before taking the
dataset median/mean; same underlying data, alternate aggregation) and the per-shape
draw-to-draw std:

| model | IoU median/mean (per-shape) | mean draw-std |
|---|---|---|
| `emis_1k_w1` | 0.0043 / 0.1419 | 0.0894 |
| `emis_1k_w5` | 0.0117 / 0.1185 | 0.0902 |
| `emis_2k_bal` | 0.0177 / 0.1170 | 0.0799 |
| `emis_2k_w5` | 0.0079 / 0.1197 | 0.0836 |
| `emis_72k_unfilt` | 0.0038 / 0.1000 | 0.0693 |

Per-seed IoU mean (all 5 models, all within ~0.02-0.03 of each other across seeds; no seed
is a systematic outlier):

| model | seed0 | seed1 | seed2 |
|---|---|---|---|
| `emis_1k_w1` | 0.1403 | 0.1295 | 0.1560 |
| `emis_1k_w5` | 0.1142 | 0.1057 | 0.1357 |
| `emis_2k_bal` | 0.1191 | 0.1019 | 0.1299 |
| `emis_2k_w5` | 0.1256 | 0.1114 | 0.1221 |
| `emis_72k_unfilt` | 0.0990 | 0.0907 | 0.1103 |

**Comparison to Dongchen's DiT baselines** (`docs/status/evaluation-results.md` /
EXECUTION_LOG `_v2` head-to-head, n=371): `albedo→emission` 0.0917/0.3031 median/mean IoU,
`pbr→emission` 0.0626/0.2991. Both beat every SegviGen row above on BOTH median and mean.
Caveat: I have not independently verified their prediction parquets carry the identical
371 shape-uuids as mine (both derive from the same `usable`+align-gate pipeline and both
report n=371, so they very likely are, but I did not intersect uuid sets to confirm it).
Read the ordering as indicative, not certified identical-population, until that check is
done.

## 5. Empty predictions (a result, not bookkeeping)

Fraction of the 381 shapes where a model's predicted mask is empty at every voxel
(pred_frac@thr0.5 == 0.0), per seed, from `pred_voxels/<model>_seed<k>/summary.json`:

| model | seed0 | seed1 | seed2 |
|---|---|---|---|
| `emis_1k_w1` | 36.7% | 37.3% | 35.2% |
| `emis_1k_w5` | 25.7% | 22.3% | 24.1% |
| `emis_2k_bal` | 12.6% | 10.8% | 9.4% |
| `emis_2k_w5` | 25.5% | 24.9% | 22.3% |
| `emis_72k_unfilt` | 30.2% | 32.0% | 31.5% |

This is broadly consistent with the fig7 11-shape gallery's qualitative read (72k model near-
empty on most of that small sample) holding at 381-shape scale: roughly a third of the
72k model's predictions are exactly empty, not "essentially nothing" on a handful of
unlucky shapes.

**Albedo-black loss, the per-model version of §3's finding**: of the voxels a model's OWN
mask turns on, the mean fraction that fall below the 1/255 threshold once multiplied by
albedo (median in parentheses):

| model | mean, range over 3 seeds (median, range over 3 seeds) |
|---|---|
| `emis_1k_w1` | 10.5-11.7% (0.05-0.42%) |
| `emis_1k_w5` | 12.0-12.3% (0.08-0.35%) |
| `emis_2k_bal` | 11.3-11.8% (0.33-0.44%) |
| `emis_2k_w5` | 11.7-12.4% (0.08-0.28%) |
| `emis_72k_unfilt` | 8.6-10.9% (0.02-0.13%) |

Similar magnitude across all five models and close to the oracle's own 7.5% mean (§3):
the loss is a property of the mask×albedo formulation on this dataset's albedo
distribution, not something a specific checkpoint is doing worse than the others.

**Downsample correspondence** (our 512-grid prediction → Dongchen's 256-grid coords, nearest-
neighbor, cap 2 finer-grid units): mean/median miss fraction ~0.00021 / 0.0 across every
model and seed: essentially perfect coordinate correspondence, confirming the shared
`glb_to_vxz.py` frame convention (bbox-center, scale=0.99999/extent, identity axes) that
`build_dataset_direct.py`'s own docstring already asserts.

## 6. Contamination

Per-model train-overlap against the 381-shape `usable` eval set, and the score with those
shapes excluded (from the SAME per-shape parquets, not a rerun):

| model | train sids checked | overlap | all-shapes mean IoU | excl.-contaminated mean IoU |
|---|---|---|---|---|
| `emis_1k_w1` | `dataset/train_1k_{fresh,reused}_sids.txt` union, 1124 | 2 / 381 | 0.1419 | 0.1427 |
| `emis_1k_w5` | same | 2 / 381 | 0.1185 | 0.1189 |
| `emis_2k_bal` | `dataset/train_2k_ef_manifest.json`, 2000 | 8 / 381 | 0.1170 | 0.1161 |
| `emis_2k_w5` | same | 8 / 381 | 0.1197 | 0.1196 |
| `emis_72k_unfilt` | directory membership, `dataset_direct/train_72k` | 311 / 381 (81.6%) | 0.1000 | 0.0839 |

The two contaminated 1k/2k shapes are `b5189396fc1c40519321eb4ee18a9938` and
`b75d432e0aef4bc980d905ba63665636` (both also inside the 8-shape 2k overlap). At 1k/2k
scale the effect on the mean is inside noise (±0.001-0.01, smaller than the seed-to-seed
spread in §4). `emis_72k_unfilt` is HEAVILY contaminated: 311 of 381 usable shapes are in
its own training split, and excluding them drops its mean IoU from 0.1000 to 0.0839 (also
drops its own val/test-only n from 371 to 68 aligned shapes). Label the `emis_72k_unfilt`
"all shapes" row as **memorization-inflated**, the same way this repo already labels
TEXGen's pinned-split row; the 68-shape excl.-contaminated row is the honest number for that
checkpoint and its median is still 0.0000.

## 7. Known deviation: zero-conditioning on four checkpoints trained with real conditioning

Verified directly in each checkpoint's own training log (`grep "\[data\]" train_*.log`):

| model | trained with |
|---|---|
| `emis_1k_w1` | `cond=real` (`train_231172.log`) |
| `emis_1k_w5` | `cond=real` (`train_231171.log`) |
| `emis_2k_bal` | `cond=real` (`train_231599.log`) |
| `emis_2k_w5` | `cond=real` (`train_231598.log`) |
| `emis_72k_unfilt` | `cond=zero` (`train_72k_unfilt.sbatch`, matches its own training) |

`emis_72k_unfilt`'s row above needs no conditioning caveat: it trained zero-cond, so
zero-cond inference is its matched, native condition. The other four rows were trained with
real DINOv3 image conditioning (an emission-free render of the asset, Path A's own recipe)
and are scored here with `dump_pred_voxels.py`'s hardcoded zero-conditioning, because the
`dataset_direct`-format eval inputs built for this run carry no `cond.pth`. This is an
out-of-distribution conditioning input for those four checkpoints and likely UNDERSTATES
their real capability. A thumbnail-conditioning bracket (build_cond_thumbnail.py backfill,
likely an OVERSTATE in the opposite direction since the TexVerse thumbnail is fully-lit and
can show where the emission is) was scoped as a second pass but NOT run in this session;
scripts are staged at
`/project/3dlg-hcvc/omages/yanxg_scratch/pointsampled/scripts/{cond_thumb_backfill.sbatch,dump_pred_voxels_cond.py,dump_pred_thumbcond.sbatch}`
but time went entirely into verifying and delivering the zero-cond table. Report the zero-cond
numbers above as a LOWER BOUND for those four checkpoints' true (Path-A-matched)
capability, not as their capability.

## 8a. EMA vs raw weights: checked, does not explain anything (added after initial delivery)

All five rows in §4 used `best.ckpt`, a symlink to the RAW (non-EMA) epoch on every model;
an `epoch_NNNN_ema.ckpt` exists alongside each. The manager flagged this because an older
qualitative page (`finetune_binary_v1`) showed coherent masks under `W1-EMA`/`W5-EMA`
labels where raw predictions here look speckled or empty, and because `train_emissive.py`'s
best-epoch selection (`train_emissive.py:294-309`) is based on the RAW checkpoint's quick-val
IoU only; EMA weights are never consulted, so `best.ckpt`'s epoch is not necessarily the best
EMA epoch either.

Decisive test run: `emis_1k_w1` with `epoch_0016_ema.ckpt` (the EMA weights paired with
`best.ckpt`'s own chosen epoch, 16), identical pipeline, identical 371 aligned shapes, zero-cond,
3 seeds:

| | IoU median | IoU mean | n_empty_pred (of 381, per seed) |
|---|---|---|---|
| `emis_1k_w1` raw (best.ckpt) | 0.0000 | 0.1419 | 140 / 142 / 134 |
| `emis_1k_w1` EMA (epoch_0016_ema.ckpt) | 0.0000 | 0.1314 | 111 / 124 / 113 |

**EMA is not better here; if anything slightly worse on mean, identical on median (both
0.0000), and this MATCHES the manager's independent 8-shape per-voxel reproduction, which
found EMA helped `w5` and hurt `w1`.** EMA does produce non-empty predictions on more shapes
(111-124 vs 134-142 empty of 381), so it is not doing nothing differently, but the extra
coverage does not turn into overlap with ground truth at the point-sampled level.

Per-voxel threshold sweep (the cheap diagnostic dump_pred_voxels.py already computes on its
own decoded voxel GT, THRS 0.2/0.3/0.4/0.5; NOT the point-sampled metric above, kept
separate deliberately):

| | thr 0.2 | thr 0.3 | thr 0.4 | thr 0.5 |
|---|---|---|---|---|
| raw, mean IoU (3 seeds) | 0.142/0.146/0.152 | 0.138/0.145/0.151 | 0.140/0.143/0.149 | 0.144/0.139/0.148 |
| EMA, mean IoU (3 seeds) | 0.150/0.141/0.153 | 0.143/0.133/0.149 | 0.138/0.128/0.137 | 0.137/0.124/0.134 |

Raw is roughly flat across thresholds; EMA is highest at 0.2 and monotonically DECREASES
through 0.5 on all 3 seeds, consistent with EMA shifting the predicted-probability
distribution lower (more voxels sit just above 0.2 than above 0.5). This says the fixed 0.5
operating point is not obviously wrong for raw, and is arguably the WORST of the four
tested points for EMA specifically, yet even at its best threshold (0.2) EMA's per-voxel
IoU is not clearly above raw's own range (raw already spans 0.138-0.152 across thresholds).

**Conclusion: EMA does not explain the SegviGen numbers in §4, and rerunning all five
checkpoints on EMA (the scoped "step 2") was cancelled** on this evidence rather than run
speculatively. The coherent-looking masks on the old `finetune_binary_v1` page most likely
come from a different cause: a different (smaller, hand-picked or differently distributed)
sample of shapes, and/or that page's own per-model threshold (0.2 for w5, 0.5 for w1) rather
than a uniform 0.5, not from EMA vs raw weights, which this session's 371-shape,
point-sampled test found is a wash to a slight regression for `w1`.

## 8. What this run does NOT claim

- Does not claim the zero-cond numbers for `emis_1k_w1`/`emis_1k_w5`/`emis_2k_bal`/`emis_2k_w5`
  represent those checkpoints' real capability (§7).
- Does not claim `emis_72k_unfilt`'s all-shapes row is a clean measurement (§6); its 68-shape
  excl.-contaminated row is the one to trust, and it is still small.
- Does not claim the SegviGen-vs-Dongchen-DiT ordering in §4 is certified same-population
  (uuid intersection not checked).
- Does not claim the 0.7177/0.6807 ceiling rows anchor against the literal 0.24497 figure
  in the original brief; that figure measures something else (§2).

## Reproduction

```
scripts under /project/3dlg-hcvc/omages/yanxg_scratch/pointsampled/scripts/:
  build_eval381.sbatch            -- build the 381 eval shapes (dataset_direct format)
  dump_pred.sbatch <tag> <ckpt> <seed>   -- zero-cond model inference (512-grid mask)
  convert_to_voxel_pred.py / convert.sbatch  -- 512->256 downsample, mask x albedo
  run_eval.sbatch <voxel_pred_root> <out_dir>  -- evaluation/evaluate_pointsampled.py
  final_aggregate.py              -- the tables in this file
contamination.json, final_report.json  -- raw numbers behind every table above
```
