# Fact sheet: SegviGen emissive experiment, current state
# Assembled and verified 2026-07-30/31 by the master session.
# EVERY number here was measured this session unless marked [PRIOR].
# The page builder MUST NOT invent, round differently, or extrapolate any number.
# If a number you want is not here, the correct action is to say so on the page
# or ask, NOT to estimate it.

## 0. What the experiment is

Fine-tune TRELLIS.2's `slat_flow_imgshape2tex` (the 1.3B sparse DiT,
`slat_flow_imgshape2tex_dit_1_3B_512_bf16`), warm-started from SegviGen's
`full_seg.ckpt`, to predict a PER-VOXEL BINARY EMISSIVE mask: for every occupied
voxel on a shape's surface, does it emit light or not.

Three latent channels. OWNER-RATIFIED GLOSSES, use these words, do not invent others:
  - `shape_slat`      = "where the surface is"
  - `input_tex_slat`  = "how it reflects light"
  - `output_tex_slat` = "what it emits"      <- the prediction target
NEVER gloss PBR as "what it's made of".

## 1. The resolution ladder (the thing that confuses everyone)

Three different grids, all live at once:
  - 256^3  Dongchen's attribute bake (his `pbr_voxels_256`, `emission_voxels_256`)
  - 512^3  OUR grid. Locked, because the pretrained TRELLIS.2 encoder's contract is 512.
  - 32^3   the latent. 16x downsample from 512.

Consequences:
  - one latent token covers 16^3 = 4,096 cells of the 512 grid
  - the same token covers 8^3 = 512 attribute values of the 256 bake
We upsample 256 -> 512 by parent lookup (`coords512 // 2`) with a cKDTree fallback
for cells with no exact parent.

## 2. The channel hijack (why no new encoder was trained)

Emission is binarized at > 1/255 (ANY emission counts) and written into the
`base_color` slot, with constants:
  OUT_METALLIC_U8 = 0, OUT_ROUGHNESS_U8 = 255, OUT_ALPHA_U8 = 255
(build_dataset_direct.py lines 97-99). So the pretrained PBR encoder is reused
byte-identically; no architecture change, no new encoder.

## 3. The emissive mip bug (commonly misstated -- state it CORRECTLY)

Dongchen's voxelize.py carries `_FINEST_MIP_OFFSET = -1e6` passed as
`mip_level_offset`. THE BUG IS ABOUT TEXTURE SIZE, NOT GRID RESOLUTION:
it affects emissive textures >= 512x512 PIXELS. Coarse mips read garbage, and
o_voxel's emissive default is white, so black/sparse emissive textures bake
near-white. The fix is one parameter and applies at ANY grid size.
DO NOT write that the bug is specific to 512^3 grids. That claim is FALSE and was
already retracted once on the pipeline_design page.

## 4. Dataset construction outcome (build array 237094)

Splits (data_splits_74k.json, indices into the emissive_thumbnails parquet):
  train 59,602 | val 7,450 | test 7,451 | TOTAL 74,503

Built and on disk:
  train 57,968 | val 7,290 | test 7,288 | TOTAL 72,546

Missing: 1,957 total, decomposing EXACTLY as:
  1,036  source never existed (permanent, known since the build was planned)
    584  in Dongchen's rebake list, his rebake COMPLETED but produced no output
    337  buildable right now (source present, ours to rebuild)
  (1,036 + 584 + 337 = 1,957)

Build failures logged: 1,791 FileNotFoundError + 70 CUDA OutOfMemoryError.
  Of the 1,791: 1,032 permanent-list, 759 rebake-list, 0 unexplained.
  Of the 759 rebake-list failures, only 177 had source reappear after his jobs cleared.
Shard 211 of 375: host-OOM killed, wrote no manifest; 103 of its 200 shapes
landed on disk anyway. 374/375 manifests exist.

CAVEAT THE PAGE MUST STATE: "1,957 missing" and "~921 missing" are both correct
under different denominators -- 1,957 against the full 74,503 split, ~921 against
the 73,467 we expected to be buildable after excluding the 1,036 permanent.

## 5. Validation of our 512 build against Dongchen's 256 bake

Geometry, our 512 dual vertices averaged down vs his `dual_grid_256`:
  72,092,657 vertex pairs across 400 shapes
  mean disagreement 0.119 voxel widths | p90 0.29 | worst 1.54
  mean signed per-axis offset [+0.0011, +0.0010, +0.0011]
  => no systematic offset, no frame bug.

Occupancy ratio 512 vs 256: median 4.10x. This is SURFACE scaling (2^2), not
volume scaling (2^3) -- the expected result for a surface-supported grid.

gap_frac (cells with no exact 256 parent): 31.3% of shapes exactly zero,
median 0.000023, but 31 of 17,894 exceed 10% and one tiny shape reaches 1.0.

## 6. What the dataset actually contains (class balance)

Sampled n=1,998 random train shapes, reading each meta.json emissive_frac:
  fraction of shapes with emissive_frac > 0       : 90.8%
  > 0.001  : 79.7%
  > 0.01   : 59.3%
  > 0.1    : 36.9%
  > 0.5    : 22.9%
  median 0.025 | mean 0.244

Read this as BIMODAL: most shapes have a small emissive region, and a large
minority is almost entirely emissive. The mean >> median gap IS the story.
The near-fully-emissive group is largely "fullbright" content, where the
emissive texture equals the base color.

## 7. The filter that EXISTS but is NOT being applied

`vis_data/emissive_filtering/stage1_survivors.txt` holds 32,121 sids.
Intersected with what we built:
  train 25,547 | val 3,262 | test 3,241 | TOTAL 32,050 survivors built

THE OWNER HAS DECIDED to train on the UNFILTERED set first. The page must
present this as a deliberate, reasoned choice (a baseline over everything, with
filtering as a planned later ablation), NOT as an oversight and NOT as a
recommendation being ignored. Do not editorialize about it.

## 8. Training configuration (as launched / to launch)

Model: TRELLIS.2-4B `slat_flow_imgshape2tex_dit_1_3B_512_bf16`
Warm start: SegviGen `full_seg.ckpt`
Gradient checkpointing: enabled on 31 modules (fits the 46GB L40S)
  --cond zero        (NO image conditioning; see below)
  --lr 1e-5          --lr_schedule const
  --pos_weight 5.0   --ema 0.999   --select_on nonzero
  --emis_oversample  OFF   <- deliberate: on unfiltered data it would
                              preferentially sample the fullbright group
                              (emissive_frac near 1.0), amplifying the noise.
                              Every PRIOR run had it ON, so this does break
                              comparability with the pilots. Say so.

CONDITIONING: zero of the 72,546 built shapes have cond.pth, and
build_dataset_direct.py has no conditioning code path at all. DINOv3-L IS now
downloaded (1.2G, model.safetensors present) so this is NO LONGER a licensing
block -- it is simply unbuilt. State it that way.

## 9. Measured cost (from smoke test job 237741, 3 epochs x 24 shapes)

  ~1.25 s/shape          (epoch 2: 27s/24 shapes; epoch 3: 34s/24 shapes)
  ~7.5 min dataset init  (~350k Lustre metadata ops: listdir + ~5 stats +
                          a meta.json open per shape, single-threaded)
  => ONE FULL EPOCH over 57,968 shapes ~ 20 HOURS

Smoke test outcome: warm-start, checkpointing, EMA, quick-val all functional.
Loss 0.292 -> 0.225 across 3 epochs.
Quick-val IoU 0.0003 at n=2.
  *** THIS IS A PLUMBING CHECK ONLY. n=2, 72 total training samples seen. ***
  *** DO NOT compare it to 0.259 or 0.203. DO NOT plot it as a result.    ***
  *** If you show it at all, label it explicitly as a pipeline check.     ***

## 10. Prior results -- ALL SUPERSEDED, mark them clearly

Every published result page (finetune_binary_v1, results_2k_v1,
training_curves_v1, fullseg_canon10*, gt_vs_pred_canon10, official_repro) is
"Path A": the somage/GLB round-trip pipeline that the direct-ovoxel design
replaced. Their numbers DO NOT describe the current experiment.

[PRIOR] best zero-cond pilot, 232 shapes: val IoU 0.203 at epoch 25
[PRIOR] v2 (232 shapes, oversample): 0.095 at ep10, 0.068 at ep20 (overfit)
[PRIOR] best honest held-out across Path A: ~0.15 and declining

## 11. The baselines, and a metric trap that MUST appear on the page

From the paper (sec/5_experiment.tex): three baselines --
  TEXGen      (works in UV space)
  TRELLIS.2   (3D latent; two variants: replace albedo with emission, or
               replace all PBR with a single emission channel)
  DiffusionNet (works on the surface)

DiffusionNet val IoU 0.259 is real and reproducible ON A CLEAN SPLIT.
BUT: it is a PER-FACE metric, and SegviGen's is PER-VOXEL. These are
INCOMMENSURABLE. eval_emissive.py already calls itself a "proxy". The 0.259
number must NOT be presented as a target SegviGen is chasing on equal terms.
This is the single most important honesty point on the page.

## 12. Open decisions (present as open, do not resolve them on the page)

  1. Real run budget: --n_per_epoch (proposed 3,000 ~= 65 min/epoch) and
     --epochs; needs a non-debug partition.
  2. --emis_oversample: currently off; confirm or flip.
  3. Mop-up of the 337 buildable shapes: recommended SKIP (+0.6%, cannot move
     a result).
  4. The 584 shapes Dongchen's rebake did not produce: a question for him.

## 13. Reusable assets

48 rendered PNGs at web/_preview/data_compare/img/ named
  <sid>_{geom,emis}{256,512}.png
i.e. matched geometry/emission renders at both resolutions for the same shape.
These are ALREADY VERIFIED and are the best available "look at the actual data"
material. Use them.
SVG diagram helpers to build on: web/_preview/data_pipeline/diagrams.py
  (svg_figure, _rect, _line, _text, _arrow, _box, _lane_band, _filecard,
   diagram_pipeline_flow, diagram_resolution_ladder)
and web/_preview/ovoxel_explained/diagrams.py (597 lines, richer vocabulary).
