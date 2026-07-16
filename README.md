# Emissive branch: binary emissive-mask fine-tuning for SegviGen

This branch is a pure addition on top of upstream Nelipot-Lee/SegviGen. No
upstream file is modified; everything new lives under `emissive/` (reorganized
2026-07-16 from an earlier flat-at-repo-root layout — see "Layout" below), so
upstream's own files (`inference_full.py`, `inference_interactive.py`,
`trellis2/`, `data_toolkit/`, `assets/`) stay exactly where upstream put them.

## What this adds

SegviGen's pretrained `full_seg` flow predicts per-voxel part-segmentation
colors. This branch repurposes that same `Gen3DSeg` wrapper and sampling
convention to predict a **binary emissive mask** (white = emissive, black =
not) instead of multi-class part colors, via a standalone fine-tuning loop
that warm-starts from a SegviGen checkpoint.

Pipeline stages:
1. **Data prep** — build a dataset of (shape, DINOv3 image condition,
   binary emissive target) triples from "somage" assets.
2. **Fine-tune** — a standalone training loop for `Gen3DSeg` with a
   per-voxel `pos_weight` (for class imbalance), EMA of weights, and
   best-checkpoint selection on a held-out split.
3. **Eval** — threshold sweep, Otsu thresholding, and stratified metrics
   (by emissive-voxel prevalence) against the fine-tuned or baseline model.

## Layout

```
emissive/
  data_prep/   somage_to_glb.py  make_emis_mask.py  build_dataset.py  gt_parts_extract.py
  train/       train_emissive.py
  eval/        eval_emissive.py  seg_covers_emissive.py  seg_to_mesh.py  make_pred_glb.py
  infer/       predict_emissive.py            # agent-facing: glb in -> mask + mesh out
  slurm/       *_v5.sbatch, eval_emissive.sbatch, eval_val96.sbatch, build_dataset*.sbatch,
               make_pred_glb.sbatch, seg_covers_*.sbatch, dl_weights.sbatch, gpu_smoke.sbatch,
               build_canon10.sbatch
  slurm/archive/  superseded train_emissive* variants (kept for history, not for use)
  env/         setup_trellis_env.sh  resume_env.sh  smoke_test.py
  docs/        EXPERIMENTS.md                 # run/checkpoint registry — read this first
  canon_overfit10.txt
```

Data prep / conversion (`emissive/data_prep/`):
- `somage_to_glb.py` — convert our "somage" asset format to GLB so
  SegviGen's `data_toolkit` (glb_to_vxz etc.) can ingest it.
- `make_emis_mask.py` — compute the binary per-voxel emissive mask from
  source asset metadata.
- `build_dataset.py` — end-to-end dataset builder: somage → GLB → voxel
  grid → binary emissive target + DINOv3 conditioning, written to a
  train/val split.
- `gt_parts_extract.py` — extract ground-truth part labels from source
  assets (used for stratified eval and sanity checks).

Training (`emissive/train/`):
- `train_emissive.py` — standalone fine-tune loop for `Gen3DSeg`, warm-started
  from a SegviGen checkpoint (`--init_ckpt`, defaults to `full_seg`).
  Per-voxel `pos_weight` (fixed or adaptive per-shape balanced), cosine LR,
  EMA, and best-checkpoint-by-eval-metric selection (`--select_on`). Reuses
  `emissive/eval/eval_emissive.py`'s `load_eval_models` / `evaluate_split` for
  periodic validation during training (cross-directory import — see
  "Import-path fix" below).

Eval (`emissive/eval/`):
- `eval_emissive.py` — evaluation: loads a checkpoint (baseline or
  fine-tuned), samples `--draws` independent draws and averages, runs
  threshold sweep + Otsu, reports metrics stratified by emissive-voxel
  prevalence (`--stratify`, `--bucket_by`).
- `seg_covers_emissive.py` — exploratory script (no training): runs the
  pretrained `full_seg.ckpt` as-is on a few assets to see how much of the
  emissive mask its part-segmentation output already covers (the oracle bar
  in `emissive/docs/EXPERIMENTS.md`).
- `seg_to_mesh.py` — renders full part-segmentation on the actual decoded
  mesh surface (vertex-colored GLB) rather than coarsened voxel cubes, for
  qualitative inspection.
- `make_pred_glb.py` — decode ONE fine-tuned prediction to a paper-style GLB
  mesh (official remesh+bake via `inference_full.slat_to_glb`), for a
  qualitative view of a checkpoint's output as a smooth surface.

Inference (`emissive/infer/`):
- `predict_emissive.py` — agent-facing CLI: any `.glb` in → predicted binary
  emissive mask (`mask.npz`) + decoded mesh (`pred_mesh.glb`) out. Reuses
  `eval_emissive.py`'s model loading/sampling and `make_pred_glb.py`'s decode
  path (see the script's own docstring for the full attribution). Untested on
  GPU as of this commit — see `emissive/docs/EXPERIMENTS.md`.

Env (`emissive/env/`):
- `smoke_test.py` — loads TRELLIS.2-4B + SegviGen checkpoints from the HF
  cache and introspects the flow's conditioning dims; used to verify the
  environment before a real run.
- `setup_trellis_env.sh`, `resume_env.sh` — cluster env setup / resume helpers.

Batch scripts (`emissive/slurm/*.sbatch`): one per script above (multiple
dated variants exist for `train_emissive*` and `build_dataset*` — later
`_v2`, `_v3`, `_v4`, `_v5` sbatch files reflect iterated
hyperparameters/paths, not supersession; check the script body for what
changed). Superseded `train_emissive*` variants (older hyperparameter/path
iterations of the same launcher, since replaced by `train_emissive_v5.sbatch`)
are archived under `emissive/slurm/archive/` (kept for history, not for use).
`eval_emissive.sbatch` itself is NOT superseded — it's iterated in place
(same filename, current v5-era args) — so it stays in the active `slurm/`
dir alongside `eval_val96.sbatch`.
**Deployment convention:** the whole repo is rsynced to the cluster and every
sbatch script is submitted from the repo root (`cd`'d to inside the script,
then `python emissive/<sub>/<script>.py ...` — repo-root-relative paths, not
the old flat `code/<script>.py` convention this project used before the
reorg).

Registry (`emissive/docs/EXPERIMENTS.md`): the run/checkpoint registry —
which arm used what data/loss, headline eval numbers, and the recommended
checkpoint for inference. Read this before picking a `--ckpt` for
`predict_emissive.py` or launching a new training run.

Data:
- `emissive/canon_overfit10.txt` — a 10-shape sid list used for a fast overfit
  sanity-check split (referenced by `emissive/slurm/build_canon10.sbatch`).

## Import-path fix

Upstream these scripts lived *next to* a separate `SegviGen/` clone, so they
did `sys.path.insert(0, os.path.join(ROOT, "SegviGen"))` to reach
`inference_full`. They were later merged into this repo's root (so
`ROOT == inference_full.py`'s directory), and are now nested two levels
further down under `emissive/<sub>/`. Each script that imports
`inference_full`/`trellis2`/`o_voxel` resolves the repo root by walking
upward from `__file__` until it finds `inference_full.py`, so it works at any
nesting depth:

```python
ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(ROOT, "inference_full.py")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError(f"could not locate SegviGen repo root (inference_full.py) above {__file__}")
    ROOT = parent
SEGVIGEN = ROOT
sys.path.insert(0, SEGVIGEN)
```

Fixed in: `emissive/data_prep/build_dataset.py`, `emissive/eval/eval_emissive.py`,
`emissive/eval/seg_covers_emissive.py`, `emissive/eval/seg_to_mesh.py`,
`emissive/eval/make_pred_glb.py`, `emissive/env/smoke_test.py`,
`emissive/train/train_emissive.py`, `emissive/infer/predict_emissive.py`.
`train_emissive.py` and `predict_emissive.py` additionally add the sibling
`emissive/eval/` dir to `sys.path` (they import from `eval_emissive.py`,
which now lives one directory over, not next to them).

## Cluster paths to adjust

All hardcoded paths point at the SFU CS `/3dlg-jupiter-project` cluster
mount (no `/local-scratch2` or `/localhome` paths were carried over).
Teammates on a different cluster/environment need to update:

- `/3dlg-jupiter-project/lightgen/hf_cache` — HF cache dir (`HF_HOME`),
  referenced in every `.py` and most `.sbatch` files.
- `/3dlg-jupiter-project/lightgen/segvigen_emissive/...` — working/output
  dirs (dataset, checkpoints, logs), referenced throughout the `.sbatch`
  files and in `emissive/data_prep/make_emis_mask.py`.
- `/3dlg-jupiter-project/lightgen/miniforge3/...` — conda env activation,
  in every `.sbatch` file.
- `/3dlg-jupiter-project/lightgen/diffusionnet_xg/...` — source label/split
  files used by `build_dataset.py` (`labels_uv_74k`,
  `data_splits_74k.json`).
- HF checkpoint hash `.../models--fenghora--SegviGen/snapshots/<hash>/full_seg.ckpt`
  in several `.sbatch` files — pin will drift if the HF repo is updated.

## Relation to upstream

Pure addition: no file from upstream `Nelipot-Lee/SegviGen` is modified.
All new files live under `emissive/` (see "Layout" above); upstream's own
files remain untouched at the repo root.

## Results

https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/index.html
