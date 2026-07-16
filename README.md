# Emissive branch: binary emissive-mask fine-tuning for SegviGen

This branch is a pure addition on top of upstream Nelipot-Lee/SegviGen. No
upstream file is modified; everything below is new, added at the repo root
so that scripts can `from inference_full import Gen3DSeg` directly (as if
they were part of the original codebase).

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

## Files added

Data prep / conversion:
- `somage_to_glb.py` — convert our "somage" asset format to GLB so
  SegviGen's `data_toolkit` (glb_to_vxz etc.) can ingest it.
- `make_emis_mask.py` — compute the binary per-voxel emissive mask from
  source asset metadata.
- `build_dataset.py` — end-to-end dataset builder: somage → GLB → voxel
  grid → binary emissive target + DINOv3 conditioning, written to a
  train/val split.
- `gt_parts_extract.py` — extract ground-truth part labels from source
  assets (used for stratified eval and sanity checks).

Training / eval:
- `train_emissive.py` — standalone fine-tune loop for `Gen3DSeg`, warm-started
  from a SegviGen checkpoint (`--init_ckpt`, defaults to `full_seg`).
  Per-voxel `pos_weight`, EMA, and best-checkpoint-by-eval-metric selection.
  Reuses `eval_emissive.py`'s `load_eval_models` / `evaluate_split` for
  periodic validation during training.
- `eval_emissive.py` — evaluation: loads a checkpoint (baseline or
  fine-tuned), runs threshold sweep + Otsu, reports metrics stratified by
  emissive-voxel prevalence.
- `seg_covers_emissive.py` — exploratory script (no training): runs the
  pretrained `full_seg.ckpt` as-is on a few assets to see how much of the
  emissive mask its part-segmentation output already covers.
- `seg_to_mesh.py` — renders full part-segmentation on the actual decoded
  mesh surface (vertex-colored GLB) rather than coarsened voxel cubes, for
  qualitative inspection.
- `smoke_test.py` — loads TRELLIS.2-4B + SegviGen checkpoints from the HF
  cache and introspects the flow's conditioning dims; used to verify the
  environment before a real run.

Batch scripts (`*.sbatch`, Slurm): one per script above (multiple dated
variants exist for `train_emissive*` and `build_dataset*` — later `_v2`,
`_v3`, `_v4` sbatch files reflect iterated hyperparameters/paths, not
supersession; check the script body for what changed).

Data:
- `canon_overfit10.txt` — a 10-shape sid list used for a fast overfit
  sanity-check split (referenced by `build_canon10.sbatch`).

## Import-path fix

Upstream these scripts lived *next to* a separate `SegviGen/` clone, so they
did `sys.path.insert(0, os.path.join(ROOT, "SegviGen"))` to reach
`inference_full`. Now that they live inside the SegviGen repo itself, that
path doesn't exist. Each script now does:

```python
SEGVIGEN = os.path.join(ROOT, "SegviGen")
if os.path.isdir(SEGVIGEN):
    sys.path.insert(0, SEGVIGEN)   # legacy: sits next to a separate clone
else:
    SEGVIGEN = ROOT                # new: lives inside the SegviGen repo root
    sys.path.insert(0, ROOT)
```

This keeps the scripts working unmodified if someone still runs them from
the old side-by-side layout, while working correctly in this repo. Fixed in:
`build_dataset.py`, `eval_emissive.py`, `seg_covers_emissive.py`,
`seg_to_mesh.py`, `smoke_test.py`, `train_emissive.py`.

## Cluster paths to adjust

All hardcoded paths point at the SFU CS `/3dlg-jupiter-project` cluster
mount (no `/local-scratch2` or `/localhome` paths were carried over).
Teammates on a different cluster/environment need to update:

- `/3dlg-jupiter-project/lightgen/hf_cache` — HF cache dir (`HF_HOME`),
  referenced in every `.py` and most `.sbatch` files.
- `/3dlg-jupiter-project/lightgen/segvigen_emissive/...` — working/output
  dirs (dataset, checkpoints, logs), referenced throughout the `.sbatch`
  files and in `make_emis_mask.py`.
- `/3dlg-jupiter-project/lightgen/miniforge3/...` — conda env activation,
  in every `.sbatch` file.
- `/3dlg-jupiter-project/lightgen/diffusionnet_xg/...` — source label/split
  files used by `build_dataset.py` (`labels_uv_74k`,
  `data_splits_74k.json`).
- HF checkpoint hash `.../models--fenghora--SegviGen/snapshots/<hash>/full_seg.ckpt`
  in several `.sbatch` files — pin will drift if the HF repo is updated.

## Relation to upstream

Pure addition: no file from upstream `Nelipot-Lee/SegviGen` is modified.
All new files sit at the repo root as siblings of `inference_full.py`.

## Results

https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/index.html
