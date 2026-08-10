#!/bin/bash
#SBATCH -J condrep
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-venus-02,cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/cond_thumb/repair_%j.log
#
# Residual pass over whatever the verbatim Path A preprocessing could not handle
# (~1% of thumbnails: TexVerse's grayscale placeholder icon, and frames where
# BiRefNet finds no foreground). --repair swaps in preprocess_image_repair's two
# documented guards; it skips every shape that already has cond.pth, so it targets
# exactly the residual. Single task: only a few hundred shapes are left.
#
# Ends by asserting FULL coverage of both splits, because the whole point is that
# train_emissive.py --cond real hard-errors on the first shape without cond.pth.
set -euo pipefail
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/3dlg-jupiter-project/lightgen/segvigen_emissive
cd $ROOT

echo "REPAIR_START $(date -Is) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for SPLIT in val_72k train_72k; do
  python emissive/data_prep/build_cond_thumbnail.py --split $SPLIT --repair
done

python - <<'PY'
import os
import random
import torch
root = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct"
bad_total = 0
all_present = {}
for sp in ("train_72k", "val_72k"):
    d = os.path.join(root, sp)
    sids = sorted(os.listdir(d))
    missing = [s for s in sids if not os.path.exists(os.path.join(d, s, "cond.pth"))]
    print(f"COVERAGE {sp}: {len(sids) - len(missing)}/{len(sids)} have cond.pth, "
          f"missing={len(missing)} {missing[:20]}", flush=True)
    all_present[sp] = [s for s in sids if s not in set(missing)]
    bad_total += len(missing)
assert bad_total == 0, f"{bad_total} shapes still without cond.pth -- train --cond real would abort"
print("COVERAGE_OK", flush=True)

# Sampled content audit. Existence is not enough: an all-zero cond is byte-valid,
# loads fine, and is indistinguishable at train time from --cond zero -- i.e. it
# would silently reproduce the exact confound this dataset exists to remove. Load
# a random sample and assert shape/dtype/nonzero rather than trusting the writer.
random.seed(0)
n_bad = 0
for sp, sids in all_present.items():
    for s in random.sample(sids, min(150, len(sids))):
        c = torch.load(os.path.join(root, sp, s, "cond.pth"), map_location="cpu")["cond"]
        if tuple(c.shape) != (1, 1029, 1024) or c.dtype != torch.float32 or float(c.abs().max()) == 0:
            print(f"AUDIT_BAD {sp}/{s} shape={tuple(c.shape)} dtype={c.dtype} "
                  f"absmax={float(c.abs().max())}", flush=True)
            n_bad += 1
assert n_bad == 0, f"{n_bad} sampled cond.pth failed the shape/dtype/nonzero audit"
print("AUDIT_OK sampled 150 per split: all (1,1029,1024) float32 and nonzero", flush=True)
PY

echo "REPAIR_OK $(date -Is)"
