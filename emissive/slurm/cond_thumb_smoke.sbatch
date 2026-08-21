#!/bin/bash
#SBATCH -J condsmoke
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/cond_thumb/smoke_%j.log
#
# Smoke: build thumbnail conditioning for 8 val_72k shapes, verify the tensor
# contract against a Path A cond.pth, then prove train_emissive.py --cond real
# actually loads them (a symlink split of exactly those 8 shapes, 2 steps, no
# quick-val). Ends in SMOKE_OK.
set -euo pipefail
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/3dlg-jupiter-project/lightgen/segvigen_emissive
cd $ROOT

echo "SMOKE_START $(date -Is) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python emissive/data_prep/build_cond_thumbnail.py --split val_72k --limit 8 --verify
echo "COND_OK"

# a throwaway split holding exactly the 8 shapes we just built cond for, so
# EmisDataset's hard "cond.pth missing" error is a real test rather than a
# whole-split precondition we cannot yet satisfy.
SM=$ROOT/dataset_direct/condsmoke8
rm -rf $SM && mkdir -p $SM
python - <<'PY'
import os
root = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct"
src = os.path.join(root, "val_72k")
dst = os.path.join(root, "condsmoke8")
sids = [s for s in sorted(os.listdir(src)) if os.path.exists(os.path.join(src, s, "cond.pth"))][:8]
assert len(sids) == 8, f"expected 8 shapes with cond.pth, got {len(sids)}"
for s in sids:
    os.symlink(os.path.join(src, s), os.path.join(dst, s))
print("smoke split:", sids)
PY

python emissive/train/train_emissive.py \
  --dataset $ROOT/dataset_direct \
  --train_split condsmoke8 \
  --val_split condsmoke8 \
  --out_dir $ROOT/outputs/cond_smoke \
  --cond real \
  --pbr_cond token \
  --init_ckpt full_seg \
  --epochs 1 \
  --n_per_epoch 2 \
  --save_every 1 \
  --val_quick 0 \
  --lr 1e-5 \
  --lr_schedule const \
  --pos_weight 5.0 \
  --ema 0.999

echo "SMOKE_OK $(date -Is)"
