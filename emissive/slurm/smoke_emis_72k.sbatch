#!/bin/bash
#SBATCH -J emis_smoke_72k
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72k_unfilt/smoke_%j.log
set -euo pipefail

mkdir -p /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72k_unfilt

source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "SMOKE_START $(date -Iseconds) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /3dlg-jupiter-project/lightgen/segvigen_emissive
python -u emissive/train/train_emissive.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct \
  --train_split train_72k \
  --val_split val_72k \
  --out_dir /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72k_unfilt/smoke \
  --init_ckpt full_seg \
  --cond zero \
  --pos_weight 5.0 \
  --lr 1e-5 --lr_schedule const --ema 0.999 \
  --n_per_epoch 24 --epochs 3 --save_every 1 \
  --select_on nonzero --val_quick 2 \
  2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }'

echo "SMOKE_DONE $(date -Iseconds)"
