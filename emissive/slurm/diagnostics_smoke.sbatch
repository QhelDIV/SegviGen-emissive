#!/bin/bash
#SBATCH -J emis_diag_smoke
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/diag_smoke_%j.log
set -euo pipefail

source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /3dlg-jupiter-project/lightgen/segvigen_emissive

python -c "
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'code', 'SegviGen'))
import torch, o_voxel, trellis2
print('IMPORT_OK', torch.cuda.is_available())
"

python emissive/diagnostics/diagnostics_run.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct \
  --split val_72k \
  --ckpt /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72k_unfilt/run1/best.ckpt \
  --train_curve /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72k_unfilt/run1/train_curve.json \
  --out /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72k_unfilt/run1/diagnostics_smoke.json \
  --n 5 --draws 3 --steps 12 --seed 0

echo SMOKE_ALL_DONE
