#!/bin/bash
#SBATCH -J emis72k
#SBATCH --partition=3dlg-hcvc-lab-long
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72k_unfilt/run1_%j.log
#
# First full-scale run: unfiltered 72k direct-ovoxel dataset, zero conditioning.
# Owner decisions: unfiltered (filtering is a later ablation), --cond zero,
# --emis_oversample OFF (on unfiltered data it would preferentially sample the
# fullbright group, emissive_frac near 1.0, and amplify label noise).
#
# Sizing: measured ~1.25 s/shape, so --n_per_epoch 3000 is ~65 min/epoch.
# 24 h wall-clock is ~22 epochs ~= 1.1 passes over the 57,968 train shapes.
# --epochs 24 deliberately exceeds what fits: the time limit is the binding
# constraint and every epoch checkpoints, so a mid-epoch kill loses <=1 epoch.
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /3dlg-jupiter-project/lightgen/segvigen_emissive

echo "RUN_START $(date -Is) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python code/train_emissive.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct \
  --train_split train_72k \
  --val_split val_72k \
  --out_dir /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72k_unfilt/run1 \
  --cond zero \
  --init_ckpt full_seg \
  --epochs 24 \
  --n_per_epoch 3000 \
  --save_every 1 \
  --val_quick 16 \
  --select_on nonzero \
  --lr 1e-5 \
  --lr_schedule const \
  --pos_weight 5.0 \
  --ema 0.999 \
  2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush() }'

echo "RUN_END $(date -Is)"
