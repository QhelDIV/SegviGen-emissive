#!/bin/bash
#SBATCH -J emis_overfit_direct1
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/train_overfit1_direct_%j.log
# Overfit-1 memorization check on the direct-ovoxel loader's dataset_direct/smoke1
# (single shape 294095f9c38d48f39b6f9b7162b963d7). This is a MEMORIZATION check only:
# loss should drive toward zero and per-voxel IoU on this same shape should approach
# ~0.9+. It is NOT a generalization result.
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
python emissive/train/train_emissive.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct \
  --init_ckpt full_seg \
  --out_dir /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/overfit1_direct \
  --epochs 200 --lr 1e-5 --train_split smoke1 --val_split smoke1 --n_per_epoch 1 \
  --cond zero --pos_weight 5.0 \
  --lr_schedule const --select_on nonzero \
  --ema 0.999 \
  --save_every 20 --val_quick 1
