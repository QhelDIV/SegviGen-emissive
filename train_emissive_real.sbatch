#!/bin/bash
#SBATCH -J emis_train_real
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/train_%j.log
# args: $1=epochs $2=train_split $3=out_subdir
# Real-cond variant of v3: uses DINOv3 image features (cond.pth) instead of zeros.
# Decisive comparison vs zero-cond (best 0.20). Keep oversample for class imbalance.
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPT=/3dlg-jupiter-project/lightgen/hf_cache/hub/models--fenghora--SegviGen/snapshots/73326b77a17b5194bcc918dbf13d8d5cefb1cca5/full_seg.ckpt
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
python code/train_emissive.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset \
  --segvigen_ckpt "$CKPT" \
  --out_dir /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/${3:-emis_real} \
  --epochs ${1:-12} --lr 1e-5 --train_split ${2:-train} --real_cond --emis_oversample --save_every 2
