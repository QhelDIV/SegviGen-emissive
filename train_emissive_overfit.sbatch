#!/bin/bash
#SBATCH -J emis_overfit
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/train_%j.log
# Sanity test: can the pipeline overfit a tiny set to ~1.0 IoU?
# args: $1=split_name (e.g. overfit_1) $2=out_subdir $3=epochs (default 50) $4=n_per_epoch (default 20)
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPT=/3dlg-jupiter-project/lightgen/hf_cache/hub/models--fenghora--SegviGen/snapshots/73326b77a17b5194bcc918dbf13d8d5cefb1cca5/full_seg.ckpt
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
python code/train_emissive.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset \
  --segvigen_ckpt "$CKPT" \
  --out_dir /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/${2:-emis_overfit} \
  --epochs ${3:-50} --lr 1e-5 --train_split "$1" --zero_cond --emis_oversample \
  --n_per_epoch ${4:-20} --save_every 10
