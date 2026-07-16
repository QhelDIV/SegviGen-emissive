#!/bin/bash
#SBATCH -J emis_train_v5
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/train_%j.log
# args: $1=epochs $2=dataset (full path, default the standard dataset root)
#       $3=out_subdir $4=cond (real|zero, default real)
#       $5=pos_weight (default 1.0 — only used when $6 (balanced_cap) is 0)
#       $6=balanced_cap (default 50; --balanced_pos_weight, 0=off falls back to fixed $5)
#       $7=lr_schedule (const|cosine, default cosine)
#       $8=val_split (default val) $9=save_every (default 4) ${10}=val_quick (default 8)
#       ${11}=select_on (all|nonzero, default nonzero) ${12}=train_split (default train)
#       ${13}=n_per_epoch (default 0=all)
# v5: per-shape balanced pos_weight (W_shape=min(CAP,(1-p)/p), p=sample's own emis_mask
# coverage — fixes the flat-mean scheme's blind spot where a fixed --pos_weight
# upweights emissive VOXELS within a shape but does nothing for shapes that are almost
# entirely non-emissive relative to half-emissive ones), cosine LR decay (--lr_schedule),
# and best.ckpt selection gated on nonzero-coverage quick-val shapes (--select_on) so a
# ckpt can't win by just predicting all-black on empty-glow shapes.
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
python emissive/train/train_emissive.py \
  --dataset "${2:-/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset}" \
  --init_ckpt full_seg \
  --out_dir /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/${3:-emis_v5} \
  --epochs ${1:-100} --lr 1e-5 --train_split ${12:-train} --val_split ${8:-val} \
  --n_per_epoch ${13:-0} \
  --cond ${4:-real} --pos_weight ${5:-1.0} --balanced_pos_weight ${6:-50} \
  --lr_schedule ${7:-cosine} --select_on ${11:-nonzero} \
  --emis_oversample --ema 0.999 \
  --save_every ${9:-4} --val_quick ${10:-8}
