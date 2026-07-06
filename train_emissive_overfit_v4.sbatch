#!/bin/bash
#SBATCH -J emis_overfit_w
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/train_%j.log
# Sanity gate for the v4 weighted-loss trainer: can it still overfit a tiny set to
# ~1.0 IoU with --pos_weight on (previous UNweighted overfit-1 baseline: 0.946@ep30)?
# args: $1=split_name (e.g. overfit_1, canon_overfit10) $2=out_subdir $3=epochs (default 30)
#       $4=pos_weight (default 5.0) $5=cond (default zero — matches the historical
#          baseline's --zero_cond exactly) $6=n_per_epoch (default 20 — MUST match the
#          original train_emissive_overfit.sbatch's default or this is not an
#          apples-to-apples step-count comparison; a single-sample split with
#          n_per_epoch=0 does only 1 grad step/epoch, not 20) $7=val_quick (default 1;
#          set to the split size for a multi-sample overfit gate, e.g. 10 for canon_overfit10)
#          $8=init_ckpt (default full_seg; pass an epoch_XXXX.ckpt path to RESUME a
#          previous run — fresh optimizer/epoch-numbering is fine for a gate extension)
#          $9=save_every (default 5)
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
python code/train_emissive.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset \
  --init_ckpt "${8:-full_seg}" \
  --out_dir /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/${2:-overfit1_w5} \
  --epochs ${3:-30} --lr 1e-5 --train_split "$1" --val_split "$1" --n_per_epoch ${6:-20} \
  --cond ${5:-zero} --pos_weight ${4:-5.0} --emis_oversample --ema 0.999 \
  --save_every ${9:-5} --val_quick ${7:-1}
