#!/bin/bash
#SBATCH -J emis_train
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/train_%j.log
# args: $1=epochs $2=train_split $3=out_subdir $4=cond (real|zero, default real)
#       $5=pos_weight (default 5.0) $6=val_split (default val) $7=save_every (default 4)
#       $8=val_quick (default 8) $9=n_per_epoch (default 0=all)
# v4: wires up the class-imbalance fixes — weighted flow loss (--pos_weight, needs
# emis_mask.pth from make_emis_mask.py / build_dataset.py), EMA shadow weights,
# best-ckpt quick-val tracking (train_curve.json), explicit --cond (no silent
# zero-cond fallback), --oversample_pow. Same partition/GPU as v3.
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
python emissive/train/train_emissive.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset \
  --init_ckpt full_seg \
  --out_dir /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/${3:-emis_v4} \
  --epochs ${1:-100} --lr 1e-5 --train_split ${2:-train} --val_split ${6:-val} \
  --n_per_epoch ${9:-0} \
  --cond ${4:-real} --pos_weight ${5:-5.0} --emis_oversample --ema 0.999 \
  --save_every ${7:-4} --val_quick ${8:-8}
