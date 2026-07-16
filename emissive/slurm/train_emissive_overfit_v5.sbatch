#!/bin/bash
#SBATCH -J emis_overfit_v5
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/train_%j.log
# Sanity gate for the v5 trainer (balanced_pos_weight + cosine lr_schedule + select_on):
# can it still overfit a tiny set to ~1.0 IoU, same shape as train_emissive_overfit_v4.sbatch?
# args: $1=split_name (e.g. train_1k_gate10) $2=out_subdir $3=epochs (default 60)
#       $4=balanced_cap (default 50) $5=cond (default real — gate10 has real cond.pth;
#          pass zero to match the older zero-cond baselines) $6=n_per_epoch (default 10,
#          i.e. one pass over a 10-sample split per epoch) $7=val_quick (default 10 — the
#          whole split, so quick-val covers every sample) $8=init_ckpt (default full_seg)
#          $9=save_every (default 5) ${10}=lr_schedule (default cosine) ${11}=select_on
#          (default nonzero) ${12}=val_split (default = $1)
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
python emissive/train/train_emissive.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset \
  --init_ckpt "${8:-full_seg}" \
  --out_dir /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/${2:-gate10_v5} \
  --epochs ${3:-60} --lr 1e-5 --train_split "$1" --val_split "${12:-$1}" --n_per_epoch ${6:-10} \
  --cond ${5:-real} --balanced_pos_weight ${4:-50} --pos_weight 1.0 \
  --lr_schedule ${10:-cosine} --select_on ${11:-nonzero} \
  --emis_oversample --ema 0.999 \
  --save_every ${9:-5} --val_quick ${7:-10}
