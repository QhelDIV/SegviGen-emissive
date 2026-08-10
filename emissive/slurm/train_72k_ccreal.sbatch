#!/bin/bash
#SBATCH -J emis72k_ccreal
#SBATCH --partition=3dlg-hcvc-lab-long
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72k_ccreal/run1_%j.log
#
# Conditioning-parity production run (2026-08-09 decision): identical to
# train_72k_unfilt.sbatch EXCEPT (a) --pbr_cond channel — lightgen/TRELLIS.2-style
# per-token PBR concat [x_t|shape|pbr]=96ch (input_layer re-init from full_seg,
# all blocks loaded), replacing upstream token-doubling; (b) --cond real — real
# DINOv3 thumbnail cond, matching the TRELLIS2 emission DiTs. Everything else
# (unfiltered 72k, pos_weight 5.0, no oversample, const lr) copies the owner
# decisions of the zero-cond token run so the two runs differ only on the two
# unified axes.
#
# Sizing: token mode measured ~1.25 s/shape at sequence 2N; channel mode runs
# sequence N so expect FASTER — recalibrate --n_per_epoch/--epochs from the
# smoke log if s/shape drops materially. Run smoke_72k_ccreal.sbatch first.
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /3dlg-jupiter-project/lightgen/segvigen_emissive

echo "RUN_START $(date -Is) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python emissive/train/train_emissive.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct \
  --train_split train_72k \
  --val_split val_72k \
  --out_dir /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72k_ccreal/run1 \
  --cond real \
  --pbr_cond channel \
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
