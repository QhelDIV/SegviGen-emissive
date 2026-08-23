#!/bin/bash
#SBATCH -J emis_ddp
#SBATCH --partition=3dlg-hcvc-lab-long
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-15,cs-venus-19
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/REPLACE_RUN_NAME/run_ddp_%j.log
#
# Template for a multi-GPU (single node, PyTorch DDP) emissive fine-tune.
# Copy it, replace REPLACE_RUN_NAME and the flags below, and submit.
#
# What differs from the single-GPU sbatch:
#   * gres asks for 4 GPUs on ONE node instead of 1
#   * the trainer is launched through torchrun, not python
#   * cpus-per-task and mem are 4x the single-GPU ask: each rank is a full process
#     with its own dataloading and its own copy of the model
# Everything else (the recipe flags) is unchanged. --n_per_epoch counts shapes
# GLOBALLY across ranks, so an existing value keeps its meaning here.
#
# Continuing an existing single-GPU run on 4 GPUs: see README_DDP.md in this
# directory. Short version: point --init_ckpt at the run's latest epoch_XXXX.ckpt,
# write to a NEW --out_dir, and keep --lr_schedule const.
set -euo pipefail
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# one OMP thread per rank: four ranks on one node otherwise oversubscribe the cores
# and the data path (torch.load of the four .pth files) gets slower, not faster
export OMP_NUM_THREADS=4
cd /3dlg-jupiter-project/lightgen/segvigen_emissive

echo "RUN_START $(date -Is) on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# --standalone: single node, torchrun picks the rendezvous port itself, so nothing
# has to be coordinated with SLURM. For a multi-NODE run this is the line that
# would change (and nothing else in the trainer).
torchrun --standalone --nproc_per_node 4 \
  emissive/train/train_emissive.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct \
  --train_split train_72k_nonzero_nocopy \
  --val_split val_72k \
  --out_dir /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/REPLACE_RUN_NAME \
  --init_ckpt /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/REPLACE_SOURCE_RUN/epoch_XXXX.ckpt \
  --cond real \
  --pos_weight 1.0 \
  --balanced_pos_weight 0.0 \
  --lr_schedule const \
  --lr 1e-5 \
  --emis_oversample \
  --ema 0.999 \
  --val_quick 8 \
  --select_on nonzero \
  --epochs 8 \
  --save_every 2 \
  2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush() }'

echo "RUN_END $(date -Is)"
