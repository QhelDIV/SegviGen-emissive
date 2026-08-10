#!/bin/bash
#SBATCH -J condfill
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-venus-02,cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-23
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/cond_thumb/fill_%A_%a.log
#
# Backfill thumbnail conditioning over the whole direct-ovoxel dataset:
# val_72k (7,290) then train_72k (57,968), each strided into 24 shards so a task
# gets an even slice of the cost. Idempotent -- a shape that already has cond.pth
# is skipped, so a preempted task can simply be resubmitted.
#
# Node choice: excludes cs-venus-02 (2080Ti/11GB, no headroom for BiRefNet at
# 1024px + DINOv3-L together), the two Blackwell nodes (sm_120, unsupported by
# this torch), and cs-venus-09. Leaves a40 / a100 / l40s.
set -euo pipefail
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/3dlg-jupiter-project/lightgen/segvigen_emissive
cd $ROOT

echo "FILL_START shard=$SLURM_ARRAY_TASK_ID $(date -Is) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# NOTE: build_cond_thumbnail.py exits 0 on per-shape preprocessing failures by
# design -- roughly 1% of TexVerse thumbnails are placeholders or empty frames
# that Path A's preprocessing cannot handle, and the --repair pass absorbs them.
# An earlier version exited nonzero, which under `set -e` aborted this loop after
# val_72k and cost the whole array its train_72k half. Coverage is asserted
# separately, by counting cond.pth against the shape count, not by exit code.
for SPLIT in val_72k train_72k; do
  python code/build_cond_thumbnail.py \
    --split $SPLIT --shard $SLURM_ARRAY_TASK_ID --nshards 24
done

echo "FILL_OK shard=$SLURM_ARRAY_TASK_ID $(date -Is)"
