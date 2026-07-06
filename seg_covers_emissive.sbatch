#!/bin/bash
#SBATCH -J seg_covers
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/segcov_%j.log
# args: $1=split (default overfit_10) $2=dump_vis subdir (default seg_vis_$1)
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEG_CKPT=/3dlg-jupiter-project/lightgen/hf_cache/hub/models--fenghora--SegviGen/snapshots/73326b77a17b5194bcc918dbf13d8d5cefb1cca5/full_seg.ckpt
SPLIT=${1:-overfit_10}
DUMP=${2:-seg_vis_$SPLIT}
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
python code/seg_covers_emissive.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset \
  --split "$SPLIT" --seg_ckpt "$SEG_CKPT" --steps 25 \
  --dump_vis /3dlg-jupiter-project/lightgen/segvigen_emissive/$DUMP
