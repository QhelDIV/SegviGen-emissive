#!/bin/bash
#SBATCH -J emis_dump
#SBATCH --partition=3dlg-hcvc-lab-long
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/logs/pred_dump__%j.log
set -euo pipefail
#
# Sample all three checkpoints on the 11 paper-figure sids and save the raw predicted
# emissive field per (model, sid). One job, three models, since each is ~11 x 7 s.
BASE=/3dlg-jupiter-project/lightgen/segvigen_emissive
OUT=$BASE/outputs/three_ckpt_eval

source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

echo "RUN_START $(date -Is) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for spec in "emis_1k_w1:$BASE/outputs/emis_1k_w1/best.ckpt" \
            "emis_1k_w5:$BASE/outputs/emis_1k_w5/best.ckpt" \
            "emis_72k_unfilt:$BASE/outputs/emis_72k_unfilt/run1/best.ckpt"; do
  TAG="${spec%%:*}"; CKPT="${spec#*:}"
  echo "MODEL_START $TAG"
  python emissive/infer/dump_pred_voxels.py \
    --dataset $BASE/dataset_direct \
    --sids_json $OUT/gallery_sids.json \
    --ckpt "$CKPT" \
    --out_dir $OUT/pred_voxels/$TAG \
    --steps 12 --seed 0
  echo "MODEL_DONE $TAG"
done

echo "JOB_ALL_DONE $(date -Is)"
