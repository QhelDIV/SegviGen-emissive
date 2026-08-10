#!/bin/bash
#SBATCH -J emis_texiou
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/texiou2_%j.log
set -euo pipefail
# args: $1 = mask dir name under pred_masks/
BASE=/3dlg-jupiter-project/lightgen/segvigen_emissive
PV3=/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3
TAG="$1"

source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
cd $BASE
echo "RUN_START $(date -Is) on $(hostname) tag=$TAG"

python code/texture_mask_iou.py \
  --masks $PV3/pred_masks/$TAG \
  --control $PV3/pred_masks/emis_1k_w5 \
  --control_summary $BASE/outputs/three_ckpt_eval/pred_voxels/emis_1k_w5/summary.json \
  --npz_root $BASE/outputs/three_ckpt_eval/pred_voxels/emis_1k_w5 \
  --glb_dir $PV3/glb_src \
  --survey $PV3/material_survey.json \
  --tol 0.87 --tex 1024 \
  --out $BASE/outputs/three_ckpt_eval/iou_ctrl_$TAG.json

echo "JOB_ALL_DONE $(date -Is)"
