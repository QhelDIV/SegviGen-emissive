#!/bin/bash
#SBATCH -J emis_texshare
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/texshare_%j.log
set -euo pipefail
BASE=/3dlg-jupiter-project/lightgen/segvigen_emissive
PV3=/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
cd $BASE
echo "RUN_START $(date -Is) on $(hostname)"
python code/textured_share.py \
  --diagnostics $BASE/outputs/emis_72k_unfilt/run1/diagnostics.json \
  --survey $PV3/material_survey.json \
  --out $BASE/outputs/three_ckpt_eval/textured_share_300.json
echo "JOB_ALL_DONE $(date -Is)"
