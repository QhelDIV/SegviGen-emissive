#!/bin/bash
#SBATCH -J emis_mask
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/logs/mask_xfer__%j.log
set -euo pipefail
#
# CPU-only: resample each model's predicted emissive field into the original asset's UV
# space, one emissive mask PNG per (model, sid, geometry).
#
# args: $1 = model tag  $2 = comma-separated sids (or "all")  $3 = source (pred|gt, default pred)
BASE=/3dlg-jupiter-project/lightgen/segvigen_emissive
OUT=$BASE/outputs/three_ckpt_eval
PV3=/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3
GLB_SRC=$PV3/glb_src
TAG="$1"; SIDS="$2"; SOURCE="${3:-pred}"; TOL="${4:-2.0}"; SUFFIX="${5:-}"
# masks land where paper-v3 picks them up; compute nodes mount /project, never /cs.
# The GT round-trip reads the same npz (gt_e is model-independent) but must not overwrite a
# model's masks, so it gets its own directory.
if [ "$SOURCE" = "gt" ]; then DEST=$PV3/pred_masks/gt_roundtrip$SUFFIX; else DEST=$PV3/pred_masks/$TAG$SUFFIX; fi

source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
cd $BASE

echo "RUN_START $(date -Is) on $(hostname) tag=$TAG"

if [ "$SIDS" = "all" ]; then
  LIST=$(python -c "import json;print(','.join(sorted(json.load(open('$OUT/gallery_sids.json')))))")
else
  LIST="$SIDS"
fi

IFS=',' read -ra ARR <<< "$LIST"
N=${#ARR[@]}
i=0
for sid in "${ARR[@]}"; do
  i=$((i+1))
  python emissive/eval/pred_mask_to_asset.py \
    --npz $OUT/pred_voxels/$TAG/$sid.npz \
    --glb $GLB_SRC/$sid.glb \
    --out_dir $DEST \
    --sid $sid \
    --survey $PV3/material_survey.json \
    --source $SOURCE \
    --thr 0.5 --tex 1024 --tol $TOL
  echo "CASE_DONE $i/$N $sid"
done

echo "JOB_ALL_DONE $(date -Is)"
