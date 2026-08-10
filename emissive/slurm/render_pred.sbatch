#!/bin/bash
#SBATCH -J emis_render
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/render_pred_%j.log
set -euo pipefail
#
# Render a model's PREDICTED emissive mask on the pristine asset, key-lit, matched to the
# published comparison columns.
#
# The treatment flags are PINNED, not defaulted. paper-v3 changed the script's bloom
# defaults after a sweep (now 7 / -0.45), but the published comparison columns were made at
# 9 / -0.15. Rendering at today's defaults would make this column differ from the four
# beside it in bloom as well as in mask, which is exactly the unattributable difference the
# figure must not contain.
#
# args: $1 = model tag   $2 = mode (method|box, default method)   $3 = key (default 8)
#
# key 8 rather than 20 (lead-approved): at key 20 the Glare node fires on 108,865 pixels of a
# shape that emits NOTHING, because a key that bright pushes pale albedo over the threshold.
# At key 8 it fires on zero pixels of an empty prediction and ~50,000 of ground truth, so
# bloom becomes a property of emission rather than of brightness. Measured by paper-v3.
PV3=/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3
TAG="$1"; MODE="${2:-method}"; KEY="${3:-8}"
OUT=$PV3/pred/${TAG}_${MODE}_k${KEY}

echo "RUN_START $(date -Is) on $(hostname) tag=$TAG mode=$MODE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p $OUT

N=$(python3 -c "import json;print(len(json.load(open('$PV3/pred/manifest_11.json'))))")
i=0
for sid in $(python3 -c "import json;[print(e['sid']) for e in json.load(open('$PV3/pred/manifest_11.json'))]"); do
  i=$((i+1))
  if [ "$MODE" = "box" ]; then
    TREAT="--view_transform Filmic --exposure 1.5 --wall 0.80 --max_bounces 32 \
           --diffuse_bounces 16 --bloom_size 7 --bloom_threshold 1.0 --bloom_mix -0.45 \
           --samples 1024"
  else
    TREAT="--view_transform AgX --exposure 0.0 --key $KEY --bg 0.012 \
           --bloom 1 --bloom_size 9 --bloom_threshold 1.0 --bloom_mix -0.15 \
           --samples 256 --samples_lit 96"
  fi
  PYTHONPATH=/project/3dlg-hcvc/omages/xgutils/src \
  /project/3dlg-hcvc/omages/omages_internal/.venv/bin/python \
    $PV3/render_emissive.py \
    --manifest $PV3/pred/manifest_11.json \
    --glb_dir $PV3/glb_src \
    --out $OUT \
    --mode $MODE --only $sid \
    --pred_masks $PV3/pred_masks/$TAG \
    --camera_json $PV3/cameras/$sid.json \
    --res 768 \
    $TREAT \
    --overwrite 1
  echo "CASE_DONE $i/$N $sid"
done

echo "JOB_ALL_DONE $(date -Is)"
