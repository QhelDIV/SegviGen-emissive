#!/bin/bash
#SBATCH -J emis_nb
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/nobloom_%j.log
set -euo pipefail
PV3=/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3
sid=658ecf9f837246509b0b1c4aa81e9e5b
PYTHONPATH=/project/3dlg-hcvc/omages/xgutils/src \
/project/3dlg-hcvc/omages/omages_internal/.venv/bin/python \
  $PV3/render_emissive.py \
  --manifest $PV3/pred/manifest_11.json --glb_dir $PV3/glb_src \
  --out $PV3/pred/_nobloom_probe --mode method --only $sid \
  --pred_masks $PV3/pred_masks/emis_72k_unfilt \
  --camera_json $PV3/cameras/$sid.json \
  --res 768 --view_transform AgX --exposure 0.0 --key 20 --bg 0.012 \
  --bloom 0 --samples 256 --samples_lit 96 --overwrite 1
echo "PROBE_DONE $(date -Is)"
