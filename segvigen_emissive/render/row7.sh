#!/bin/bash
#SBATCH --job-name=lg_row7
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --gres=gpu:0 --cpus-per-task=16 --mem=64G --time=01:00:00
#SBATCH --output=/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3/logs/row7_%j.out
set -o pipefail
D=/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3
PY=/project/3dlg-hcvc/omages/omages_internal/.venv/bin/python
export PYTHONPATH=/project/3dlg-hcvc/omages/xgutils/src:$D
hostname

# The sci-fi weapon joins the comparison as a seventh row. It is the only shape
# where the model predicts a PLAUSIBLE amount of light (0.103 area against a
# true 0.069) and still scores essentially zero, because the light is in the
# wrong place. Every other row shows the model either doing nothing or flooding
# the object at 16x to 144x, which a reader can dismiss as a model that does not
# work; this one cannot be dismissed and is the failure an aggregate hides.
SID=51a60b164e874bf891597d9c6c1941af
COMMON="--res 768 --samples 256 --samples_lit 96 --key 8 --bg 0.012 \
  --view_transform AgX --exposure 0.0 \
  --bloom 1 --bloom_size 9 --bloom_threshold 1.0 --bloom_mix -0.15"

# 1. the random baseline at this shape's own emissive density
$PY $D/render_emissive.py --manifest $D/manifest12.json --glb_dir $D/glb_src \
    --out $D/baselines_k8 $COMMON --mode random --seed 7 --only $SID --overwrite 1
echo "RANDOM_EXIT=$?"

# 2. the coverage-matched albedo mask, in the prediction format
$PY $D/albedo_heuristic.py --manifest $D/manifest12.json --glb_dir $D/glb_src \
    --out $D/pred_masks/albedo_matched --only $SID 2>&1 \
  | grep -E "^OK|^FAIL|ALL_DONE|Traceback"
echo "MASK_EXIT=$?"

# 3. that mask rendered through the same path a model prediction takes
$PY $D/render_emissive.py --manifest $D/manifest12.json --glb_dir $D/glb_src \
    --out $D/pred/albedo_matched_k8 --mode method --only $SID $COMMON \
    --pred_masks $D/pred_masks/albedo_matched \
    --camera_json $D/cameras/$SID.json --overwrite 1
echo "HEUR_EXIT=$?"
echo ROW7_DONE
