#!/bin/bash
#SBATCH -J emis_3ck
#SBATCH --partition=3dlg-hcvc-lab-long
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/three_ckpt_%j.log
set -euo pipefail
#
# Re-score one checkpoint on the SAME 300 val_72k shapes, same cond (zero), same threshold
# sweep and same draws=3 as emis_72k_unfilt/run1's diagnostics.json, then aggregate through
# diagnostics_run.py's own stratification pass so the output is schema-identical.
#
# args: $1 = model tag (output subdir name)
#       $2 = ckpt path (absolute)
#       $3 = train_curve.json path (absolute)
BASE=/3dlg-jupiter-project/lightgen/segvigen_emissive
TAG="$1"; CKPT="$2"; CURVE="$3"
OUT_DIR=$BASE/outputs/three_ckpt_eval/$TAG
REF_DIR=$BASE/outputs/emis_72k_unfilt/run1

source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=$BASE/../hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

echo "RUN_START $(date -Is) on $(hostname) tag=$TAG ckpt=$CKPT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p $OUT_DIR
# diag1/diag2 are data properties, identical for every checkpoint on these shapes.
cp -n $REF_DIR/_diag12_raw_seed0_n300.json $OUT_DIR/_diag12_raw_seed0_n300.json
echo "DIAG12_CACHE_READY"

python code/diag3_only.py \
  --dataset $BASE/dataset_direct --split val_72k \
  --ckpt "$CKPT" --out_dir $OUT_DIR \
  --n 300 --draws 3 --steps 12 --seed 0

python code/diagnostics_run.py --aggregate_only \
  --dataset $BASE/dataset_direct --split val_72k \
  --ckpt "$CKPT" --train_curve "$CURVE" \
  --out $OUT_DIR/diagnostics.json \
  --n 300 --draws 3 --steps 12 --seed 0

echo "JOB_ALL_DONE $(date -Is)"
