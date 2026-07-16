#!/bin/bash
#SBATCH -J pred_glb
#SBATCH -p 3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH -o /3dlg-jupiter-project/lightgen/segvigen_emissive/predglb_%j.log
# Decode ONE emissive prediction to a paper-style GLB mesh (official remesh+bake).
# args: $1=ckpt  $2=sid  $3=out_glb  $4=thr(default 0.5)  $5=cond(default real)  $6=seed(default 0)
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
python emissive/eval/make_pred_glb.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset \
  --split val_96 \
  --ckpt "$1" --sid "$2" --out "$3" \
  --thr "${4:-0.5}" --cond "${5:-real}" --seed "${6:-0}"
