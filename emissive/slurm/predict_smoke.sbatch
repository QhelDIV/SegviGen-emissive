#!/bin/bash
#SBATCH -J predict_smoke
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/repo_reorg_smoke/predict_smoke_%j.log
# One-off smoke test for emissive/infer/predict_emissive.py after the emissive/ reorg
# (2026-07-16). NOT part of the normal pipeline — see emissive/docs/EXPERIMENTS.md for
# the tested-on-GPU status this run establishes. args: $1=input glb (abs path)
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /3dlg-jupiter-project/lightgen/segvigen_emissive/repo_reorg_smoke
python emissive/infer/predict_emissive.py \
  --glb "$1" \
  --out /3dlg-jupiter-project/lightgen/segvigen_emissive/repo_reorg_smoke/smoke_out \
  --draws 2 --steps 12 --zero_cond
