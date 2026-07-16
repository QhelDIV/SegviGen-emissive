#!/bin/bash
#SBATCH -J emis_eval96
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/eval96_%j.log
# Full val_96 eval for a Phase 4 ckpt: fixed threshold sweep + Otsu + stratified-by-
# emissive_frac breakdown (0, (0,0.05], (0.05,0.3], >0.3), to separate the flat
# DiffusionNet-comparable mean from the near-zero-emissive failure mode seen in the
# canon10 autopsy. NOT auto-submitted — run manually once a Phase 4 best.ckpt lands:
#   sbatch emissive/slurm/eval_val96.sbatch outputs/emis_1k_w5/best.ckpt
#   sbatch emissive/slurm/eval_val96.sbatch outputs/emis_1k_w1/best.ckpt
# args: $1=ckpt $2=dump_vis dir (or '', relative to segvigen_emissive/)
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
DUMP_ARG=""
if [ -n "$2" ]; then DUMP_ARG="--dump_vis /3dlg-jupiter-project/lightgen/segvigen_emissive/$2"; fi
python emissive/eval/eval_emissive.py --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset \
  --ckpt "$1" --split val_96 --cond real --otsu --stratify $DUMP_ARG
