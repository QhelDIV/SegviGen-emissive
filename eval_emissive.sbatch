#!/bin/bash
#SBATCH -J emis_eval
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/eval_%j.log
# args: $1=ckpt $2=split $3=dump_vis dir (or '') $4=n (or '') $5(optional)="real"|"zero" (default zero)
# NOTE (v4): eval_emissive.py now requires an explicit --cond {real,zero} — no silent
# zero-cond fallback — and gained --otsu (pass $6=otsu to enable).
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
DUMP_ARG=""; N_ARG=""; OTSU_ARG=""
if [ -n "$3" ]; then DUMP_ARG="--dump_vis /3dlg-jupiter-project/lightgen/segvigen_emissive/$3"; fi
if [ -n "$4" ]; then N_ARG="--n $4"; fi
if [ "$6" = "otsu" ]; then OTSU_ARG="--otsu"; fi
python code/eval_emissive.py --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset \
  --ckpt "$1" --split "${2:-val}" --cond "${5:-zero}" $DUMP_ARG $N_ARG $OTSU_ARG
