#!/bin/bash
#SBATCH -J emis_eval96
#SBATCH -p 3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=1:30:00
#SBATCH -o /3dlg-jupiter-project/lightgen/segvigen_emissive/eval_%j.log
# honest-protocol val_96 eval: averaged (K=4) + voxel buckets + otsu + stratify. arg: $1=ckpt
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
python emissive/eval/eval_emissive.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset \
  --ckpt "$1" --split val_96 --cond real --draws 4 --bucket_by voxel --otsu --stratify
