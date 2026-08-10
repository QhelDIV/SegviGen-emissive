#!/bin/bash
#SBATCH -J emis_dump8
#SBATCH -p 3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=0:30:00
#SBATCH -o /3dlg-jupiter-project/lightgen/segvigen_emissive/dump8_%j.log
# dump_vis eval on the 8-shape results_2k_v1 prediction subset. args: $1=ckpt $2=dump_vis_dir
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
python emissive/eval/eval_emissive.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset \
  --ckpt "$1" --split val_pred8_2k --cond real --draws 1 --bucket_by voxel --dump_vis "$2"
