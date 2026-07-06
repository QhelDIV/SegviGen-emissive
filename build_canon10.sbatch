#!/bin/bash
#SBATCH -J build_canon10
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/build_%j.log
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
python code/build_dataset.py --real_cond \
  --sid_file /3dlg-jupiter-project/lightgen/segvigen_emissive/canon_overfit10.txt \
  --out_split_name canon_overfit10 \
  --out_root /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset
