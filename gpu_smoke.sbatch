#!/bin/bash
#SBATCH -J emis_smoke
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/smoke_%j.log
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
python code/smoke_test.py
