#!/bin/bash
#SBATCH -J gt_parts
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/gtparts_%j.log
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh; conda activate trellis2
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
python emissive/data_prep/gt_parts_extract.py --sid_file canon_overfit10.txt --out_dir gt_parts_canon10
