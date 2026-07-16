#!/bin/bash
#SBATCH -J emis_build_sf
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/build_%j.log
# sid_file-driven chunk variant (mirrors build_canon10.sbatch), for building
# train_1k / val_96 from explicit pbr-filtered sid lists (avoids the buggy
# --pbr_only path in _load_split_sids, which crashes on pandas StringDtype NA).
# args: $1=sid_file $2=out_split_name
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
python emissive/data_prep/build_dataset.py --real_cond \
  --sid_file "$1" \
  --out_split_name "$2" \
  --out_root /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset
