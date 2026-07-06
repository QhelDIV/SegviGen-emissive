#!/bin/bash
#SBATCH -J dl_weights
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/dl_weights_%j.log

# Download HF weights on a memory'd compute node (login node OOM'd the Rust hf_transfer).
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0   # use the python downloader (low memory, robust)
/3dlg-jupiter-project/lightgen/miniforge3/bin/python /3dlg-jupiter-project/lightgen/segvigen_emissive/dl_weights.py
