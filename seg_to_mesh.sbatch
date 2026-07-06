#!/bin/bash
#SBATCH -J seg2mesh
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/seg2mesh_%j.log
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEG_CKPT=/3dlg-jupiter-project/lightgen/hf_cache/hub/models--fenghora--SegviGen/snapshots/73326b77a17b5194bcc918dbf13d8d5cefb1cca5/full_seg.ckpt
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
python code/seg_to_mesh.py --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset \
  --split ${1:-canon_overfit10} --seg_ckpt "$SEG_CKPT" \
  --out_glb_dir /3dlg-jupiter-project/lightgen/segvigen_emissive/${2:-seg_mesh_canon10} --steps 25
