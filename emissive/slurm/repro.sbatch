#!/bin/bash
#SBATCH -J segvigen_repro
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/logs/repro__%j.log
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh; conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# the deployed tree IS the SegviGen fork: inference_full.py and data_toolkit/ live at its root
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
# satisfy the literal relative open("microsoft/TRELLIS.2-4B/pipeline.json")
SNAP=$(ls -d /3dlg-jupiter-project/lightgen/hf_cache/hub/models--microsoft--TRELLIS.2-4B/snapshots/*/ | head -1)
mkdir -p microsoft/TRELLIS.2-4B
ln -sf ${SNAP%/}/pipeline.json microsoft/TRELLIS.2-4B/pipeline.json
CKPT=/3dlg-jupiter-project/lightgen/hf_cache/hub/models--fenghora--SegviGen/snapshots/73326b77a17b5194bcc918dbf13d8d5cefb1cca5/full_seg.ckpt
python inference_full.py \
  --ckpt_path "$CKPT" \
  --glb ./data_toolkit/assets/example.glb \
  --input_vxz ./data_toolkit/assets/repro_input.vxz \
  --transforms ./data_toolkit/transforms.json \
  --img ./data_toolkit/assets/repro_img.png \
  --export_glb ./data_toolkit/assets/repro_output.glb
echo REPRO_DONE
