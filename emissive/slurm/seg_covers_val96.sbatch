#!/bin/bash
#SBATCH -J oracle_val96
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/oracle_val96_%j.log
# Oracle-on-val_96 (all 111 shapes with output_tex_slat.pth+emis_mask.pth): pretrained
# full_seg zero-shot -> parts -> label each part emissive iff >50% GT-emissive -> union
# -> per-shape IoU. Same conventions as eval_emissive.py: zero-glow union==0 -> IoU=1.0,
# voxel (emis_mask.pth) bucketing, nonzero-only aggregate. Output: dataset/oracle_val96.json
# (also printed table). ~6h budget for 111 shapes @ steps=25 (2h was eval_val96's budget
# for 111 shapes @ steps=12 — this job samples the heavier 25-step schedule).
# args: $1=shard_idx (default 0) $2=num_shards (default 1) $3=dump_vis subdir (optional)
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEG_CKPT=/3dlg-jupiter-project/lightgen/hf_cache/hub/models--fenghora--SegviGen/snapshots/73326b77a17b5194bcc918dbf13d8d5cefb1cca5/full_seg.ckpt
SHARD_IDX=${1:-0}
NUM_SHARDS=${2:-1}
OUT_JSON=/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset/oracle_val96.json
if [ "$NUM_SHARDS" -gt 1 ]; then
  OUT_JSON=/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset/oracle_val96.shard${SHARD_IDX}of${NUM_SHARDS}.json
fi
DUMP_ARG=""
if [ -n "$3" ]; then DUMP_ARG="--dump_vis /3dlg-jupiter-project/lightgen/segvigen_emissive/$3"; fi
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
python emissive/eval/seg_covers_emissive.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset \
  --split val_96 --seg_ckpt "$SEG_CKPT" --steps 25 \
  --shard_idx "$SHARD_IDX" --num_shards "$NUM_SHARDS" \
  --out_json "$OUT_JSON" $DUMP_ARG
