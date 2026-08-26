#!/bin/bash
#SBATCH -J emis_ag37k_fseg_solar
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --nodelist=cs-venus-17
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=66:00:00
#SBATCH --output=/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/emis_agentic37k_fromfullseg_pw1b_solar/run_%j.log
#
# Solar replicate of the fir fair arm: agentic 36,226-shape fine-tune FROM THE
# SEGVIGEN-PROVIDED full_seg CHECKPOINT (no 72k exposure). Same recipe as the
# fir run (effective batch 4, one shape per rank, no grad accumulation, lr 1e-5
# const), so the two runs are hardware/seed replicates of one arm; whichever
# finishes first is used. venus-17 measured 1.99 shapes/s on this recipe
# (ddp4 run 249196), ~5.1 h/epoch, so 8 epochs ~41 h; lab-short's 3-day cap
# holds that with margin, and per-epoch saves plus the resume block below make
# a resubmit continue loss-free from the newest checkpoint + sidecar.
set -euo pipefail
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
cd /3dlg-jupiter-project/lightgen/segvigen_emissive

TARGET_EPOCHS=8
OUT_DIR=/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/emis_agentic37k_fromfullseg_pw1b_solar
DATASET=/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct

echo "RUN_START $(date -Is) on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ---- resume bookkeeping: what is already done? ----
LATEST=""
COMPLETED=0
for c in $(ls "$OUT_DIR"/epoch_????.ckpt 2>/dev/null | sort -r); do
  if [ -f "${c%.ckpt}_state.pt" ]; then
    LATEST=$c
    COMPLETED=$((10#$(basename "$c" .ckpt | cut -d_ -f2)))
    break
  fi
done
if [ "$COMPLETED" -ge "$TARGET_EPOCHS" ]; then
  echo "CHAIN_DONE $COMPLETED/$TARGET_EPOCHS epochs already trained; nothing to do"
  exit 0
fi
REMAINING=$((TARGET_EPOCHS - COMPLETED))
echo "CHAIN_STATE completed=$COMPLETED remaining=$REMAINING latest=${LATEST:-none}"
if [ -n "$LATEST" ]; then
  INIT_ARGS=(--resume "$LATEST" --epochs "$REMAINING")
else
  INIT_ARGS=(--init_ckpt full_seg --epochs "$TARGET_EPOCHS")
fi
echo "INIT_ARGS ${INIT_ARGS[*]}"

# NFS preflight with the consumer's own admission predicate (negative-dentry law)
python3 /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72kv2_cond_pw1b_agentic_smoke/preflight_split.py \
  "$DATASET" train_72k_agentic

torchrun --standalone --nproc_per_node 4 \
  emissive/train/train_emissive.py \
  --dataset "$DATASET" \
  --train_split train_72k_agentic \
  --val_split val_72k \
  --out_dir "$OUT_DIR" \
  "${INIT_ARGS[@]}" \
  --cond real \
  --pos_weight 1.0 \
  --balanced_pos_weight 0.0 \
  --lr_schedule const \
  --lr 1e-5 \
  --emis_oversample \
  --ema 0.999 \
  --val_quick 8 \
  --select_on nonzero \
  --save_every 1 \
  2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush() }'

echo "RUN_END $(date -Is)"
