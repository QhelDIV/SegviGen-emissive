#!/bin/bash
#SBATCH -J ddp_parity
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --exclude=cs-venus-02,cs-venus-05,cs-venus-09,cs-venus-15,cs-venus-19
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=05:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/ddp_probe/parity_%j.log
#
# Correctness probe for the DDP support added to train_emissive.py. Two GPUs is
# enough for every question here; the throughput table is a separate 4-GPU job.
# cs-venus-02 is excluded on top of the standing set because its 2080_ti cards
# have 11GB and this model needs ~30GB.
#
# Phases:
#   P1  1 GPU, 16 shapes, fixed seed        -> per-step loss reference
#   P2  2 GPUs, same 16 shapes (8 per rank) -> per-step loss to compare against P1
#   P3  1 GPU, 200 shapes                   -> loss trend reference
#   P4  2 GPUs, same 200 shapes             -> loss trend to compare against P3
#   P5  2 GPUs, saves a checkpoint + runs quick-val
#   P6  eval_emissive.py loads P5's checkpoint
set -euo pipefail
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
cd /3dlg-jupiter-project/lightgen/segvigen_emissive

# Which trainer file to exercise. Defaults to the real one; the pre-merge test
# round overrides it so a staged copy can be validated without touching the file
# that queued production jobs will pick up when they start.
TRAINER="${TRAINER:-emissive/train/train_emissive.py}"
echo "TRAINER=$TRAINER"

DS=/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct
OUT=/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/ddp_probe
mkdir -p "$OUT"

echo "RUN_START $(date -Is) on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
nvidia-smi topo -m || true

# val_72k is the training split for these probes on purpose: 387 shapes, every one
# of them has cond.pth (excluded_v2 is missing it for 41 of 286), the directory scan
# is instant, and the checkpoints written here are thrown away. No quality claim is
# made from any number below -- these phases test plumbing and agreement, not IoU.
COMMON="--dataset $DS --train_split val_72k --val_split val_72k --cond real \
  --pos_weight 1.0 --balanced_pos_weight 0.0 --lr 1e-5 --lr_schedule const \
  --init_ckpt /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72kv2_cond_pw1b/epoch_0008.ckpt"

echo "=============== P1: 1 GPU, 16 shapes, seed 1234 ==============="
python "$TRAINER" $COMMON \
  --out_dir "$OUT/p1_1gpu" --seed 1234 --n_per_epoch 16 --epochs 1 \
  --log_step_loss --ema 0 --val_quick 0 --save_every 999 2>&1 | sed 's/^/[P1] /'

echo "=============== P2: 2 GPUs, same 16 shapes, seed 1234 ==============="
torchrun --standalone --nproc_per_node 2 "$TRAINER" $COMMON \
  --out_dir "$OUT/p2_2gpu" --seed 1234 --n_per_epoch 16 --epochs 1 \
  --log_step_loss --ema 0 --val_quick 0 --save_every 999 2>&1 | sed 's/^/[P2] /'

echo "=============== P3: 1 GPU, 200 shapes (20 x 10 epochs), seed 1234 ==============="
python "$TRAINER" $COMMON \
  --out_dir "$OUT/p3_1gpu" --seed 1234 --n_per_epoch 20 --epochs 10 \
  --ema 0 --val_quick 0 --save_every 999 2>&1 | sed 's/^/[P3] /'

echo "=============== P4: 2 GPUs, same 200 shapes, seed 1234 ==============="
torchrun --standalone --nproc_per_node 2 "$TRAINER" $COMMON \
  --out_dir "$OUT/p4_2gpu" --seed 1234 --n_per_epoch 20 --epochs 10 \
  --ema 0 --val_quick 0 --save_every 999 2>&1 | sed 's/^/[P4] /'

echo "=============== P5: 2 GPUs, 40 shapes, saves + quick-val (full recipe) ==============="
torchrun --standalone --nproc_per_node 2 "$TRAINER" $COMMON \
  --out_dir "$OUT/p5_smoke" --n_per_epoch 40 --epochs 1 \
  --emis_oversample --ema 0.999 --val_quick 4 --select_on nonzero \
  --save_every 1 2>&1 | sed 's/^/[P5] /'

echo "--- P5 checkpoint files ---"
ls -la "$OUT/p5_smoke"

echo "=============== P6: eval_emissive.py loads the DDP checkpoint ==============="
python emissive/eval/eval_emissive.py \
  --dataset "$DS" --ckpt "$OUT/p5_smoke/epoch_0001.ckpt" \
  --split val_72k --cond real --n 4 --steps 12 2>&1 | sed 's/^/[P6] /'

echo "RUN_END $(date -Is)"
