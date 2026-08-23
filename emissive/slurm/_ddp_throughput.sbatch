#!/bin/bash
#SBATCH -J ddp_thru
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --exclude=cs-venus-02,cs-venus-05,cs-venus-09,cs-venus-15,cs-venus-19
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=05:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/ddp_probe/thru_%j.log
#
# Throughput of the emissive fine-tune at 1, 2 and 4 GPUs, all three measured in
# ONE allocation on ONE node so the numbers are comparable. Also times a bare
# NCCL all-reduce of the model's gradient size, which is what tells us whether a
# gap from linear scaling is the interconnect or something else.
#
# gres asks for 4 GPUs of any type: l40s is what production uses, but every 48GB
# node on this cluster stays fully booked for days at a time and the SCALING
# question does not depend on which of them we land on. The log records the card.
# cs-venus-02 is excluded on top of the standing set (11GB cards, model needs ~30GB).
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

# Same 96 shapes per epoch at every width, drawn from the same 387-shape split, so
# shapes/second compares like with like. 4 epochs; epoch 1 is discarded as warmup
# (first CUDA kernels, NCCL channel setup, cold page cache) and epochs 2-4 are the
# measurement. EMA stays on because production runs with it and it is real work.
COMMON="--dataset $DS --train_split val_72k --val_split val_72k --cond real \
  --pos_weight 1.0 --balanced_pos_weight 0.0 --lr 1e-5 --lr_schedule const \
  --emis_oversample --ema 0.999 --val_quick 0 --save_every 999 \
  --n_per_epoch 96 --epochs 4 \
  --init_ckpt /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72kv2_cond_pw1b/epoch_0008.ckpt \
  ${EXTRA_FLAGS:-}"

# EXTRA_FLAGS exists so the same script can be run a second time with --seed set.
# Without a seed each width draws its own random shapes, and since per-shape cost
# varies the widths are then compared on different work; with a seed all three see
# the identical shapes in the identical order.

echo "=============== T1: 1 GPU ==============="
python "$TRAINER" $COMMON --out_dir "$OUT/t1" 2>&1 | sed 's/^/[T1] /'

echo "=============== T2: 2 GPUs ==============="
torchrun --standalone --nproc_per_node 2 "$TRAINER" $COMMON \
  --out_dir "$OUT/t2" 2>&1 | sed 's/^/[T2] /'

echo "=============== T4: 4 GPUs ==============="
torchrun --standalone --nproc_per_node 4 "$TRAINER" $COMMON \
  --out_dir "$OUT/t4" 2>&1 | sed 's/^/[T4] /'

echo "=============== T4a: 4 GPUs, grad_accum 4 (comm amortized 4x) ==============="
torchrun --standalone --nproc_per_node 4 "$TRAINER" $COMMON \
  --grad_accum 4 --out_dir "$OUT/t4a" 2>&1 | sed 's/^/[T4a] /'

echo "=============== T4b: 4 GPUs, bf16 gradient compression ==============="
torchrun --standalone --nproc_per_node 4 "$TRAINER" $COMMON \
  --ddp_bf16_comm --out_dir "$OUT/t4b" 2>&1 | sed 's/^/[T4b] /'

echo "=============== NCCL all-reduce cost, 2 then 4 GPUs ==============="
torchrun --standalone --nproc_per_node 2 emissive/probes/nccl_allreduce_probe.py 2>&1 | sed 's/^/[NC2] /'
torchrun --standalone --nproc_per_node 4 emissive/probes/nccl_allreduce_probe.py 2>&1 | sed 's/^/[NC4] /'

echo "RUN_END $(date -Is)"
