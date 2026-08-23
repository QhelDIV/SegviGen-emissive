#!/bin/bash
#SBATCH -J ddp_identity
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --exclude=cs-venus-02,cs-venus-05,cs-venus-09,cs-venus-15,cs-venus-19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/ddp_probe/identity_%j.log
#
# Does the DDP commit change what a SINGLE-GPU run does?
#
# The DDP support (commit b1742dc) is meant to leave the single-GPU path alone:
# without torchrun WORLD_SIZE is unset and every distributed branch is skipped.
# But the refactor did move the SparseTensor packing into a FlowStep module and
# reorganised the step loop, so "should be identical" wants checking rather than
# asserting, especially while multi-day single-GPU runs are launching against it.
#
# Both versions are run through pinned_rng_launcher.py, which fixes the global
# torch seed first. Without that the two runs would differ for reasons that have
# nothing to do with the change (the trainer seeds nothing by default). With it,
# the two consume the generators in the same order and any difference in the loss
# sequence is a real behavior change.
#
# --emis_oversample is ON so the weighted multinomial draw is exercised, not just
# the randperm path.
set -euo pipefail
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /3dlg-jupiter-project/lightgen/segvigen_emissive

DS=/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct
OUT=/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/ddp_probe
OLD=${OLD_TRAINER:?set OLD_TRAINER to the pre-DDP train_emissive.py}
NEW=emissive/train/train_emissive.py
mkdir -p "$OUT"

echo "RUN_START $(date -Is) on $(hostname)"
echo "OLD=$OLD"
echo "NEW=$NEW"
md5sum "$OLD" "$NEW"

COMMON="--dataset $DS --train_split val_72k --val_split val_72k --cond real \
  --pos_weight 1.0 --balanced_pos_weight 0.0 --lr 1e-5 --lr_schedule const \
  --emis_oversample --ema 0 --val_quick 0 --save_every 999 \
  --n_per_epoch 12 --epochs 3 \
  --init_ckpt /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72kv2_cond_pw1b/epoch_0008.ckpt"

# THE CONTROL MATTERS HERE. flash_attn and the sparse ops reduce with atomics, so
# the same file run twice on the same GPU with the same seed does NOT reproduce
# itself bit for bit. Comparing old against new without that reference would
# report a "behavior change" that is really just floating-point nondeterminism.
# So the old trainer is run TWICE, and old-vs-new is judged against old-vs-old.
echo "=============== OLD trainer (pre-DDP), seed pinned, run A ==============="
python emissive/probes/pinned_rng_launcher.py 1234 "$OLD" $COMMON \
  --out_dir "$OUT/id_old_a" 2>&1 | sed 's/^/[OLDA] /'

echo "=============== OLD trainer again, identical seed, run B (control) ==============="
python emissive/probes/pinned_rng_launcher.py 1234 "$OLD" $COMMON \
  --out_dir "$OUT/id_old_b" 2>&1 | sed 's/^/[OLDB] /'

echo "=============== NEW trainer (post-DDP), same pinned seed ==============="
python emissive/probes/pinned_rng_launcher.py 1234 "$NEW" $COMMON \
  --out_dir "$OUT/id_new" 2>&1 | sed 's/^/[NEW] /'

echo "=============== VERDICT ==============="
python - "$OUT/id_old_a/log.json" "$OUT/id_old_b/log.json" "$OUT/id_new/log.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
n = json.load(open(sys.argv[3]))
print(f"{'epoch':>6} {'old A':>14} {'old B':>14} {'new':>14} {'B-A (control)':>15} {'new-A (test)':>14}")
ctrl, test = [], []
for x, y, z in zip(a, b, n):
    c = y["loss"] - x["loss"]; t = z["loss"] - x["loss"]
    ctrl.append(abs(c)); test.append(abs(t))
    print(f"{x['epoch']:>6} {x['loss']:>14.10f} {y['loss']:>14.10f} {z['loss']:>14.10f} "
          f"{c:>15.2e} {t:>14.2e}")
mc, mt = max(ctrl), max(test)
print(f"\nmax |old B - old A| (same file twice) = {mc:.3e}")
print(f"max |new   - old A| (across the change) = {mt:.3e}")
if mc == 0.0 and mt == 0.0:
    print("VERDICT: bit-identical, and the run is bitwise reproducible")
elif mc == 0.0:
    print("VERDICT: the old file IS reproducible and the new one differs -> REAL CHANGE")
elif mt <= 3 * mc:
    print("VERDICT: the difference across the change is within the same-file "
          "nondeterminism -> NO BEHAVIOR CHANGE DETECTED")
else:
    print("VERDICT: difference across the change exceeds the same-file "
          "nondeterminism -> INVESTIGATE")
PY

echo "RUN_END $(date -Is)"
