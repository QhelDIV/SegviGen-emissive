#!/bin/bash
#SBATCH -J ddp_id_steps
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --exclude=cs-venus-02,cs-venus-05,cs-venus-09,cs-venus-15,cs-venus-19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/ddp_probe/idsteps_%j.log
#
# PER-STEP identity check of the single-GPU path across the DDP commit (b1742dc).
#
# Getting a per-step loss sequence out of the PRE-DDP trainer without editing it:
# set --n_per_epoch 1, so one epoch is exactly one shape and the per-epoch loss
# line the old trainer already prints IS that step's loss. Both files are run
# completely unmodified; only the flags differ from a normal run.
#
# Step 1 is the measurement that matters most. At step 1 both runs start from
# byte-identical weights on byte-identical data, so any difference there is pure
# kernel nondeterminism with no optimizer divergence mixed in. Later steps
# accumulate that floor through the optimizer and so must drift regardless.
#
# The old file is run TWICE. flash_attn and the sparse ops reduce with atomics,
# so this model does not reproduce itself bit for bit, and a bare old-vs-new
# comparison against zero would report a behavior change that is really the
# hardware. old-vs-old is the reference the test is judged against.
#
# --emis_oversample is ON because that is what the production recipe uses, so the
# weighted multinomial draw is the code path under test, not randperm.
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
md5sum "$OLD" "$NEW"
wc -l "$OLD" "$NEW"

# 20 epochs of exactly 1 shape = 20 single-shape optimizer steps.
COMMON="--dataset $DS --train_split val_72k --val_split val_72k --cond real \
  --pos_weight 1.0 --balanced_pos_weight 0.0 --lr 1e-5 --lr_schedule const \
  --emis_oversample --ema 0 --val_quick 0 --save_every 999 \
  --n_per_epoch 1 --epochs 20 \
  --init_ckpt /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72kv2_cond_pw1b/epoch_0008.ckpt"

echo "=============== OLD (pre-DDP), pinned seed, run A ==============="
python emissive/probes/pinned_rng_launcher.py 1234 "$OLD" $COMMON \
  --out_dir "$OUT/st_old_a" 2>&1 | sed 's/^/[OLDA] /'

echo "=============== OLD again, same pinned seed, run B (control) ==============="
python emissive/probes/pinned_rng_launcher.py 1234 "$OLD" $COMMON \
  --out_dir "$OUT/st_old_b" 2>&1 | sed 's/^/[OLDB] /'

echo "=============== NEW (post-DDP, WORLD_SIZE unset), same pinned seed ==============="
python emissive/probes/pinned_rng_launcher.py 1234 "$NEW" $COMMON \
  --out_dir "$OUT/st_new" 2>&1 | sed 's/^/[NEW] /'

echo "=============== VERDICT ==============="
python - "$OUT/st_old_a/log.json" "$OUT/st_old_b/log.json" "$OUT/st_new/log.json" <<'PY'
import json, sys
a, b, n = (json.load(open(p)) for p in sys.argv[1:4])
print(f"{'step':>5} {'old A':>15} {'old B':>15} {'new':>15} {'B-A ctrl':>11} {'new-A test':>11}")
ctrl, test = [], []
for x, y, z in zip(a, b, n):
    c, t = y["loss"] - x["loss"], z["loss"] - x["loss"]
    ctrl.append(abs(c)); test.append(abs(t))
    print(f"{x['epoch']:>5} {x['loss']:>15.10f} {y['loss']:>15.10f} {z['loss']:>15.10f} "
          f"{c:>11.2e} {t:>11.2e}")
print()
print(f"STEP 1 (identical starting weights, no optimizer divergence yet):")
print(f"  |old B - old A| = {ctrl[0]:.3e}   |new - old A| = {test[0]:.3e}")
print(f"max over all {len(ctrl)} steps:")
print(f"  |old B - old A| = {max(ctrl):.3e}   |new - old A| = {max(test):.3e}")
# relative to the loss scale, so the numbers mean something
scale = sum(x["loss"] for x in a) / len(a)
print(f"mean loss scale = {scale:.4f}; control {max(ctrl)/scale:.2e} rel, test {max(test)/scale:.2e} rel")
if max(ctrl) == 0.0 and max(test) == 0.0:
    print("VERDICT: bit-identical AND the run is bitwise reproducible")
elif max(ctrl) == 0.0:
    print("VERDICT: old reproduces itself exactly but new does not match it -> REAL CHANGE")
elif max(test) <= 3 * max(ctrl):
    print("VERDICT: bit-identity is not achievable for ANY version on this hardware; "
          "the across-change difference sits inside the same-file floor -> "
          "NO BEHAVIOR CHANGE DETECTED")
else:
    print("VERDICT: across-change difference exceeds the same-file floor -> INVESTIGATE")
PY

echo "RUN_END $(date -Is)"
