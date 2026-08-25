#!/bin/bash
#SBATCH -J optstate_demo
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --exclude=cs-venus-02,cs-venus-05,cs-venus-09,cs-venus-15,cs-venus-19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/optstate_probe/demo_%j.log
#
# Does resuming WITH optimizer state actually remove the resettle, and does
# resuming WITHOUT it actually show one?
#
# Four runs, one seed, identical data order in all of them:
#   A   24 steps straight through                      the reference curve
#   B1  12 steps, saves weights + state at step 12
#   B2  --resume B1  -> steps 13..24, optimizer restored
#   C   --resume B1 --fresh_opt -> steps 13..24, optimizer reset
#
# C is the control and it is why --fresh_opt exists: it restores the position and
# the EMA, so it draws exactly the shapes A and B2 draw, and differs from B2 in
# one thing only, whether AdamW keeps its moments. Comparing C against A without
# that flag would confound the resettle with a different data order.
#
# --n_per_epoch 1 makes one epoch exactly one shape, so the per-epoch line the
# trainer already prints IS the per-step loss and log.json is a per-step curve.
# Epoch numbering continues across a resume, which is also what makes B2 and C
# draw A's steps 13..24 rather than re-drawing steps 1..12.
#
# Observed at N=100: B2 sits at 1.06x the nondeterminism floor, i.e. indistinguishable
# from rerunning the same code; C sits at 5.16x, separated from B2 by 3-6x in every
# window, and lands 5.6x further from A in weight space. The penalty is a SUSTAINED
# elevation while AdamW rebuilds its moments, not a spike in the first few steps, so
# N must be large enough to see it: an N=12 run could not.
set -euo pipefail
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /3dlg-jupiter-project/lightgen/segvigen_emissive

TRAINER="${TRAINER:-emissive/train/train_emissive.py}"
DS=/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct
OUT=/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/optstate_probe
N=${N:-12}
SEED=${SEED:-4242}
mkdir -p "$OUT"

echo "RUN_START $(date -Is) on $(hostname)"
echo "TRAINER=$TRAINER  N=$N  SEED=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

COMMON="--dataset $DS --train_split val_72k --val_split val_72k --cond real \
  --pos_weight 1.0 --balanced_pos_weight 0.0 --lr 1e-5 --lr_schedule const \
  --emis_oversample --ema 0.999 --val_quick 0 --seed $SEED --n_per_epoch 1 \
  --keep_state 0"
INIT=/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72kv2_cond_pw1b/epoch_0008.ckpt

echo "=============== A: $((N*2)) steps straight through ==============="
python "$TRAINER" $COMMON --init_ckpt "$INIT" \
  --out_dir "$OUT/a_full" --epochs $((N*2)) --save_every 999 2>&1 | sed 's/^/[A] /'

echo "=============== B1: first $N steps, saves state ==============="
python "$TRAINER" $COMMON --init_ckpt "$INIT" \
  --out_dir "$OUT/b1_first" --epochs $N --save_every 999 2>&1 | sed 's/^/[B1] /'

echo "--- what B1 left behind ---"
ls -la "$OUT/b1_first"

echo "=============== B2: resume WITH optimizer state ==============="
python "$TRAINER" $COMMON --resume "$OUT/b1_first/epoch_$(printf %04d $N).ckpt" \
  --out_dir "$OUT/b2_resume" --epochs $N --save_every 999 2>&1 | sed 's/^/[B2] /'

echo "=============== C: resume with a FRESH optimizer (the control) ==============="
python "$TRAINER" $COMMON --resume "$OUT/b1_first/epoch_$(printf %04d $N).ckpt" --fresh_opt \
  --out_dir "$OUT/c_freshopt" --epochs $N --save_every 999 2>&1 | sed 's/^/[C] /'

echo "=============== VERDICT ==============="
python - "$OUT" "$N" <<'PY'
import json, os, sys
out, N = sys.argv[1], int(sys.argv[2])
def curve(run):
    return {e["epoch"]: e["loss"] for e in json.load(open(os.path.join(out, run, "log.json")))}
A, B1, B2, C = (curve(r) for r in ("a_full", "b1_first", "b2_resume", "c_freshopt"))

pre = [s for s in sorted(B1) if s <= N]
d_pre = max(abs(B1[s] - A[s]) for s in pre)
print(f"sanity, steps 1..{N} before the seam: max |B1 - A| = {d_pre:.2e}")
print("(that is the floor: same code, same seed, two processes)\n")

steps = sorted(s for s in A if s > N)
print(f"{'step':>5} {'A (ref)':>13} {'B2 resumed':>13} {'C fresh opt':>13} "
      f"{'B2-A':>11} {'C-A':>11}")
db, dc = [], []
for s in steps:
    b, c = B2.get(s), C.get(s)
    if b is None or c is None:
        continue
    db.append(abs(b - A[s])); dc.append(abs(c - A[s]))
    print(f"{s:>5} {A[s]:>13.8f} {b:>13.8f} {c:>13.8f} {b-A[s]:>11.2e} {c-A[s]:>11.2e}")

# The floor is the reference scale for both numbers: two runs of the SAME code with
# the same seed already differ by this much, so "resumed matches A" can only ever
# mean "within the floor", never "identical".
floor = sum(abs(B1[s] - A[s]) for s in pre) / len(pre)
print(f"\nnondeterminism floor (mean |B1 - A| before the seam) = {floor:.3e}\n")
print("divergence from the reference, by window after the seam")
print("%14s %12s %12s %8s" % ("window", "resumed", "fresh opt", "ratio"))
for lo, hi, lab in ((0, 10, "steps 1-10"), (10, 25, "steps 11-25"),
                    (25, 50, "steps 26-50"), (50, len(db), "steps 51+")):
    b, c = db[lo:hi], dc[lo:hi]
    if not b:
        continue
    mb_, mc_ = sum(b) / len(b), sum(c) / len(c)
    print("%14s %12.3e %12.3e %7.1fx" % (lab, mb_, mc_, mc_ / max(mb_, 1e-30)))
mb, mc = sum(db) / len(db), sum(dc) / len(dc)
print("%14s %12.3e %12.3e %7.1fx" % ("all steps", mb, mc, mc / max(mb, 1e-30)))
print(f"\nresumed vs the floor: {mb/floor:.2f}x   (1.0 = indistinguishable from a rerun)")
print(f"fresh   vs the floor: {mc/floor:.2f}x")

# Weight-space distance is the sensitive measure. Per-step loss on one shape is a
# noisy scalar read of where the weights are; the weights themselves are the thing.
# All three runs save at their last step, so this costs nothing extra.
import torch
def weights(run, step):
    p = os.path.join(out, run, f"epoch_{step:04d}.ckpt")
    return torch.load(p, map_location="cpu")["state_dict"]
try:
    WA, WB, WC = (weights(r, 2*N) for r in ("a_full", "b2_resume", "c_freshopt"))
    num_b = num_c = den = 0.0
    for k_ in WA:
        a_ = WA[k_].float()
        num_b += (WB[k_].float() - a_).pow(2).sum().item()
        num_c += (WC[k_].float() - a_).pow(2).sum().item()
        den += a_.pow(2).sum().item()
    rb, rc = (num_b/den) ** 0.5, (num_c/den) ** 0.5
    print(f"\nweight-space distance from A after {2*N} steps (relative L2):")
    print(f"  resumed with state : {rb:.3e}")
    print(f"  fresh optimizer    : {rc:.3e}")
    print(f"  fresh optimizer lands {rc/max(rb,1e-30):.1f}x further from the reference")
except Exception as e:
    print(f"\n(weight-space comparison unavailable: {type(e).__name__}: {e})")
# Judged on the whole post-seam stretch, not on the first few steps. An earlier
# version of this script keyed the verdict on a 4-step window and reported
# "inconclusive" while the full-run numbers separated cleanly by 5x: the fresh
# optimizer's penalty is a SUSTAINED elevation as it rebuilds its moments, not an
# instantaneous spike, and four steps is far too small a sample to see it.
resumed_ok = mb <= 2 * floor
fresh_separates = mc >= 2.5 * mb
if resumed_ok and fresh_separates:
    print("\nVERDICT: resume-with-state reproduces the uninterrupted run to within the "
          "nondeterminism floor, and the fresh optimizer is measurably further out")
elif resumed_ok:
    print("\nVERDICT: resume-with-state reproduces the uninterrupted run, but the "
          "fresh-optimizer control did not separate; inconclusive about the size of the tax")
else:
    print("\nVERDICT: resume-with-state does NOT reproduce the reference -> INVESTIGATE")
PY

echo "RUN_END $(date -Is)"
