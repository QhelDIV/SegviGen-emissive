#!/bin/bash
#SBATCH -J build_72k_direct
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --array=0-374%20
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/logs/build_72k_%A_%a.log
#
# RATIFIED by owner (via team-lead), launching. Design notes:
#
# - Scope: ALL SPLITS in one array, not two passes -- owner's call: train 59,602 +
#   val 7,450 + test 7,451 = 74,503 (data_splits_74k.json total). One submission now
#   rather than deferring val/test to a second mobilization that tends not to happen
#   promptly.
# - Shard size 200/task, per split:
#     train: ceil(59602/200)=299 shards (task ids   0..298, last shard n=2)
#     val:   ceil( 7450/200)= 38 shards (task ids 299..336, last shard n=50)
#     test:  ceil( 7451/200)= 38 shards (task ids 337..374, last shard n=51)
#   total 375 tasks -> --array=0-374. build_dataset_direct.py slices sids_all[start:
#   start+n] with plain Python list slicing, which does not error on overrun, so the
#   short last shard per split needs no special-casing here.
# - Throttle %20, not the 42 confirmed-usable GPUs: owner's other project (somages,
#   arxiv_sprint) is actively running on Solar right now; this build is not urgent to
#   the hour, so it does not take every GPU.
# - --exclude nodes confirmed NOT usable (tested with mount_check_*.sbatch, jobs
#   237041/237042): cs-venus-19 confirmed NOT mounting OVOX_ROOT; cs-venus-09 is DOWN
#   cluster-wide (sinfo: down on every partition) so it cannot be tested or scheduled,
#   excluded on that harder basis rather than an unconfirmed mount. cs-venus-02 tested
#   and CONFIRMED mounted (job 237041, stat/ls both succeeded) -- included in the pool.
# - cs-venus-05 EXCLUDED (added after job 237072's first run): its RTX PRO 6000 Blackwell
#   Max-Q GPUs are sm_120, and the trellis2 conda env's PyTorch build only supports up to
#   sm_90 -- every shape on that node hard-fails with "CUDA error: no kernel image is
#   available for execution on the device" (confirmed: task 237072_12 logged 11/11
#   failures, 0 successes, all this exact error, before this node was excluded). Not a
#   contention or data issue -- a hardware/environment incompatibility, 100% failure rate
#   for any task landing there. Revisit only if the trellis2 env is rebuilt against a
#   newer PyTorch/CUDA toolkit with sm_120 support.
# - CORRECTED after first launch (job 237050, cancelled): the 10.6s/shape projection was
#   measured on an idle node running one process, and does not hold with Dongchen's rebake
#   (8 workers, 64 CPUs/node) and other concurrent account jobs sharing the same nodes and
#   NFS. Position-in-shard analysis of 125 completed shapes from the first attempt: shapes
#   1-3 per task mean 48.6s (one-time CUDA/kernel warm-up, confirmed by the encode-stage
#   timing specifically, mesh/dualgrid stay small throughout), shapes 4+ mean 17.7s steady
#   state -- 67% over the original 10.6s projection, with ~28% of steady-state shapes still
#   over 20s (open question: correlates with n512 = intrinsic size tail, vs uncorrelated =
#   node contention -- not resolved before this resubmit, doesn't gate it).
#   Revised shard estimate: 3*48.6 + 197*17.7 ~= 3633s ~= 60.5min. --time=3:00:00 (was
#   50:00 on 3dlg-hcvc-lab-debug, which is also the wrong partition for a 375-task
#   production array -- moved to -short) gives ~3x margin over the revised estimate,
#   covering the measured size tail (zstd write_vxz speedup shrinks from ~2-7x at typical
#   sizes to ~1.15x at the largest, ~1.9M voxels) and further contention.
#   Revised totals: ~378 node-hours, ~19 hours wall clock at throttle %20 (was 265
#   node-hours / 10-11 hours from the original idle-node projection).
# - --shuffle_seed 42 passed to build_dataset_direct.py, per split independently: per-shape
#   cost correlates with shape size, so shard-by-slice on the UNSHUFFLED split risks one
#   shard drawing a disproportionate share of the slow tail. Shuffle once per split, slice
#   after -- straggler risk is a within-split-shard concern, so shuffling within each split
#   separately (rather than pooling all three splits into one shuffle) is sufficient and
#   keeps each split's own sid-to-index mapping simple to audit.
# - Idempotent: build_one's caller already skips any sid with output_tex_slat.pth +
#   emis_mask.pth on disk, so resubmitting this exact array after a partial/preempted
#   run only rebuilds what's missing -- no separate resume logic needed.
# - Failure manifest: each array task writes its own <out_split>_failures_shard<N>.json
#   (sid + error string) and <out_split>_manifest_shard<N>.json (ok sids) -- suffixed by
#   SLURM_ARRAY_TASK_ID so concurrent tasks never race on one shared file. Merge per split
#   after all tasks finish, e.g. `cat train_72k_manifest_shard*.json` (jq -s or similar) for
#   the ok list, same for failures. Note shard numbering restarts at 0 for the val and test
#   blocks (see SPLIT/START derivation below), so merge glob patterns should not assume
#   shard<N> is globally unique across splits -- it is unique per out_split_name, which IS
#   what the filenames are keyed on.
# - No filtering: every shape in the pool gets built, including albedo-lit ones -- that
#   decision stays with the owner, not implicit in this script.

source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
cd /3dlg-jupiter-project/lightgen/segvigen_emissive

SHARD_SIZE=200
TID=$SLURM_ARRAY_TASK_ID

# Map the single combined array index to (split, per-split shard index), then to
# (--split, --start, --out_split_name). Ranges: train 0-298, val 299-336, test 337-374.
if [ "$TID" -le 298 ]; then
  SPLIT=train
  OUT_NAME=train_72k
  SHARD=$TID
elif [ "$TID" -le 336 ]; then
  SPLIT=val
  OUT_NAME=val_72k
  SHARD=$((TID - 299))
else
  SPLIT=test
  OUT_NAME=test_72k
  SHARD=$((TID - 337))
fi
START=$((SHARD * SHARD_SIZE))

echo "task=$TID split=$SPLIT shard=$SHARD start=$START" >&2

python emissive/data_prep/build_dataset_direct.py \
  --split "$SPLIT" --shuffle_seed 42 \
  --start "$START" --n "$SHARD_SIZE" \
  --out_root /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct \
  --out_split_name "$OUT_NAME"
