#!/bin/bash
#SBATCH --job-name=lg_k8
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --gres=gpu:0 --cpus-per-task=16 --mem=64G --time=03:00:00
#SBATCH --array=0-10
#SBATCH --output=/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3/logs/k8_%A_%a.out
set -o pipefail
D=/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3
PY=/project/3dlg-hcvc/omages/omages_internal/.venv/bin/python
export PYTHONPATH=/project/3dlg-hcvc/omages/xgutils/src
hostname

# THE KEY DROPS FROM 20 TO 8, EVERYWHERE THE PANELS ARE KEY-LIT.
#
# Not a taste change. At key 20 the Glare node fired on 108,865 pixels of a
# panel whose prediction was EMPTY, against 123,948 on the ground truth, so the
# glow was reporting the lamp rather than the object. At key 8 nothing
# non-emissive reaches the node's linear threshold of 1.0: it fires on 0 pixels
# of the empty panel and 50,458 of the ground truth. Bloom becomes a property of
# emission instead of a property of brightness.
#
# Every flag is written out rather than defaulted, so this run records what
# produced it and a later default change cannot alter a re-run silently.
COMMON="--res 768 --samples 256 --samples_lit 96 --key 8 --bg 0.012 \
  --view_transform AgX --exposure 0.0 \
  --bloom 1 --bloom_size 9 --bloom_threshold 1.0 --bloom_mix -0.15"
CMP=48af42db48c44cd9bfab32bbb057a39c,1e9c6545b4da42e0ba4e5dbcd2e0e8ff,9418a924a50d44c186dd499006b62424,8f4c281aef1b4563b6103efbcd77fac1,b7709a651d144134a5babce33223380a,658ecf9f837246509b0b1c4aa81e9e5b

# 1. the gallery: every shape, its studio reference, its ground truth and its
#    mask-times-albedo panel
$PY $D/render_emissive.py --manifest $D/manifest12.json --glb_dir $D/glb_src \
    --out $D/final3 $COMMON \
    --shard $SLURM_ARRAY_TASK_ID --nshards 11 --overwrite 1
G=$?
echo "GALLERY_EXIT=$G"

# 2. the random baseline, comparison shapes only. Sharded over the same array,
#    so a shard holding no comparison shape simply renders nothing.
$PY $D/render_emissive.py --manifest $D/manifest12.json --glb_dir $D/glb_src \
    --out $D/baselines_k8 $COMMON --mode random --seed 7 --only $CMP \
    --shard $SLURM_ARRAY_TASK_ID --nshards 11 --overwrite 1
B=$?
echo "BASELINE_EXIT=$B"

# 3. the coverage-matched albedo baseline. The MASKS are unchanged and are not
#    rebuilt: they are a property of the asset's texture, not of the lighting.
#    Only the render moves to the new key.
for SID in $(echo $CMP | tr ',' ' '); do
  H=$(( $(echo $SID | cksum | cut -d' ' -f1) % 11 ))
  [ "$H" = "$SLURM_ARRAY_TASK_ID" ] || continue
  $PY $D/render_emissive.py --manifest $D/manifest12.json --glb_dir $D/glb_src \
      --out $D/pred/albedo_matched_k8 --mode method --only $SID $COMMON \
      --pred_masks $D/pred_masks/albedo_matched \
      --camera_json $D/cameras/$SID.json --overwrite 1
  echo "HEUR_EXIT_$SID=$?"
done
echo "ALL_STAGES_DONE"
