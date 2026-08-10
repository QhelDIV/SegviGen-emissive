#!/bin/bash
#SBATCH -J emis_smoke_ccreal
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72k_ccreal/smoke_%j.log
# Smoke for the 2026-08 conditioning-parity run: --pbr_cond channel (lightgen-style
# per-token concat, 96ch, input_layer re-init from full_seg) + --cond real.
# Success = SMOKE_DONE with (a) "[pbr_cond=channel] warm start: re-initialized
# ['flow_model.input_layer.weight', 'flow_model.input_layer.bias']" in the log,
# (b) finite decreasing flow_loss over 3 epochs, (c) quick-val runs.
set -euo pipefail

mkdir -p /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72k_ccreal

source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "SMOKE_START $(date -Iseconds) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /3dlg-jupiter-project/lightgen/segvigen_emissive

# Preflight: with --cond real EmisDataset hard-errors on the first admitted
# sample without cond.pth (and, since --pos_weight 5.0 => require_mask=True, on
# missing emis_mask.pth). Mirror its admission rule exactly (all three core
# files) and fail up front with an actionable count instead.
python - <<'EOF'
import os
root = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct"
core = ("shape_slat.pth", "input_tex_slat.pth", "output_tex_slat.pth")
for split in ("train_72k", "val_72k"):
    sdir = os.path.join(root, split)
    sids = [s for s in sorted(os.listdir(sdir))
            if all(os.path.exists(os.path.join(sdir, s, f)) for f in core)]
    no_cond = [s for s in sids if not os.path.exists(os.path.join(sdir, s, "cond.pth"))]
    no_mask = [s for s in sids if not os.path.exists(os.path.join(sdir, s, "emis_mask.pth"))]
    print(f"[preflight] {split}: {len(sids)} admitted samples, "
          f"{len(no_cond)} missing cond.pth, {len(no_mask)} missing emis_mask.pth")
    assert not no_cond, (f"{split}: {len(no_cond)} admitted samples lack cond.pth — run "
                         f"emissive/data_prep/build_cond_thumbnail.py (see cond_thumb_repair.sbatch) first")
    assert not no_mask, (f"{split}: {len(no_mask)} admitted samples lack emis_mask.pth — run "
                         f"make_emis_mask.py (or drop --pos_weight) first")
EOF

python -u emissive/train/train_emissive.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct \
  --train_split train_72k \
  --val_split val_72k \
  --out_dir /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72k_ccreal/smoke \
  --init_ckpt full_seg \
  --cond real \
  --pbr_cond channel \
  --pos_weight 5.0 \
  --lr 1e-5 --lr_schedule const --ema 0.999 \
  --n_per_epoch 24 --epochs 3 --save_every 1 \
  --select_on nonzero --val_quick 2 \
  2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }'

echo "SMOKE_DONE $(date -Iseconds)"
