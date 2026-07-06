#!/bin/bash
#SBATCH -J emis_build_1k
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/build_%j.log
# Chunking variant of build_dataset.sbatch — adds $6=start offset so the same
# split can be built in parallel array-style chunks (e.g. train_1k / val_96).
# args: $1=split $2=n $3(optional)="real"|"" $4(optional)=out_split_name $5(optional)="pbr"|"" $6(optional)=start
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
cd /3dlg-jupiter-project/lightgen/segvigen_emissive
COND_FLAG="--zero_cond"
if [ "$3" = "real" ]; then COND_FLAG="--real_cond"; fi
OUT_NAME_ARG=""
if [ -n "$4" ]; then OUT_NAME_ARG="--out_split_name $4"; fi
PBR_FLAG=""
if [ "$5" = "pbr" ]; then PBR_FLAG="--pbr_only"; fi
python code/build_dataset.py --split "${1:-val}" --n "${2:-16}" --start "${6:-0}" \
    --out_root /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset \
    $COND_FLAG $OUT_NAME_ARG $PBR_FLAG
