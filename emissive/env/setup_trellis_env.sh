#!/bin/bash
#SBATCH -J trellis_env_build
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/env_build_%j.log

# NOTE: built on an idle 2080ti just to have a GPU node for nvcc; arch list below
# cross-compiles for the GPUs we actually train on (a40/a5000=8.6, a100=8.0, l40s=8.9).

# Self-contained, sudo-free TRELLIS.2 env build (mirrors setup.sh with fixes:
# conda cuda-toolkit for nvcc, libjpeg-turbo instead of `sudo apt`, skip pillow-simd).
set -x
ENVNAME=trellis2
TRELLIS=/3dlg-jupiter-project/lightgen/segvigen_emissive/TRELLIS.2
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
nvidia-smi
cd "$TRELLIS"

export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"
export MAX_JOBS=8

# ── 1. env + torch (cu124) ────────────────────────────────────────────────────
conda create -y -n $ENVNAME python=3.10 || { echo "FAIL: env create"; exit 1; }
conda activate $ENVNAME
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 || { echo "FAIL: torch"; exit 1; }

# nvcc toolkit (match cu124) for compiling extensions + libjpeg (no sudo)
conda install -y -c "nvidia/label/cuda-12.4.0" cuda-toolkit || conda install -y -c nvidia cuda-toolkit=12.4.* || echo "WARN: cuda-toolkit install issue"
conda install -y -c conda-forge libjpeg-turbo || echo "WARN: libjpeg-turbo"
export CUDA_HOME=$CONDA_PREFIX
which nvcc && nvcc --version

# ── 2. basic deps (no sudo, no pillow-simd) ────────────────────────────────────
pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh tensorboard pandas lpips zstandard kornia timm pillow
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 || echo "WARN: utils3d"
# SegviGen extras
pip install mathutils
pip install transformers==4.57.6
pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/ || echo "WARN: bpy"
pip install spconv-cu124 || echo "WARN: spconv"

# ── 3. compiled extensions (need nvcc + GPU arch) ──────────────────────────────
pip install flash-attn==2.7.3 || { echo "FAIL: flash-attn"; exit 1; }
mkdir -p /tmp/extensions
git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && pip install /tmp/extensions/nvdiffrast --no-build-isolation || echo "WARN: nvdiffrast"
git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec && pip install /tmp/extensions/nvdiffrec --no-build-isolation || echo "WARN: nvdiffrec"
git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive && pip install /tmp/extensions/CuMesh --no-build-isolation || echo "WARN: cumesh"
git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive && pip install /tmp/extensions/FlexGEMM --no-build-isolation || echo "WARN: flexgemm"
cp -r o-voxel /tmp/extensions/o-voxel && pip install /tmp/extensions/o-voxel --no-build-isolation || { echo "FAIL: o-voxel"; exit 1; }

# ── 4. smoke test ──────────────────────────────────────────────────────────────
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import o_voxel; print('o_voxel OK')" && echo "ENV_BUILD_OK" || echo "ENV_BUILD_PARTIAL: o_voxel import failed"
echo "=== build finished ==="
