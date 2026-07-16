#!/bin/bash
#SBATCH -J trellis_env_resume
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/3dlg-jupiter-project/lightgen/segvigen_emissive/env_resume_%j.log

# Resume the partially-built trellis2 env (torch + cuda-toolkit + basics already installed;
# it died at flash-attn). Install EXTENSIONS FIRST (o_voxel etc. — what data-prep needs),
# then flash-attn LAST with --no-build-isolation (the fix for the ModuleNotFoundError).
set -x
source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh
conda activate trellis2
cd /3dlg-jupiter-project/lightgen/segvigen_emissive/TRELLIS.2
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"
export CUDA_HOME=$CONDA_PREFIX
export MAX_JOBS=4
which nvcc && nvcc --version | tail -1
python -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available())"
pip install packaging ninja wheel

mkdir -p /tmp/extensions
# extensions needed for DATA-PREP (encoders + voxelization) — prioritize
[ -d /tmp/extensions/nvdiffrast ] || git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install /tmp/extensions/nvdiffrast --no-build-isolation || echo "WARN nvdiffrast"
[ -d /tmp/extensions/nvdiffrec ] || git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
pip install /tmp/extensions/nvdiffrec --no-build-isolation || echo "WARN nvdiffrec"
[ -d /tmp/extensions/CuMesh ] || git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive
pip install /tmp/extensions/CuMesh --no-build-isolation || echo "WARN cumesh"
[ -d /tmp/extensions/FlexGEMM ] || git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive
pip install /tmp/extensions/FlexGEMM --no-build-isolation || echo "WARN flexgemm"
cp -r o-voxel /tmp/extensions/o-voxel 2>/dev/null; pip install /tmp/extensions/o-voxel --no-build-isolation || echo "WARN o-voxel"

# flash-attn LAST (needed for the FLOW model = train/eval, not data-prep). Source build
# with --no-build-isolation (uses installed torch). If it OOMs, try prebuilt wheel fallback.
pip install flash-attn==2.7.3 --no-build-isolation || \
  pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir || echo "WARN flash-attn (flow/train blocked; data-prep still OK)"

# smoke tests
python -c "import o_voxel; print('o_voxel OK')" && echo "OVOXEL_OK" || echo "OVOXEL_FAIL"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)" && echo "FLASHATTN_OK" || echo "FLASHATTN_FAIL"
python -c "from trellis2 import models; print('trellis2 models import OK')" && echo "TRELLIS_IMPORT_OK" || echo "TRELLIS_IMPORT_FAIL"
echo "=== resume finished ==="
