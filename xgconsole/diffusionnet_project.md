# DiffusionNet Emissive Segmentation — Project Reference

## Goal

Train DiffusionNet as a **geometry-only baseline** for predicting emissive regions on 3D object meshes.
- Input: triangle mesh (geometry + baked PBR textures as per-vertex features)
- Output: per-face binary mask (emissive / not emissive)
- Context: Dongchen is working on TEXGen+TRELLIS (image-conditioned, UV-space diffusion) for the same task. This baseline is geometry-conditioned only (no image/CLIP), same train/test split and metrics.

---

## Data

### Storage paths
| Location | Path |
|---|---|
| Workstation data | `/cs/3dlg-jupiter-project/lightgen/data/baked_uv_local_subset_uv/` |
| Solar data (same NFS, different mount) | `/3dlg-jupiter-project/lightgen/data/baked_uv_local_subset_uv/` |
| Parquet index | `.../baked_uv_local_subset/df_SomgProc_emission_filtered.parquet` |
| Overfit 1 sample split | `.../baked_uv_local_subset/overfit_split_single.json` |
| Overfit 10 sample split | `.../baked_uv_local_subset/overfit_split_10.json` |
| Full 1k split | `.../baked_uv_local_subset/data_splits_emission_filtered.json` |

**Important:** Split JSONs reference `baked_uv_local_subset` (no mesh), but we use `baked_uv_local_subset_uv` which has the same sample IDs plus mesh files.

### Per-sample file structure
Each sample is at `{data_root}/{shard}/{sample_id}/`, e.g. `000-046/e5db.../`

| File | Contents |
|---|---|
| `somage.npz` | UV-baked maps: position, objnormal, color, metal, rough, emission_color, occupancy (all 512×512) |
| `somage_original_mesh.npz` | `vert (V×3)`, `face (F×3)`, `uv (F×3×2)` — the 3D mesh |
| `somage_repacked_trimesh.npz` | Alternative mesh repacking (not used) |
| `somage.png` | Preview thumbnail |

### UV map encoding
| Array | Shape | Dtype | Decode |
|---|---|---|---|
| `position` | 512×512×3 | uint16 | `x / 65535 * 4 - 2` (range [-2, 2]) |
| `objnormal` | 512×512×3 | uint16 | `x / 65535 * 2 - 1` (range [-1, 1]) |
| `color`, `emission_color`, `metal`, `rough` | 512×512×C | uint8 | `x / 255.0` |
| `occupancy` | 512×512×1 | bool | as-is |

### Ground truth emission mask
```python
gt_emission_mask = (emission_color.max(axis=-1) > 0.001) & occupancy  # per UV pixel
```
- ~13% of occupied UV pixels are emissive (moderate class imbalance)
- The `_emission_filtered` split already excludes 55 zero-emission samples

### Split sizes (1k)
- Train: 878, Val: ~111, Test: ~110

---

## DiffusionNet Input Format

### UV → per-face labels
- UV coordinates: `uv` in `original_mesh.npz` is shape `(F, 3, 2)` — per face-corner
- Map UV `(u, v)` → pixel `(col=u*511, row=v*511)` in the 512×512 emission map
- **Per-face label**: emissive if **any** of the 3 face-corner UV pixels has emission > 0.001 (**any-emissive-wins**)

### UV → per-vertex features (11D)
For each vertex, average UV samples across all incident face-corners:
- `xyz` (3): directly from `vert` array (mesh coordinates)
- `normal` (3): area-weighted per-vertex normals computed from mesh faces (not UV-baked normals)
- `color` (3): average of UV-sampled albedo at incident face-corners
- `metal` (1): average of UV-sampled metalness
- `rough` (1): average of UV-sampled roughness

### Multi-mesh scenes
Each scene may have multiple disconnected sub-meshes. Concatenate all into one mesh (offset face indices). `patch2mesh` in `somage.npz` maps UV patches to mesh IDs.

### DiffusionNet operator cache
Call `diffusion_net.geometry.get_all_operators(verts, faces, k_eig=128, op_cache_dir=...)` once per mesh; cache to `/project/3dlg-hcvc/lightgen/diffusionnet/op_cache/`.

---

## Model & Training

### Architecture
```python
model = DiffusionNet(
    C_in=11,           # xyz + normal + color + metal + rough
    C_out=1,           # binary logit
    C_width=128,
    outputs_at='faces',
    last_activation=None  # raw logit, apply sigmoid outside
)
```

### Loss
`BCEWithLogitsLoss(pos_weight=torch.tensor([7.0]))` — compensates for ~13% positive rate.

### Metrics
IoU and F1 on binarized predictions (threshold 0.5).

### Training order
1. Overfit 10 samples → validate pipeline (expect val IoU > 0.8 on train set)
2. Full 1k training

---

## Solar Cluster

### Key facts
- **Login node** `/home/sya225` ≠ **compute node** `/home/sya225` — SEPARATE local filesystems
- **`/project/`** is NFS (saturn), shared across login + all compute nodes → put everything here
- **`/3dlg-jupiter-project/`** is NFS (jupiter), shared across login + all compute nodes → data lives here
- **Partition**: `3dlg-hcvc`
- **GPU nodes**: cs-venus-15/16/17/18 (L40S, 48GB), cs-venus-07/09/13/14 (A40, 48GB), cs-venus-08 (A100, 80GB)
- **uv**: `/home/sya225/.local/bin/uv` (installed, v0.11.7)
- **Python**: 3.12 via uv

### Connecting from workstation
```bash
python3 /tmp/cluster_skill_tmp/cluster_ssh.py run "<cmd>"
python3 /tmp/cluster_skill_tmp/cluster_ssh.py write /project/path/file.py   # pipe content in
python3 /tmp/cluster_skill_tmp/cluster_ssh.py ls /project/path/
```
(requires `paramiko`: `uv pip install paramiko`)

### Project layout on Solar
```
/project/3dlg-hcvc/lightgen/diffusionnet/
├── code/
│   ├── uv_baker.py        # UV maps → per-face labels + per-vertex features
│   ├── dataset.py         # Dataset class with operator caching
│   ├── train.py           # Training + eval loop
│   ├── submit_overfit.sh  # sbatch for 10-sample overfit
│   ├── submit_full.sh     # sbatch for full 1k training
│   └── diffusion-net/     # DiffusionNet source (git clone)
├── venv/                  # uv venv (Python 3.12)
├── op_cache/              # Precomputed Laplacian eigenvectors
├── outputs/
│   ├── overfit10/         # checkpoints + metrics
│   └── full1k/
└── logs/                  # %N-%j.out from sbatch
```

### Typical sbatch script
```bash
#!/bin/bash
#SBATCH -J diffnet_overfit
#SBATCH --partition=3dlg-hcvc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/3dlg-hcvc/lightgen/diffusionnet/logs/%N-%j.out

export PATH=/home/sya225/.local/bin:$PATH
source /project/3dlg-hcvc/lightgen/diffusionnet/venv/bin/activate

python /project/3dlg-hcvc/lightgen/diffusionnet/code/train.py \
    --split /3dlg-jupiter-project/lightgen/data/baked_uv_local_subset/overfit_split_10.json \
    --data_root /3dlg-jupiter-project/lightgen/data/baked_uv_local_subset_uv \
    --output_dir /project/3dlg-hcvc/lightgen/diffusionnet/outputs/overfit10
```

### Environment setup (one-time, run from login node)
```bash
cd /project/3dlg-hcvc/lightgen/diffusionnet
~/.local/bin/uv venv venv --python 3.12
~/.local/bin/uv pip install -p venv torch numpy scipy potpourri3d robust-laplacian scikit-learn
~/.local/bin/uv pip install -p venv -e code/diffusion-net/
```

### Monitoring
```bash
python3 /tmp/cluster_skill_tmp/cluster_ssh.py run "squeue -u sya225"
python3 /tmp/cluster_skill_tmp/cluster_ssh.py run "tail -50 /project/3dlg-hcvc/lightgen/diffusionnet/logs/<logfile>"
```

---

## Implementation Phases

| Phase | What | Where |
|---|---|---|
| 0 | Bootstrap: mkdir, git clone diffusion-net, create venv, install deps | Solar (via cluster_ssh.py) |
| 1 | Write uv_baker.py, dataset.py, train.py | Write locally → push via cluster_ssh.py write |
| 2 | Overfit 10-sample test | Solar sbatch |
| 3 | Full 1k training | Solar sbatch |

---

## Reference Code

### Dongchen's emission mask (TEXGen)
`/localhome/xya120/studio/misc/lightgen/TEXGen/spuv/data/lightgen_uv.py` — line 247:
```python
gt_emission_mask = ((emission_color.max(dim=0, keepdim=True)[0] > 0.001) * occupancy).float()
```

### DiffusionNet reference experiment
`/local-scratch2/xya120/studio/misc/lightgen/diffusion-net/experiments/human_segmentation_original/`
