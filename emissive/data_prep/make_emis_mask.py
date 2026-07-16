"""
Per-sample emissive coverage mask aligned to output_tex_slat's latent coords, for the
class-imbalance loss weighting in train_emissive.py (--pos_weight).

For each output_tex_slat coord (32-res sparse latent grid, i.e. 512/16 downsample from
the raw vxz voxel grid) compute the fraction of that 16^3 raw block's SURFACE voxels
(as stored in output.vxz — sparse, occupied-only) that are white/emissive. Saved as
emis_mask.pth: a float32 tensor (N,), aligned 1:1 (same order) with output_tex_slat's
coords.

Verified attr layout (canon_overfit10 sample 0007deb6..., GPU node cs-venus-16, via ssh):
  o_voxel.io.read(output.vxz) -> coords (M,3) int32 @512-res (occupied surface voxels
  only, NOT dense) + data dict incl. base_color (M,3) uint8. Overwhelmingly binary:
  (0,0,0) black / (255,255,255) white — set by somage_to_glb.build_emissive_target_glb's
  baseColorFactor/emissiveFactor on the two solid submeshes. A handful of dual-grid
  boundary voxels get intermediate blended values (e.g. 191, 170) -> threshold at
  mean>127.5 to binarize. Block correspondence confirmed exact: floor(raw_coord/16) ==
  slat_coord for both the min and max of every axis on the sample checked.

o_voxel can't be imported on a GPU-less node as-is: its __init__.py eagerly does
`from . import postprocess`, which pulls in flex_gemm/triton and needs a live CUDA
driver even for CPU-only work (confirmed: fails with "0 active drivers" on the login
node). We bypass this by registering a stub 'o_voxel' package in sys.modules (pointing
at the real package dir) before importing o_voxel.io/serialize directly, so
__init__.py never runs. Those submodules only touch o_voxel._C (CPU/CUDA-dispatched
compiled extension) and stdlib/numpy/torch — verified CPU-only on the login node. On a
GPU node where trellis2/o_voxel is already fully imported, this is a no-op (uses the
existing sys.modules entry).

CLI (pure CPU/torch — runs on the cluster login node or a CPU job):
  python make_emis_mask.py --data_root /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset \
      --split canon_overfit10 [--sid <sid>]
"""
import os
import sys
import json
import types
import argparse
import importlib
import importlib.util

import torch


def _cpu_o_voxel_io():
    """Import o_voxel.io (+its o_voxel.serialize/._C deps) without triggering
    o_voxel/__init__.py's eager triton import. See module docstring."""
    if "o_voxel" not in sys.modules:
        spec = importlib.util.find_spec("o_voxel")
        assert spec is not None, "o_voxel not installed (wrong conda env? need trellis2)"
        pkg = types.ModuleType("o_voxel")
        pkg.__path__ = spec.submodule_search_locations
        pkg.__spec__ = spec
        sys.modules["o_voxel"] = pkg
    return importlib.import_module("o_voxel.io")


LATENT_RES = 32   # output_tex_slat coord grid resolution
BLOCK = 16         # raw voxels per latent cell per axis (512 raw / 32 latent)


def compute_emis_mask(output_vxz_path, slat_coords, io_mod=None):
    """slat_coords: (N,4) int tensor [batch,x,y,z] @32-res, in output_tex_slat's coord
    order (as saved by vxz_to_slat.py). Returns float32 (N,): fraction of output.vxz
    surface voxels in that coord's 16^3 raw block that are white/emissive (0 for a
    latent coord whose block happens to have no surface voxels — shouldn't occur for
    coords that came out of vxz_to_slat, but guarded rather than dividing by zero)."""
    io_mod = io_mod or _cpu_o_voxel_io()
    # num_threads=1: the generic io.read() dispatcher spawns os.cpu_count() ThreadPool
    # workers per call with no override, which exhausted the login node's `ulimit -u`
    # (max user processes, shared across all our sessions there) after a handful of
    # samples — call read_vxz directly so we can pin it down.
    coords_raw, data = io_mod.read_vxz(output_vxz_path, num_threads=1)
    white = data["base_color"].float().mean(dim=-1) > 127.5

    n = LATENT_RES ** 3
    block = (coords_raw.long() // BLOCK).clamp(0, LATENT_RES - 1)
    lin = (block[:, 0] * LATENT_RES + block[:, 1]) * LATENT_RES + block[:, 2]
    total = torch.bincount(lin, minlength=n).float()
    whit = torch.bincount(lin[white], minlength=n).float()
    frac_grid = torch.where(total > 0, whit / total.clamp(min=1), torch.zeros_like(total))

    sx, sy, sz = slat_coords[:, 1].long(), slat_coords[:, 2].long(), slat_coords[:, 3].long()
    slin = (sx * LATENT_RES + sy) * LATENT_RES + sz
    return frac_grid[slin].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--sid", default=None, help="single sample id (default: whole split)")
    args = ap.parse_args()

    io_mod = _cpu_o_voxel_io()
    sdir = os.path.join(args.data_root, args.split)
    sids = [args.sid] if args.sid else sorted(os.listdir(sdir))

    n_done = n_skip = n_fail = 0
    covs, metas = [], []
    for sid in sids:
        d = os.path.join(sdir, sid)
        vxz_p = os.path.join(d, "output.vxz")
        slat_p = os.path.join(d, "output_tex_slat.pth")
        out_p = os.path.join(d, "emis_mask.pth")
        if not (os.path.isdir(d) and os.path.exists(vxz_p) and os.path.exists(slat_p)):
            continue
        if os.path.exists(out_p):
            n_skip += 1
            continue
        try:
            slat = torch.load(slat_p, map_location="cpu")
            mask = compute_emis_mask(vxz_p, slat["coords"], io_mod=io_mod)
            torch.save(mask, out_p)
            n_done += 1
            cov = mask.mean().item()
            covs.append(cov)
            mp = os.path.join(d, "meta.json")
            mfrac = json.load(open(mp)).get("emissive_frac") if os.path.exists(mp) else None
            metas.append(mfrac)
            mfrac_s = f"{mfrac:.3f}" if mfrac is not None else "n/a"
            print(f"[ok] {sid} N={mask.shape[0]} mask_cov={cov:.3f} meta_frac={mfrac_s}", flush=True)
        except Exception as e:
            n_fail += 1
            print(f"[fail] {sid}: {repr(e)[:200]}", flush=True)

    print(f"DONE split={args.split} done={n_done} skip={n_skip} fail={n_fail}", flush=True)
    if len(covs) >= 2 and all(m is not None for m in metas):
        import numpy as np
        r = float(np.corrcoef(covs, metas)[0, 1])
        print(f"[corr] mask_cov vs meta.emissive_frac: r={r:.4f} (n={len(covs)})", flush=True)


if __name__ == "__main__":
    main()
