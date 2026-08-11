#!/usr/bin/env python3
"""Read each shape's 256^3 PBR and emission voxels and stage them for rendering.

The voxel files live on /cs/3dlg-jupiter-project, which the solar compute nodes
cannot see, and reading them needs o_voxel's compiled extension, which is built
against the trellis2 environment. So the read happens HERE, on the workstation,
and the result is written to /project as a plain npz the render job can load with
nothing but numpy.

Run (the LD_PRELOAD is not optional: the workstation's system libstdc++ is
GLIBCXX_3.4.30 and o_voxel's _C.so needs 3.4.32):

  LD_PRELOAD=/cs/3dlg-jupiter-project/lightgen/miniforge3/lib/libstdc++.so.6 \
  /cs/3dlg-jupiter-project/lightgen/miniforge3/envs/trellis2/bin/python \
      extract_voxels.py --sids <a,b,c> --out <dir>

Downsampling: the raw grid is 256^3, which at a gallery tile's size is well
under a pixel per voxel and reads as a smooth surface rather than as voxels. The
staged grid is therefore 64^3, four raw voxels per side. A display cell counts as
emissive if ANY raw voxel inside it emits, so a small emissive region stays
visible instead of being eroded away; that dilates thin regions by up to four
voxels, which is why the true 256^3 fraction is recorded alongside and is the
number to quote.
"""
import argparse
import importlib
import importlib.util
import json
import os
import sys
import types

import numpy as np

OVOXEL = ("/cs/3dlg-jupiter-project/lightgen/segvigen_emissive/direct_pilot/"
          "o-voxel-build")
BAKE = "/cs/3dlg-jupiter-project/lightgen/uv_voxel_pipeline/out_uv_voxel_74k"
RES = 256


def o_voxel_io():
    """o_voxel.io without running o_voxel/__init__.py, which eagerly imports
    triton and needs a live CUDA driver."""
    sys.path.insert(0, OVOXEL)
    if "o_voxel" not in sys.modules:
        spec = importlib.util.find_spec("o_voxel")
        assert spec is not None, "o_voxel not importable"
        pkg = types.ModuleType("o_voxel")
        pkg.__path__ = spec.submodule_search_locations
        pkg.__spec__ = spec
        sys.modules["o_voxel"] = pkg
    return importlib.import_module("o_voxel.io")


def one(io_mod, sid, out_dir, block):
    pbr = os.path.join(BAKE, sid, "pbr_voxels_256", f"{sid}.vxz")
    emi = os.path.join(BAKE, sid, "emission_voxels_256", f"{sid}.vxz")
    coord, attr = io_mod.read_vxz(pbr, num_threads=4)
    coord_e, attr_e = io_mod.read_vxz(emi, num_threads=4)
    coord = coord.numpy().astype(np.int32)
    color = attr["base_color"].numpy().astype(np.uint8)
    emissive = attr_e["emissive"].numpy()
    # The two files are written from one pass over the same surface voxels, so
    # they must agree row for row; if they ever stop agreeing, the emissive
    # attribute would be attached to the wrong voxel and nothing downstream
    # would notice.
    assert coord_e.shape[0] == coord.shape[0], f"{sid}: voxel count mismatch"
    assert np.array_equal(coord_e.numpy().astype(np.int32), coord), \
        f"{sid}: pbr and emission voxel orders differ"

    lit = emissive.max(axis=1) > 0        # any emission at all
    frac256 = float(lit.mean())

    # ---- downsample to a legible display grid
    cell = coord // block
    res = RES // block
    key = (cell[:, 0].astype(np.int64) * res + cell[:, 1]) * res + cell[:, 2]
    order = np.argsort(key, kind="stable")
    key_s, color_s, lit_s = key[order], color[order], lit[order]
    uniq, start, count = np.unique(key_s, return_index=True, return_counts=True)
    sums = np.add.reduceat(color_s.astype(np.int64), start, axis=0)
    mean_color = (sums / count[:, None]).astype(np.uint8)
    any_lit = np.maximum.reduceat(lit_s.astype(np.uint8), start)

    cells = np.stack([uniq // (res * res), (uniq // res) % res, uniq % res],
                     axis=1).astype(np.uint8)
    meta = {
        "sid": sid, "res_raw": RES, "res_display": res, "block": block,
        "n_voxels_raw": int(coord.shape[0]),
        "n_cells_display": int(cells.shape[0]),
        "emissive_frac_raw": frac256,
        "emissive_frac_display": float(any_lit.mean()),
    }
    np.savez_compressed(os.path.join(out_dir, f"{sid}.npz"),
                        cells=cells, color=mean_color, lit=any_lit,
                        meta=json.dumps(meta))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sids", required=True, help="comma separated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    io_mod = o_voxel_io()
    for sid in args.sids.split(","):
        m = one(io_mod, sid, args.out, args.block)
        print(f"{sid[:8]}  raw {m['n_voxels_raw']:>8,} voxels  "
              f"display {m['n_cells_display']:>7,} cells at {m['res_display']}^3  "
              f"emissive raw {m['emissive_frac_raw']:.4f}  "
              f"display {m['emissive_frac_display']:.4f}")


if __name__ == "__main__":
    main()
