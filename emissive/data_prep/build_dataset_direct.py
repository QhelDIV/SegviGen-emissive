"""
Direct-ovoxel dataset builder (CANONICAL pipeline, "Option B").

Builds the training tuple (shape_slat, input_tex_slat, output_tex_slat, emis_mask)
DIRECTLY from Dongchen's preprocessed 256^3 o-voxels -- no somage/GLB round-trip.
Supersedes build_dataset.py (Path A), which is kept only as the tensor-contract
reference. See project memory: project_direct_ovoxel_pipeline.

Ground-truth ovoxel data (read-only):
  /3dlg-jupiter-project/lightgen/uv_voxel_pipeline/out_uv_voxel_74k/<sha>/
    pbr_voxels_256/<sha>.vxz       -- base_color(N,3) u8, metallic(N,1) u8,
                                       roughness(N,1) u8, alpha(N,1) u8
    emission_voxels_256/<sha>.vxz  -- emissive(N,3) u8
  Both share the SAME coords (N,3) int32 in [0,255], SAME order (verified).
  (Path is /cs/3dlg-jupiter-project/... from the cs-3dlg-25 workstation, but
  /3dlg-jupiter-project/... -- no /cs prefix -- on Solar compute nodes. Not all
  Solar nodes mount it: cs-venus-19 does not; cs-venus-{05,07,08,13,14,15,16,17}
  do, confirmed 2026-07-29.)

Frame (VERIFIED empirically against Dongchen's coords: re-voxelizing the raw glb
of the "simple" fixture 00001dbc99db48efb16f81945dcc9999 at grid=256 with the
recipe below reproduces Dongchen's exact voxel coordinate SET, IoU=1.0000):
identity axes, NO permutation -- center = bbox center of the raw glb's merged
geometry, scale = 0.99999 / max_extent -> vertices in [-0.5, 0.5]. This is the
same single-step recipe SegviGen's glb_to_vxz.py uses, and is mathematically
equivalent to uv_voxel_pipeline's two-stage canonical[-1,1] -> tight[-0.5,0.5]
normalize (composing two linear normalizes of the same shape gives one).
CORRECTION vs. prior project notes: there is no "dong = (x, z, 1-y)" axis
permutation in this pipeline -- that must refer to a different/older convention.

THE GAP this module fills: Dongchen's ovoxels carry attribute channels only,
never the dual-grid GEOMETRY channel (dual_vertices, intersected) full_seg's
shape_encoder needs, and never at the 512^3 resolution full_seg was pretrained
at (Dongchen's grid is 256^3). This builder re-extracts the dual-grid geometry
NATIVELY at 512 from the source mesh (o_voxel.convert.mesh_to_flexible_dual_grid,
same call/params as glb_to_vxz.py -- CPU, no GPU needed for this step alone),
then UPSAMPLES Dongchen's 256 attribute values onto those 512 coords by nearest
-neighbor (parent = coord512 // 2), falling back to a nearest-occupied-256-voxel
KD-tree lookup for the rare 512 voxel whose parent cell has no 256 attribute
(the QEF dual-grid and texture-attr rasterization occupancy sets don't nest
perfectly -- see meta.json's "gap_frac" per shape).

Binarization: emission is binarized at >1/255 (any nonzero channel), per project
memory -- NOT the older 0.04 per-face threshold the superseded build_dataset.py
path used.

Output target ("output_tex_slat") follows the same channel-hijack convention
already used by SegviGen's somage_to_glb.build_emissive_target_glb (white=
emissive / black=non in base_color, metallic=0, roughness=1(255), alpha=1(255))
and matches TRELLIS2's EmissionPbrFinetuneTrainer channel-hijack constants.

Design: build a REAL (input.vxz, output.vxz) pair per shape -- identical coords
(the freshly-extracted 512 dual grid) in both -- then hand off to SegviGen's
UNCHANGED vxz_to_slat() for encode + get_common_coords + save. The on-disk
contract train_emissive.py already reads (shape_slat.pth, input_tex_slat.pth,
output_tex_slat.pth, emis_mask.pth, meta.json under <out_root>/<split>/<sid>/)
needs ZERO changes -- EmisDataset works unmodified.

Requires the TRELLIS.2 env (o_voxel, trellis2, scipy) + weights. Run on a GPU node.

Usage:
  python build_dataset_direct.py --split train --n 20 --out_root .../dataset_direct
  python build_dataset_direct.py --sid_file shas.txt --out_split_name smoke20 \
      --out_root .../dataset_direct
"""
import os
import sys
import json
import time
import argparse
import traceback

import numpy as np
import torch
import trimesh

ROOT = os.path.dirname(os.path.abspath(__file__))
SEGVIGEN = os.path.join(ROOT, "SegviGen")
sys.path.insert(0, SEGVIGEN)
sys.path.insert(0, os.path.join(SEGVIGEN, "data_toolkit"))

os.environ.setdefault("HF_HOME", "/3dlg-jupiter-project/lightgen/hf_cache")

import o_voxel  # noqa: E402
from vxz_to_slat import vxz_to_slat  # noqa: E402
from make_emis_mask import compute_emis_mask  # noqa: E402

OVOX_ROOT = "/3dlg-jupiter-project/lightgen/uv_voxel_pipeline/out_uv_voxel_74k"
GLB_ROOT = "/3dlg-falas/datasets/TexVerse-1K"
PARQUET = "/3dlg-falas/project/omages/datasets/TexVerse/lightgen/emissive_thumbnails_obj_ids_df.parquet"
SPLIT_JSON = "/3dlg-jupiter-project/lightgen/diffusionnet_xg/data/data_splits_74k.json"
TRELLIS = "microsoft/TRELLIS.2-4B/ckpts"

GRID = 512
# channel-hijack constants for the binary emissive target (matches
# somage_to_glb.build_emissive_target_glb / TRELLIS2 EmissionPbrFinetuneTrainer)
OUT_METALLIC_U8 = 0
OUT_ROUGHNESS_U8 = 255
OUT_ALPHA_U8 = 255


def load_split_sids(split, shuffle_seed=None):
    """shuffle_seed: if set, shuffle the split's sid list (fixed seed, reproducible)
    before the caller slices it with --start/--n. Needed for SLURM-array sharding of
    the full split -- per-shape build cost correlates with shape size (measured: the
    slowest smoke20 shapes were also the largest by n_voxels_512), so slicing the
    UNSHUFFLED split risks one array shard drawing a disproportionate share of the
    slow tail and becoming a straggler. Shuffling once with a fixed seed spreads that
    tail evenly across shards while keeping the assignment reproducible/idempotent
    across resubmits."""
    import pandas as pd
    df = pd.read_parquet(PARQUET)
    if "success" in df.columns:
        df = df[df["success"] == True]  # noqa: E712
    sp = json.load(open(SPLIT_JSON))
    idx = list(sp[split]["indices"])
    if shuffle_seed is not None:
        import random
        random.Random(shuffle_seed).shuffle(idx)
    sids = [df.iloc[i].name for i in idx]
    glb_rel = {df.iloc[i].name: df.iloc[i]["glb_1k_path"] for i in idx}
    return sids, glb_rel


def resolve_glb_rel(sids):
    """--sid_file path: look up glb_1k_path for an explicit sid list."""
    import pandas as pd
    df = pd.read_parquet(PARQUET)
    missing = [s for s in sids if s not in df.index]
    if missing:
        raise KeyError(f"{len(missing)} sids not in {PARQUET}: {missing[:3]}...")
    return {s: df.loc[s, "glb_1k_path"] for s in sids}


def merged_mesh_512_frame(glb_path):
    """Load+merge a glb into (vertices, faces) tensors in the tight [-0.5, 0.5]
    voxel frame -- verified byte-identical (voxel-set IoU=1.0000 at grid=256) to
    Dongchen's ovoxel frame. Same single-step recipe as SegviGen's glb_to_vxz.py."""
    asset = trimesh.load(glb_path, force="scene")
    aabb = asset.bounding_box.bounds
    center = (aabb[0] + aabb[1]) / 2
    scale = 0.99999 / (aabb[1] - aabb[0]).max()
    asset.apply_translation(-center)
    asset.apply_scale(scale)
    mesh = asset.to_mesh()
    vertices = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32))
    faces = torch.from_numpy(np.asarray(mesh.faces, dtype=np.int64))
    return vertices, faces


def dual_grid_512(vertices, faces):
    """o_voxel dual-grid GEOMETRY extraction at 512 (CPU, no GPU needed) -- same
    call/params/post-processing as SegviGen's glb_to_vxz.py at its native 512 grid."""
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices, faces, grid_size=GRID, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False,
    )
    vid = o_voxel.serialize.encode_seq(voxel_indices)
    mapping = torch.argsort(vid)
    voxel_indices = voxel_indices[mapping]
    dual_vertices = dual_vertices[mapping]
    intersected = intersected[mapping]

    dual_vertices = dual_vertices * GRID - voxel_indices
    dual_vertices = (torch.clamp(dual_vertices, 0, 1) * 255).type(torch.uint8)
    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)
    return voxel_indices, dual_vertices, intersected


class Upsampler256to512:
    """coord512 -> attribute row index at 256, by nearest-neighbor block parent
    (coord512 // 2), falling back to a nearest-OCCUPIED-256-voxel KD-tree lookup
    when the exact parent cell has no attribute (the QEF dual-grid occupancy at
    512 and the texture-attr rasterization occupancy at 256 don't nest exactly)."""

    def __init__(self, coords256):
        self.coords256_np = coords256.numpy()
        self.exact = {tuple(c): i for i, c in enumerate(self.coords256_np.tolist())}
        from scipy.spatial import cKDTree
        self.tree = cKDTree(self.coords256_np)

    def lookup(self, coords512):
        parent = (coords512 // 2).numpy()
        idx = np.empty(len(parent), dtype=np.int64)
        exact_hit = np.zeros(len(parent), dtype=bool)
        gaps = []
        for i, p in enumerate(map(tuple, parent.tolist())):
            j = self.exact.get(p)
            if j is None:
                gaps.append(i)
            else:
                idx[i] = j
                exact_hit[i] = True
        if gaps:
            gap_parent = parent[gaps]
            _, nn = self.tree.query(gap_parent, k=1)
            idx[gaps] = nn
        return torch.from_numpy(idx), exact_hit


def build_one(sid, glb_rel_path, save_dir, shape_encoder, tex_encoder):
    t0 = time.perf_counter()
    glb_path = os.path.join(GLB_ROOT, glb_rel_path)
    ovox_dir = os.path.join(OVOX_ROOT, sid)
    pbr_vxz = os.path.join(ovox_dir, "pbr_voxels_256", f"{sid}.vxz")
    emis_vxz = os.path.join(ovox_dir, "emission_voxels_256", f"{sid}.vxz")
    if not os.path.exists(pbr_vxz):
        raise FileNotFoundError(f"missing pbr ovoxel: {pbr_vxz}")
    if not os.path.exists(emis_vxz):
        raise FileNotFoundError(f"missing emission ovoxel: {emis_vxz}")

    coords256_pbr, pbr = o_voxel.io.read(pbr_vxz)
    coords256_emis, emis = o_voxel.io.read(emis_vxz)
    if not torch.equal(coords256_pbr, coords256_emis):
        raise ValueError(f"{sid}: pbr/emission ovoxel coords differ (expected identical)")
    t_load_ovox = time.perf_counter()

    vertices, faces = merged_mesh_512_frame(glb_path)
    t_load_mesh = time.perf_counter()

    voxel_indices, dual_vertices, intersected = dual_grid_512(vertices, faces)
    t_dualgrid = time.perf_counter()

    up = Upsampler256to512(coords256_pbr)
    idx, exact_hit = up.lookup(voxel_indices)
    gap_frac = float((~exact_hit).mean()) if len(exact_hit) else 0.0

    base_color = pbr["base_color"][idx]
    metallic = pbr["metallic"][idx]
    roughness = pbr["roughness"][idx]
    alpha = pbr["alpha"][idx]
    emissive = emis["emissive"][idx]
    t_upsample = time.perf_counter()

    # binarize at >1/255 (any nonzero channel)
    is_emis = (emissive > 0).any(dim=1)
    emissive_frac = float(is_emis.float().mean())
    out_base_color = (is_emis.unsqueeze(1).expand(-1, 3).to(torch.uint8)) * 255
    out_metallic = torch.full_like(metallic, OUT_METALLIC_U8)
    out_roughness = torch.full_like(roughness, OUT_ROUGHNESS_U8)
    out_alpha = torch.full_like(alpha, OUT_ALPHA_U8)

    os.makedirs(save_dir, exist_ok=True)
    input_vxz = os.path.join(save_dir, "input.vxz")
    output_vxz = os.path.join(save_dir, "output.vxz")
    # zstd (default level) instead of the project-wide lzma default: measured 2.6x faster
    # write and 2.85x faster read than lzma for +6.9% size (vxz_compression_probe_v2.py,
    # single fixture: lzma 4.632s write/0.057s read/561792B vs zstd 1.760s/0.020s/600467B).
    # Scoped to THIS file only -- o_voxel's own default and Path A's glb_to_vxz.py are left
    # untouched. Safe to mix with existing lzma files: the reader takes codec+level from
    # each file's own JSON header (vxz.py:read_vxz), never a global assumption.
    o_voxel.io.write(input_vxz, voxel_indices, {
        "base_color": base_color, "metallic": metallic, "roughness": roughness, "alpha": alpha,
        "dual_vertices": dual_vertices, "intersected": intersected,
    }, compression="zstd")
    o_voxel.io.write(output_vxz, voxel_indices, {
        "base_color": out_base_color, "metallic": out_metallic, "roughness": out_roughness,
        "alpha": out_alpha, "dual_vertices": dual_vertices, "intersected": intersected,
    }, compression="zstd")
    t_write = time.perf_counter()

    vxz_to_slat(shape_encoder, tex_encoder, input_vxz, output_vxz, save_dir, interactive=False)
    t_encode = time.perf_counter()

    otx = torch.load(os.path.join(save_dir, "output_tex_slat.pth"), map_location="cpu")
    mask = compute_emis_mask(output_vxz, otx["coords"])
    torch.save(mask, os.path.join(save_dir, "emis_mask.pth"))
    t_mask = time.perf_counter()

    meta = {
        "sid": sid,
        "emissive_frac": emissive_frac,
        "gap_frac": gap_frac,
        "n_voxels_512": int(voxel_indices.shape[0]),
        "n_voxels_256": int(coords256_pbr.shape[0]),
        "n_common_coords": int(otx["coords"].shape[0]),
        "timing_s": {
            "load_ovox": t_load_ovox - t0,
            "load_mesh": t_load_mesh - t_load_ovox,
            "dual_grid": t_dualgrid - t_load_mesh,
            "upsample": t_upsample - t_dualgrid,
            "write_vxz": t_write - t_upsample,
            "encode_slat": t_encode - t_write,
            "emis_mask": t_mask - t_encode,
            "total": t_mask - t0,
        },
    }
    json.dump(meta, open(os.path.join(save_dir, "meta.json"), "w"))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--out_split_name", default=None,
                     help="dir name under out_root (defaults to --split, or to 'sidfile' "
                          "when --sid_file is used without an explicit name)")
    ap.add_argument("--sid_file", default=None,
                     help="build an explicit list of sids (one per line) instead of a "
                          "split slice, e.g. for a fixed smoke-test set")
    ap.add_argument("--shuffle_seed", type=int, default=None,
                     help="shuffle the split's sid list with this fixed seed before "
                          "--start/--n slicing (for balanced SLURM-array sharding of "
                          "the full split; see load_split_sids docstring). Ignored "
                          "with --sid_file. None (default) preserves the split's "
                          "original on-disk order, matching prior smoke-test runs.")
    args = ap.parse_args()

    from trellis2 import models
    shape_encoder = models.from_pretrained(f"{TRELLIS}/shape_enc_next_dc_f16c32_fp16").cuda().eval()
    tex_encoder = models.from_pretrained(f"{TRELLIS}/tex_enc_next_dc_f16c32_fp16").cuda().eval()

    if args.sid_file:
        sids = [l.strip() for l in open(args.sid_file) if l.strip()]
        glb_rel = resolve_glb_rel(sids)
        out_split = args.out_split_name or "sidfile"
    else:
        sids_all, glb_rel = load_split_sids(args.split, shuffle_seed=args.shuffle_seed)
        sids = sids_all[args.start:args.start + args.n]
        out_split = args.out_split_name or args.split

    os.makedirs(args.out_root, exist_ok=True)
    ok, fail = 0, 0
    manifest, failures = [], []
    for i, sid in enumerate(sids):
        save_dir = os.path.join(args.out_root, out_split, sid)
        done = (os.path.exists(os.path.join(save_dir, "output_tex_slat.pth"))
                and os.path.exists(os.path.join(save_dir, "emis_mask.pth")))
        if done:
            ok += 1
            manifest.append(sid)
            print(f"[skip] {sid} already built ({i + 1}/{len(sids)})", flush=True)
            continue
        try:
            meta = build_one(sid, glb_rel[sid], save_dir, shape_encoder, tex_encoder)
            ok += 1
            manifest.append(sid)
            t = meta["timing_s"]
            print(f"[ok] {sid} frac={meta['emissive_frac']:.4f} gap={meta['gap_frac']:.4f} "
                  f"n512={meta['n_voxels_512']} common={meta['n_common_coords']} "
                  f"total={t['total']:.1f}s (mesh={t['load_mesh']:.1f} dualgrid={t['dual_grid']:.1f} "
                  f"encode={t['encode_slat']:.1f}) ({i + 1}/{len(sids)})", flush=True)
        except Exception as e:
            fail += 1
            failures.append({"sid": sid, "error": repr(e)[:500]})
            print(f"[fail] {sid}: {repr(e)[:300]}", flush=True)
            traceback.print_exc()

    # SLURM-array-safe output naming: many concurrent array tasks can share the same
    # out_split_name (they write to different <out_split>/<sid>/ subdirs, which is fine),
    # but a single shared "<out_split>_manifest.json" path would race between tasks and
    # silently lose all but the last writer. Suffix by SLURM_ARRAY_TASK_ID when present so
    # every shard gets its own manifest; falls back to the old unsuffixed name for
    # non-array runs (unchanged behavior for all smoke-test invocations so far).
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    suffix = f"_shard{array_id}" if array_id is not None else ""
    json.dump(manifest, open(os.path.join(args.out_root, f"{out_split}_manifest{suffix}.json"), "w"))
    json.dump(failures, open(os.path.join(args.out_root, f"{out_split}_failures{suffix}.json"), "w"))
    print(f"DONE split={out_split}{suffix} ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
