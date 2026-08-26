#!/usr/bin/env python3
"""Voxel-native emission export: mask x albedo computed per voxel at 512, then
exported through o_voxel's own postprocess (fresh unwrap, texture baked from the
volume). No artist UVs are read or written anywhere in this path.

Why this exists
---------------
The incumbent visualization pipeline takes the predicted per-voxel emissive mask,
projects it back onto the ORIGINAL asset, and writes it into a new texture through
the asset's authored UV layout (emissive/eval/pred_mask_to_asset.py +
emissive/render/bpy_rebake.py). Writing through someone else's UVs inherits every
pathology of that layout: tiles outside the unit square, overlapping charts,
materials with no UVs at all. One such asset (the laptop cd9c020fb2a9..., whose
screen UVs live in glTF REPEAT tile [2-3]x[1-2]) baked to solid black.

This script never touches those UVs. It works entirely in the 512 voxel grid the
model already predicts in, and hands the finished volume to the same exporter the
generative stack uses for its own results (inference_full.slat_to_glb ->
o_voxel.postprocess.to_glb). The exporter reads the volume; it never writes into a
layout it did not create.

Stages
------
  1. decode   shape_slat.pth -> mesh + subdivision structure at 512 (shape decoder)
              input_tex_slat.pth -> per-voxel PBR (tex decoder, guided by subs)
  2. mask     load the per-voxel emissive mask (GT or a sampled prediction) and
              multiply it into the decoded base color -> per-voxel emission
  3. export   ONE call to ovoxel_to_glb_emissive.to_glb_emissive: o_voxel's own
              exporter (fresh unwrap, texture baked from the volume) with an
              emissive channel added, so albedo and emission are sampled through
              the same unwrap in the same pass. A two-pass version that reused
              upstream unchanged was written first and rejected: the remesh and
              unwrap are not deterministic, so the second pass's texture would not
              register with the first pass's geometry.
  4. align    undo the exporter's final glTF axis conversion, which is one rotation
              too many for this input, and check the result against the volume's own
              coordinates so a frame change cannot pass silently
  5. write    <sid>.glb, <sid>__mat0__emis.png, <sid>__stats.json (slot names, so
              render_emissive_closest.py --pred_emission can verify the keying),
              <sid>__timing.json

The output is rendered by the standard dark box setup, unchanged:

  python emissive/render/render_emissive_closest.py \\
      --manifest <manifest.json> --glb_dir <out_dir> --pred_emission <out_dir> \\
      --out <box_out> --mode box --res 768 --samples 256 \\
      --view_transform Filmic --exposure 1.5 \\
      --bloom 1 --bloom_size 7 --bloom_threshold 1.0 --bloom_mix -0.45 \\
      --wall 0.80 --box_scale 2.0 --emit_strength 4.0 --emission_strength_ours 1 \\
      --export_glb 0 --overwrite 0

Environment (trellis2 conda env; on this workstation the system libstdc++ is older
than the one o_voxel's extension was built against, hence the LD_PRELOAD):

  LD_PRELOAD=/cs/3dlg-jupiter-project/lightgen/miniforge3/lib/libstdc++.so.6 \\
  HF_HOME=/cs/3dlg-jupiter-project/lightgen/hf_cache \\
  /cs/3dlg-jupiter-project/lightgen/miniforge3/envs/trellis2/bin/python \\
      emissive/render/render_voxel_native.py \\
      --dataset /cs/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct \\
      --npz_dir <converted_npz> --panels cd9c020fb2a94143b4b7eb59af49f406_gt \\
      --out_dir <out>

A panel id is "<sid>_gt" or "<sid>_draw<k>", matching the npz files that
eval_emissive.py --dump_vis writes.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(ROOT, "inference_full.py")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError("could not locate the SegviGen repo root above this file")
    ROOT = parent
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/cs/3dlg-jupiter-project/lightgen/hf_cache")

import trellis2.modules.sparse as sp            # noqa: E402
from trellis2 import models                     # noqa: E402
from trellis2.representations import MeshWithVoxel  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ovoxel_to_glb_emissive import to_glb_emissive  # noqa: E402

RESOLUTION = 512

# Channels 0:6 are the layout o_voxel already understands, in its own order, so the
# albedo half of this export is byte-for-byte what the generative stack produces for
# its own results. Channels 6:9 are the addition.
PBR_LAYOUT = {
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}
PBR_EMIS_LAYOUT = dict(PBR_LAYOUT, emissive=slice(6, 9))

# One material, named here so the render script's slot-name check has something
# stable to verify against instead of whatever trimesh would default to.
MATERIAL_NAME = "voxelnative"


class Timer:
    """Wall clock per named stage, so the study's runtime table is measured."""

    def __init__(self):
        self.t = {}

    def __call__(self, name):
        return _Stage(self, name)


class _Stage:
    def __init__(self, timer, name):
        self.timer, self.name = timer, name

    def __enter__(self):
        torch.cuda.synchronize()
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        torch.cuda.synchronize()
        self.timer.t[self.name] = self.timer.t.get(self.name, 0.0) + time.time() - self.t0
        return False


def load_decoders(device="cuda"):
    """The same two decoders eval_emissive.load_eval_models uses, and for the same
    reason: the mask the model predicts lives on the coords these produce, so the
    albedo it is multiplied against has to come from the same decode."""
    tex_decoder = models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16").to(device).eval()
    shape_decoder = models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16").to(device).eval()
    return shape_decoder, tex_decoder


def find_sample_dir(dataset_root, sid, splits=("val_72k", "test_72k", "train_72k")):
    for s in splits:
        d = os.path.join(dataset_root, s, sid)
        if os.path.isdir(d):
            return d, s
    raise FileNotFoundError(f"{sid} not found under {dataset_root} in {splits}")


def decode_shape_and_pbr(shape_decoder, tex_decoder, sample_dir, timer, device="cuda"):
    """shape_slat -> mesh (+subs); input_tex_slat -> per-voxel PBR on those coords."""
    shp = torch.load(os.path.join(sample_dir, "shape_slat.pth"), map_location=device)
    itx = torch.load(os.path.join(sample_dir, "input_tex_slat.pth"), map_location=device)
    coords = shp["coords"].to(device)
    with timer("decode_shape"), torch.no_grad():
        shape_decoder.set_resolution(RESOLUTION)
        meshes, subs = shape_decoder(
            sp.SparseTensor(shp["feats"].to(device), coords), return_subs=True)
    with timer("decode_tex"), torch.no_grad():
        tex_voxels = tex_decoder(
            sp.SparseTensor(itx["feats"].to(device), coords), guide_subs=subs) * 0.5 + 0.5
    return meshes, tex_voxels


def resolve_npz(panel_id, args):
    """Where this panel's mask lives, in either of the two layouts we produce.

    --npz_dir is the converted layout eval_emissive.py --dump_vis writes: one file
    per panel, <panel_id>.npz, carrying `gt_e` (bool) and `pred_bc` (per-voxel mean
    base color).

    --gen_root is the generation job's own tree, one directory per draw:
        <gen_root>/draw<k>/<sid>.npz  with `pred` and `gt_recon`, both (N, 3) RGB.
    A "_gt" panel reads gt_recon out of draw0, because the ground truth is a decode
    of the shape's own stored target and does not depend on which draw it sits next
    to. Verified rather than assumed: on three shapes the two layouts agree on the
    mask fraction to four decimals and their coords arrays are equal.
    """
    if args.gen_root:
        sid, _, kind = panel_id.rpartition("_")
        if kind == "gt":
            return os.path.join(args.gen_root, "draw0", f"{sid}.npz"), "gt_recon"
        assert kind.startswith("draw"), f"{panel_id}: not a draw or gt panel"
        return os.path.join(args.gen_root, kind, f"{sid}.npz"), "pred"
    return os.path.join(args.npz_dir, f"{panel_id}.npz"), None


def load_mask(npz_path, tex_voxels, thr=0.5, max_miss_frac=2e-2, raw_key=None):
    """Read the per-voxel emissive mask and transfer it onto THIS decode's coords.

    The npz was written by a different process (the generation job's dump), and the
    shape decoder is not bit-reproducible across runs: on the laptop asset the two
    decodes differ by 7 voxels out of 773929. So the mask is matched by voxel
    COORDINATE, not by row order. Positional indexing would have been off by those
    seven rows for every voxel after the first divergence, which paints emission
    onto the wrong surface and still renders as a perfectly plausible picture.

    Voxels the mask does not cover are treated as non-emissive, and the miss rate is
    both returned and capped: a large miss rate means the two decodes are not the
    same shape at all, not a rounding difference.

    The cap is 2%, not the 0.1% it started at. That first value was calibrated on one
    shape whose two decodes differed by 7 voxels in 773929 and it rejected two panels
    of the epoch-8 gallery that differ by 0.15% (601 of 389787), which is plainly the
    same shape decoded twice. The guard is here to catch a mismatch of tens of
    percent, the kind that means the mask belongs to a different asset; the measured
    rate is recorded per panel either way, so a creeping increase stays visible
    instead of being hidden by the looser bound.
    """
    d = np.load(npz_path)
    npz_coords = d["coords"].astype(np.int64)
    dec_coords = tex_voxels.coords[:, 1:].cpu().numpy().astype(np.int64)
    if raw_key is not None:
        # the generation tree's layout: (N, 3) RGB, thresholded on the channel mean,
        # exactly as eval_emissive.py derives gt_e and pred_bc from a decode
        vals = d[raw_key].astype(np.float32).mean(-1) > thr
        src = f"{raw_key}.mean>{thr}"
    elif "pred_bc" in d.files and not npz_path.endswith("_gt.npz"):
        vals = d["pred_bc"].astype(np.float32) > thr
        src = f"pred_bc>{thr}"
    else:
        vals = d["gt_e"].astype(bool)
        src = "gt_e"

    R = RESOLUTION
    key_npz = (npz_coords[:, 0] * R + npz_coords[:, 1]) * R + npz_coords[:, 2]
    key_dec = (dec_coords[:, 0] * R + dec_coords[:, 1]) * R + dec_coords[:, 2]
    order = np.argsort(key_npz)
    key_sorted = key_npz[order]
    pos = np.searchsorted(key_sorted, key_dec)
    pos_clipped = np.clip(pos, 0, len(key_sorted) - 1)
    hit = key_sorted[pos_clipped] == key_dec
    mask = np.zeros(len(key_dec), dtype=bool)
    mask[hit] = vals[order[pos_clipped[hit]]]
    miss_frac = float((~hit).mean())
    assert miss_frac <= max_miss_frac, (
        f"{os.path.basename(npz_path)}: {miss_frac:.4%} of this decode's voxels have "
        f"no counterpart in the mask ({len(key_dec)} decoded vs {len(key_npz)} in the "
        f"mask); these are not the same shape")
    return torch.from_numpy(mask).cuda(), src, miss_frac, len(key_npz)


def prepare_mesh(m, tex_voxels, attrs):
    """The mesh preparation inference_full.slat_to_glb does before exporting.

    Returned as plain vertex/face tensors rather than the MeshWithVoxel, because
    fill_holes() and simplify() both mutate the decoder's mesh object in place and
    the caller should not have to know that.
    """
    m.fill_holes()
    mv = MeshWithVoxel(
        m.vertices, m.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1 / RESOLUTION,
        coords=tex_voxels.coords[:, 1:],
        attrs=attrs,
        voxel_shape=torch.Size([*tex_voxels.shape, *tex_voxels.spatial_shape]),
        layout=PBR_LAYOUT,
    )
    mv.simplify(10000000)
    return mv.vertices.clone(), mv.faces.clone(), mv.coords, mv.voxel_size


def export_glb(vertices, faces, coords, voxel_size, attr_volume,
               decimation_target, texture_size):
    """One export call, parameterized exactly as inference_full.slat_to_glb does."""
    return to_glb_emissive(
        vertices=vertices,
        faces=faces,
        attr_volume=attr_volume,
        coords=coords,
        attr_layout=PBR_EMIS_LAYOUT,
        voxel_size=voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )


def align_to_source_frame(glb):
    """Undo the exporter's final glTF axis conversion, which is one rotation too many
    for this input.

    o_voxel's exporter ends by mapping (x, y, z) -> (x, z, -y), the conversion its own
    inference path needs because there the attribute volume is built from a mesh in a
    different up-axis convention. The volume here comes from the dataset's stored
    latents, which are already in the source asset's own frame, so the conversion tips
    the object onto its side and the render looks at a different face of it than the
    incumbent pipeline's render of the same shape does.

    Measured, not assumed: voxelising the exported mesh and the source GLB at 32 and
    comparing occupancy, the inverse rotation (x, -z, y) scores 0.93 to 1.00 IoU across
    five shapes while the identity and the two other candidates score 0.04 to 0.17.
    verify_frame() below re-checks the result on every panel against the mask's own
    voxel coordinates, so a silent frame change would fail the run rather than produce
    a comparison of two different viewpoints.
    """
    m = np.eye(4)
    m[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]   # (x, y, z) -> (x, -z, y)
    glb.apply_transform(m)                            # rotates the normals too
    return glb


def verify_frame(glb, coords, panel_id, res=32, min_iou=0.7, n_samples=200000):
    """Check the exported mesh occupies the same space as the volume it was baked from.

    Both are put on a coarse grid over the same unit box, and the test is what
    fraction of the VOLUME's cells the mesh covers, not a symmetric overlap. A mesh in
    the wrong frame still renders, and still renders something that looks like the
    shape, so this is checked rather than eyeballed.

    Coverage rather than IoU because the exporter's remesh legitimately ADDS surface:
    it fills holes and closes shells, so the mesh is a superset of the sparse
    surface-voxel set. A crate in the epoch-8 gallery was falsely rejected at 0.616
    symmetric IoU while covering every one of the volume's 2312 occupied cells and
    adding 1440 of its own; voxel-only cells numbered zero. Coverage scores that 1.000
    and still collapses under a genuine rotation, which is the failure this guards.

    The mesh is also sampled over its SURFACE by area rather than at its vertices, so
    a shape with few vertices and large flat faces is not undersampled.
    """
    v = np.asarray(glb.vertices, dtype=np.float64)
    f = np.asarray(glb.faces)
    tris = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0],
                                         tris[:, 2] - tris[:, 0]), axis=1)
    if area.sum() > 0:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(tris), size=n_samples, p=area / area.sum())
        t = tris[pick]
        u = rng.random((n_samples, 1)); w = rng.random((n_samples, 1))
        over = (u + w) > 1
        u[over] = 1 - u[over]; w[over] = 1 - w[over]
        pts = t[:, 0] + u * (t[:, 1] - t[:, 0]) + w * (t[:, 2] - t[:, 0])
    else:
        pts = v

    c = (pts.min(0) + pts.max(0)) / 2
    sc = (pts.max(0) - pts.min(0)).max()
    vi = np.clip((((pts - c) / sc) + 0.5) * res, 0, res - 1).astype(int)
    g_mesh = np.zeros((res, res, res), bool)
    g_mesh[vi[:, 0], vi[:, 1], vi[:, 2]] = True

    p = coords.astype(np.float64)
    c2 = (p.min(0) + p.max(0)) / 2
    sc2 = (p.max(0) - p.min(0)).max()
    pi = np.clip((((p - c2) / sc2) + 0.5) * res, 0, res - 1).astype(int)
    g_vox = np.zeros((res, res, res), bool)
    g_vox[pi[:, 0], pi[:, 1], pi[:, 2]] = True

    covered = float((g_mesh & g_vox).sum()) / max(int(g_vox.sum()), 1)
    assert covered >= min_iou, (
        f"{panel_id}: the exported mesh does not cover the voxel volume it was baked "
        f"from ({covered:.3f} of the volume's occupied cells at {res}^3, expected >= "
        f"{min_iou}); the emission would be painted on a surface in a different "
        f"orientation")
    return covered


def run_panel(panel_id, args, shape_decoder, tex_decoder):
    sid = panel_id.split("_")[0]
    timer = Timer()
    sample_dir, split = find_sample_dir(args.dataset, sid)
    npz_path, raw_key = resolve_npz(panel_id, args)
    assert os.path.isfile(npz_path), f"mask npz missing: {npz_path}"

    meshes, tex_voxels = decode_shape_and_pbr(
        shape_decoder, tex_decoder, sample_dir, timer)
    assert len(meshes) == 1, f"{panel_id}: expected one mesh, got {len(meshes)}"

    mask, mask_src, miss_frac, n_mask_vox = load_mask(
        npz_path, tex_voxels, thr=args.thr, raw_key=raw_key)
    pbr = tex_voxels.feats.float()
    base_color = pbr[:, PBR_LAYOUT["base_color"]]
    emission = base_color * mask[:, None].float()
    # One volume, nine channels: the six o_voxel already bakes plus the three the
    # extended exporter turns into the emissive texture.
    attrs = torch.cat([pbr[:, 0:6], emission], dim=-1)

    stats = {
        "panel": panel_id, "sid": sid, "split": split,
        "n_voxels": int(pbr.shape[0]),
        "mask_source": mask_src,
        "n_mask_voxels": int(n_mask_vox),
        "mask_coord_miss_frac": round(miss_frac, 8),
        "mask_frac": float(mask.float().mean().item()),
        "n_mesh_vertices_decoded": int(meshes[0].vertices.shape[0]),
        "n_mesh_faces_decoded": int(meshes[0].faces.shape[0]),
        "texture_size": args.texture_size,
        "decimation_target": args.decimation_target,
    }

    with timer("prepare_mesh"):
        verts, faces, vox_coords, voxel_size = prepare_mesh(meshes[0], tex_voxels, attrs)
    with timer("export"):
        glb, emis_img = export_glb(verts, faces, vox_coords, voxel_size, attrs,
                                   args.decimation_target, args.texture_size)
    assert emis_img is not None, f"{panel_id}: exporter returned no emissive texture"
    align_to_source_frame(glb)
    stats["frame_iou_vs_volume"] = round(
        verify_frame(glb, tex_voxels.coords[:, 1:].cpu().numpy(), panel_id,
                     min_iou=args.min_frame_iou), 4)

    stats["n_mesh_vertices_exported"] = int(len(glb.vertices))
    stats["n_mesh_faces_exported"] = int(len(glb.faces))
    uv = np.asarray(glb.visual.uv, dtype=np.float64)
    # The point of this pipeline is that the UV layout is the exporter's own, so
    # check the property the incumbent could not guarantee: every UV inside the unit
    # square, which is what made the laptop asset bake to black on the other path.
    stats["uv_min"] = [round(float(x), 6) for x in uv.min(0)]
    stats["uv_max"] = [round(float(x), 6) for x in uv.max(0)]
    stats["uv_outside_unit_frac"] = float(
        ((uv < -1e-6) | (uv > 1 + 1e-6)).any(axis=1).mean())

    os.makedirs(args.out_dir, exist_ok=True)
    glb.visual.material.name = MATERIAL_NAME
    glb_path = os.path.join(args.out_dir, f"{panel_id}.glb")
    with timer("write"):
        glb.export(glb_path)
        emis_img = emis_img.convert("RGB")
        emis_img.save(os.path.join(args.out_dir, f"{panel_id}__mat0__emis.png"))

    arr = np.asarray(emis_img, dtype=np.float32) / 255.0
    stats["materials"] = [MATERIAL_NAME]
    stats["emis_texture_nonzero_frac"] = float((arr.max(-1) > 1.0 / 255).mean())
    stats["emis_texture_mean_rgb"] = [round(float(x), 5) for x in arr.reshape(-1, 3).mean(0)]
    stats["emis_texture_max_rgb"] = [round(float(x), 5) for x in arr.reshape(-1, 3).max(0)]
    stats["timing_s"] = {k: round(v, 2) for k, v in timer.t.items()}
    stats["timing_total_s"] = round(sum(timer.t.values()), 2)
    json.dump(stats, open(os.path.join(args.out_dir, f"{panel_id}__stats.json"), "w"), indent=1)
    print(f"[ok] {panel_id} mask={mask_src} frac={stats['mask_frac']:.4f} "
          f"tex_nonzero={stats['emis_texture_nonzero_frac']:.4f} "
          f"total={stats['timing_total_s']}s {stats['timing_s']}", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="dataset_direct root (holds val_72k/test_72k/train_72k)")
    ap.add_argument("--npz_dir", default=None,
                    help="dir of per-voxel mask npz files (<panel>.npz), the "
                         "converted layout")
    ap.add_argument("--gen_root", default=None,
                    help="the generation job's own tree instead: <gen_root>/draw<k>/"
                         "<sid>.npz with `pred` and `gt_recon`")
    ap.add_argument("--panels", default=None,
                    help="comma-separated panel ids (<sid>_gt / <sid>_draw2)")
    ap.add_argument("--panels_file", default=None, help="one panel id per line")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--thr", type=float, default=0.5,
                    help="threshold on pred_bc for prediction panels")
    ap.add_argument("--texture_size", type=int, default=2048)
    ap.add_argument("--decimation_target", type=int, default=100000)
    ap.add_argument("--overwrite", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0,
                    help="take panels[shard::nshards] only, for a SLURM job array")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--min_frame_iou", type=float, default=0.9,
                    help="reject a panel whose exported mesh does not line up with "
                         "the voxel volume it was baked from; lower only to "
                         "INVESTIGATE a rejection, never to make one go away")
    args = ap.parse_args()

    panels = []
    if args.panels:
        panels += [p.strip() for p in args.panels.split(",") if p.strip()]
    if args.panels_file:
        panels += [l.strip() for l in open(args.panels_file) if l.strip()]
    assert panels, "no panels given"
    assert bool(args.npz_dir) != bool(args.gen_root), \
        "give exactly one of --npz_dir or --gen_root"
    if args.nshards > 1:
        panels = panels[args.shard::args.nshards]
        print(f"shard {args.shard}/{args.nshards}: {len(panels)} panels", flush=True)

    shape_decoder, tex_decoder = load_decoders()
    os.makedirs(args.out_dir, exist_ok=True)
    n_ok = n_fail = n_skip = 0
    for p in panels:
        sp_path = os.path.join(args.out_dir, f"{p}__stats.json")
        if os.path.exists(sp_path) and not args.overwrite:
            n_skip += 1
            print(f"[skip] {p}", flush=True)
            continue
        try:
            run_panel(p, args, shape_decoder, tex_decoder)
            n_ok += 1
        except Exception:
            n_fail += 1
            print(f"[FAIL] {p}", flush=True)
            traceback.print_exc()
        torch.cuda.empty_cache()
    print(f"DONE ok={n_ok} fail={n_fail} skip={n_skip}", flush=True)


if __name__ == "__main__":
    main()
