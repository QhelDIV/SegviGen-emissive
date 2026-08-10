"""
Round-trip probe v2. Two additions over v1 (roundtrip_probe.py), per team-lead review:

1. THRESHOLD SWEEP instead of a fixed 0.5 cut on the decoded base_color. A single
   arbitrary threshold can make a fine encoder look broken (or vice versa) if the decoded
   scale is offset/rescaled. Report IoU across a sweep, the best-achievable IoU + its
   threshold, and the decoded-value distribution (mean/std/percentiles) split by the TRUE
   label (voxels that were emissive in the input vs not) -- if those two distributions are
   cleanly separated the representation is fine regardless of any single threshold choice.

2. SEMANTIC correspondence check, replacing v1's plumbing-only check (which only proved
   shape_slat/input_tex_slat/output_tex_slat share a coords tensor -- true by construction
   since the loader builds them that way on purpose). This version independently re-derives
   correspondence from Dongchen's RAW 256^3 emission data (untouched by the loader's
   internal idx/upsample arrays): pick voxels independently known emissive/dark at 256-res,
   map each to its 512-res children via coord512//2 (recomputed fresh here, not reused from
   the build), and confirm those children are labeled emissive/dark in the built output.vxz.
   Also reports raw-256 emissive fraction vs built-512 emissive fraction per sample.
"""
import os
import sys
import json

ROOT = "/3dlg-jupiter-project/lightgen/segvigen_emissive/code"
SEGVIGEN = os.path.join(ROOT, "SegviGen")
sys.path.insert(0, SEGVIGEN)
os.environ.setdefault("HF_HOME", "/3dlg-jupiter-project/lightgen/hf_cache")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import o_voxel  # noqa: E402
import trellis2.modules.sparse as sp  # noqa: E402
from trellis2 import models  # noqa: E402

DATASET = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct"
OVOX_ROOT = "/3dlg-jupiter-project/lightgen/uv_voxel_pipeline/out_uv_voxel_74k"
SAMPLES = [
    ("smoke1", "294095f9c38d48f39b6f9b7162b963d7"),   # low frac 0.0143 -- the overfit-1 target
    ("smoke20", "94124b539e714bd29d7889c3cb4c5325"),   # median frac 0.088
    ("smoke20", "9acd6bd8c0c1453d9d2bea771ee3941f"),   # near-full frac 0.957 (albedo-lit candidate)
]
THRS = np.round(np.arange(0.05, 1.0, 0.05), 2)

device = "cuda"
tex_decoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16").to(device).eval()
shape_decoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16").to(device).eval()
print("[init] decoders loaded", flush=True)


def to_map(coords_t, val_t):
    c = coords_t.cpu().numpy()
    v = val_t.cpu().numpy()
    return {tuple(c[i].tolist()): v[i] for i in range(len(c))}


for split, sid in SAMPLES:
    d = os.path.join(DATASET, split, sid)
    meta = json.load(open(os.path.join(d, "meta.json")))
    print(f"\n=== {split}/{sid} (meta emissive_frac={meta['emissive_frac']:.4f}) ===", flush=True)

    shp = torch.load(os.path.join(d, "shape_slat.pth"), map_location=device)
    otx = torch.load(os.path.join(d, "output_tex_slat.pth"), map_location=device)
    coords = shp["coords"].to(device)

    with torch.no_grad():
        shape_decoder.set_resolution(512)
        _, subs = shape_decoder(sp.SparseTensor(shp["feats"].to(device), coords), return_subs=True)
        gt_vox = tex_decoder(sp.SparseTensor(otx["feats"].to(device), coords), guide_subs=subs) * 0.5 + 0.5

    pred_val = gt_vox.feats[:, :3].mean(-1)          # continuous decoded value, per voxel
    pred_coords = gt_vox.coords[:, 1:]

    output_vxz = os.path.join(d, "output.vxz")
    coords_raw, data = o_voxel.io.read(output_vxz)
    true_e = (data["base_color"].float().mean(dim=-1) > 127.5)

    pred_map = to_map(pred_coords, pred_val)
    true_map = to_map(coords_raw, true_e)
    common_keys = list(set(pred_map.keys()) & set(true_map.keys()))
    print(f"[coords] decoded_vox N={len(pred_map)} raw_vox N={len(true_map)} common={len(common_keys)}", flush=True)

    pv = np.array([pred_map[k] for k in common_keys])
    tv = np.array([bool(true_map[k]) for k in common_keys])

    # --- 1. threshold sweep ---
    best_iou, best_t = -1.0, None
    for t in THRS:
        pe = pv > t
        inter = (pe & tv).sum()
        union = (pe | tv).sum()
        iou = inter / union if union > 0 else 1.0
        if iou > best_iou:
            best_iou, best_t = iou, t
    iou_at_05 = None
    pe05 = pv > 0.5
    inter05 = (pe05 & tv).sum(); union05 = (pe05 | tv).sum()
    iou_at_05 = inter05 / union05 if union05 > 0 else 1.0
    print(f"[SWEEP] best IoU={best_iou:.4f} @thr={best_t:.2f} | IoU@thr=0.5={iou_at_05:.4f}", flush=True)

    pos_vals = pv[tv]
    neg_vals = pv[~tv]
    def stats(x):
        if len(x) == 0:
            return "n=0"
        p = np.percentile(x, [5, 25, 50, 75, 95])
        return f"n={len(x)} mean={x.mean():.3f} std={x.std():.3f} p5/25/50/75/95={p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f}/{p[3]:.3f}/{p[4]:.3f}"
    print(f"[DIST] decoded value | true POSITIVE voxels: {stats(pos_vals)}", flush=True)
    print(f"[DIST] decoded value | true NEGATIVE voxels: {stats(neg_vals)}", flush=True)
    sep = (pos_vals.mean() - neg_vals.mean()) if len(pos_vals) and len(neg_vals) else float("nan")
    print(f"[DIST] mean separation (pos_mean - neg_mean) = {sep:.4f}", flush=True)

    # --- 2. semantic correspondence check against RAW 256 data ---
    sha = sid
    emis_vxz = os.path.join(OVOX_ROOT, sha, "emission_voxels_256", f"{sha}.vxz")
    coords256, emis256 = o_voxel.io.read(emis_vxz)
    is_emis_256 = (emis256["emissive"] > 0).any(dim=1)
    raw_frac = float(is_emis_256.float().mean())
    print(f"[FRAC] raw-256 emissive_frac={raw_frac:.4f} vs built-512 emissive_frac={meta['emissive_frac']:.4f}", flush=True)

    c256 = coords256.numpy()
    emis_mask_np = is_emis_256.numpy()
    rng = np.random.default_rng(0)
    pos_idx = np.where(emis_mask_np)[0]
    neg_idx = np.where(~emis_mask_np)[0]
    n_check = 200
    pos_sample = rng.choice(pos_idx, size=min(n_check, len(pos_idx)), replace=False) if len(pos_idx) else np.array([], dtype=int)
    neg_sample = rng.choice(neg_idx, size=min(n_check, len(neg_idx)), replace=False) if len(neg_idx) else np.array([], dtype=int)

    built_true_map = true_map  # coord(512-tuple) -> bool emissive, from output.vxz (already loaded)

    def check_group(idx_group, expect_label, name):
        checked, hits, no_children = 0, 0, 0
        for i in idx_group:
            parent = tuple(c256[i].tolist())
            children = [(parent[0]*2+dx, parent[1]*2+dy, parent[2]*2+dz)
                        for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
            present = [c for c in children if c in built_true_map]
            if not present:
                no_children += 1
                continue
            checked += 1
            agree = all(built_true_map[c] == expect_label for c in present)
            hits += int(agree)
        rate = hits / checked if checked else float("nan")
        print(f"[SEMANTIC] {name}: sampled={len(idx_group)} with_512_children={checked} "
              f"no_children_built={no_children} agree_rate={rate:.4f}", flush=True)

    check_group(pos_sample, True, "known-EMISSIVE-256 -> built-512 children also emissive")
    check_group(neg_sample, False, "known-DARK-256 -> built-512 children also dark")

print("\nROUNDTRIP_PROBE_V2_DONE", flush=True)
