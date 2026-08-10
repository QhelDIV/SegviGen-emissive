"""
Decisive round-trip probe (no training): for a handful of already-built samples, take
the GT binary emissive target as encoded on disk (output_tex_slat.pth), decode it straight
back through the pretrained tex/shape decoders (guide_subs from shape_decoder), and compare
the decoded binary mask against the TRUE binary target read directly from output.vxz
(pre-encode). If the f16c32 encoder/decoder cannot represent a binary 0/1 indicator well,
this round-trip IoU will be low regardless of any training -- this isolates a representation
question from an optimization question.

Also runs the cheap positional-correspondence check: confirm shape_slat, input_tex_slat,
output_tex_slat share the IDENTICAL coords tensor (so per-row alignment across the three
saved tensors holds by construction, not by luck).

Read-only against dataset_direct/{smoke1,smoke20}. Writes nothing outside scratch.
"""
import os
import sys
import json

ROOT = "/3dlg-jupiter-project/lightgen/segvigen_emissive/code"
SEGVIGEN = os.path.join(ROOT, "SegviGen")
sys.path.insert(0, SEGVIGEN)
os.environ.setdefault("HF_HOME", "/3dlg-jupiter-project/lightgen/hf_cache")

import torch  # noqa: E402
import o_voxel  # noqa: E402
import trellis2.modules.sparse as sp  # noqa: E402
from trellis2 import models  # noqa: E402

DATASET = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct"
SAMPLES = [
    ("smoke1", "294095f9c38d48f39b6f9b7162b963d7"),   # low frac 0.0143 -- the overfit-1 target
    ("smoke20", "94124b539e714bd29d7889c3cb4c5325"),   # median frac 0.088
    ("smoke20", "9acd6bd8c0c1453d9d2bea771ee3941f"),   # near-full frac 0.957 (albedo-lit candidate)
]

device = "cuda"
tex_decoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16").to(device).eval()
shape_decoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16").to(device).eval()
print("[init] decoders loaded", flush=True)


def to_map(coords_t, mask_t):
    c = coords_t.cpu().numpy()
    m = mask_t.cpu().numpy()
    return {tuple(c[i].tolist()): bool(m[i]) for i in range(len(c))}


for split, sid in SAMPLES:
    d = os.path.join(DATASET, split, sid)
    meta = json.load(open(os.path.join(d, "meta.json")))
    print(f"\n=== {split}/{sid} (meta emissive_frac={meta['emissive_frac']:.4f}) ===", flush=True)

    shp = torch.load(os.path.join(d, "shape_slat.pth"), map_location=device)
    itx = torch.load(os.path.join(d, "input_tex_slat.pth"), map_location=device)
    otx = torch.load(os.path.join(d, "output_tex_slat.pth"), map_location=device)

    # --- check 2: positional correspondence between the three saved tensors ---
    c_ok_1 = torch.equal(shp["coords"], otx["coords"])
    c_ok_2 = torch.equal(shp["coords"], itx["coords"])
    print(f"[check2] shape_slat/output_tex_slat coords identical: {c_ok_1}", flush=True)
    print(f"[check2] shape_slat/input_tex_slat  coords identical: {c_ok_2}", flush=True)
    print(f"[check2] N_common (latent, 32-res) = {shp['coords'].shape[0]}", flush=True)

    coords = shp["coords"].to(device)
    with torch.no_grad():
        shape_decoder.set_resolution(512)
        _, subs = shape_decoder(sp.SparseTensor(shp["feats"].to(device), coords), return_subs=True)
        gt_vox = tex_decoder(sp.SparseTensor(otx["feats"].to(device), coords), guide_subs=subs) * 0.5 + 0.5

    pred_e = (gt_vox.feats[:, :3].mean(-1) > 0.5)
    pred_coords = gt_vox.coords[:, 1:]

    output_vxz = os.path.join(d, "output.vxz")
    coords_raw, data = o_voxel.io.read(output_vxz)
    true_e = (data["base_color"].float().mean(dim=-1) > 127.5)

    pred_map = to_map(pred_coords, pred_e)
    true_map = to_map(coords_raw, true_e)
    pred_keys = set(pred_map.keys())
    true_keys = set(true_map.keys())
    common_keys = pred_keys & true_keys
    print(f"[coords] decoded_vox N={len(pred_keys)} raw_vox N={len(true_keys)} common={len(common_keys)} "
          f"(decoded_only={len(pred_keys - true_keys)} raw_only={len(true_keys - pred_keys)})", flush=True)

    inter = sum(1 for k in common_keys if pred_map[k] and true_map[k])
    union = sum(1 for k in common_keys if pred_map[k] or true_map[k])
    iou = inter / union if union > 0 else float("nan")
    true_frac = sum(true_map[k] for k in common_keys) / len(common_keys) if common_keys else float("nan")
    pred_frac = sum(pred_map[k] for k in common_keys) / len(common_keys) if common_keys else float("nan")
    print(f"[ROUNDTRIP] per-voxel IoU (decode(encode(true_target)) vs true_target) = {iou:.4f} "
          f"over {len(common_keys)} common voxels", flush=True)
    print(f"[ROUNDTRIP] true_frac_emissive={true_frac:.4f} decoded_frac_emissive={pred_frac:.4f}", flush=True)

print("\nROUNDTRIP_PROBE_DONE", flush=True)
