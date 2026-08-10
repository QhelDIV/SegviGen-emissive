"""
Round-trip IoU as a function of emissive_frac, across all 23 already-built samples
(smoke1's 1 + smoke20's 22). No training. For each sample: decode the saved
output_tex_slat straight back (same recipe as roundtrip_probe_v2.py), sweep the
threshold, and report (sid, emissive_frac, best_iou, best_thr, iou_at_0.5).
One line per sample so the caller can fit a curve / read off the value at any frac.
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
THRS = np.round(np.arange(0.05, 1.0, 0.05), 2)

device = "cuda"
tex_decoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16").to(device).eval()
shape_decoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16").to(device).eval()
print("[init] decoders loaded", flush=True)

samples = [("smoke1", sid) for sid in os.listdir(os.path.join(DATASET, "smoke1"))]
samples += [("smoke20", sid) for sid in os.listdir(os.path.join(DATASET, "smoke20"))]

results = []
for split, sid in samples:
    d = os.path.join(DATASET, split, sid)
    if not os.path.exists(os.path.join(d, "output_tex_slat.pth")):
        continue
    meta = json.load(open(os.path.join(d, "meta.json")))
    shp = torch.load(os.path.join(d, "shape_slat.pth"), map_location=device)
    otx = torch.load(os.path.join(d, "output_tex_slat.pth"), map_location=device)
    coords = shp["coords"].to(device)

    with torch.no_grad():
        shape_decoder.set_resolution(512)
        _, subs = shape_decoder(sp.SparseTensor(shp["feats"].to(device), coords), return_subs=True)
        gt_vox = tex_decoder(sp.SparseTensor(otx["feats"].to(device), coords), guide_subs=subs) * 0.5 + 0.5

    pred_val = gt_vox.feats[:, :3].mean(-1)
    pred_coords = gt_vox.coords[:, 1:].cpu().numpy()

    coords_raw, data = o_voxel.io.read(os.path.join(d, "output.vxz"))
    true_e = (data["base_color"].float().mean(dim=-1) > 127.5)

    pred_map = {tuple(pred_coords[i].tolist()): pred_val[i].item() for i in range(len(pred_coords))}
    true_np = coords_raw.numpy()
    true_map = {tuple(true_np[i].tolist()): bool(true_e[i].item()) for i in range(len(true_np))}
    common_keys = list(set(pred_map.keys()) & set(true_map.keys()))
    pv = np.array([pred_map[k] for k in common_keys])
    tv = np.array([true_map[k] for k in common_keys])

    best_iou, best_t = -1.0, None
    for t in THRS:
        pe = pv > t
        inter = (pe & tv).sum(); union = (pe | tv).sum()
        iou = inter / union if union > 0 else 1.0
        if iou > best_iou:
            best_iou, best_t = iou, t
    pe05 = pv > 0.5
    inter05 = (pe05 & tv).sum(); union05 = (pe05 | tv).sum()
    iou_at_05 = inter05 / union05 if union05 > 0 else 1.0

    frac = meta["emissive_frac"]
    results.append((sid, frac, best_iou, best_t, iou_at_05))
    print(f"[frac_sweep] sid={sid} frac={frac:.5f} best_iou={best_iou:.4f} best_thr={best_t:.2f} "
          f"iou@0.5={iou_at_05:.4f}", flush=True)

results.sort(key=lambda r: r[1])
print("\n--- sorted by emissive_frac ---", flush=True)
for sid, frac, bi, bt, i05 in results:
    print(f"frac={frac:8.5f}  best_iou={bi:.4f}  iou@0.5={i05:.4f}  sid={sid}", flush=True)

fracs = np.array([r[1] for r in results])
best_ious = np.array([r[2] for r in results])
i05s = np.array([r[4] for r in results])
# simple monotone-fit summary: correlation of iou with log(frac) (sparsity is the driver)
logf = np.log10(np.clip(fracs, 1e-6, None))
if len(fracs) > 2:
    r_best = np.corrcoef(logf, best_ious)[0, 1]
    r_05 = np.corrcoef(logf, i05s)[0, 1]
    print(f"\n[FIT] corr(log10(frac), best_iou) = {r_best:.4f}", flush=True)
    print(f"[FIT] corr(log10(frac), iou@0.5)   = {r_05:.4f}", flush=True)
    # nearest-neighbor readout at the dataset median frac (0.033, per smoke20 stats)
    target = 0.033
    idx = np.argsort(np.abs(fracs - target))[:3]
    print(f"[READOUT] 3 nearest samples to dataset median frac={target}:", flush=True)
    for i in idx:
        print(f"  frac={fracs[i]:.5f} best_iou={best_ious[i]:.4f} iou@0.5={i05s[i]:.4f} sid={results[i][0]}", flush=True)

json.dump([{"sid": s, "frac": f, "best_iou": bi, "best_thr": bt, "iou_at_05": i5}
           for s, f, bi, bt, i5 in results],
          open("/3dlg-jupiter-project/lightgen/segvigen_emissive/roundtrip_frac_sweep.json", "w"), indent=2)

print("\nFRAC_SWEEP_DONE", flush=True)
