"""
Sample a checkpoint on an explicit list of sids and save the raw predicted emissive field
at the 512-res voxel coordinates, one npz per (model, sid).

This is the common input to every way of putting a real prediction into the comparison
figure: it is the model's output before any meshing, baking or thresholding choice, so a
render decision made later does not require re-sampling on a GPU. The npz carries the same
three arrays `eval_emissive.eval_sample(--dump_vis)` writes -- coords, pred_bc, gt_e -- plus
a per-shape summary json with the predicted and GT emissive fractions at each threshold, so
"this prediction is empty" is a recorded number rather than something noticed at render time.

Fixed seed per (sid, draw): the figure shows ONE representative draw, and it has to be the
same draw every time the renders are regenerated. The scored IoU is the K=3 average from
diag3 and is a different number by construction; both get reported.

Conditioning is "zero" -- the direct-ovoxel dataset carries no cond.pth.

Usage (GPU node, trellis2 env):
  python emissive/infer/dump_pred_voxels.py --dataset .../dataset_direct --sids_json sids.json \
      --ckpt .../outputs/emis_72k_unfilt/run1/best.ckpt --out_dir .../pred_voxels/emis_72k --seed 0
"""
import os
import sys
import json
import time
import argparse
from collections import OrderedDict

ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(ROOT, "inference_full.py")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError(f"could not locate SegviGen repo root (inference_full.py) above {__file__}")
    ROOT = parent   # walk up: this script now lives nested under emissive/infer/, not repo root
SEGVIGEN = ROOT
sys.path.insert(0, SEGVIGEN)
sys.path.insert(0, os.path.join(ROOT, "emissive", "eval"))  # sibling dir holding eval_emissive.py
os.environ.setdefault("HF_HOME", "/3dlg-jupiter-project/lightgen/hf_cache")

import numpy as np
import torch
import trellis2.modules.sparse as sp
from trellis2 import models
from inference_full import Gen3DSeg

import eval_emissive as ee

THRS = [0.2, 0.3, 0.4, 0.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sids_json", required=True,
                    help='json: {"<sid>": "<split>", ...} -- the split each sid lives in, '
                         'since the gallery shapes are spread across val_72k and test_72k')
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda"

    os.makedirs(args.out_dir, exist_ok=True)
    sid_split = json.load(open(args.sids_json))
    print(f"SIDS_OK n={len(sid_split)}", flush=True)

    models_d = ee.load_eval_models(device)
    print("ENV_OK models loaded", flush=True)

    flow = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")
    gen = Gen3DSeg(flow).to(device)
    sd = torch.load(args.ckpt, map_location=device)["state_dict"]
    sd = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in sd.items()])
    missing, unexpected = gen.load_state_dict(sd, strict=False)
    print(f"CKPT_LOADED missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing or unexpected:
        raise SystemExit("checkpoint does not match the current Gen3DSeg architecture")
    gen.eval()

    tex_decoder, shape_decoder, sampler = models_d["tex_decoder"], models_d["shape_decoder"], models_d["sampler"]
    sm, ss, tm, ts = models_d["sm"], models_d["ss"], models_d["tm"], models_d["ts"]
    sp_params = dict(models_d["pipeline_args"]["tex_slat_sampler"]["params"])
    sp_params["steps"] = args.steps

    summary = {}
    t0 = time.time()
    for i, (sid, split) in enumerate(sorted(sid_split.items())):
        d = os.path.join(args.dataset, split, sid)
        # per-sid seed, so adding a sid later does not change any existing sid's draw
        torch.manual_seed(args.seed * 1000003 + (int(sid[:8], 16) % 100000))
        np.random.seed(args.seed)

        shp = torch.load(os.path.join(d, "shape_slat.pth"), map_location=device)
        itx = torch.load(os.path.join(d, "input_tex_slat.pth"), map_location=device)
        otx = torch.load(os.path.join(d, "output_tex_slat.pth"), map_location=device)
        coords = shp["coords"].to(device)
        cond = torch.zeros(1, ee.COND_T, ee.COND_D, device=device)
        cond_dict = {"cond": cond, "neg_cond": torch.zeros_like(cond)}
        shp_n = sp.SparseTensor((shp["feats"].to(device) - sm) / ss, coords)
        itx_n = sp.SparseTensor((itx["feats"].to(device) - tm) / ts, coords)

        with torch.no_grad():
            shape_decoder.set_resolution(512)
            _, subs = shape_decoder(sp.SparseTensor(shp["feats"].to(device), coords), return_subs=True)
            gt_vox = tex_decoder(sp.SparseTensor(otx["feats"].to(device), coords), guide_subs=subs) * 0.5 + 0.5
            noise = sp.SparseTensor(torch.randn_like(itx_n.feats), coords)
            out = sampler.sample(gen, noise, itx_n, shp_n, [coords.shape[0]], cond_dict, sp_params)
            out = out * ts + tm
            pred_vox = tex_decoder(out, guide_subs=subs) * 0.5 + 0.5

        gt_e = (gt_vox.feats[:, :3].mean(-1) > 0.5)
        pred_bc = pred_vox.feats[:, :3].mean(-1)
        vc = pred_vox.coords[:, 1:].cpu().numpy().astype(np.int16)

        np.savez_compressed(os.path.join(args.out_dir, f"{sid}.npz"),
                            coords=vc,
                            pred_bc=pred_bc.float().cpu().numpy().astype(np.float16),
                            gt_e=gt_e.cpu().numpy())

        rec = {"sid": sid, "split": split, "n_voxels": int(vc.shape[0]),
               "gt_frac": float(gt_e.float().mean().item()),
               "coord_min": [int(x) for x in vc.min(0)], "coord_max": [int(x) for x in vc.max(0)],
               "pred_frac_by_thr": {}, "iou_by_thr": {}}
        for t in THRS:
            pe = pred_bc > t
            inter = (pe & gt_e).sum().item(); union = (pe | gt_e).sum().item()
            rec["pred_frac_by_thr"][str(t)] = float(pe.float().mean().item())
            rec["iou_by_thr"][str(t)] = inter / union if union > 0 else 1.0
        summary[sid] = rec
        print(f"SHAPE_DONE {i + 1}/{len(sid_split)} {sid} split={split} "
              f"gt_frac={rec['gt_frac']:.4f} pred_frac@0.5={rec['pred_frac_by_thr']['0.5']:.4f} "
              f"iou@0.5={rec['iou_by_thr']['0.5']:.4f} elapsed={time.time() - t0:.1f}s", flush=True)

    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=1)
    n_empty = sum(1 for r in summary.values() if r["pred_frac_by_thr"]["0.5"] == 0.0)
    print(f"DUMP_DONE n={len(summary)} n_empty_at_0.5={n_empty} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
