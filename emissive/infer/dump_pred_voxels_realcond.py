"""
fbv1_repro real-conditioning test (2026-08-07): dump_pred_voxels_repro.py always fed
zero conditioning, matching what "tonight's" runs used. But emis_1k_w1 / emis_1k_w5
trained on Path A data with cond=real (see code/three_ck_table.py's MODELS list), so
scoring them zero-cond may be starving them of conditioning they were never trained
without. This script is dump_pred_voxels_repro.py with exactly one change: cond is
loaded from a Path A dataset dir's own cond.pth instead of zeros. Everything else --
shape_slat/input_tex_slat/output_tex_slat, still read from --dataset (dataset_direct,
the canonical direct-ovoxel data) -- is unchanged, so this isolates the conditioning
variable rather than also swapping the underlying sample representation.

Usage (GPU node, trellis2 env):
  python code/dump_pred_voxels_realcond.py --dataset .../dataset_direct \
      --cond_dir .../dataset/val_96 --sids_json sids.json \
      --ckpt .../outputs/emis_1k_w1/epoch_0016_ema.ckpt \
      --out_dir .../pred_voxels/w1_ema_realcond --seed 0 --draws 3
"""
import os
import sys
import json
import time
import argparse
from collections import OrderedDict

ROOT = os.path.dirname(os.path.abspath(__file__))
SEGVIGEN = os.path.join(ROOT, "SegviGen")
sys.path.insert(0, SEGVIGEN)
sys.path.insert(0, ROOT)
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
    ap.add_argument("--dataset", required=True, help="dataset_direct: shape/tex slats")
    ap.add_argument("--cond_dir", required=True,
                    help="Path A dataset split dir (e.g. dataset/val_96): <sid>/cond.pth")
    ap.add_argument("--sids_json", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=3)
    args = ap.parse_args()
    device = "cuda"

    os.makedirs(args.out_dir, exist_ok=True)
    sid_split = json.load(open(args.sids_json))
    print(f"SIDS_OK n={len(sid_split)} draws={args.draws} cond=REAL from {args.cond_dir}", flush=True)

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
        cond_p = os.path.join(args.cond_dir, sid, "cond.pth")
        if not os.path.exists(cond_p):
            print(f"SKIP {sid}: no cond.pth at {cond_p}", flush=True)
            summary[sid] = {"sid": sid, "error": "cond.pth missing", "cond_path": cond_p}
            continue
        base_seed = args.seed * 1000003 + (int(sid[:8], 16) % 100000)

        shp = torch.load(os.path.join(d, "shape_slat.pth"), map_location=device)
        itx = torch.load(os.path.join(d, "input_tex_slat.pth"), map_location=device)
        otx = torch.load(os.path.join(d, "output_tex_slat.pth"), map_location=device)
        coords = shp["coords"].to(device)
        cond_raw = torch.load(cond_p, map_location=device)["cond"]
        if cond_raw.dim() == 2:
            cond_raw = cond_raw.unsqueeze(0)
        cond = cond_raw.to(device)
        cond_dict = {"cond": cond, "neg_cond": torch.zeros_like(cond)}
        shp_n = sp.SparseTensor((shp["feats"].to(device) - sm) / ss, coords)
        itx_n = sp.SparseTensor((itx["feats"].to(device) - tm) / ts, coords)

        with torch.no_grad():
            shape_decoder.set_resolution(512)
            _, subs = shape_decoder(sp.SparseTensor(shp["feats"].to(device), coords), return_subs=True)
            gt_vox = tex_decoder(sp.SparseTensor(otx["feats"].to(device), coords), guide_subs=subs) * 0.5 + 0.5
        gt_e = (gt_vox.feats[:, :3].mean(-1) > 0.5)

        draw_frac = {str(t): [] for t in THRS}
        draw_iou = {str(t): [] for t in THRS}
        vc_draw0 = pred_bc_draw0 = None
        for k in range(args.draws):
            torch.manual_seed(base_seed + k)
            np.random.seed(args.seed)
            with torch.no_grad():
                noise = sp.SparseTensor(torch.randn_like(itx_n.feats), coords)
                out = sampler.sample(gen, noise, itx_n, shp_n, [coords.shape[0]], cond_dict, sp_params)
                out = out * ts + tm
                pred_vox = tex_decoder(out, guide_subs=subs) * 0.5 + 0.5
            pred_bc = pred_vox.feats[:, :3].mean(-1)
            vc = pred_vox.coords[:, 1:].cpu().numpy().astype(np.int16)

            if k == 0:
                vc_draw0 = vc
                pred_bc_draw0 = pred_bc.float().cpu().numpy().astype(np.float16)
                np.savez_compressed(os.path.join(args.out_dir, f"{sid}.npz"),
                                    coords=vc_draw0, pred_bc=pred_bc_draw0,
                                    gt_e=gt_e.cpu().numpy())

            for t in THRS:
                pe = pred_bc > t
                inter = (pe & gt_e).sum().item(); union = (pe | gt_e).sum().item()
                draw_frac[str(t)].append(float(pe.float().mean().item()))
                draw_iou[str(t)].append(inter / union if union > 0 else 1.0)

        rec = {"sid": sid, "split": split, "n_voxels": int(vc_draw0.shape[0]),
               "gt_frac": float(gt_e.float().mean().item()), "draws": args.draws,
               "cond_path": cond_p,
               "pred_frac_by_thr": {t: float(np.mean(v)) for t, v in draw_frac.items()},
               "pred_frac_std_by_thr": {t: float(np.std(v)) for t, v in draw_frac.items()},
               "iou_by_thr": {t: float(np.mean(v)) for t, v in draw_iou.items()},
               "iou_std_by_thr": {t: float(np.std(v)) for t, v in draw_iou.items()}}
        summary[sid] = rec
        print(f"SHAPE_DONE {i + 1}/{len(sid_split)} {sid} split={split} draws={args.draws} "
              f"gt_frac={rec['gt_frac']:.4f} pred_frac@0.5={rec['pred_frac_by_thr']['0.5']:.4f}"
              f"(+-{rec['pred_frac_std_by_thr']['0.5']:.4f}) "
              f"iou@0.5={rec['iou_by_thr']['0.5']:.4f}(+-{rec['iou_std_by_thr']['0.5']:.4f}) "
              f"elapsed={time.time() - t0:.1f}s", flush=True)

    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=1)
    n_ok = sum(1 for r in summary.values() if "error" not in r)
    print(f"DUMP_DONE n={len(summary)} n_ok={n_ok} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
