"""
fbv1_repro variant of dump_pred_voxels.py: adds --draws (default 3), so the reported
IoU/frac match the diag3 convention used elsewhere in this project (K independent flow
draws, scored and averaged), while the npz written for rendering still holds exactly ONE
representative draw (draw index 0, the same one dump_pred_voxels.py always wrote), so a
render made from it reproduces bit-for-bit if this script is rerun.

Written for the finetune_binary_v1 8-shape EMA-vs-raw reproduction (2026-08-07): the
question is whether raw checkpoints (what "tonight" used, since best.ckpt -> the raw
epoch file on every model) score differently from their EMA siblings, and whether that
gap is threshold-sensitive. Everything else about the inference path is copied unchanged
from dump_pred_voxels.py -- same models, same zero-cond, same per-sid base seeding.

Usage (GPU node, trellis2 env):
  python code/dump_pred_voxels_repro.py --dataset .../dataset_direct --sids_json sids.json \
      --ckpt .../outputs/emis_1k_w1/epoch_0016_ema.ckpt --out_dir .../pred_voxels/w1_ema \
      --seed 0 --draws 3
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
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sids_json", required=True,
                    help='json: {"<sid>": "<split>", ...}')
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0, help="base seed for draw 0")
    ap.add_argument("--draws", type=int, default=3,
                    help="independent flow draws per shape, scored and averaged; "
                         "draw 0 is also dumped to npz for rendering")
    args = ap.parse_args()
    device = "cuda"

    os.makedirs(args.out_dir, exist_ok=True)
    sid_split = json.load(open(args.sids_json))
    print(f"SIDS_OK n={len(sid_split)} draws={args.draws}", flush=True)

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
        base_seed = args.seed * 1000003 + (int(sid[:8], 16) % 100000)

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
               "coord_min": [int(x) for x in vc_draw0.min(0)],
               "coord_max": [int(x) for x in vc_draw0.max(0)],
               "pred_frac_by_thr": {t: float(np.mean(v)) for t, v in draw_frac.items()},
               "pred_frac_std_by_thr": {t: float(np.std(v)) for t, v in draw_frac.items()},
               "iou_by_thr": {t: float(np.mean(v)) for t, v in draw_iou.items()},
               "iou_std_by_thr": {t: float(np.std(v)) for t, v in draw_iou.items()},
               "iou_by_thr_per_draw": draw_iou,
               "pred_frac_by_thr_per_draw": draw_frac,
               "draw0_pred_frac_at_0.5": float((pred_bc_draw0 > 0.5).mean())}
        summary[sid] = rec
        print(f"SHAPE_DONE {i + 1}/{len(sid_split)} {sid} split={split} draws={args.draws} "
              f"gt_frac={rec['gt_frac']:.4f} pred_frac@0.5={rec['pred_frac_by_thr']['0.5']:.4f}"
              f"(+-{rec['pred_frac_std_by_thr']['0.5']:.4f}) "
              f"iou@0.5={rec['iou_by_thr']['0.5']:.4f}(+-{rec['iou_std_by_thr']['0.5']:.4f}) "
              f"elapsed={time.time() - t0:.1f}s", flush=True)

    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=1)
    n_empty = sum(1 for r in summary.values() if r["pred_frac_by_thr"]["0.5"] == 0.0)
    print(f"DUMP_DONE n={len(summary)} n_empty_at_0.5={n_empty} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
