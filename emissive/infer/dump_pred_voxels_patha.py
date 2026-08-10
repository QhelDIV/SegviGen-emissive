"""
fbv1_repro full Path A reproduction (2026-08-07): dump_pred_voxels_realcond.py mixed
dataset_direct shape/tex slats with Path A conditioning, and that failed outright --
dataset_direct got re-partitioned to a v2 split tonight by another workstream, so the
old bucket paths this script's sids_json pointed at (val_72k) no longer hold these sids.

The fix is also the more faithful test: read EVERYTHING -- shape_slat.pth,
input_tex_slat.pth, output_tex_slat.pth, AND cond.pth -- from one Path A sample dir,
e.g. dataset/val_96/<sid>/. That is byte-for-byte what the old finetune_binary_v1 page's
eval consumed: Path A inputs, Path A conditioning, the same checkpoint. No split
subdirectory inside --dataset; each sid is a direct child.

Usage (GPU node, trellis2 env):
  python emissive/infer/dump_pred_voxels_patha.py \
      --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset/val_96 \
      --sids sid1,sid2,... \
      --ckpt .../outputs/emis_1k_w1/epoch_0016_ema.ckpt \
      --out_dir .../pred_voxels/w1_ema_patha --seed 0 --draws 3
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
    ap.add_argument("--dataset", required=True,
                    help="Path A split dir, e.g. dataset/val_96 -- sids are direct children")
    ap.add_argument("--sids", required=True, help="comma-separated")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--cond", default="real", choices=["real", "zero"],
                    help="real (default): use the sid's own cond.pth. zero: for an "
                         "inputs-only-isolation follow-up, still reading Path A "
                         "shape/tex slats but zeroing the conditioning.")
    args = ap.parse_args()
    device = "cuda"

    os.makedirs(args.out_dir, exist_ok=True)
    sids = args.sids.split(",")
    print(f"SIDS_OK n={len(sids)} draws={args.draws} cond={args.cond} dataset={args.dataset}", flush=True)

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
    for i, sid in enumerate(sorted(sids)):
        d = os.path.join(args.dataset, sid)
        if not os.path.exists(os.path.join(d, "output_tex_slat.pth")):
            print(f"SKIP {sid}: no output_tex_slat.pth under {d}", flush=True)
            summary[sid] = {"sid": sid, "error": f"missing sample dir {d}"}
            continue
        cond_p = os.path.join(d, "cond.pth")
        if args.cond == "real" and not os.path.exists(cond_p):
            print(f"SKIP {sid}: no cond.pth at {cond_p}", flush=True)
            summary[sid] = {"sid": sid, "error": "cond.pth missing", "cond_path": cond_p}
            continue
        base_seed = args.seed * 1000003 + (int(sid[:8], 16) % 100000)

        shp = torch.load(os.path.join(d, "shape_slat.pth"), map_location=device)
        itx = torch.load(os.path.join(d, "input_tex_slat.pth"), map_location=device)
        otx = torch.load(os.path.join(d, "output_tex_slat.pth"), map_location=device)
        coords = shp["coords"].to(device)
        if args.cond == "real":
            cond_raw = torch.load(cond_p, map_location=device)["cond"]
            cond = cond_raw.to(device) if cond_raw.dim() == 3 else cond_raw.unsqueeze(0).to(device)
        else:
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

        rec = {"sid": sid, "n_voxels": int(vc_draw0.shape[0]),
               "gt_frac": float(gt_e.float().mean().item()), "draws": args.draws,
               "cond": args.cond, "sample_dir": d,
               "pred_frac_by_thr": {t: float(np.mean(v)) for t, v in draw_frac.items()},
               "pred_frac_std_by_thr": {t: float(np.std(v)) for t, v in draw_frac.items()},
               "iou_by_thr": {t: float(np.mean(v)) for t, v in draw_iou.items()},
               "iou_std_by_thr": {t: float(np.std(v)) for t, v in draw_iou.items()}}
        summary[sid] = rec
        print(f"SHAPE_DONE {i + 1}/{len(sids)} {sid} draws={args.draws} cond={args.cond} "
              f"gt_frac={rec['gt_frac']:.4f} pred_frac@0.5={rec['pred_frac_by_thr']['0.5']:.4f}"
              f"(+-{rec['pred_frac_std_by_thr']['0.5']:.4f}) "
              f"iou@0.5={rec['iou_by_thr']['0.5']:.4f}(+-{rec['iou_std_by_thr']['0.5']:.4f}) "
              f"elapsed={time.time() - t0:.1f}s", flush=True)

    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=1)
    n_ok = sum(1 for r in summary.values() if "error" not in r)
    print(f"DUMP_DONE n={len(summary)} n_ok={n_ok} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
