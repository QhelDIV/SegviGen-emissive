"""Local, test-only patched copy of dump_pred_fbv1.py that saves EVERY draw's
full voxel array (not just draw k=0), so a genuinely representative draw can be
picked for a panel instead of whichever one happened to land in slot 0. The
shared script is untouched; this is a standalone copy for this specific need.
Only the save block changed (saves {sid}__draw{k}.npz for every k, plus keeps
writing {sid}.npz for draw 0 to stay compatible with existing consumers).
"""
import os
import sys
import json
import time
import argparse
from collections import OrderedDict

ROOT = "/3dlg-jupiter-project/lightgen/segvigen_emissive"
CODE = os.path.join(ROOT, "code")
SEGVIGEN = os.path.join(CODE, "SegviGen")
sys.path.insert(0, SEGVIGEN)
sys.path.insert(0, CODE)
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
    ap.add_argument("--split", default="paper3_11")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cond", required=True, choices=["real", "zero"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--only", default=None,
                    help="comma-separated sids to restrict to (rescue runs "
                         "only need a subset of the split, not the whole dir)")
    args = ap.parse_args()
    device = "cuda"

    os.makedirs(args.out_dir, exist_ok=True)
    sdir = os.path.join(args.dataset, args.split)
    sids = sorted(os.listdir(sdir))
    if args.only:
        keep = set(args.only.split(","))
        sids = [s for s in sids if s in keep]
    print(f"SIDS_OK n={len(sids)} draws={args.draws} cond={args.cond}", flush=True)

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
    for i, sid in enumerate(sids):
        d = os.path.join(sdir, sid)
        base_seed = args.seed * 1000003 + (int(sid[:8], 16) % 100000)

        shp = torch.load(os.path.join(d, "shape_slat.pth"), map_location=device)
        itx = torch.load(os.path.join(d, "input_tex_slat.pth"), map_location=device)
        otx = torch.load(os.path.join(d, "output_tex_slat.pth"), map_location=device)
        coords = shp["coords"].to(device)
        if args.cond == "zero":
            cond = torch.zeros(1, ee.COND_T, ee.COND_D, device=device)
        else:
            cond_raw = torch.load(os.path.join(d, "cond.pth"), map_location=device)["cond"]
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
            pred_bc_np = pred_bc.float().cpu().numpy().astype(np.float16)

            # SAVE EVERY DRAW (the only change from the shared script)
            np.savez_compressed(os.path.join(args.out_dir, f"{sid}__draw{k}.npz"),
                                coords=vc, pred_bc=pred_bc_np, gt_e=gt_e.cpu().numpy())
            if k == 0:
                np.savez_compressed(os.path.join(args.out_dir, f"{sid}.npz"),
                                    coords=vc, pred_bc=pred_bc_np, gt_e=gt_e.cpu().numpy())

            frac05 = float((pred_bc_np > 0.5).mean())
            print(f"DRAW sid={sid} k={k} pred_frac@0.5={frac05:.4f} max={pred_bc_np.max():.4f}", flush=True)

            for t in THRS:
                pe = pred_bc > t
                inter = (pe & gt_e).sum().item(); union = (pe | gt_e).sum().item()
                draw_frac[str(t)].append(float(pe.float().mean().item()))
                draw_iou[str(t)].append(inter / union if union > 0 else 1.0)

        rec = {"sid": sid, "gt_frac": float(gt_e.float().mean().item()), "draws": args.draws,
               "cond": args.cond,
               "pred_frac_by_thr": {t: float(np.mean(v)) for t, v in draw_frac.items()},
               "pred_frac_std_by_thr": {t: float(np.std(v)) for t, v in draw_frac.items()},
               "per_draw_frac_0.5": draw_frac["0.5"]}
        summary[sid] = rec
        print(f"SHAPE_DONE {i + 1}/{len(sids)} {sid} elapsed={time.time() - t0:.1f}s", flush=True)

    json.dump(summary, open(os.path.join(args.out_dir, "summary_alldraws.json"), "w"), indent=1)
    print(f"DUMP_DONE n={len(summary)} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
