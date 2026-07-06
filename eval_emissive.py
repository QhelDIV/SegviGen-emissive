"""
Evaluate the emissive fine-tune: sample the flow on val assets, decode the predicted
coloring to per-voxel base color, threshold to binary emissive, and compute IoU vs the
GT emissive coloring. Comparable (proxy) to the DiffusionNet baseline (~0.259 val IoU).

Per-voxel IoU on the shared (common) voxel coords is the v1 metric. The rigorous
per-face mesh IoU (decode→mesh→majority vote, like inference_full) is a TODO refinement.

The heavy-lifting (`load_eval_models`, `evaluate_split`) is factored out so
train_emissive.py can reuse it for cheap quick-val tracking during training without
reloading the tex/shape decoders every call.

Usage (GPU node, trellis2 env):
  python eval_emissive.py --dataset .../dataset --ckpt .../outputs/emis_pilot/last.ckpt \
      --split val --steps 12 --cond zero [--otsu]
"""
import os, sys, json, argparse
ROOT = os.path.dirname(os.path.abspath(__file__))
SEGVIGEN = os.path.join(ROOT, "SegviGen")
if os.path.isdir(SEGVIGEN):
    sys.path.insert(0, SEGVIGEN)   # legacy layout: script sits next to a separate SegviGen/ clone
else:
    SEGVIGEN = ROOT                # this script now lives inside the SegviGen repo root
    sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/3dlg-jupiter-project/lightgen/hf_cache")

import torch
import numpy as np
from collections import OrderedDict
import trellis2.modules.sparse as sp
from trellis2 import models
from inference_full import Gen3DSeg, Sampler
from huggingface_hub import hf_hub_download

COND_T, COND_D = 1024, 1024
THRS = [0.2, 0.3, 0.4, 0.5]   # sweep pred threshold (GT is white/black → fixed 0.5)


def load_pipeline_args():
    pj = hf_hub_download(repo_id="microsoft/TRELLIS.2-4B", filename="pipeline.json")
    return json.load(open(pj))["args"]


def load_eval_models(device="cuda"):
    """Load the fixed decoders + norm stats + sampler shared by every eval/quick-val
    call. Heavy (loads two decoder networks) — call once, reuse across calls."""
    tex_decoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16").to(device).eval()
    # tex decoder needs the shape VAE's subdivision structure (guide_subs) to upsample
    shape_decoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16").to(device).eval()
    pa = load_pipeline_args()
    sm = torch.tensor(pa["shape_slat_normalization"]["mean"])[None].to(device)
    ss = torch.tensor(pa["shape_slat_normalization"]["std"])[None].to(device)
    tm = torch.tensor(pa["tex_slat_normalization"]["mean"])[None].to(device)
    ts = torch.tensor(pa["tex_slat_normalization"]["std"])[None].to(device)
    sampler = Sampler()
    return {"tex_decoder": tex_decoder, "shape_decoder": shape_decoder, "pipeline_args": pa,
            "sm": sm, "ss": ss, "tm": tm, "ts": ts, "sampler": sampler}


def otsu_threshold(vals, bins=256):
    """2-cluster Otsu threshold on a 1D array of values in [0,1] (pure numpy — no
    cv2/skimage dependency). Maximizes inter-class variance of the histogram."""
    vals = np.asarray(vals, dtype=np.float64)
    hist, edges = np.histogram(vals, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = total - w0
    sum0 = np.cumsum(hist * centers)
    sum_all = sum0[-1]
    mu0 = np.divide(sum0, w0, out=np.zeros_like(sum0), where=w0 > 0)
    mu1 = np.divide(sum_all - sum0, w1, out=np.zeros_like(sum0), where=w1 > 0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[int(np.argmax(between))])


def eval_sample(gen, models_d, d, cond_mode, device, steps=12, thrs=THRS, otsu=False, dump_vis=None, sid=None):
    """Sample the flow on one sample dir `d`, decode, and score vs GT. Returns a dict
    with per-threshold IoU (+ optional Otsu IoU) and the GT emissive fraction."""
    tex_decoder, shape_decoder, sampler = models_d["tex_decoder"], models_d["shape_decoder"], models_d["sampler"]
    sm, ss, tm, ts = models_d["sm"], models_d["ss"], models_d["tm"], models_d["ts"]
    sp_params = dict(models_d["pipeline_args"]["tex_slat_sampler"]["params"]); sp_params["steps"] = steps

    shp = torch.load(os.path.join(d, "shape_slat.pth"), map_location=device)
    itx = torch.load(os.path.join(d, "input_tex_slat.pth"), map_location=device)
    otx = torch.load(os.path.join(d, "output_tex_slat.pth"), map_location=device)
    if cond_mode == "zero":
        cond = torch.zeros(1, COND_T, COND_D, device=device)
    else:
        cond_p = os.path.join(d, "cond.pth")
        if not os.path.exists(cond_p):
            raise FileNotFoundError(f"--cond real but cond.pth missing: {cond_p}")
        cond = torch.load(cond_p, map_location=device)["cond"]
    coords = shp["coords"].to(device)

    shp_n = sp.SparseTensor((shp["feats"].to(device) - sm) / ss, coords)
    itx_n = sp.SparseTensor((itx["feats"].to(device) - tm) / ts, coords)
    cond_dict = {"cond": cond, "neg_cond": torch.zeros_like(cond)}
    with torch.no_grad():
        shape_decoder.set_resolution(512)
        _, subs = shape_decoder(sp.SparseTensor(shp["feats"].to(device), coords), return_subs=True)
        noise = sp.SparseTensor(torch.randn_like(itx_n.feats), coords)
        out = sampler.sample(gen, noise, itx_n, shp_n, [coords.shape[0]], cond_dict, sp_params)
        out = out * ts + tm
        pred_vox = tex_decoder(out, guide_subs=subs) * 0.5 + 0.5           # base color in [0,1]-ish
        gt_vox = tex_decoder(sp.SparseTensor(otx["feats"].to(device), coords), guide_subs=subs) * 0.5 + 0.5
    # base_color = channels 0:3 ; emissive = bright (white). GT fixed at 0.5.
    pred_bc = pred_vox.feats[:, :3].mean(-1)
    gt_e = (gt_vox.feats[:, :3].mean(-1) > 0.5)

    if dump_vis:
        os.makedirs(dump_vis, exist_ok=True)
        np.savez_compressed(
            os.path.join(dump_vis, f"{sid}.npz"),
            coords=pred_vox.coords[:, 1:].cpu().numpy().astype(np.int16),
            pred_bc=pred_bc.float().cpu().numpy().astype(np.float16),
            gt_e=gt_e.cpu().numpy())

    res = {"gt_frac": gt_e.float().mean().item(), "iou_by_thr": {}}
    for t in thrs:
        pe = pred_bc > t
        inter = (pe & gt_e).sum().item(); union = (pe | gt_e).sum().item()
        res["iou_by_thr"][t] = inter / union if union > 0 else 1.0
    if otsu:
        t_otsu = otsu_threshold(pred_bc.float().cpu().numpy())
        pe = pred_bc > t_otsu
        inter = (pe & gt_e).sum().item(); union = (pe | gt_e).sum().item()
        res["otsu_thr"] = t_otsu
        res["otsu_iou"] = inter / union if union > 0 else 1.0
    return res


def evaluate_split(gen, models_d, dataset_root, split, cond_mode, device="cuda", steps=12,
                    thrs=THRS, n=0, otsu=False, dump_vis=None, verbose=True):
    """Eval `gen` (in .eval() mode) over `split`. Returns {ious_by_thr, best_thr,
    best_iou, otsu_iou (if requested), n}."""
    sdir = os.path.join(dataset_root, split)
    sids = sorted(os.listdir(sdir))
    if n and n > 0:
        sids = sids[:n]
    ious_by_thr = {t: [] for t in thrs}
    otsu_ious = []
    per_sample = []   # [{sid, gt_frac, best_iou, best_thr}, ...] — per-sample memorization
    n_eval = 0
    for sid in sids:
        d = os.path.join(sdir, sid)
        if not os.path.exists(os.path.join(d, "output_tex_slat.pth")):
            continue
        res = eval_sample(gen, models_d, d, cond_mode, device, steps=steps, thrs=thrs,
                          otsu=otsu, dump_vis=dump_vis, sid=sid)
        n_eval += 1
        line = [f"{sid} gt_frac={res['gt_frac']:.3f}"]
        for t in thrs:
            ious_by_thr[t].append(res["iou_by_thr"][t])
            line.append(f"IoU@{t}={res['iou_by_thr'][t]:.3f}")
        if otsu:
            otsu_ious.append(res["otsu_iou"])
            line.append(f"IoU@otsu({res['otsu_thr']:.2f})={res['otsu_iou']:.3f}")
        if verbose:
            print(" ".join(line), flush=True)
        s_best_thr = max(res["iou_by_thr"], key=res["iou_by_thr"].get)
        per_sample.append({"sid": sid, "gt_frac": res["gt_frac"], "iou_by_thr": res["iou_by_thr"],
                           "best_iou": res["iou_by_thr"][s_best_thr], "best_thr": s_best_thr})

    mean_by_thr = {t: float(np.mean(v)) if v else 0.0 for t, v in ious_by_thr.items()}
    best_thr = max(mean_by_thr, key=mean_by_thr.get) if mean_by_thr else None
    out = {"ious_by_thr": mean_by_thr, "best_thr": best_thr,
           "best_iou": mean_by_thr[best_thr] if best_thr is not None else 0.0, "n": n_eval,
           "per_sample": per_sample}
    if otsu:
        out["otsu_iou"] = float(np.mean(otsu_ious)) if otsu_ious else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--cond", required=True, choices=["real", "zero"],
                    help="explicit — no silent zero-cond fallback")
    ap.add_argument("--otsu", action="store_true", default=False,
                    help="also report per-shape 2-cluster Otsu-threshold IoU alongside the fixed sweep")
    ap.add_argument("--dump_vis", default=None,
                    help="dir to dump per-sample {coords,pred_bc,gt_e} npz for visualization")
    ap.add_argument("--n", type=int, default=0,
                    help="evaluate only the first n samples of the split (0 = all). "
                         "Use a small subset of the train split to track train-vs-val IoU "
                         "(overfitting monitor) cheaply.")
    ap.add_argument("--stratify", action="store_true", default=False,
                    help="also report mean IoU (at the fixed best_thr) bucketed by GT "
                         "emissive_frac: 0, (0,0.05], (0.05,0.3], >0.3 — diagnoses whether "
                         "error concentrates on near-zero-emissive shapes (see canon10 autopsy)")
    args = ap.parse_args()
    device = "cuda"

    flow = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")
    gen = Gen3DSeg(flow).to(device)
    sd = torch.load(args.ckpt, map_location=device)["state_dict"]
    sd = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in sd.items()])
    gen.load_state_dict(sd); gen.eval()

    models_d = load_eval_models(device)
    result = evaluate_split(gen, models_d, args.dataset, args.split, args.cond, device=device,
                            steps=args.steps, thrs=THRS, n=args.n, otsu=args.otsu,
                            dump_vis=args.dump_vis)

    print(f"\n=== {args.split} ({result['n']} samples; DiffusionNet ~0.259) ===", flush=True)
    for t in THRS:
        print(f"  mean IoU @pred>{t} = {result['ious_by_thr'][t]:.4f}", flush=True)
    print(f"  BEST (fixed thr): @pred>{result['best_thr']} = {result['best_iou']:.4f}", flush=True)
    if args.otsu:
        print(f"  Otsu (per-shape adaptive thr): mean IoU = {result['otsu_iou']:.4f}", flush=True)

    strat_out = None
    if args.stratify:
        buckets = [("0", lambda f: f == 0), ("(0,0.05]", lambda f: 0 < f <= 0.05),
                   ("(0.05,0.3]", lambda f: 0.05 < f <= 0.3), (">0.3", lambda f: f > 0.3)]
        bt = result["best_thr"]
        print(f"\n  --- stratified by GT emissive_frac (IoU @ global best_thr={bt}) ---", flush=True)
        strat_out = {}
        for name, pred in buckets:
            ious = [p["iou_by_thr"][bt] for p in result["per_sample"] if pred(p["gt_frac"])]
            mean_iou = float(np.mean(ious)) if ious else None
            strat_out[name] = {"n": len(ious), "mean_iou": mean_iou}
            print(f"  {name:12s} n={len(ious):3d}  mean IoU={'n/a' if mean_iou is None else f'{mean_iou:.4f}'}", flush=True)

    out_json = dict(result["ious_by_thr"])
    if args.otsu:
        out_json["otsu"] = result["otsu_iou"]
    if strat_out:
        out_json["stratified"] = strat_out
    json.dump(out_json, open(os.path.join(args.dataset, f"eval_{args.split}.json"), "w"))


if __name__ == "__main__":
    main()
