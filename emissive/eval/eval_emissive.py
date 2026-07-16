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
while not os.path.isfile(os.path.join(ROOT, "inference_full.py")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError(f"could not locate SegviGen repo root (inference_full.py) above {__file__}")
    ROOT = parent   # walk up: this script now lives nested under emissive/eval/, not repo root
SEGVIGEN = ROOT
sys.path.insert(0, SEGVIGEN)
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


def bucket_frac_for(d, bucket_by):
    """Coverage fraction used for --stratify bucketing. 'voxel' loads emis_mask.pth
    (mean coverage over the sample's 32-res latent grid, built from actual surface-voxel
    occupancy in make_emis_mask.py) — not tied to mesh tessellation. 'face' uses
    meta.json's emissive_frac (a mesh face-area fraction — biased by how finely a shape
    happens to be tessellated, kept only for comparison). Falls back face-wards with
    `used_fallback=True` if bucket_by='voxel' but emis_mask.pth is missing.
    Returns (frac, used_fallback)."""
    if bucket_by == "voxel":
        mp = os.path.join(d, "emis_mask.pth")
        if os.path.exists(mp):
            return float(torch.load(mp, map_location="cpu").mean().item()), False
    mj = os.path.join(d, "meta.json")
    frac = json.load(open(mj)).get("emissive_frac", 0.0) if os.path.exists(mj) else 0.0
    return float(frac), (bucket_by == "voxel")


def eval_sample(gen, models_d, d, cond_mode, device, steps=12, thrs=THRS, otsu=False, dump_vis=None,
                 sid=None, draws=1):
    """Sample the flow on one sample dir `d`, decode, and score vs GT, `draws` independent
    times (fresh flow-matching noise each draw; GT/shape-decode subs are computed once and
    reused). Returns a dict with per-threshold mean+std IoU across draws (+ optional Otsu)
    and the GT emissive fraction. draws=1 (default) reproduces the old single-draw behavior
    (std=0)."""
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
        # GT decode is independent of the sampled draw — decode once, reuse across draws.
        gt_vox = tex_decoder(sp.SparseTensor(otx["feats"].to(device), coords), guide_subs=subs) * 0.5 + 0.5
    gt_e = (gt_vox.feats[:, :3].mean(-1) > 0.5)   # base_color channels 0:3; GT fixed at 0.5
    gt_frac = gt_e.float().mean().item()

    draw_iou = {t: [] for t in thrs}
    draw_otsu_iou, draw_otsu_thr = [], []
    for k in range(draws):
        with torch.no_grad():
            noise = sp.SparseTensor(torch.randn_like(itx_n.feats), coords)
            out = sampler.sample(gen, noise, itx_n, shp_n, [coords.shape[0]], cond_dict, sp_params)
            out = out * ts + tm
            pred_vox = tex_decoder(out, guide_subs=subs) * 0.5 + 0.5   # base color in [0,1]-ish
        pred_bc = pred_vox.feats[:, :3].mean(-1)

        if dump_vis and k == 0:   # dump only the first draw to keep npz volume bounded
            os.makedirs(dump_vis, exist_ok=True)
            np.savez_compressed(
                os.path.join(dump_vis, f"{sid}.npz"),
                coords=pred_vox.coords[:, 1:].cpu().numpy().astype(np.int16),
                pred_bc=pred_bc.float().cpu().numpy().astype(np.float16),
                gt_e=gt_e.cpu().numpy())

        for t in thrs:
            pe = pred_bc > t
            inter = (pe & gt_e).sum().item(); union = (pe | gt_e).sum().item()
            draw_iou[t].append(inter / union if union > 0 else 1.0)
        if otsu:
            t_otsu = otsu_threshold(pred_bc.float().cpu().numpy())
            pe = pred_bc > t_otsu
            inter = (pe & gt_e).sum().item(); union = (pe | gt_e).sum().item()
            draw_otsu_thr.append(t_otsu)
            draw_otsu_iou.append(inter / union if union > 0 else 1.0)

    res = {"gt_frac": gt_frac,
           "iou_by_thr": {t: float(np.mean(v)) for t, v in draw_iou.items()},
           "iou_std_by_thr": {t: float(np.std(v)) for t, v in draw_iou.items()},
           "draws": draws}
    if otsu:
        res["otsu_iou"] = float(np.mean(draw_otsu_iou))
        res["otsu_iou_std"] = float(np.std(draw_otsu_iou))
        res["otsu_thr"] = float(np.mean(draw_otsu_thr))
    return res


def evaluate_split(gen, models_d, dataset_root, split, cond_mode, device="cuda", steps=12,
                    thrs=THRS, n=0, otsu=False, dump_vis=None, verbose=True, draws=1,
                    bucket_by="voxel"):
    """Eval `gen` (in .eval() mode) over `split`. Returns per-threshold means for ALL
    shapes and, separately, for shapes with GT coverage>0 ("nonzero" — the timidity-proof
    aggregate that a ckpt can't game by predicting all-black on empty-glow shapes), plus
    the IoU@0.5 headline, a threshold-sensitivity number (best-sweep minus @0.5), and (if
    draws>1) a split-level draw-std diagnostic."""
    sdir = os.path.join(dataset_root, split)
    sids = sorted(os.listdir(sdir))
    if n and n > 0:
        sids = sids[:n]
    ious_by_thr = {t: [] for t in thrs}
    otsu_ious = []
    per_sample = []   # [{sid, gt_frac, bucket_frac, best_iou, best_thr, iou_by_thr, iou_std_by_thr}, ...]
    n_eval = 0
    warned_fallback = False
    for sid in sids:
        d = os.path.join(sdir, sid)
        if not os.path.exists(os.path.join(d, "output_tex_slat.pth")):
            continue
        res = eval_sample(gen, models_d, d, cond_mode, device, steps=steps, thrs=thrs,
                          otsu=otsu, dump_vis=dump_vis, sid=sid, draws=draws)
        n_eval += 1
        bucket_frac, used_fallback = bucket_frac_for(d, bucket_by)
        if used_fallback and not warned_fallback:
            print(f"[warn] emis_mask.pth missing for {sid} (bucket_by=voxel) — "
                  f"falling back to face-frac (meta.json) for bucketing", flush=True)
            warned_fallback = True
        line = [f"{sid} gt_frac={res['gt_frac']:.3f}"]
        for t in thrs:
            ious_by_thr[t].append(res["iou_by_thr"][t])
            std_s = f"±{res['iou_std_by_thr'][t]:.3f}" if draws > 1 else ""
            line.append(f"IoU@{t}={res['iou_by_thr'][t]:.3f}{std_s}")
        if otsu:
            otsu_ious.append(res["otsu_iou"])
            line.append(f"IoU@otsu({res['otsu_thr']:.2f})={res['otsu_iou']:.3f}")
        if verbose:
            print(" ".join(line), flush=True)
        s_best_thr = max(res["iou_by_thr"], key=res["iou_by_thr"].get)
        per_sample.append({"sid": sid, "gt_frac": res["gt_frac"], "bucket_frac": bucket_frac,
                           "iou_by_thr": res["iou_by_thr"], "iou_std_by_thr": res["iou_std_by_thr"],
                           "best_iou": res["iou_by_thr"][s_best_thr], "best_thr": s_best_thr})

    def _agg(samples):
        mbt = {t: float(np.mean([s["iou_by_thr"][t] for s in samples])) if samples else 0.0 for t in thrs}
        bt = max(mbt, key=mbt.get) if samples else None
        bi = mbt[bt] if bt is not None else 0.0
        i5 = mbt.get(0.5, 0.0)
        return mbt, bt, bi, i5

    mean_by_thr, best_thr, best_iou, iou_at_5 = _agg(per_sample)
    nonzero_sample = [s for s in per_sample if s["gt_frac"] > 0]
    mean_by_thr_nz, best_thr_nz, best_iou_nz, iou_at_5_nz = _agg(nonzero_sample)

    draw_std = None
    if draws > 1 and best_thr is not None:
        stds = [s["iou_std_by_thr"][best_thr] for s in per_sample]
        draw_std = float(np.mean(stds)) if stds else None

    out = {"ious_by_thr": mean_by_thr, "best_thr": best_thr, "best_iou": best_iou,
           "iou_at_5": iou_at_5, "threshold_sensitivity": best_iou - iou_at_5,
           "ious_by_thr_nonzero": mean_by_thr_nz, "best_thr_nonzero": best_thr_nz,
           "best_iou_nonzero": best_iou_nz, "iou_at_5_nonzero": iou_at_5_nz,
           "threshold_sensitivity_nonzero": best_iou_nz - iou_at_5_nz,
           "n": n_eval, "n_nonzero": len(nonzero_sample), "draws": draws, "draw_std": draw_std,
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
                         "coverage (see --bucket_by): 0, (0,0.05], (0.05,0.3], >0.3 — "
                         "diagnoses whether error concentrates on near-zero-emissive shapes "
                         "(see canon10 autopsy)")
    ap.add_argument("--draws", type=int, default=1,
                    help="per-shape independent samplings (K). Reports per-shape mean+std "
                         "IoU at each threshold and a split-level draw-std diagnostic — "
                         "same ckpt re-scored with K=1 can swing appreciably (e.g. "
                         "0.096->0.128 seen historically), so K>1 is the honest number for "
                         "reporting. Use K=4 for full evals; runtime scales ~linearly with K.")
    ap.add_argument("--bucket_by", choices=["face", "voxel"], default="voxel",
                    help="coverage metric for --stratify bucketing. 'voxel' (default) loads "
                         "emis_mask.pth per shape — built from actual surface-voxel occupancy, "
                         "not tessellation-biased. 'face' uses meta.json's mesh-face-area "
                         "fraction (the old default; kept for comparison).")
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
                            dump_vis=args.dump_vis, draws=args.draws, bucket_by=args.bucket_by)

    draws_s = f", draws={args.draws}" if args.draws > 1 else ""
    print(f"\n=== {args.split} ({result['n']} samples, {result['n_nonzero']} with GT coverage>0"
          f"{draws_s}; DiffusionNet ~0.259) ===", flush=True)
    print(f"  HEADLINE  IoU@0.5          = {result['iou_at_5']:.4f}", flush=True)
    print(f"  HEADLINE  IoU@0.5 nonzero  = {result['iou_at_5_nonzero']:.4f}", flush=True)
    if result["draw_std"] is not None:
        print(f"  draw-std @best_thr (mean over shapes, K={args.draws}) = {result['draw_std']:.4f}", flush=True)
    print("\n  --- calibration diagnostic: threshold sweep (fixed thr may overfit the sweep) ---", flush=True)
    for t in THRS:
        print(f"  mean IoU @pred>{t} = {result['ious_by_thr'][t]:.4f}  "
              f"(nonzero: {result['ious_by_thr_nonzero'][t]:.4f})", flush=True)
    print(f"  BEST (fixed thr): @pred>{result['best_thr']} = {result['best_iou']:.4f}  "
          f"(nonzero: @pred>{result['best_thr_nonzero']} = {result['best_iou_nonzero']:.4f})", flush=True)
    print(f"  threshold sensitivity (best_sweep - @0.5) = {result['threshold_sensitivity']:.4f}  "
          f"(nonzero: {result['threshold_sensitivity_nonzero']:.4f})", flush=True)
    if args.otsu:
        print(f"  Otsu (per-shape adaptive thr): mean IoU = {result['otsu_iou']:.4f}", flush=True)

    strat_out = None
    if args.stratify:
        buckets = [("0", lambda f: f == 0), ("(0,0.05]", lambda f: 0 < f <= 0.05),
                   ("(0.05,0.3]", lambda f: 0.05 < f <= 0.3), (">0.3", lambda f: f > 0.3)]
        bt = result["best_thr"]
        print(f"\n  --- stratified by GT coverage (bucket_by={args.bucket_by}; "
              f"IoU @ global best_thr={bt}) ---", flush=True)
        strat_out = {}
        for name, pred in buckets:
            ious = [p["iou_by_thr"][bt] for p in result["per_sample"] if pred(p["bucket_frac"])]
            mean_iou = float(np.mean(ious)) if ious else None
            strat_out[name] = {"n": len(ious), "mean_iou": mean_iou}
            print(f"  {name:12s} n={len(ious):3d}  mean IoU={'n/a' if mean_iou is None else f'{mean_iou:.4f}'}", flush=True)

    out_json = dict(result["ious_by_thr"])
    out_json["iou_at_5"] = result["iou_at_5"]
    out_json["best_iou"] = result["best_iou"]
    out_json["best_thr"] = result["best_thr"]
    out_json["threshold_sensitivity"] = result["threshold_sensitivity"]
    out_json["n"] = result["n"]
    out_json["n_nonzero"] = result["n_nonzero"]
    out_json["nonzero"] = {"ious_by_thr": result["ious_by_thr_nonzero"], "iou_at_5": result["iou_at_5_nonzero"],
                            "best_iou": result["best_iou_nonzero"], "best_thr": result["best_thr_nonzero"],
                            "threshold_sensitivity": result["threshold_sensitivity_nonzero"]}
    if args.draws > 1:
        out_json["draws"] = args.draws
        out_json["draw_std"] = result["draw_std"]
    if args.otsu:
        out_json["otsu"] = result["otsu_iou"]
    if strat_out:
        out_json["stratified"] = strat_out
        out_json["bucket_by"] = args.bucket_by
    json.dump(out_json, open(os.path.join(args.dataset, f"eval_{args.split}.json"), "w"))


if __name__ == "__main__":
    main()
