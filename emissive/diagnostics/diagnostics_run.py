"""
Diagnose why emis_72k_unfilt/run1 (and every other emissive-seg model) scores ~0.1 IoU.
Measures only -- does not touch the model, does not retrain, does not tune anything.

Four diagnostics, all on the SAME N random val_72k shapes (fixed seed):
  1. VAE round-trip ceiling: decode the GT latent (output_tex_slat.pth) through
     tex_decoder, threshold at each of THRS, score against the TRUE binary target
     (output.vxz's base_color, thresholded >127.5 -- the same recipe make_emis_mask.py
     uses). This is the best any model could score under the current metric.
  2. Trivial baselines on the same metric path: all-zero, all-one, random at matched
     density, and a percentile-of-PBR-brightness heuristic (from input.vxz).
  3. best.ckpt evaluated with --draws 3 --cond zero on the same shapes (per-shape
     mean+std IoU across draws), to separate sampling noise from model signal.
  4. Everything above stratified by meta.json's emissive_frac (a raw-voxel fraction
     in this direct-ovoxel pipeline, not a face-area fraction -- see build_dataset_direct.py).

GT-vs-truth coordinate note: tex_decoder(otx, guide_subs=subs) decodes back to the
raw 512-res voxel grid. Whether this decoded coordinate set matches output.vxz's raw
coords exactly is verified per-shape (not assumed) via an exact coordinate join;
match statistics are recorded per shape and summarized.

Usage (GPU node, trellis2 env):
  python diagnostics_run.py --dataset dataset_direct --split val_72k \
      --ckpt outputs/emis_72k_unfilt/run1/best.ckpt \
      --train_curve outputs/emis_72k_unfilt/run1/train_curve.json \
      --out outputs/emis_72k_unfilt/run1/diagnostics.json --n 300 --draws 3 --seed 0
"""
import os
import sys
import json
import time
import random
import argparse
from collections import OrderedDict

ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(ROOT, "inference_full.py")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError(f"could not locate SegviGen repo root (inference_full.py) above {__file__}")
    ROOT = parent   # walk up: this script now lives nested under emissive/diagnostics/, not repo root
SEGVIGEN = ROOT
sys.path.insert(0, SEGVIGEN)
sys.path.insert(0, os.path.join(ROOT, "emissive", "eval"))  # sibling dir holding eval_emissive.py
os.environ.setdefault("HF_HOME", "/3dlg-jupiter-project/lightgen/hf_cache")

import numpy as np
import torch
import o_voxel
import trellis2.modules.sparse as sp
from trellis2 import models
from inference_full import Gen3DSeg, Sampler

import eval_emissive as ee

THRS = ee.THRS  # [0.2, 0.3, 0.4, 0.5]
PCTS = [50, 70, 80, 90, 95, 97, 99]
BUCKETS = [("[0,0.01)", 0.0, 0.01), ("[0.01,0.05)", 0.01, 0.05), ("[0.05,0.2)", 0.05, 0.2),
           ("[0.2,0.5)", 0.2, 0.5), ("[0.5,0.8)", 0.5, 0.8), ("[0.8,1.0]", 0.8, 1.0000001)]
NEEDED_FILES = ["shape_slat.pth", "input_tex_slat.pth", "output_tex_slat.pth",
                "emis_mask.pth", "meta.json", "input.vxz", "output.vxz"]


def select_sids(dataset_root, split, n, seed):
    sdir = os.path.join(dataset_root, split)
    all_sids = sorted(os.listdir(sdir))
    random.Random(seed).shuffle(all_sids)
    chosen, skipped = [], []
    for sid in all_sids:
        d = os.path.join(sdir, sid)
        missing = [f for f in NEEDED_FILES if not os.path.exists(os.path.join(d, f))]
        if missing:
            skipped.append({"sid": sid, "reason": f"missing {missing}"})
            continue
        chosen.append(sid)
        if len(chosen) >= n:
            break
    return chosen, skipped


def iou_np(pred_bool, true_bool):
    inter = int((pred_bool & true_bool).sum())
    union = int((pred_bool | true_bool).sum())
    return inter / union if union > 0 else 1.0


def analyze_shape_ceiling_and_baselines(d, meta, models_d, device, seed):
    """Diagnostics 1+2 for one shape. Returns a dict of per-shape numbers, or a dict
    with 'error' if something failed (e.g. zero matched coords)."""
    tex_decoder, shape_decoder = models_d["tex_decoder"], models_d["shape_decoder"]
    shp = torch.load(os.path.join(d, "shape_slat.pth"), map_location=device)
    otx = torch.load(os.path.join(d, "output_tex_slat.pth"), map_location=device)
    coords = shp["coords"].to(device)
    with torch.no_grad():
        shape_decoder.set_resolution(512)
        _, subs = shape_decoder(sp.SparseTensor(shp["feats"].to(device), coords), return_subs=True)
        gt_vox = tex_decoder(sp.SparseTensor(otx["feats"].to(device), coords), guide_subs=subs) * 0.5 + 0.5
    gt_bc = gt_vox.feats[:, :3].mean(-1).float().cpu().numpy()
    gt_coords = gt_vox.coords[:, 1:].cpu().numpy().astype(np.int64)

    raw_coords_out, data_out = o_voxel.io.read_vxz(os.path.join(d, "output.vxz"), num_threads=1)
    raw_coords_in, data_in = o_voxel.io.read_vxz(os.path.join(d, "input.vxz"), num_threads=1)
    raw_coords_out_np = raw_coords_out.numpy().astype(np.int64)
    raw_coords_in_np = raw_coords_in.numpy().astype(np.int64)
    same_raw_coords = bool(raw_coords_out_np.shape == raw_coords_in_np.shape
                            and np.array_equal(raw_coords_out_np, raw_coords_in_np))

    white_true = (data_out["base_color"].float().mean(dim=-1) > 127.5).numpy()
    pbr_bright = (data_in["base_color"].float().mean(dim=-1) / 255.0).numpy()

    # exact coordinate join: gt_coords (decoded, 512-res) vs raw_coords_out (vxz, 512-res)
    key = lambda c: c[:, 0].astype(np.int64) * 512 * 512 + c[:, 1] * 512 + c[:, 2]
    raw_key = key(raw_coords_out_np)
    order = np.argsort(raw_key)
    raw_key_sorted = raw_key[order]
    gt_key = key(gt_coords)
    pos = np.clip(np.searchsorted(raw_key_sorted, gt_key), 0, max(len(raw_key_sorted) - 1, 0))
    matched = (len(raw_key_sorted) > 0) & (raw_key_sorted[pos] == gt_key) if len(raw_key_sorted) else np.zeros_like(gt_key, dtype=bool)
    matched_raw_idx = order[pos[matched]]

    n_gt, n_raw, n_matched = int(len(gt_key)), int(len(raw_key)), int(matched.sum())
    out = {"n_gt_vox": n_gt, "n_raw_vox": n_raw, "n_matched": n_matched,
           "match_frac_of_gt": (n_matched / n_gt) if n_gt else None,
           "same_raw_coords_in_out": same_raw_coords}
    if n_matched == 0:
        out["error"] = "zero matched coords between decoded GT and raw output.vxz"
        return out

    gt_bc_m = gt_bc[matched]
    true_e_m = white_true[matched_raw_idx]
    pbr_bright_m = pbr_bright[matched_raw_idx]
    n = len(true_e_m)
    gt_frac_true = float(true_e_m.mean())
    out["gt_frac_true_matched"] = gt_frac_true
    out["gt_frac_meta"] = meta.get("emissive_frac")

    # --- diagnostic 1: ceiling ---
    ceiling = {str(t): iou_np(gt_bc_m > t, true_e_m) for t in THRS}
    out["ceiling_iou_by_thr"] = ceiling
    out["ceiling_best_iou"] = max(ceiling.values())

    # secondary cross-check: emis_mask.pth (32-res block-mean) broadcast to this shape's
    # matched voxels vs the fine true_e_m -- should agree well if block-averaging is sane.
    mask_p = os.path.join(d, "emis_mask.pth")
    if os.path.exists(mask_p):
        emis_mask = torch.load(mask_p, map_location="cpu")
        slat_coords = shp["coords"][:, 1:].cpu().numpy().astype(np.int64)  # 32-res block coords
        block = (raw_coords_out_np[matched_raw_idx] // 16).clip(0, 31)
        blk_key = block[:, 0] * 32 * 32 + block[:, 1] * 32 + block[:, 2]
        slat_key = slat_coords[:, 0] * 32 * 32 + slat_coords[:, 1] * 32 + slat_coords[:, 2]
        s_order = np.argsort(slat_key)
        s_key_sorted = slat_key[s_order]
        s_pos = np.clip(np.searchsorted(s_key_sorted, blk_key), 0, max(len(s_key_sorted) - 1, 0))
        s_matched = (len(s_key_sorted) > 0) & (s_key_sorted[s_pos] == blk_key)
        mask_vals = np.zeros(n, dtype=np.float32)
        mask_vals[s_matched] = emis_mask.numpy()[s_order[s_pos[s_matched]]]
        mask_bin = mask_vals > 0.5
        out["emis_mask_agreement_with_true"] = float((mask_bin == true_e_m).mean())
        out["emis_mask_iou_vs_true"] = iou_np(mask_bin, true_e_m)
    else:
        out["emis_mask_agreement_with_true"] = None
        out["emis_mask_iou_vs_true"] = None

    # --- diagnostic 2: trivial baselines ---
    zero_pred = np.zeros(n, dtype=bool)
    one_pred = np.ones(n, dtype=bool)
    out["baseline_all_zero"] = iou_np(zero_pred, true_e_m)
    out["baseline_all_one"] = iou_np(one_pred, true_e_m)

    rng = np.random.RandomState((seed * 1000003 + abs(hash(os.path.basename(d.rstrip("/"))))) % (2 ** 32))
    rand_pred = rng.rand(n) < gt_frac_true
    out["baseline_random_matched"] = iou_np(rand_pred, true_e_m)

    heur = {}
    for pct in PCTS:
        thr_val = float(np.percentile(pbr_bright_m, pct)) if n > 0 else 0.0
        heur[str(pct)] = {"iou": iou_np(pbr_bright_m > thr_val, true_e_m), "thr_val": thr_val}
    out["baseline_pbr_heuristic_by_pct"] = heur
    best_pct = max(heur, key=lambda k: heur[k]["iou"])
    out["baseline_pbr_heuristic_best_per_shape"] = {"pct": best_pct, "iou": heur[best_pct]["iou"]}
    # also store raw arrays' iou-vs-pct so a GLOBAL best pct can be picked later
    out["_pbr_bright_m_for_global"] = None  # placeholder; global pct is recomputed from heur dict only
    return out


def bucket_of(frac):
    if frac is None:
        return None
    for name, lo, hi in BUCKETS:
        if lo <= frac < hi:
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val_72k")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_curve", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aggregate_only", action="store_true", default=False,
                     help="skip both compute loops; re-run only the aggregation/stratification "
                          "pass off the checkpoint files a prior run already wrote (crash "
                          "recovery -- a bug in aggregation should never cost re-computing "
                          "diag1/2/3, which are the expensive part)")
    args = ap.parse_args()
    device = "cuda"

    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    ckpt12_path = os.path.join(out_dir, f"_diag12_raw_seed{args.seed}_n{args.n}.json")
    ckpt3_path = os.path.join(out_dir, f"_diag3_raw_seed{args.seed}_n{args.n}_draws{args.draws}.json")

    def summarize(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        a = np.array(vals, dtype=np.float64)
        return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
                "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}

    def print_summary(tag, key, s):
        if s is None:
            print(f"{tag} {key} n=0 (no data)", flush=True)
        else:
            print(f"{tag} {key} n={s['n']} mean={s['mean']:.4f} median={s['median']:.4f} "
                  f"p10={s['p10']:.4f} p90={s['p90']:.4f}", flush=True)

    if args.aggregate_only:
        assert os.path.exists(ckpt12_path), f"--aggregate_only but missing checkpoint {ckpt12_path}"
        assert os.path.exists(ckpt3_path), f"--aggregate_only but missing checkpoint {ckpt3_path}"
        ck12 = json.load(open(ckpt12_path))
        ck3 = json.load(open(ckpt3_path))
        sids, skipped, metas = ck12["sids"], ck12["skipped"], ck12["metas"]
        per_shape_12, n_err = ck12["per_shape_12"], ck12["n_err"]
        per_shape_3 = ck3["per_shape_3"]
        print(f"AGGREGATE_ONLY loaded checkpoints n={len(sids)}", flush=True)
    else:
        sids, skipped = select_sids(args.dataset, args.split, args.n, args.seed)
        print(f"SELECT_OK n={len(sids)} skipped={len(skipped)}", flush=True)

        models_d = ee.load_eval_models(device)
        print("ENV_OK models loaded", flush=True)

        sdir = os.path.join(args.dataset, args.split)
        metas = {}
        for sid in sids:
            metas[sid] = json.load(open(os.path.join(sdir, sid, "meta.json")))

        # ---- diagnostics 1 + 2 ----
        per_shape_12 = {}
        n_err = 0
        t0 = time.time()
        for i, sid in enumerate(sids):
            d = os.path.join(sdir, sid)
            try:
                res = analyze_shape_ceiling_and_baselines(d, metas[sid], models_d, device, args.seed)
            except Exception as e:
                res = {"error": repr(e)[:300]}
            if "error" in res and "ceiling_iou_by_thr" not in res:
                n_err += 1
            per_shape_12[sid] = {k: v for k, v in res.items() if not k.startswith("_")}
            if (i + 1) % 50 == 0 or i == 0:
                print(f"DIAG12_PROGRESS {i + 1}/{len(sids)} elapsed={time.time() - t0:.1f}s errs={n_err}", flush=True)
        print(f"DIAG1_2_DONE n={len(sids)} errs={n_err} elapsed={time.time() - t0:.1f}s", flush=True)

        # persist BEFORE aggregation so a later crash never costs this compute
        json.dump({"sids": sids, "skipped": skipped, "metas": metas,
                    "per_shape_12": per_shape_12, "n_err": n_err}, open(ckpt12_path, "w"))
        print(f"CHECKPOINT_WRITTEN {ckpt12_path}", flush=True)

        ok_preview = [s for s in sids if "ceiling_iou_by_thr" in per_shape_12[s]]
        zero_preview = [s for s in ok_preview if metas[s].get("emissive_frac", 0.0) == 0.0]
        nz_preview = [s for s in ok_preview if (metas[s].get("emissive_frac") or 0.0) > 0.0]
        print(f"GT_COMPOSITION n_ok={len(ok_preview)} n_zero_gt={len(zero_preview)} "
              f"n_nonzero_gt={len(nz_preview)}", flush=True)
        print("IOU_CONVENTION empty_pred_and_empty_gt -> IoU=1.0 (matches eval_sample: "
              "inter/union if union>0 else 1.0). This inflates all-shapes ceiling/all_zero "
              "means on zero-GT shapes -- see the *_NONZERO_ONLY lines for the honest number.",
              flush=True)
        for t in THRS:
            print_summary("DIAG1_CEILING_ALL_SHAPES", f"thr={t}",
                           summarize([per_shape_12[s]["ceiling_iou_by_thr"][str(t)] for s in ok_preview]))
            print_summary("DIAG1_CEILING_NONZERO_ONLY", f"thr={t}",
                           summarize([per_shape_12[s]["ceiling_iou_by_thr"][str(t)] for s in nz_preview]))
        print_summary("DIAG2_BASELINE_ALL_ZERO_ALL_SHAPES", "-",
                       summarize([per_shape_12[s]["baseline_all_zero"] for s in ok_preview]))
        print_summary("DIAG2_BASELINE_ALL_ZERO_NONZERO_ONLY", "-",
                       summarize([per_shape_12[s]["baseline_all_zero"] for s in nz_preview]))
        print_summary("DIAG2_BASELINE_ALL_ONE_ALL_SHAPES", "-",
                       summarize([per_shape_12[s]["baseline_all_one"] for s in ok_preview]))
        print_summary("DIAG2_BASELINE_ALL_ONE_NONZERO_ONLY", "-",
                       summarize([per_shape_12[s]["baseline_all_one"] for s in nz_preview]))
        print_summary("DIAG2_BASELINE_RANDOM_ALL_SHAPES", "-",
                       summarize([per_shape_12[s]["baseline_random_matched"] for s in ok_preview]))
        print_summary("DIAG2_BASELINE_RANDOM_NONZERO_ONLY", "-",
                       summarize([per_shape_12[s]["baseline_random_matched"] for s in nz_preview]))
        for pct in PCTS:
            print_summary("DIAG2_BASELINE_PBR_HEUR_ALL_SHAPES", f"pct={pct}",
                           summarize([per_shape_12[s]["baseline_pbr_heuristic_by_pct"][str(pct)]["iou"]
                                      for s in ok_preview]))
            print_summary("DIAG2_BASELINE_PBR_HEUR_NONZERO_ONLY", f"pct={pct}",
                           summarize([per_shape_12[s]["baseline_pbr_heuristic_by_pct"][str(pct)]["iou"]
                                      for s in nz_preview]))

        # ---- diagnostic 3: model eval with draws ----
        flow = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")
        gen = Gen3DSeg(flow).to(device)
        sd = torch.load(args.ckpt, map_location=device)["state_dict"]
        sd = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in sd.items()])
        gen.load_state_dict(sd)
        gen.eval()
        print("CKPT_LOADED", flush=True)

        per_shape_3 = {}
        t0 = time.time()
        smoke_n = 5
        for i, sid in enumerate(sids):
            d = os.path.join(sdir, sid)
            res = ee.eval_sample(gen, models_d, d, "zero", device, steps=args.steps, thrs=THRS,
                                  draws=args.draws, sid=sid)
            per_shape_3[sid] = {"gt_frac_decoded": res["gt_frac"],
                                 "iou_by_thr_mean": {str(k): v for k, v in res["iou_by_thr"].items()},
                                 "iou_by_thr_std": {str(k): v for k, v in res["iou_std_by_thr"].items()}}
            if i + 1 == smoke_n:
                elapsed = time.time() - t0
                est_total = elapsed / smoke_n * len(sids)
                print(f"DIAG3_SMOKE_TIMING n={smoke_n} elapsed={elapsed:.1f}s "
                      f"est_total_for_{len(sids)}={est_total:.1f}s", flush=True)
            if (i + 1) % 50 == 0:
                print(f"DIAG3_PROGRESS {i + 1}/{len(sids)} elapsed={time.time() - t0:.1f}s", flush=True)
        print(f"DIAG3_DONE n={len(sids)} elapsed={time.time() - t0:.1f}s", flush=True)

        # persist BEFORE aggregation
        json.dump({"per_shape_3": per_shape_3}, open(ckpt3_path, "w"))
        print(f"CHECKPOINT_WRITTEN {ckpt3_path}", flush=True)

        for t in THRS:
            print_summary("DIAG3_MODEL", f"thr={t}",
                           summarize([per_shape_3[s]["iou_by_thr_mean"][str(t)] for s in sids]))
            print_summary("DIAG3_DRAW_STD", f"thr={t}",
                           summarize([per_shape_3[s]["iou_by_thr_std"][str(t)] for s in sids]))

    # ---- assemble diag4 stratification + overall summaries (fresh run or --aggregate_only) ----
    ok_sids = [s for s in sids if "ceiling_iou_by_thr" in per_shape_12[s]]
    # exact-zero GT shapes make ceiling/all_zero trivially 1.0 by the 0/0->1.0 convention below;
    # every "all shapes" aggregate must be paired with a nonzero-only one or it's misleading.
    zero_gt_sids = [s for s in ok_sids if metas[s].get("emissive_frac", 0.0) == 0.0]
    nonzero_gt_sids = [s for s in ok_sids if (metas[s].get("emissive_frac") or 0.0) > 0.0]
    gt_composition = {"n_ok": len(ok_sids), "n_zero_gt": len(zero_gt_sids), "n_nonzero_gt": len(nonzero_gt_sids)}

    def dual_summary(getter, sids_all=ok_sids, sids_nz=nonzero_gt_sids):
        return {"all_shapes": summarize([getter(s) for s in sids_all]),
                "nonzero_gt_only": summarize([getter(s) for s in sids_nz])}

    ceiling_summary = {str(t): dual_summary(lambda s, t=t: per_shape_12[s]["ceiling_iou_by_thr"][str(t)])
                        for t in THRS}
    baseline_summary = {
        "all_zero": dual_summary(lambda s: per_shape_12[s]["baseline_all_zero"]),
        "all_one": dual_summary(lambda s: per_shape_12[s]["baseline_all_one"]),
        "random_matched": dual_summary(lambda s: per_shape_12[s]["baseline_random_matched"]),
    }
    # global best PBR-heuristic percentile (chosen once across the whole eval set, not per-shape),
    # picked separately on all-shapes and on nonzero-GT-only since the zero-GT shapes trivially
    # favor whichever percentile predicts closest to empty.
    pct_means = {}
    pct_means_nz = {}
    for pct in PCTS:
        pct_means[str(pct)] = float(np.mean(
            [per_shape_12[s]["baseline_pbr_heuristic_by_pct"][str(pct)]["iou"] for s in ok_sids]))
        pct_means_nz[str(pct)] = float(np.mean(
            [per_shape_12[s]["baseline_pbr_heuristic_by_pct"][str(pct)]["iou"] for s in nonzero_gt_sids])
        ) if nonzero_gt_sids else None
    best_global_pct = max(pct_means, key=pct_means.get)
    best_global_pct_nz = max(pct_means_nz, key=pct_means_nz.get) if nonzero_gt_sids else None
    baseline_summary["pbr_heuristic_by_pct_global_mean"] = pct_means
    baseline_summary["pbr_heuristic_by_pct_global_mean_nonzero_gt_only"] = pct_means_nz
    baseline_summary["pbr_heuristic_best_global"] = {"pct": best_global_pct, "mean_iou": pct_means[best_global_pct]}
    baseline_summary["pbr_heuristic_best_global_nonzero_gt_only"] = (
        {"pct": best_global_pct_nz, "mean_iou": pct_means_nz[best_global_pct_nz]} if best_global_pct_nz else None)

    model_summary = {str(t): summarize([per_shape_3[s]["iou_by_thr_mean"][str(t)] for s in sids]) for t in THRS}
    draw_std_summary = {str(t): summarize([per_shape_3[s]["iou_by_thr_std"][str(t)] for s in sids]) for t in THRS}

    # stratify by meta.json emissive_frac. The [0,0.01) bucket mixes exact-zero-GT shapes (where
    # ceiling/all_zero are trivially 1.0/1.0 by the 0/0 convention) with genuinely-small-but-nonzero
    # ones -- report the exact-zero count within that bucket explicitly rather than let it hide.
    strat = {}
    for name, lo, hi in BUCKETS:
        b_sids = [s for s in sids if metas[s].get("emissive_frac") is not None and lo <= metas[s]["emissive_frac"] < hi]
        b_ok = [s for s in b_sids if s in ok_sids]
        b_zero = [s for s in b_ok if metas[s]["emissive_frac"] == 0.0]
        b_nz = [s for s in b_ok if metas[s]["emissive_frac"] > 0.0]
        entry = {"n": len(b_sids), "n_with_valid_ceiling": len(b_ok),
                 "n_exact_zero_gt": len(b_zero), "n_nonzero_gt": len(b_nz)}
        if b_ok:
            entry["ceiling_iou_at_0.5"] = float(np.mean([per_shape_12[s]["ceiling_iou_by_thr"]["0.5"] for s in b_ok]))
            entry["ceiling_best_iou"] = float(np.mean([per_shape_12[s]["ceiling_best_iou"] for s in b_ok]))
            entry["baseline_all_zero"] = float(np.mean([per_shape_12[s]["baseline_all_zero"] for s in b_ok]))
            entry["baseline_all_one"] = float(np.mean([per_shape_12[s]["baseline_all_one"] for s in b_ok]))
            entry["baseline_random_matched"] = float(np.mean([per_shape_12[s]["baseline_random_matched"] for s in b_ok]))
            entry["baseline_pbr_heuristic_global_pct"] = float(np.mean(
                [per_shape_12[s]["baseline_pbr_heuristic_by_pct"][best_global_pct]["iou"] for s in b_ok]))
        if b_nz:
            entry["ceiling_iou_at_0.5_nonzero_only"] = float(np.mean(
                [per_shape_12[s]["ceiling_iou_by_thr"]["0.5"] for s in b_nz]))
            entry["baseline_all_zero_nonzero_only"] = float(np.mean(
                [per_shape_12[s]["baseline_all_zero"] for s in b_nz]))
        if b_sids:
            entry["model_iou_at_0.5"] = float(np.mean([per_shape_3[s]["iou_by_thr_mean"]["0.5"] for s in b_sids]))
            entry["model_best_iou"] = float(np.mean(
                [max(per_shape_3[s]["iou_by_thr_mean"].values()) for s in b_sids]))
        strat[name] = entry
        print(f"BUCKET_DONE {name} n={entry['n']} n_exact_zero_gt={entry['n_exact_zero_gt']} "
              f"n_nonzero_gt={entry['n_nonzero_gt']}", flush=True)

    # coordinate-match sanity summary
    match_fracs = [per_shape_12[s]["match_frac_of_gt"] for s in sids
                   if per_shape_12[s].get("match_frac_of_gt") is not None]
    same_coords_rate = float(np.mean([per_shape_12[s]["same_raw_coords_in_out"] for s in sids
                                       if "same_raw_coords_in_out" in per_shape_12[s]]))
    mask_agree = [per_shape_12[s]["emis_mask_agreement_with_true"] for s in ok_sids
                  if per_shape_12[s].get("emis_mask_agreement_with_true") is not None]

    train_curve_raw = json.load(open(args.train_curve))
    train_curve_summary = [{k: v for k, v in epoch.items() if k != "per_sample"} for epoch in train_curve_raw]

    out = {
        "config": {"dataset": args.dataset, "split": args.split, "ckpt": args.ckpt, "n": len(sids),
                    "draws": args.draws, "steps": args.steps, "seed": args.seed, "thrs": THRS},
        "n_requested": args.n, "n_selected": len(sids), "n_skipped": len(skipped),
        "skipped": skipped,
        "n_ceiling_errors": n_err,
        "gt_composition": gt_composition,
        "iou_convention": "empty prediction AND empty GT -> IoU=1.0 (matches eval_sample's "
                           "own convention: inter/union if union>0 else 1.0). Every 'all_shapes' "
                           "aggregate below is paired with a 'nonzero_gt_only' one because this "
                           "convention trivially inflates ceiling and all_zero-baseline scores on "
                           "exact-zero-GT shapes (see gt_composition.n_zero_gt for the count).",
        "diag1_ceiling_summary_by_thr": ceiling_summary,
        "diag2_baseline_summary": baseline_summary,
        "diag3_model_summary_by_thr": model_summary,
        "diag3_draw_std_summary_by_thr": draw_std_summary,
        "diag4_stratified": strat,
        "coord_match_diagnostics": {
            "mean_match_frac_of_decoded_gt": float(np.mean(match_fracs)) if match_fracs else None,
            "min_match_frac_of_decoded_gt": float(np.min(match_fracs)) if match_fracs else None,
            "same_raw_coords_input_output_rate": same_coords_rate,
            "mean_emis_mask_agreement_with_true_voxel": float(np.mean(mask_agree)) if mask_agree else None,
        },
        "per_shape_diag1_diag2": {s: {k: v for k, v in per_shape_12[s].items() if not k.startswith("_")}
                                   for s in sids},
        "per_shape_diag3": per_shape_3,
        "per_shape_meta_emissive_frac": {s: metas[s].get("emissive_frac") for s in sids},
        "train_curve": train_curve_summary,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=None)
    print(f"WROTE {args.out}", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
