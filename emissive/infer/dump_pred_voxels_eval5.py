"""Five-draw sample generation over the shared 381-shape evaluation set (usable.txt),
for the lightgen team's shared evaluator (evaluation/evaluate_texge.py via
TRELLIS2/data_toolkit/bridge_voxel_to_uv.py). Job: eval5_samples.

Differs from every existing dump_pred_voxels* variant in the details the bridge needs:
- Real conditioning read straight from the dataset_direct sample's own cond.pth
  (not a separate --cond_dir the way dump_pred_voxels_realcond.py does for Path A data;
  dataset_direct/val_72k already carries cond.pth per shape).
- Saves the FULL [N,3] RGB emission array under the key `pred`, not the brightness-mean
  scalar `pred_bc` every prior script wrote. bridge_voxel_to_uv.py's --source flag only
  accepts {pred, gt_recon, gt_emission} and resamples emission[idx] into a [H,W,3] UV
  image -- a 1-D scalar array does not broadcast into that; it has to be [N,3].
- All 5 draws are saved, one directory per draw (draw0/ .. draw4/, each holding a flat
  {sha}.npz), not just draw 0. This is the cleanest fit onto make_pred_dir.py, which
  globs one flat directory and re-keys a single source column to `pred` -- five draw
  directories let the team's existing bridge/eval tooling run unmodified per draw, then
  average the 5 resulting IoUs (the project's K-draw measurement doctrine), rather than
  requiring make_pred_dir.py to learn a new multi-draw npz schema.
- gt_recon (this model's own VAE decode of the GT output_tex_slat, the same ceiling
  dump_pred_voxels.py already computes as `gt_vox`) is written into every draw dir too,
  identical across draws -- cheap (already computed once per shape) and lets the team
  score our decode ceiling with the same unmodified per-draw tooling.

Usage (GPU node, trellis2 env):
  python emissive/infer/dump_pred_voxels_eval5.py \
      --dataset /cs/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct \
      --split val_72k --sids_file usable.txt \
      --ckpt .../outputs/emis_72kv2_cond_pw1b/epoch_0008.ckpt \
      --out_dir .../outputs/eval5_samples_ep8 --seed 0 --draws 5
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
    ROOT = parent   # walk up: this script lives nested under emissive/infer/, not repo root
SEGVIGEN = ROOT
sys.path.insert(0, SEGVIGEN)
sys.path.insert(0, os.path.join(ROOT, "emissive", "eval"))  # sibling dir holding eval_emissive.py
os.environ.setdefault("HF_HOME", "/3dlg-jupiter-project/lightgen/hf_cache")

# inference_full.py unconditionally imports data_toolkit.bpy_render for its render helper,
# which this script never calls (generation only, no rendering) -- bpy/mathutils are not
# installed in the trellis2 env on this workstation. Every bpy/mathutils use in bpy_render.py
# is inside a function body, not at module level, so a dummy stub is safe here (same trick
# TRELLIS2/data_toolkit/predict_emission_voxels.py uses for unused o_voxel submodules).
import types
for _m in ("bpy", "mathutils"):
    sys.modules.setdefault(_m, types.ModuleType(_m))

import numpy as np
import torch
import trellis2.modules.sparse as sp
from trellis2 import models
from inference_full import Gen3DSeg

import eval_emissive as ee

THRS = [0.2, 0.3, 0.4, 0.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset_direct root")
    ap.add_argument("--split", default="val_72k")
    ap.add_argument("--sids_file", required=True, help="text file, one sha per line (e.g. usable.txt)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cond", default="real", choices=["real", "zero"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--only", default=None, help="comma-separated sids to restrict to (pilot runs)")
    args = ap.parse_args()
    device = "cuda"

    os.makedirs(args.out_dir, exist_ok=True)
    sids = [l.strip() for l in open(args.sids_file) if l.strip()]
    if args.only:
        keep = set(args.only.split(","))
        sids = [s for s in sids if s in keep]

    draw_dirs = [os.path.join(args.out_dir, f"draw{k}") for k in range(args.draws)]
    for dd in draw_dirs:
        os.makedirs(dd, exist_ok=True)

    # Resume support: a prior run of this exact (--out_dir, --sids_file, --seed, --draws)
    # may have died partway (e.g. an external kill with no error trail -- observed once,
    # 2026-08-18). Per-shape seeds are a pure function of (args.seed, sid) alone -- see
    # base_seed below, computed fresh from sid with no dependence on loop position or
    # which other shapes ran -- so skipping already-done shapes reproduces bit-identical
    # seeds to a from-scratch run; this is not an approximation.
    summary_path = os.path.join(args.out_dir, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

    def load_completed_shape(sid):
        """Returns the 5 loaded draw npz dicts if this shape is fully, validly on disk;
        None otherwise. The npz files -- not summary.json -- are the source of truth for
        "is this shape done": summary.json is a derived convenience file and did not
        survive the 2026-08-18 kill even though 48 shapes' npz were intact, so gating
        resume on summary.json presence (an earlier, wrong version of this function)
        silently redid all 48 already-complete shapes."""
        loaded = []
        for dd in draw_dirs:
            p = os.path.join(dd, f"{sid}.npz")
            if not os.path.exists(p):
                return None
            try:
                d = np.load(p)
                if "coords" not in d or "pred" not in d or "gt_recon" not in d:
                    return None
                loaded.append(d)
            except Exception:
                return None  # truncated/corrupt npz from a mid-write kill -- redo it
        return loaded

    def rebuild_summary_rec(sid, loaded):
        """Recompute the same per-draw/per-threshold stats main()'s loop would have
        logged, from already-on-disk npz -- so a resumed run's summary.json ends up
        complete even for shapes whose original stats were lost with the old summary.json."""
        gt_recon = loaded[0]["gt_recon"].astype(np.float32)
        gt_e_bool = gt_recon.mean(-1) > 0.5
        draw_frac = {str(t): [] for t in THRS}
        draw_iou = {str(t): [] for t in THRS}
        for d in loaded:
            pred_bc = d["pred"].astype(np.float32).mean(-1)
            for t in THRS:
                pe_t = pred_bc > t
                inter_t = int((pe_t & gt_e_bool).sum()); union_t = int((pe_t | gt_e_bool).sum())
                draw_frac[str(t)].append(float(pe_t.mean()))
                draw_iou[str(t)].append(inter_t / union_t if union_t > 0 else 1.0)
        return {"sid": sid, "split": args.split, "draws": len(loaded), "cond": args.cond,
                "seeds": None, "note": "recomputed from on-disk npz after a resume",
                "gt_frac": float(gt_e_bool.mean()),
                "pred_frac_by_thr": {t: float(np.mean(v)) for t, v in draw_frac.items()},
                "pred_frac_std_by_thr": {t: float(np.std(v)) for t, v in draw_frac.items()},
                "iou_by_thr": {t: float(np.mean(v)) for t, v in draw_iou.items()},
                "iou_std_by_thr": {t: float(np.std(v)) for t, v in draw_iou.items()},
                "iou_by_thr_per_draw": draw_iou}

    todo = []
    n_reconstructed = 0
    for s in sids:
        loaded = load_completed_shape(s)
        if loaded is None:
            todo.append(s)
        elif s not in summary:
            summary[s] = rebuild_summary_rec(s, loaded)
            n_reconstructed += 1
    n_skipped = len(sids) - len(todo)
    print(f"SIDS_OK n={len(sids)} draws={args.draws} cond={args.cond} split={args.split} "
          f"resume_skip={n_skipped} (summary_reconstructed={n_reconstructed}) todo={len(todo)}", flush=True)
    sids = todo
    if n_reconstructed:
        json.dump(summary, open(summary_path, "w"), indent=1)
    if not sids:
        print(f"DUMP_DONE n={len(summary)} elapsed=0.0s (nothing to do, already complete)", flush=True)
        return

    models_d = ee.load_eval_models(device)
    print("ENV_OK models loaded", flush=True)

    flow = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")
    gen = Gen3DSeg(flow).to(device)
    sd = torch.load(args.ckpt, map_location=device)["state_dict"]
    sd = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in sd.items()])
    missing, unexpected = gen.load_state_dict(sd, strict=False)
    print(f"CKPT_LOADED ckpt={args.ckpt} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing or unexpected:
        raise SystemExit("checkpoint does not match the current Gen3DSeg architecture")
    gen.eval()

    tex_decoder, shape_decoder, sampler = models_d["tex_decoder"], models_d["shape_decoder"], models_d["sampler"]
    sm, ss, tm, ts = models_d["sm"], models_d["ss"], models_d["tm"], models_d["ts"]
    sp_params = dict(models_d["pipeline_args"]["tex_slat_sampler"]["params"])
    sp_params["steps"] = args.steps

    # NB: `summary` is already populated above (loaded from disk + resume-reconstructed
    # entries) -- do NOT reset it here. An earlier version of this line did `summary = {}`,
    # silently discarding every reconstructed/pre-loaded entry once the main loop's first
    # incremental flush overwrote summary.json with only the newly-generated shapes (found
    # 2026-08-18 while verifying the completed 381-shape sweep: summary.json had 332/381
    # entries, missing exactly the 49 that were already done at the run's start -- the npz
    # data itself was unaffected, only this bookkeeping file).
    t0 = time.time()
    for i, sid in enumerate(sids):
        d = os.path.join(args.dataset, args.split, sid)

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
        gt_e_bool = (gt_vox.feats[:, :3].mean(-1) > 0.5)
        gt_recon_rgb = gt_vox.feats[:, :3].float().cpu().numpy().astype(np.float16)

        # per-shape, per-draw fixed base seed (independent of run order / added sids later)
        base_seed = args.seed * 1000003 + (int(sid[:8], 16) % 100000)

        draw_frac = {str(t): [] for t in THRS}
        draw_iou = {str(t): [] for t in THRS}
        seeds_used = []
        for k in range(args.draws):
            seed_k = base_seed + k
            seeds_used.append(seed_k)
            torch.manual_seed(seed_k)
            np.random.seed(args.seed)
            with torch.no_grad():
                noise = sp.SparseTensor(torch.randn_like(itx_n.feats), coords)
                out = sampler.sample(gen, noise, itx_n, shp_n, [coords.shape[0]], cond_dict, sp_params)
                out = out * ts + tm
                pred_vox = tex_decoder(out, guide_subs=subs) * 0.5 + 0.5
            pred_rgb = pred_vox.feats[:, :3].float().cpu().numpy().astype(np.float16)
            pred_bc = pred_vox.feats[:, :3].mean(-1)
            vc = pred_vox.coords[:, 1:].cpu().numpy().astype(np.int16)

            # atomic write (temp + rename): on this project's NFS-backed jupiter storage,
            # a kill mid-savez can leave a truncated file at the final path; write-then-
            # rename means shape_already_done() only ever sees a complete file or none.
            # NB: np.savez_compressed silently appends ".npz" to any name not already
            # ending in it, so the temp name must end in ".npz" itself or the rename below
            # targets a file that was never written.
            final_p = os.path.join(draw_dirs[k], f"{sid}.npz")
            tmp_p = os.path.join(draw_dirs[k], f"{sid}.tmp.npz")
            np.savez_compressed(tmp_p, coords=vc, pred=pred_rgb, gt_recon=gt_recon_rgb)
            os.replace(tmp_p, final_p)

            pe = pred_bc > 0.5
            inter = (pe & gt_e_bool).sum().item(); union = (pe | gt_e_bool).sum().item()
            frac05 = float(pe.float().mean().item())
            iou05 = inter / union if union > 0 else 1.0
            print(f"DRAW sid={sid} k={k} seed={seed_k} pred_frac@0.5={frac05:.4f} iou@0.5={iou05:.4f}", flush=True)

            for t in THRS:
                pe_t = pred_bc > t
                inter_t = (pe_t & gt_e_bool).sum().item(); union_t = (pe_t | gt_e_bool).sum().item()
                draw_frac[str(t)].append(float(pe_t.float().mean().item()))
                draw_iou[str(t)].append(inter_t / union_t if union_t > 0 else 1.0)

        rec = {"sid": sid, "split": args.split, "draws": args.draws, "cond": args.cond,
               "seeds": seeds_used,
               "gt_frac": float(gt_e_bool.float().mean().item()),
               "pred_frac_by_thr": {t: float(np.mean(v)) for t, v in draw_frac.items()},
               "pred_frac_std_by_thr": {t: float(np.std(v)) for t, v in draw_frac.items()},
               "iou_by_thr": {t: float(np.mean(v)) for t, v in draw_iou.items()},
               "iou_std_by_thr": {t: float(np.std(v)) for t, v in draw_iou.items()},
               "iou_by_thr_per_draw": draw_iou}
        summary[sid] = rec
        # flush after every shape (atomic write via temp+rename) so a mid-sweep kill loses
        # at most the in-flight shape's stats, not every stat computed so far -- this is
        # what made the previous death lose all 48 shapes' summary.json entries even though
        # their npz files were intact on disk.
        tmp_path = os.path.join(args.out_dir, "summary.json.tmp")
        json.dump(summary, open(tmp_path, "w"), indent=1)
        os.replace(tmp_path, summary_path)
        print(f"SHAPE_DONE {i + 1}/{len(sids)} {sid} gt_frac={rec['gt_frac']:.4f} "
              f"iou@0.5={rec['iou_by_thr']['0.5']:.4f}(+-{rec['iou_std_by_thr']['0.5']:.4f}) "
              f"elapsed={time.time() - t0:.1f}s", flush=True)

    n_ok = len(summary)
    print(f"DUMP_DONE n={n_ok} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
