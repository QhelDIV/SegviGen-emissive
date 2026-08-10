"""
Run ONLY diagnostic 3 (model sampling + per-draw IoU) of diagnostics_run.py, for a
checkpoint other than the one whose diagnostics.json already exists.

Diagnostics 1 and 2 (VAE round-trip ceiling, trivial baselines) are properties of the
DATA, not of any checkpoint, so re-running them per checkpoint would burn ~5 min of GPU
per model to recompute a bit-identical answer. This script reuses the existing
`_diag12_raw_seed{seed}_n{n}.json` cache and writes only the diag-3 half, in exactly the
file name and schema `diagnostics_run.py --aggregate_only` expects. The shape list comes
from `diagnostics_run.select_sids`, so the shapes are the same 300 by construction rather
than by coincidence.

Conditioning is a flag (--cond). diagnostics_run.py hardcodes "zero"; that is correct for
the unconditioned dataset, which carries no cond.pth, but wrong for a checkpoint trained with
real conditioning, which would be handicapped rather than measured.

Usage (GPU node, trellis2 env):
  python code/diag3_only.py --dataset dataset_direct --split val_72k \
      --ckpt outputs/emis_1k_w1/best.ckpt --out_dir outputs/three_ckpt_eval/emis_1k_w1 \
      --n 300 --draws 3 --steps 12 --seed 0
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

import torch
from trellis2 import models
from inference_full import Gen3DSeg

import eval_emissive as ee
from diagnostics_run import select_sids, THRS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val_72k")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond", choices=["zero", "real"], default="zero",
                    help="explicit, because a checkpoint trained with real conditioning scored "
                         "at zero-cond is being handicapped rather than measured. "
                         "diagnostics_run.py hardcodes 'zero' and cannot do this.")
    ap.add_argument("--sids_file", default=None,
                    help="frozen sid list (json: a list, or an object with a 'sids' key). "
                         "STRONGLY preferred over re-deriving: select_sids() shuffles "
                         "sorted(os.listdir(split)), so it returns a DIFFERENT 300 shapes if the "
                         "split directory's contents differ at all between two runs. Two models "
                         "scored on re-derived lists are not necessarily scored on the same shapes.")
    args = ap.parse_args()
    device = "cuda"

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir,
                            f"_diag3_raw_seed{args.seed}_n{args.n}_draws{args.draws}.json")

    if args.sids_file:
        j = json.load(open(args.sids_file))
        sids = j["sids"] if isinstance(j, dict) else j
        sdir = os.path.join(args.dataset, args.split)
        missing = [s for s in sids if not os.path.isdir(os.path.join(sdir, s))]
        if missing:
            raise SystemExit(f"{len(missing)} frozen sids absent from {sdir}: {missing[:5]}")
        print(f"SELECT_OK n={len(sids)} from frozen list {args.sids_file}", flush=True)
    else:
        sids, skipped = select_sids(args.dataset, args.split, args.n, args.seed)
        print(f"SELECT_OK n={len(sids)} skipped={len(skipped)} (RE-DERIVED, not frozen)", flush=True)

    models_d = ee.load_eval_models(device)
    print("ENV_OK models loaded", flush=True)

    flow = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")
    gen = Gen3DSeg(flow).to(device)
    sd = torch.load(args.ckpt, map_location=device)["state_dict"]
    sd = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in sd.items()])
    missing, unexpected = gen.load_state_dict(sd, strict=False)
    print(f"CKPT_LOADED missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing or unexpected:
        # strict=False above is only so the mismatch can be REPORTED rather than raised as
        # a bare KeyError; a real mismatch still has to stop the run.
        print(f"CKPT_MISMATCH missing[:10]={list(missing)[:10]} "
              f"unexpected[:10]={list(unexpected)[:10]}", flush=True)
        raise SystemExit("checkpoint does not match the current Gen3DSeg architecture")
    gen.eval()

    sdir = os.path.join(args.dataset, args.split)
    per_shape_3 = {}
    t0 = time.time()
    for i, sid in enumerate(sids):
        d = os.path.join(sdir, sid)
        res = ee.eval_sample(gen, models_d, d, args.cond, device, steps=args.steps, thrs=THRS,
                             draws=args.draws, sid=sid)
        per_shape_3[sid] = {"gt_frac_decoded": res["gt_frac"],
                            "iou_by_thr_mean": {str(k): v for k, v in res["iou_by_thr"].items()},
                            "iou_by_thr_std": {str(k): v for k, v in res["iou_std_by_thr"].items()}}
        if i + 1 == 5:
            el = time.time() - t0
            print(f"DIAG3_SMOKE_TIMING n=5 elapsed={el:.1f}s "
                  f"est_total_for_{len(sids)}={el / 5 * len(sids):.1f}s", flush=True)
        if (i + 1) % 50 == 0:
            print(f"DIAG3_PROGRESS {i + 1}/{len(sids)} elapsed={time.time() - t0:.1f}s", flush=True)
    print(f"DIAG3_DONE n={len(sids)} elapsed={time.time() - t0:.1f}s", flush=True)

    json.dump({"per_shape_3": per_shape_3}, open(out_path, "w"))
    print(f"CHECKPOINT_WRITTEN {out_path}", flush=True)
    print("DIAG3_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
