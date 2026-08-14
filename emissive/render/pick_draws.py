"""Pick the best draw per shape from an alldraws dump: read every
{sid}__draw{k}.npz in raw_real/ and ema_real/, compute frac@0.5, and pick
the draw (across BOTH weight families) whose frac is closest to GT frac
among NON-DEGENERATE draws (frac > 0, i.e. not a totally empty prediction).
If a shape has no non-degenerate draw across both families, it is flagged
for a fresh-seed rescue (16 seeds) rather than silently accepted.

Usage:
  python pick_draws.py --raw_dir <raw_real dir> --ema_dir <ema_real dir> \
      --sids sid1,sid2,... --out picks.json
"""
import argparse
import glob
import json
import os

import numpy as np


def load_draws(d, sid):
    """[(weight, draw_idx, frac, gt_frac, npz_path)] for every draw found."""
    out = []
    for p in sorted(glob.glob(os.path.join(d, f"{sid}__draw*.npz"))):
        k = int(os.path.basename(p).split("__draw")[1].split(".npz")[0])
        z = np.load(p)
        frac = float((z["pred_bc"] > 0.5).mean())
        gt_frac = float(z["gt_e"].astype(bool).mean())
        out.append((k, frac, gt_frac, p))
    # draw 0 is ALSO saved as {sid}.npz without a __drawN suffix; that's the
    # same content as __draw0.npz, so no separate entry needed here.
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--ema_dir", required=True)
    ap.add_argument("--sids", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sids = args.sids.split(",")
    picks = {}
    for sid in sids:
        raw_draws = load_draws(args.raw_dir, sid)
        ema_draws = load_draws(args.ema_dir, sid)
        if not raw_draws and not ema_draws:
            picks[sid] = {"status": "MISSING_DUMP", "table": []}
            continue
        gt_frac = (raw_draws or ema_draws)[0][2]
        table = ([{"weight": "raw", "draw": k, "frac": f} for k, f, _, _ in raw_draws]
                + [{"weight": "ema", "draw": k, "frac": f} for k, f, _, _ in ema_draws])
        candidates = ([("raw", k, f, p) for k, f, _, p in raw_draws if f > 0]
                      + [("ema", k, f, p) for k, f, _, p in ema_draws if f > 0])
        if not candidates:
            picks[sid] = {"status": "NO_NONDEGENERATE_DRAW", "gt_frac": gt_frac, "table": table}
            continue
        weight, draw, frac, npz_path = min(candidates, key=lambda c: abs(c[2] - gt_frac))
        picks[sid] = {"status": "OK", "gt_frac": gt_frac, "picked_weight": weight,
                      "picked_draw": draw, "picked_frac": frac,
                      "abs_diff": abs(frac - gt_frac), "npz_path": npz_path,
                      "table": table}
        print(f"{sid}: picked {weight} draw{draw} frac={frac:.4f} "
              f"(GT={gt_frac:.4f}, |diff|={abs(frac-gt_frac):.4f})", flush=True)

    n_ok = sum(1 for v in picks.values() if v["status"] == "OK")
    n_bad = sum(1 for v in picks.values() if v["status"] != "OK")
    print(f"PICK_SUMMARY n_ok={n_ok} n_needs_rescue={n_bad} / {len(sids)}", flush=True)
    for sid, v in picks.items():
        if v["status"] != "OK":
            print(f"  NEEDS_RESCUE {sid}: {v['status']}", flush=True)

    with open(args.out, "w") as f:
        json.dump(picks, f, indent=1)
    print(f"PICKS_WRITTEN {args.out}", flush=True)


if __name__ == "__main__":
    main()
