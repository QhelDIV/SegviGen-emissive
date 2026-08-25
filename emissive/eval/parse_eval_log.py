"""Recover per-shape eval rows from a verbose eval log, for points that predate
out_json["per_sample"].

Why this exists. Until 2026-08-25 out_json carried only aggregates, so every
historical eval point lost its per-shape data when the process exited, and a paired
comparison against those points looked like it needed the eval re-run at hours per
point. It does not: evaluate_split already PRINTS every row when verbose=True, and
those lines carry sid, gt_frac, and per-threshold mean and across-draw std:

    00ff03ac403141dbadcbfd943dc3ff4c gt_frac=0.001 IoU@0.2=0.000+-0.000 ... IoU@0.5=0.000+-0.000

So the whole campaign's inferential basis is recoverable from surviving logs at zero
GPU cost. This emits rows in the same shape evaluate_split builds, so a retrofitted
point and a natively written one can be analysed by the same code.

What it CANNOT recover: bucket_frac (never printed) and the raw per-draw IoUs (only
their mean and std are printed). Rows are marked source="log" so nothing downstream
mistakes a retrofit for a native row.

  python emissive/eval/parse_eval_log.py <log> [<log> ...]            # summary per log
  python emissive/eval/parse_eval_log.py --json out.json <log>        # per_sample rows

As a library:
  from parse_eval_log import parse_rows, summarize
"""
import argparse
import json
import math
import re
import statistics
import sys

ROW = re.compile(r"^([0-9a-f]{32}) gt_frac=([\d.]+)(.*)$")
CELL = re.compile(r"IoU@(0\.\d)=([\d.]+)(?:±|\+-)?([\d.]+)?")


def parse_rows(path):
    """Per-shape rows from one eval log, shaped like evaluate_split's per_sample."""
    rows = []
    for line in open(path, errors="ignore"):
        m = ROW.match(line.strip())
        if not m:
            continue
        sid, gt_frac, rest = m.group(1), float(m.group(2)), m.group(3)
        iou, std = {}, {}
        for t, mu, sd in CELL.findall(rest):
            iou[t] = float(mu)
            std[t] = float(sd) if sd else 0.0
        if not iou:
            continue
        best = max(iou, key=iou.get)
        rows.append({"sid": sid, "gt_frac": gt_frac, "iou_by_thr": iou,
                     "iou_std_by_thr": std, "best_iou": iou[best], "best_thr": float(best),
                     "bucket_frac": None, "source": "log"})
    return rows


def summarize(rows, thr="0.5", draws=None, ddof=1):
    """The two standard errors, per the definitions agreed 2026-08-25.

    They answer different questions and neither is a substitute for the other:
      se_rerun         conditional on this frozen shape set. Only draw noise is
                       random. Uses the ROOT MEAN SQUARE of the per-shape draw
                       stds, because variances add and standard deviations do not;
                       the mean form understates it by about 2x on these sets.
      se_unpaired_full general claim about shapes like these, against a fixed
                       constant. Between-shape variance dominates. Draw noise is
                       ALREADY inside each per-shape mean and so already inside
                       this; adding it again double counts."""
    X = [r["iou_by_thr"][thr] for r in rows if thr in r["iou_by_thr"]]
    S = [r["iou_std_by_thr"].get(thr, 0.0) for r in rows if thr in r["iou_by_thr"]]
    n = len(X)
    if n < 2:
        return None
    between = statistics.stdev(X) if ddof == 1 else statistics.pstdev(X)
    out = {"n_shapes": n, "thr": thr, "mean_iou": statistics.mean(X),
           "between_shape_std": between, "se_unpaired_full": between / math.sqrt(n),
           "draw_std_mean_form": statistics.mean(S),
           "rms_draw_std": math.sqrt(statistics.mean(s * s for s in S))}
    if draws:
        out["draws"] = draws
        out["se_rerun"] = out["rms_draw_std"] / math.sqrt(n * draws)
        out["se_rerun_mean_form_DO_NOT_USE"] = out["draw_std_mean_form"] / math.sqrt(n * draws)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--thr", default="0.5")
    ap.add_argument("--draws", type=int, default=None,
                    help="draws per shape (K). Needed for se_rerun; it is not in the log.")
    ap.add_argument("--json", default=None, help="write the per-shape rows here")
    args = ap.parse_args()
    allrows = {}
    for path in args.logs:
        rows = parse_rows(path)
        allrows[path] = rows
        s = summarize(rows, args.thr, args.draws)
        print(f"== {path.split('/')[-1]}  rows={len(rows)}")
        if not s:
            print("   too few rows to summarize")
            continue
        print(f"   mean IoU@{args.thr}      {s['mean_iou']:.4f}   n={s['n_shapes']}")
        print(f"   between_shape_std    {s['between_shape_std']:.4f}")
        print(f"   se_unpaired_full     {s['se_unpaired_full']:.5f}")
        if args.draws:
            print(f"   se_rerun (rms form)  {s['se_rerun']:.5f}")
            print(f"   se_rerun (mean form) {s['se_rerun_mean_form_DO_NOT_USE']:.5f}"
                  f"   <- understates by {s['se_rerun']/s['se_rerun_mean_form_DO_NOT_USE']:.2f}x")
    if args.json:
        json.dump(allrows if len(allrows) > 1 else next(iter(allrows.values())),
                  open(args.json, "w"), indent=2)
        print(f"\nwrote rows -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
