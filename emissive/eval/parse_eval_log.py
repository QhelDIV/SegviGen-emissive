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

What it CANNOT recover:
  * bucket_frac, never printed
  * the raw per-draw IoUs, only their mean and std are printed
  * gt_frac and the IoUs at better than THREE DECIMALS, because that is what the
    log prints

Rows are marked source="log" so nothing downstream mistakes a retrofit for a
native row.

THE 3-DECIMAL LIMIT IS HARMLESS EXCEPT IN ONE PLACE, and that place matters.
Rounding perturbs each value by at most 5e-4, which is nothing against IoUs
spanning 0 to 1 and averages out across shapes: the aggregates and both standard
errors are unaffected to well past the digits anyone quotes. But the NONZERO
SUBSET is a threshold at exactly zero, and a shape whose true gt_frac is 3e-4
prints as 0.000 and is then excluded. Measured on eval 249199: this parser finds
79 shapes with gt_frac>0 where the native per_sample rows find 85. So a
nonzero-subset comparison mixing a retrofitted arm with a native one silently
compares different subsets.

Take the nonzero mask from a native json when one exists, and when it does not,
say which subset a retrofitted number was computed over. nonzero_rows() below
warns rather than letting this pass quietly.

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
                     "bucket_frac": None, "source": "log", "gt_frac_precision": "3dp"})
    return rows


def nonzero_rows(rows, warn=True):
    """The gt_frac>0 subset, with a warning when the rows came from a log.

    A threshold at exactly zero is the one statistic the log's 3-decimal printing
    can change: true coverage below 5e-4 prints as 0.000 and drops out. On eval
    249199 that is 79 shapes here against 85 from the native rows."""
    out = [r for r in rows if r["gt_frac"] > 0]
    if warn and any(r.get("gt_frac_precision") == "3dp" for r in rows):
        print(f"[warn] nonzero subset taken from log-derived rows: gt_frac is rounded to "
              f"3 decimals, so shapes with true coverage below 5e-4 are excluded here but "
              f"included by a native json. Got {len(out)} of {len(rows)}. Use a native "
              f"per_sample for the mask if one exists, or state which subset this is.",
              file=sys.stderr)
    return out


def summarize(rows, thr="0.5", draws=None, ddof=1, draw_ddof_correct=True):
    """The two standard errors, per the definitions agreed 2026-08-25.

    They answer different questions and neither is a substitute for the other:
      se_rerun         conditional on this frozen shape set. Only draw noise is
                       random. Uses the ROOT MEAN SQUARE of the per-shape draw
                       stds, because variances add and standard deviations do not;
                       the mean form understates it by about 2x on these sets.
      se_unpaired_full general claim about shapes like these, against a fixed
                       constant. Between-shape variance dominates. Draw noise is
                       ALREADY inside each per-shape mean and so already inside
                       this; adding it again double counts.

    draw_ddof_correct applies sqrt(K/(K-1)) to the per-shape draw stds. The stored
    iou_std_by_thr uses np.std's default ddof=0, which over K draws biases each s_i
    low by sqrt((K-1)/K) and so biases se_rerun toward overconfidence. Convention
    agreed 2026-08-25 with agentic-train: leave the STORED value at ddof=0 so every
    existing json stays comparable, and correct it in the DERIVED se. This flag
    exists so that convention is visible and reversible rather than baked in."""
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
        # ddof correction first, then rms, then /sqrt(nK). Every s_i scales by the
        # same factor so the order does not matter numerically, but doing it here
        # keeps the stored rms_draw_std reporting what the log literally contains.
        bump = math.sqrt(draws / (draws - 1)) if (draw_ddof_correct and draws > 1) else 1.0
        out["draw_ddof_correction"] = bump
        out["se_rerun"] = bump * out["rms_draw_std"] / math.sqrt(n * draws)
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
            print(f"   se_rerun (rms, ddof-corrected x{s['draw_ddof_correction']:.3f})  {s['se_rerun']:.5f}")
            rms_only = s["rms_draw_std"] / math.sqrt(s["n_shapes"] * args.draws)
            print(f"   se_rerun (mean form) {s['se_rerun_mean_form_DO_NOT_USE']:.5f}"
                  f"   <- understates the agreed se_rerun by "
                  f"{s['se_rerun']/s['se_rerun_mean_form_DO_NOT_USE']:.2f}x "
                  f"({rms_only/s['se_rerun_mean_form_DO_NOT_USE']:.2f}x of it from mean-vs-rms, "
                  f"{s['draw_ddof_correction']:.3f}x from ddof)")
    if args.json:
        json.dump(allrows if len(allrows) > 1 else next(iter(allrows.values())),
                  open(args.json, "w"), indent=2)
        print(f"\nwrote rows -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
