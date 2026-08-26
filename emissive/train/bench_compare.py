"""Compare a fresh bench_results.jsonl row against a stored baseline snapshot,
so a future change to train_emissive.py can be regression-tested for speed the
same way parity tests it for numerics.

A "stored baseline" is one JSON object (see --write_baseline) holding the
recommended config's measured shapes_per_s, peak_vram_gb and phase_ms -- the
same shape as one line of bench_results.jsonl. This script does not run
anything: point it at a bench_results.jsonl produced by train_emissive.py
--bench_steps (see that flag's help) and a baseline file, and it reports
whether the matching config regressed.

Usage:
  # after a grid run, snapshot the recommended config as the baseline
  python bench_compare.py --write_baseline --results bench_results.jsonl \
      --tag p2_condslim_ckptoff_ga1 --out bench_baseline.json

  # later, after touching train_emissive.py, rerun that one config and check
  python bench_compare.py --results new_bench_results.jsonl \
      --tag p2_condslim_ckptoff_ga1 --baseline bench_baseline.json
"""
import argparse
import json
import sys

REGRESSION_PCT = 5.0   # flag anything more than this much slower than baseline


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_by_tag(rows, tag):
    matches = [r for r in rows if r.get("tag") == tag]
    if not matches:
        raise SystemExit(f"no row with tag={tag!r} in the given results file "
                         f"(tags present: {sorted({r.get('tag') for r in rows})})")
    return matches[-1]   # most recent if duplicated


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="bench_results.jsonl to read from")
    ap.add_argument("--tag", required=True, help="--bench_tag value identifying the config to check")
    ap.add_argument("--write_baseline", action="store_true", default=False,
                    help="write the matching row from --results as a new baseline (to --out) "
                         "instead of comparing")
    ap.add_argument("--out", default=None, help="with --write_baseline: path to write")
    ap.add_argument("--baseline", default=None, help="without --write_baseline: baseline file to compare against")
    ap.add_argument("--regression_pct", type=float, default=REGRESSION_PCT,
                    help=f"flag if shapes_per_s drops more than this percent vs baseline (default {REGRESSION_PCT})")
    args = ap.parse_args()

    rows = load_jsonl(args.results)
    row = find_by_tag(rows, args.tag)

    if args.write_baseline:
        if not args.out:
            raise SystemExit("--write_baseline requires --out")
        with open(args.out, "w") as f:
            json.dump(row, f, indent=2)
        print(f"[bench_compare] wrote baseline for tag={args.tag!r}: "
             f"{row['shapes_per_s']:.3f} shapes/s, {row['peak_vram_gb']:.2f} GiB peak -> {args.out}")
        return

    if not args.baseline:
        raise SystemExit("comparing requires --baseline (or pass --write_baseline to create one)")
    with open(args.baseline) as f:
        base = json.load(f)
    if base.get("tag") != args.tag:
        print(f"[bench_compare] WARNING baseline tag={base.get('tag')!r} != requested tag={args.tag!r}; "
             f"comparing anyway since both are keyed by --tag on the caller's side", file=sys.stderr)

    base_sps = base["shapes_per_s"]
    new_sps = row["shapes_per_s"]
    pct_change = 100.0 * (new_sps - base_sps) / base_sps
    regressed = pct_change < -args.regression_pct

    base_peak = base["peak_vram_gb"]
    new_peak = row["peak_vram_gb"]
    peak_pct = 100.0 * (new_peak - base_peak) / base_peak if base_peak else 0.0

    print(f"[bench_compare] tag={args.tag}")
    print(f"  shapes/s   baseline={base_sps:.3f}  new={new_sps:.3f}  change={pct_change:+.1f}%")
    print(f"  peak VRAM  baseline={base_peak:.2f}GiB  new={new_peak:.2f}GiB  change={peak_pct:+.1f}%")
    for phase, ms in row.get("phase_ms_per_microstep", {}).items():
        base_ms = base.get("phase_ms_per_microstep", {}).get(phase)
        if ms is not None and base_ms is not None:
            print(f"  {phase:10s} baseline={base_ms:.2f}ms  new={ms:.2f}ms")
    if regressed:
        print(f"[bench_compare] REGRESSION: shapes/s dropped {-pct_change:.1f}% "
             f"(threshold {args.regression_pct}%)")
        sys.exit(1)
    print(f"[bench_compare] OK: within {args.regression_pct}% of baseline")


if __name__ == "__main__":
    main()
