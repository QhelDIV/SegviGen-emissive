"""
Compare the real-cond run against the zero-cond run on the frozen 300 val shapes,
and write the cond_invalid_sids.json sidecar.

Joins on sid over the two runs' `per_shape_diag3` (both keyed by sid, same 300
shapes via outputs/three_ckpt_eval/eval300_sids_frozen.json), so it touches neither
diagnostics_run.py nor diag3_only.py -- per three-ckpt, the "conditioning invalid"
notion is a property of THIS run, not of the data or of any checkpoint, so it stays
out of the shared stratification and lives here as a sid join.

Reports each aggregate three ways, because the five shapes with unusable thumbnails
are two different problems and collapsing them into one "excluded" number would read
as one category:
  - all 300
  - excluding the 3 PLACEHOLDER shapes (conditioning is a constant vector identical
    across all three, carrying no information about the shape -- a third condition,
    neither real conditioning nor the zero baseline)
  - excluding those 3 plus the 2 SPECK shapes (conditioning is genuine but nearly
    contentless: the object is a few pixels in a badly framed render)

Reference points measured by three-ckpt on the zero-cond run, at threshold 0.5:
0.0885 over all 300, 0.0892 excluding all five (delta +0.0007, i.e. these shapes are
not intrinsically easy or hard), and 0.0358 on the fully-textured nonzero-GT subset.
Note 1b98038d is emissive_frac 1.0000 and scores 0.2267 zero-cond: on a fully
emissive shape the answer is "paint everything" and needs no conditioning at all, so
a gain there is not evidence conditioning helped. It is printed on its own line.

Usage:
  python cond_compare_300.py \
      --cond_run  outputs/emis_72k_cond/run1 \
      --zero_run  outputs/emis_72k_unfilt/run1 \
      --out       outputs/emis_72k_cond/run1/cond_vs_zero_300.json
"""
import os
import json
import argparse

# The five frozen-300 sids whose TexVerse thumbnail cannot carry conditioning,
# split by cause. Determined from the thumbnails themselves (build_cond_thumbnail.py
# --repair records which fallback each took); the three placeholders are byte-
# identical to each other, md5 1777c382b0b8...
PLACEHOLDER_SIDS = [
    "013bdc7019584f0b8d8b5264d5da4dcc",
    "cfd4e277f0054c6783110a5db69e2df1",
    "f14d122e015445c28046474f32144af9",
]
SPECK_SIDS = [
    "1b98038d95c845068926db741c29b9d8",
    "30567c38761642f2988555df33e04bba",
]
# fully emissive (emissive_frac 1.0000): "paint everything" is free, so this shape's
# score is not evidence about conditioning either way. Reported separately.
FREE_ANSWER_SIDS = ["1b98038d95c845068926db741c29b9d8"]


def load_per_shape(run_dir):
    """per_shape_diag3 from a run's diagnostics.json, sid -> {iou_by_thr_mean: {...}}."""
    p = os.path.join(run_dir, "diagnostics.json")
    d = json.load(open(p))
    for key in ("per_shape_diag3", "per_shape_3"):
        if key in d:
            return d[key]
    raise KeyError(f"no per-shape diag3 block in {p} (keys: {sorted(d)[:12]})")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond_run", required=True)
    ap.add_argument("--zero_run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", default="0.5")
    args = ap.parse_args()

    cond = load_per_shape(args.cond_run)
    zero = load_per_shape(args.zero_run)
    sids = sorted(set(cond) & set(zero))
    only_cond, only_zero = sorted(set(cond) - set(zero)), sorted(set(zero) - set(cond))
    # the whole point is a matched comparison; a silent partial overlap would
    # produce two means over different shape sets that look comparable.
    assert not only_cond and not only_zero, (
        f"shape sets differ: {len(only_cond)} only in cond run, {len(only_zero)} only in zero run")
    print(f"MATCHED n={len(sids)} shapes", flush=True)

    def iou(block, s):
        return block[s]["iou_by_thr_mean"][args.thr]

    subsets = {
        "all_300": sids,
        "excl_placeholder": [s for s in sids if s not in set(PLACEHOLDER_SIDS)],
        "excl_placeholder_and_speck": [s for s in sids
                                       if s not in set(PLACEHOLDER_SIDS) | set(SPECK_SIDS)],
    }
    out = {"thr": args.thr, "n_matched": len(sids),
           "placeholder_sids": PLACEHOLDER_SIDS, "speck_sids": SPECK_SIDS,
           "subsets": {}, "per_shape_of_interest": {}}
    for name, ss in subsets.items():
        mc, mz = mean(iou(cond, s) for s in ss), mean(iou(zero, s) for s in ss)
        out["subsets"][name] = {"n": len(ss), "cond": mc, "zero": mz, "delta": mc - mz}
        print(f"{name:30s} n={len(ss):4d}  cond={mc:.4f}  zero={mz:.4f}  delta={mc - mz:+.4f}",
              flush=True)

    for s in PLACEHOLDER_SIDS + SPECK_SIDS:
        if s in cond:
            out["per_shape_of_interest"][s] = {
                "kind": "placeholder" if s in PLACEHOLDER_SIDS else "speck",
                "free_answer_fully_emissive": s in FREE_ANSWER_SIDS,
                "cond": iou(cond, s), "zero": iou(zero, s)}
            k = out["per_shape_of_interest"][s]
            note = "  [fully emissive: answer is free, not evidence about conditioning]" \
                if s in FREE_ANSWER_SIDS else ""
            print(f"  {s[:8]} {k['kind']:12s} cond={k['cond']:.4f} zero={k['zero']:.4f}{note}",
                  flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    sidecar = os.path.join(os.path.dirname(args.out), "cond_invalid_sids.json")
    json.dump({"placeholder": PLACEHOLDER_SIDS, "speck": SPECK_SIDS,
               "note": "frozen-300 shapes whose TexVerse thumbnail cannot carry conditioning; "
                       "placeholder = constant image identical across all three (md5 "
                       "1777c382b0b8...), speck = real render with the object a few pixels "
                       "wide. Both received a real (nonzero) cond.pth so the shape set stays "
                       "identical to the zero-cond run."},
              open(sidecar, "w"), indent=2)
    print(f"WROTE {args.out}\nWROTE {sidecar}", flush=True)


if __name__ == "__main__":
    main()
