"""
Assemble the three-checkpoint comparison from the three schema-identical diagnostics.json
files (one per checkpoint, all on the same 300 val_72k shapes, cond zero, draws 3, seed 0).

Emits the flat aggregate, the per-shape draw std, and the stratification by GT
emissive_frac into the same six buckets diagnostics_run.py uses. The flat mean on this
data is dominated by the [0,0.01) bucket (121 of 300 shapes, 28 of them exact-zero GT
scoring 1.0 by the 0/0 convention), so the stratified rows are the reportable ones and
the flat mean is carried only to show what it hides.

Run locally -- reads json, no GPU, no torch.

  python emissive/diagnostics/three_ckpt_table.py --out_dir outputs/three_ckpt_eval
"""
import os
import json
import argparse
import statistics

BUCKETS = ["[0,0.01)", "[0.01,0.05)", "[0.05,0.2)", "[0.2,0.5)", "[0.5,0.8)", "[0.8,1.0]"]
THRS = ["0.2", "0.3", "0.4", "0.5"]

MODELS = [
    ("emis_1k_w1", "1k, pos_weight 1", "OOD: Path A data, trained cond=real"),
    ("emis_1k_w5", "1k, pos_weight 5", "OOD: Path A data, trained cond=real"),
    ("emis_72k_unfilt", "72k unfiltered, pos_weight 5", "in-distribution, trained cond=zero"),
]


def load(out_dir, tag, ref_dir):
    p = os.path.join(out_dir, tag, "diagnostics.json")
    if not os.path.exists(p) and tag == "emis_72k_unfilt":
        p = os.path.join(ref_dir, "diagnostics.json")
    if not os.path.exists(p):
        return None, p
    return json.load(open(p)), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ref_dir", required=True,
                    help="emis_72k_unfilt/run1, whose diagnostics.json is the reference run")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    loaded = {}
    for tag, _, _ in MODELS:
        d, p = load(args.out_dir, tag, args.ref_dir)
        if d is None:
            print(f"MISSING {tag}: {p}")
        else:
            loaded[tag] = d
            print(f"loaded {tag}: {p}")
    if not loaded:
        raise SystemExit("no diagnostics found")

    # shape-set identity is a claim, not an assumption -- check it
    sid_sets = {t: set(d["per_shape_diag3"].keys()) for t, d in loaded.items()}
    ref = next(iter(sid_sets.values()))
    same = all(s == ref for s in sid_sets.values())
    print(f"\nSHAPE_SET_IDENTICAL={same}  n={len(ref)}")
    if not same:
        for t, s in sid_sets.items():
            print(f"  {t}: n={len(s)} sym_diff_vs_first={len(s ^ ref)}")
        raise SystemExit("shape sets differ -- the table would not be a common footing")

    print("\n=== flat aggregate over 300 shapes (IoU, mean over 3 draws per shape) ===")
    hdr = f"{'model':<18}" + "".join(f"{'@' + t:>10}" for t in THRS) + f"{'median@0.5':>12}{'draw_std':>10}"
    print(hdr)
    for tag, label, note in MODELS:
        if tag not in loaded:
            continue
        d = loaded[tag]
        row = f"{tag:<18}"
        for t in THRS:
            row += f"{d['diag3_model_summary_by_thr'][t]['mean']:>10.4f}"
        row += f"{d['diag3_model_summary_by_thr']['0.5']['median']:>12.4f}"
        row += f"{d['diag3_draw_std_summary_by_thr']['0.5']['mean']:>10.4f}"
        print(row + f"   [{note}]")

    print("\n=== stratified by GT emissive_frac (model IoU @0.5) ===")
    print(f"{'bucket':<14}{'n':>5}" + "".join(f"{t:>16}" for t, _, _ in MODELS)
          + f"{'ceiling':>10}{'all_zero':>10}{'albedo_pct':>12}")
    for b in BUCKETS:
        ent_ref = loaded[MODELS[-1][0]]["diag4_stratified"].get(b) if MODELS[-1][0] in loaded else None
        any_ent = next(loaded[t]["diag4_stratified"][b] for t, _, _ in MODELS if t in loaded)
        line = f"{b:<14}{any_ent['n']:>5}"
        for tag, _, _ in MODELS:
            if tag not in loaded:
                line += f"{'-':>16}"
                continue
            e = loaded[tag]["diag4_stratified"][b]
            v = e.get("model_iou_at_0.5")
            line += f"{v:>16.4f}" if v is not None else f"{'n/a':>16}"
        for key, w in [("ceiling_iou_at_0.5", 10), ("baseline_all_zero", 10),
                       ("baseline_pbr_heuristic_global_pct", 12)]:
            v = (ent_ref or any_ent).get(key)
            line += f"{v:>{w}.4f}" if v is not None else f"{'n/a':>{w}}"
        print(line)

    print("\n=== per-shape draw std @0.5 (sampling noise, mean over shapes) ===")
    for tag, _, _ in MODELS:
        if tag in loaded:
            s = loaded[tag]["diag3_draw_std_summary_by_thr"]["0.5"]
            print(f"  {tag:<18} mean={s['mean']:.4f} median={s['median']:.4f} p90={s['p90']:.4f}")

    print("\n=== reported best val IoU vs re-measured (the point of the exercise) ===")
    for tag, _, _ in MODELS:
        if tag not in loaded:
            continue
        d = loaded[tag]
        tc = d.get("train_curve") or []
        vals = [e.get("val_iou") for e in tc if e.get("val_iou") is not None]
        best = max(vals) if vals else None
        n_val = None
        print(f"  {tag:<18} train_curve best val_iou (N=16 quick-val) = "
              f"{best if best is None else round(best, 4)}   "
              f"re-measured on 300 shapes @0.5 = {d['diag3_model_summary_by_thr']['0.5']['mean']:.4f}"
              f"  (median {d['diag3_model_summary_by_thr']['0.5']['median']:.4f})")
        _ = n_val

    if args.json_out:
        out = {tag: {"flat": loaded[tag]["diag3_model_summary_by_thr"],
                     "draw_std": loaded[tag]["diag3_draw_std_summary_by_thr"],
                     "stratified": {b: {"n": loaded[tag]["diag4_stratified"][b]["n"],
                                        "model_iou_at_0.5": loaded[tag]["diag4_stratified"][b].get("model_iou_at_0.5"),
                                        "model_best_iou": loaded[tag]["diag4_stratified"][b].get("model_best_iou")}
                                    for b in BUCKETS}}
               for tag in loaded}
        json.dump(out, open(args.json_out, "w"), indent=1)
        print(f"\nWROTE {args.json_out}")


if __name__ == "__main__":
    main()
