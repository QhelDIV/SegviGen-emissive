"""
Merge evidence_stats.json (per-shape direct/true/somage fracs) with ab_stats.json
(per-shape overlap/agreement/IoU) into final_stats.json + headline numbers + a
picked example list. Exports somage GT npz (coords+white mask) for the picks so the
local bpy renderer can draw them without o_voxel.
"""
import os, sys, json, types, importlib, importlib.util
import numpy as np


def _cpu_io():
    if "o_voxel" not in sys.modules:
        spec = importlib.util.find_spec("o_voxel")
        pkg = types.ModuleType("o_voxel"); pkg.__path__ = spec.submodule_search_locations; pkg.__spec__ = spec
        sys.modules["o_voxel"] = pkg
    return importlib.import_module("o_voxel.io")


PD = "/3dlg-jupiter-project/lightgen/segvigen_emissive/direct_pilot"
DS = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset"
BUCKETS = ["zero", "tiny", "small", "medium", "large"]


def corr(xs, ys):
    xs, ys = np.array(xs, float), np.array(ys, float)
    if len(xs) < 2: return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def export_somage_npz(sid, out_npz):
    IO = _cpu_io()
    for split in ["train", "train_2k_ef"]:
        p = os.path.join(DS, split, sid, "output.vxz")
        if os.path.exists(p):
            coords, data = IO.read_vxz(p, num_threads=1)
            white = (data["base_color"].float().mean(dim=-1) > 127.5).cpu().numpy()
            np.savez_compressed(out_npz, coords=coords.cpu().numpy().astype(np.int32), mask=white)
            return True
    return False


def main():
    ev = {r["sid"]: r for r in json.load(open(os.path.join(PD, "evidence_stats.json")))}
    ab = json.load(open(os.path.join(PD, "ab_stats.json")))
    ab_ps = {r["sid"]: r for r in ab["per_shape"]}

    rows = []
    for sid, r in ev.items():
        a = ab_ps.get(sid, {})
        rows.append({**r, **{k: a.get(k) for k in
                     ["overlap_jaccard", "cov_direct", "cov_somage", "agree", "iou",
                      "new_only_frac", "old_only_frac"]}})

    # headline: false-positive rate on genuinely non-emissive shapes (emis texture black)
    black = [r for r in rows if r.get("emis_tex_black")]
    black_direct = [r["direct_frac"] for r in black]
    fp = dict(n=len(black),
              mean_direct_frac=float(np.mean(black_direct)) if black else 0.0,
              median_direct_frac=float(np.median(black_direct)) if black else 0.0,
              n_direct_gt_005=int(sum(x > 0.05 for x in black_direct)),
              n_direct_gt_020=int(sum(x > 0.20 for x in black_direct)),
              max_direct_frac=float(np.max(black_direct)) if black else 0.0)

    # correlations over TEXTURED shapes (drop factor-only n_emis_tex==0, unreliable true)
    tex = [r for r in rows if r["n_emis_tex"] > 0]
    cors = dict(n=len(tex),
                true_vs_direct=corr([r["true_frac"] for r in tex], [r["direct_frac"] for r in tex]),
                somage_vs_direct=corr([r["somage_frac"] or 0 for r in tex], [r["direct_frac"] for r in tex]),
                true_vs_somage=corr([r["true_frac"] for r in tex], [r["somage_frac"] or 0 for r in tex]))

    # per-bucket aggregates
    by_bucket = {}
    for b in BUCKETS:
        br = [r for r in rows if r["bucket"] == b]
        if not br: continue
        def m(k): return float(np.mean([r[k] for r in br if r.get(k) is not None])) if any(r.get(k) is not None for r in br) else None
        by_bucket[b] = dict(n=len(br), direct_frac=m("direct_frac"), somage_frac=m("somage_frac"),
                            true_frac=m("true_frac"), overlap_jaccard=m("overlap_jaccard"),
                            agree=m("agree"), iou=m("iou"),
                            new_only_frac=m("new_only_frac"), old_only_frac=m("old_only_frac"),
                            n_texblack=sum(1 for r in br if r.get("emis_tex_black")))

    # PICK examples: strongest false positives (texblack, high direct), a false negative
    # (bright texture true>0.3 but direct<0.1), and a correctly-tracked glow (true & direct high).
    fps = sorted([r for r in black], key=lambda r: -r["direct_frac"])[:3]
    fns = sorted([r for r in tex if r["true_frac"] > 0.3 and r["direct_frac"] < 0.1],
                 key=lambda r: r["direct_frac"])[:2]
    goods = sorted([r for r in tex if r["true_frac"] > 0.3 and r["direct_frac"] > 0.3],
                   key=lambda r: -r["true_frac"])[:2]
    picks = {"false_positive": [r["sid"] for r in fps],
             "false_negative": [r["sid"] for r in fns],
             "tracked": [r["sid"] for r in goods]}

    out = dict(overall=ab["overall"], false_positive=fp, correlations=cors,
               by_bucket=by_bucket, picks=picks, per_shape=rows)
    json.dump(out, open(os.path.join(PD, "final_stats.json"), "w"), indent=2)

    # export somage npz for all picks
    exdir = os.path.join(PD, "examples"); os.makedirs(exdir, exist_ok=True)
    allpicks = [s for v in picks.values() for s in v]
    for sid in allpicks:
        ok = export_somage_npz(sid, os.path.join(exdir, f"{sid}_somage.npz"))
        print(f"somage npz {sid}: {ok}")

    print("\nFALSE POSITIVE (texblack):", json.dumps(fp))
    print("CORRELATIONS:", json.dumps(cors))
    print("PICKS:", json.dumps(picks))
    print("\nBY BUCKET:")
    for b, v in by_bucket.items():
        print(f"  {b:<7} n={v['n']} direct={v['direct_frac']:.3f} somage={(v['somage_frac'] or 0):.3f} "
              f"true={(v['true_frac'] or 0):.3f} overlapJ={(v['overlap_jaccard'] or 0):.3f} "
              f"agree={(v['agree'] or 0):.3f} IoU={(v['iou'] or 0):.3f} texblack={v['n_texblack']}")
    print("MERGE_DONE")


if __name__ == "__main__":
    main()
