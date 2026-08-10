"""
Run on the cluster (login node OK for meta.json walk; needs `trellis2` conda env for the
torch.load latent-token-count subsample). Computes:
  - per-split (train_1k, val_96) emissive_frac distribution: mean/median/%zero/%>0.3 +
    4-bucket counts (0 / (0,0.05] / (0.05,0.3] / >0.3)
  - a random ~200-sample latent-token-count subsample (coords length from shape_slat.pth),
    pooled across both splits
  - 48 gallery sids from train_1k: 12 per bucket, seeded random (documented seed)
Writes one JSON to stdout-adjacent file for a single small rsync back.

  python compute_dataset_stats.py --out dataset_stats.json
"""
import os, json, random, argparse

ROOT = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset"
SPLITS = ["train_1k", "val_96"]
SEED = 42
N_TOKEN_SUBSAMPLE = 200
N_PER_BUCKET = 12

BUCKETS = [("0", lambda f: f == 0), ("(0,0.05]", lambda f: 0 < f <= 0.05),
           ("(0.05,0.3]", lambda f: 0.05 < f <= 0.3), (">0.3", lambda f: f > 0.3)]


def load_split(split):
    sdir = os.path.join(ROOT, split)
    out = []
    for sid in sorted(os.listdir(sdir)):
        mp = os.path.join(sdir, sid, "meta.json")
        if not os.path.exists(mp):
            continue
        try:
            m = json.load(open(mp))
        except Exception:
            continue
        out.append((sid, float(m["emissive_frac"])))
    return out


def stats_for(samples):
    import statistics
    fracs = [f for _, f in samples]
    n = len(fracs)
    bucket_counts = {}
    for name, pred in BUCKETS:
        bucket_counts[name] = sum(1 for f in fracs if pred(f))
    return {
        "n": n,
        "mean": statistics.fmean(fracs) if fracs else None,
        "median": statistics.median(fracs) if fracs else None,
        "pct_zero": 100.0 * bucket_counts["0"] / n if n else None,
        "pct_gt0p3": 100.0 * bucket_counts[">0.3"] / n if n else None,
        "bucket_counts": bucket_counts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    per_split_samples = {s: load_split(s) for s in SPLITS}
    per_split_stats = {s: stats_for(per_split_samples[s]) for s in SPLITS}

    # gallery: 48 from train_1k, 12/bucket, seeded random
    rng = random.Random(SEED)
    train_samples = per_split_samples["train_1k"]
    gallery = {}
    for name, pred in BUCKETS:
        pool = [sid for sid, f in train_samples if pred(f)]
        pool_sorted = sorted(pool)  # deterministic order before sampling
        k = min(N_PER_BUCKET, len(pool_sorted))
        gallery[name] = rng.sample(pool_sorted, k)

    # latent token-count subsample (pooled across both splits), via shape_slat.pth coords
    all_samples = [(s, sid) for s in SPLITS for sid, f in per_split_samples[s]]
    rng2 = random.Random(SEED)
    pool = sorted(all_samples)
    sub = rng2.sample(pool, min(N_TOKEN_SUBSAMPLE, len(pool)))
    token_counts = []
    try:
        import torch
        for split, sid in sub:
            p = os.path.join(ROOT, split, sid, "shape_slat.pth")
            if not os.path.exists(p):
                continue
            d = torch.load(p, map_location="cpu")
            token_counts.append(int(d["coords"].shape[0]))
    except Exception as e:
        print(f"[WARN] token-count subsample failed: {e}")

    out = {
        "seed": SEED,
        "per_split_stats": per_split_stats,
        "per_split_fracs": {s: [f for _, f in per_split_samples[s]] for s in SPLITS},
        "gallery_sids": gallery,
        "gallery_bucket_pool_sizes": {name: sum(1 for _, f in train_samples if pred(f))
                                       for name, pred in BUCKETS},
        "token_counts_subsample": token_counts,
        "token_counts_n_requested": N_TOKEN_SUBSAMPLE,
    }
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")
    print(json.dumps({"per_split_stats": per_split_stats,
                       "gallery_bucket_pool_sizes": out["gallery_bucket_pool_sizes"],
                       "n_token_counts": len(token_counts)}, indent=2))


if __name__ == "__main__":
    main()
