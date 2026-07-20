"""
A/B: direct-GLB emissive GT (emis_gt_direct.npz) vs current somage GT (output.vxz).

Per shape:
  - coord overlap: the two voxelizations come from DIFFERENT geometry provenance
    (original glb vs somage-rebuilt mesh), so the occupied surface-voxel SETS differ.
    Report |A∩B|/|A∪B| (Jaccard of occupancy) and coverage of each.
  - on the INTERSECTED coords only: emissive agreement %, IoU(emissive), and
    disagreement direction (new-only vs old-only emissive voxels).
  - stratify by the somage GT glow bucket.

Runs on the login node (CPU): o_voxel.io via the stub trick (no triton/CUDA import).
"""
import os, sys, json, types, importlib, importlib.util
import numpy as np
import torch


def _cpu_o_voxel_io():
    if "o_voxel" not in sys.modules:
        spec = importlib.util.find_spec("o_voxel")
        assert spec is not None, "o_voxel not installed (need trellis2 env)"
        pkg = types.ModuleType("o_voxel")
        pkg.__path__ = spec.submodule_search_locations
        pkg.__spec__ = spec
        sys.modules["o_voxel"] = pkg
    return importlib.import_module("o_voxel.io")


IO = _cpu_o_voxel_io()
DS = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset"


def _enc(coords):
    """pack int32 (M,3) @512 coords -> unique int64 keys for set ops."""
    c = coords.astype(np.int64)
    return (c[:, 0] * 512 + c[:, 1]) * 512 + c[:, 2]


def somage_gt(sid):
    for split in ["train", "train_2k_ef"]:
        p = os.path.join(DS, split, sid, "output.vxz")
        if os.path.exists(p):
            coords, data = IO.read_vxz(p, num_threads=1)
            coords = coords.cpu().numpy().astype(np.int32)
            white = (data["base_color"].float().mean(dim=-1) > 127.5).cpu().numpy()
            return coords, white, split
    return None, None, None


def bucket_of(frac):
    if frac <= 0: return "zero"
    if frac <= 0.05: return "tiny"
    if frac <= 0.15: return "small"
    if frac <= 0.30: return "medium"
    return "large"


def compare_one(sid, direct_dir):
    npz = os.path.join(direct_dir, sid, "emis_gt_direct.npz")
    if not os.path.exists(npz):
        return None
    d = np.load(npz)
    dc, dm = d["coords"].astype(np.int32), d["mask"].astype(bool)
    sc, sm, split = somage_gt(sid)
    if sc is None:
        return None

    dk, sk = _enc(dc), _enc(sc)
    dset, sset = set(dk.tolist()), set(sk.tolist())
    inter = dset & sset
    union = dset | sset
    ov_jac = len(inter) / max(1, len(union))
    cov_direct = len(inter) / max(1, len(dset))   # frac of direct voxels also in somage
    cov_somage = len(inter) / max(1, len(sset))

    # emissive labels on intersected coords
    dmap = dict(zip(dk.tolist(), dm.tolist()))
    smap = dict(zip(sk.tolist(), sm.tolist()))
    inter_l = list(inter)
    de = np.array([dmap[k] for k in inter_l], bool)
    se = np.array([smap[k] for k in inter_l], bool)
    n = len(inter_l)
    agree = float((de == se).mean()) if n else 0.0
    tp = int((de & se).sum()); fp = int((de & ~se).sum())   # fp = new-only
    fn = int((~de & se).sum())                              # fn = old-only
    iou = tp / max(1, tp + fp + fn)

    somage_frac = float(sm.mean())
    direct_frac = float(dm.mean())
    return dict(sid=sid, split=split,
                n_direct=len(dset), n_somage=len(sset), n_inter=n,
                overlap_jaccard=ov_jac, cov_direct=cov_direct, cov_somage=cov_somage,
                somage_frac=somage_frac, direct_frac=direct_frac,
                agree=agree, iou=iou,
                new_only=fp, old_only=fn, both=tp,
                new_only_frac=fp / max(1, n), old_only_frac=fn / max(1, n),
                bucket=bucket_of(somage_frac))


def main():
    direct_dir = sys.argv[1]           # .../direct_pilot/out
    out_json = sys.argv[2]
    sids = sorted([s for s in os.listdir(direct_dir)
                   if os.path.isdir(os.path.join(direct_dir, s))])
    rows = []
    for sid in sids:
        r = compare_one(sid, direct_dir)
        if r is None:
            print(f"[skip] {sid}", flush=True); continue
        rows.append(r)
        print(f"{sid} bkt={r['bucket']:<6} ovJ={r['overlap_jaccard']:.3f} "
              f"agree={r['agree']:.3f} IoU={r['iou']:.3f} "
              f"somF={r['somage_frac']:.3f} dirF={r['direct_frac']:.3f} "
              f"new_only={r['new_only_frac']:.3f} old_only={r['old_only_frac']:.3f}", flush=True)

    # aggregate by bucket
    agg = {}
    for b in ["zero", "tiny", "small", "medium", "large"]:
        br = [r for r in rows if r["bucket"] == b]
        if not br: continue
        agg[b] = dict(n=len(br),
                      overlap_jaccard=float(np.mean([r["overlap_jaccard"] for r in br])),
                      cov_direct=float(np.mean([r["cov_direct"] for r in br])),
                      cov_somage=float(np.mean([r["cov_somage"] for r in br])),
                      agree=float(np.mean([r["agree"] for r in br])),
                      iou=float(np.mean([r["iou"] for r in br])),
                      new_only_frac=float(np.mean([r["new_only_frac"] for r in br])),
                      old_only_frac=float(np.mean([r["old_only_frac"] for r in br])),
                      somage_frac=float(np.mean([r["somage_frac"] for r in br])),
                      direct_frac=float(np.mean([r["direct_frac"] for r in br])))
    out = dict(per_shape=rows, by_bucket=agg,
               overall=dict(n=len(rows),
                            overlap_jaccard=float(np.mean([r["overlap_jaccard"] for r in rows])) if rows else 0,
                            agree=float(np.mean([r["agree"] for r in rows])) if rows else 0,
                            iou=float(np.mean([r["iou"] for r in rows])) if rows else 0))
    json.dump(out, open(out_json, "w"), indent=2)
    print("\nAB_DONE ->", out_json)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
