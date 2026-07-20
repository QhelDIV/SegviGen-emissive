"""
Evidence batch: for each selected shape, build the DIRECT emissive GT and record the
three fractions that expose the o_voxel emissive bug:
  - direct_frac : direct.vxz emissive frac (lum>0.04)          [the broken attr]
  - true_frac   : original glb emissiveTexture x factor frac    [ground truth]
  - somage_frac : current somage output.vxz white-voxel frac    [current GT]
Keeps direct.vxz per shape (for later voxel rendering). Resumable (skips done).
"""
import os, sys, json
import numpy as np
import torch
import o_voxel

REPO = "/3dlg-jupiter-project/lightgen/segvigen_emissive/direct_pilot/repo"
sys.path.insert(0, os.path.join(REPO, "emissive", "data_prep"))
sys.path.insert(0, "/3dlg-jupiter-project/lightgen/segvigen_emissive/direct_pilot")
from build_dataset_direct import build_one, resolve_glb
from diag_emissive import true_emission          # UV-texture ground truth

DS = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset"
OUT = "/3dlg-jupiter-project/lightgen/segvigen_emissive/direct_pilot/out"


def somage_frac(sid):
    for split in ["train", "train_2k_ef"]:
        p = os.path.join(DS, split, sid, "output.vxz")
        if os.path.exists(p):
            coords, data = o_voxel.io.read(p)
            white = (data["base_color"].float().mean(dim=-1) > 127.5)
            return float(white.float().mean()), split, int(coords.shape[0])
    return None, None, 0


def main():
    sel = json.load(open("/3dlg-jupiter-project/lightgen/segvigen_emissive/direct_pilot/evidence_sids.json"))
    bucket_of = {}
    for b, lst in sel["buckets"].items():
        for s in lst:
            bucket_of[s] = b
    sids = sel["all"]
    stats = []
    for i, sid in enumerate(sids):
        odir = os.path.join(OUT, sid)
        try:
            glb = resolve_glb(sid)
            npz = os.path.join(odir, "emis_gt_direct.npz")
            if not os.path.exists(npz):
                build_one(sid, glb, odir)          # writes direct.vxz + npz + meta_direct.json
            m = json.load(open(os.path.join(odir, "meta_direct.json")))
            tf, tm, ntex, factors = true_emission(glb)
            sf, ssplit, snv = somage_frac(sid)
            row = dict(sid=sid, bucket=bucket_of.get(sid, "?"),
                       direct_frac=m["emissive_frac"], n_direct=m["n_voxels"],
                       true_frac=tf, true_mean=tm, n_emis_tex=ntex,
                       emis_tex_black=bool(tm < 1e-4),
                       somage_frac=sf, somage_split=ssplit, n_somage=snv)
            stats.append(row)
            print(f"[{i+1}/{len(sids)}] {sid} bkt={row['bucket']:<6} "
                  f"direct={row['direct_frac']:.3f} true={tf:.3f} somage={(sf or 0):.3f} "
                  f"texblack={row['emis_tex_black']}", flush=True)
        except Exception as e:
            print(f"[fail] {sid}: {repr(e)[:200]}", flush=True)
    json.dump(stats, open(os.path.join(OUT, "..", "evidence_stats.json"), "w"), indent=2)
    print(f"\nEVIDENCE_DONE n={len(stats)}")


if __name__ == "__main__":
    main()
