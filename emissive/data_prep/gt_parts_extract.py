"""
Extract the GROUND-TRUTH part decomposition for each canonical shape and attach it to
the ORIGINAL (GT) mesh, for comparison against the model's predicted segmentation.

Per sid: load somage (occupancy/position + omg_patch_index + patch2{component,mesh}) and
somage_original_mesh (vert/face). Map each GT-mesh vertex → nearest occupied somage pixel
→ patch → component & submesh label. Dump npz {verts, faces, gt_comp, gt_submesh} for
local high-res rendering.

GT part definitions available:
  - submesh (patch2mesh): the asset's authored separate objects (coarser, ~tens)
  - component (patch2component): connected components (finer, can be 100s)

Usage (cluster, trellis2 env):
  python gt_parts_extract.py --sid_file canon_overfit10.txt --out_dir gt_parts_canon10
"""
import os, argparse
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DATA_ROOT = "/3dlg-falas/project/omages/datasets/TexVerse/lightgen/somages_corresp_dc80k"
PARQUET   = "/3dlg-falas/project/omages/datasets/TexVerse/lightgen/emissive_thumbnails_obj_ids_df.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid_file", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_parquet(PARQUET)
    if "success" in df.columns:
        df = df[df["success"] == True]
    sids = [l.strip() for l in open(args.sid_file) if l.strip()]

    for sid in sids:
        dd = df.loc[sid, "ditem_dir"]
        d = os.path.join(DATA_ROOT, dd)
        som = np.load(os.path.join(d, "somage.npz"))
        mesh = np.load(os.path.join(d, "somage_original_mesh.npz"))
        verts = mesh["vert"].astype(np.float64)         # (V,3) original coords
        faces = mesh["face"].astype(np.int64)

        occ = som["occupancy"][..., 0].astype(bool)     # (512,512)
        pos = som["position"].astype(np.float64) / 65535.0   # (512,512,3) in [0,1]
        pidx = som["omg_patch_index"][..., 0].astype(np.int64)   # (512,512) patch id
        p2c = som["patch2component"].astype(np.int64)
        p2m = som["patch2mesh"].astype(np.int64)

        occ_pos = pos[occ]                              # (P,3) surface points
        occ_patch = pidx[occ]
        occ_comp = p2c[occ_patch]
        occ_smesh = p2m[occ_patch]

        # align mesh verts → the [0,1] position frame by bbox (O-mage normalizes to unit)
        vlo, vhi = verts.min(0), verts.max(0)
        plo, phi = occ_pos.min(0), occ_pos.max(0)
        verts_n = (verts - vlo) / np.maximum(vhi - vlo, 1e-9) * (phi - plo) + plo
        _, idx = cKDTree(occ_pos).query(verts_n, k=1)
        gt_comp = occ_comp[idx]
        gt_smesh = occ_smesh[idx]

        np.savez_compressed(os.path.join(args.out_dir, f"{sid}.npz"),
                            verts=verts.astype(np.float32), faces=faces.astype(np.int32),
                            gt_comp=gt_comp.astype(np.int32), gt_submesh=gt_smesh.astype(np.int32))
        print(f"[ok] {sid} V={len(verts)} F={len(faces)} "
              f"n_comp={len(np.unique(gt_comp))} n_submesh={len(np.unique(gt_smesh))}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
