"""
DIAGNOSTIC (gate, pre-pilot): is the direct-vxz `emissive` attr a trustworthy GT?

For a stratified set of shapes, compare THREE things per shape:
  1. TRUE glb emission  : load the original glb, for each geom compute the effective
                          emissive = emissiveTexture(RGB) * emissiveFactor, normalized
                          to [0,1]. Report the texture-pixel mean and the fraction of
                          texels with max-channel luminance > 0.04. If a shape's TRUE
                          emission texture is ~all-zero, it genuinely does NOT glow —
                          any emissive the voxelizer reports for it is a false positive.
  2. DIRECT vxz emissive: voxelize the original glb (glb_to_vxz), read back the emissive
                          attr, report frac of surface voxels with max-channel lum>0.04.
  3. SOMAGE GT          : the current pipeline's output.vxz white-voxel fraction.

The decisive column is (1) vs (2): a shape with true emission ~0 but direct frac high
means o_voxel emits spurious emissive -> the direct GT is broken (finding (a)).
"""
import os, sys, json
import numpy as np
import torch
import trimesh
import o_voxel
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_emissive import voxelize  # exact glb_to_vxz copy

import pandas as pd
PARQUET = "/3dlg-falas/project/omages/datasets/TexVerse/lightgen/emissive_thumbnails_obj_ids_df.parquet"
GLB_ROOT = "/3dlg-falas/project/omages/datasets/TexVerse/TexVerse-1K"
DS = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset"
THR = 0.04


def true_emission(glb_path):
    """Per-geom effective emissive = emissiveTexture * emissiveFactor, normalized.
    Returns (tex_pixel_frac_gt_thr, tex_mean_lum, has_emissive_tex, factors)."""
    sc = trimesh.load(glb_path, force="scene")
    lums = []
    factors = []
    n_tex = 0
    for g in sc.geometry.values():
        mat = getattr(getattr(g, "visual", None), "material", None)
        if mat is None:
            continue
        ef = getattr(mat, "emissiveFactor", None)
        et = getattr(mat, "emissiveTexture", None)
        f = np.array(ef, dtype=np.float32) if ef is not None else np.zeros(3, np.float32)
        factors.append(f.tolist())
        if et is not None:
            n_tex += 1
            img = et if isinstance(et, Image.Image) else Image.fromarray(np.asarray(et))
            a = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0   # [H,W,3] in [0,1]
            eff = a * f.reshape(1, 1, 3)                                    # texture * factor
            lums.append(eff.max(axis=-1).reshape(-1))                       # max-channel per texel
        elif ef is not None:
            # factor-only emissive (uniform): one "texel" of value = factor
            lums.append(np.array([float(f.max())], np.float32))
    if not lums:
        return 0.0, 0.0, n_tex, factors
    lum = np.concatenate(lums)
    return float((lum > THR).mean()), float(lum.mean()), n_tex, factors


def vxz_emissive_frac(vxz_path):
    coords, data = o_voxel.io.read(vxz_path)
    em = data["emissive"].float() / 255.0
    lum = em.max(dim=-1).values
    return float((lum > THR).float().mean()), int(coords.shape[0]), float(lum.mean())


def somage_frac(sid):
    """white-voxel fraction of the current somage output.vxz (across known splits)."""
    for split in ["train", "train_2k_ef"]:
        p = os.path.join(DS, split, sid, "output.vxz")
        if os.path.exists(p):
            coords, data = o_voxel.io.read(p)
            white = (data["base_color"].float().mean(dim=-1) > 127.5)
            return float(white.float().mean()), split
    return None, None


if __name__ == "__main__":
    sids = sys.argv[1].split(",")
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(PARQUET)
    rows = []
    print(f"{'sid':<34} {'true_fr':>8} {'true_mn':>8} {'ntex':>4} {'direct_fr':>9} {'somage_fr':>9} {'nvox':>8}")
    for sid in sids:
        glb = os.path.join(GLB_ROOT, df.loc[sid, "glb_1k_path"])
        tf, tm, ntex, factors = true_emission(glb)
        vxz = os.path.join(out_dir, f"{sid}.vxz")
        voxelize(glb, vxz)
        dfrac, nvox, dmean = vxz_emissive_frac(vxz)
        sfrac, split = somage_frac(sid)
        row = dict(sid=sid, true_frac=tf, true_mean=tm, n_emis_tex=ntex,
                   direct_frac=dfrac, direct_mean=dmean, somage_frac=sfrac,
                   somage_split=split, n_vox=nvox, emissive_factors=factors)
        rows.append(row)
        sfr = f"{sfrac:.4f}" if sfrac is not None else "  n/a  "
        print(f"{sid:<34} {tf:>8.4f} {tm:>8.4f} {ntex:>4} {dfrac:>9.4f} {sfr:>9} {nvox:>8}", flush=True)
        os.remove(vxz)  # keep dir small; we only need stats
    json.dump(rows, open(os.path.join(out_dir, "diag_stats.json"), "w"), indent=2)
    print("\nDIAG_DONE " + json.dumps({r["sid"]: dict(true_frac=r["true_frac"], direct_frac=r["direct_frac"], somage_frac=r["somage_frac"]) for r in rows}))
