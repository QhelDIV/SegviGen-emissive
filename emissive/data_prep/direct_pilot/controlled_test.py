"""
Controlled test: is the spurious emissive from preprocess_scene_textures (fixable) or
from o_voxel's C voxelizer itself (upstream)?

For each genuinely-non-emissive shape (emissive texture confirmed all-black), voxelize:
  (A) WITH preprocess_scene_textures (what glb_to_vxz does now)
  (B) WITHOUT preprocess
  (C) with emissiveFactor forcibly zeroed (isolates factor-vs-texture in the C combine)
and report emissive frac (lum>0.04) each way. Also print the raw emissive-texture mean
to nail the ground truth (true emission == 0).
"""
import sys
import numpy as np
import torch
import trimesh
import o_voxel
from PIL import Image
from verify_emissive import preprocess_scene_textures


def _load_norm(glb_path):
    asset = trimesh.load(glb_path, force='scene')
    return asset


def _emis_tex_mean(asset):
    means = []
    for g in asset.geometry.values():
        mat = getattr(getattr(g, "visual", None), "material", None)
        et = getattr(mat, "emissiveTexture", None) if mat else None
        ef = getattr(mat, "emissiveFactor", None) if mat else None
        if et is not None:
            a = np.asarray(et.convert("RGB")).astype(np.float32)
            means.append((round(float(a.mean()), 2), ef.tolist() if ef is not None else None))
    return means


def _normalize(asset):
    aabb = asset.bounding_box.bounds
    center = (aabb[0] + aabb[1]) / 2
    scale = 0.99999 / (aabb[1] - aabb[0]).max()
    asset.apply_translation(-center)
    asset.apply_scale(scale)
    return asset


def _voxelize_emissive_frac(asset):
    """run o_voxel voxelization, return emissive frac (lum>0.04) + per-channel mean."""
    _, attributes = o_voxel.convert.textured_mesh_to_volumetric_attr(
        asset, grid_size=512, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], timing=False)
    em = attributes["emissive"].float() / 255.0
    lum = em.max(dim=-1).values
    frac = float((lum > 0.04).float().mean())
    chan = [round(float(attributes["emissive"].float()[:, i].mean()), 1) for i in range(3)]
    return frac, chan, int(em.shape[0])


def zero_emissive_factor(asset):
    for g in asset.geometry.values():
        mat = getattr(getattr(g, "visual", None), "material", None)
        if mat is not None and getattr(mat, "emissiveFactor", None) is not None:
            mat.emissiveFactor = np.zeros(3, dtype=np.float64)
    return asset


if __name__ == "__main__":
    specs = sys.argv[1:]  # sid||glb
    for spec in specs:
        sid, glb = spec.split("||")
        print(f"\n===== {sid} =====")
        a0 = _load_norm(glb)
        print("  emissive textures (mean, factor):", _emis_tex_mean(a0))

        # (A) WITH preprocess
        aA = preprocess_scene_textures(_normalize(_load_norm(glb)))
        fA, cA, n = _voxelize_emissive_frac(aA)
        print(f"  (A) with preprocess : frac>{0.04}={fA:.4f} chanRGB={cA} n={n}")

        # (B) WITHOUT preprocess
        aB = _normalize(_load_norm(glb))
        fB, cB, _ = _voxelize_emissive_frac(aB)
        print(f"  (B) no preprocess   : frac>{0.04}={fB:.4f} chanRGB={cB}")

        # (C) emissiveFactor zeroed (no preprocess)
        aC = zero_emissive_factor(_normalize(_load_norm(glb)))
        fC, cC, _ = _voxelize_emissive_frac(aC)
        print(f"  (C) factor=0        : frac>{0.04}={fC:.4f} chanRGB={cC}")
    print("\nCTL_DONE")
