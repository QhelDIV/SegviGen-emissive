"""
CRITICAL GATE: does o_voxel's textured_mesh_to_volumetric_attr actually write a
per-voxel `emissive` attr into a vxz built from an ORIGINAL TexVerse glb?

Upstream TRELLIS.2 prep does `del attr['emissive']`; OUR data_toolkit/glb_to_vxz.py
does not, so vxz written from an original glb should carry emissive. Verify on a
glow shape (emissive nonzero) and a zero-glow shape (emissive ~zero) before building
the pilot on top of this assumption.

Runs on a GPU node (o_voxel full import needs a live CUDA driver).
"""
import sys, json
import numpy as np
import torch
import trimesh
import o_voxel
from PIL import Image


def make_texture_square_pow2(img, target_size=None):
    w, h = img.size
    max_side = max(w, h)
    pow2 = 1
    while pow2 < max_side:
        pow2 *= 2
    if target_size is not None:
        pow2 = target_size
    pow2 = min(pow2, 2048)
    return img.resize((pow2, pow2), Image.BILINEAR)


def preprocess_scene_textures(asset):
    if not isinstance(asset, trimesh.Scene):
        return asset
    TEX_KEYS = ["baseColorTexture", "normalTexture", "metallicRoughnessTexture",
                "emissiveTexture", "occlusionTexture"]
    for geom in asset.geometry.values():
        visual = getattr(geom, "visual", None)
        mat = getattr(visual, "material", None)
        if mat is None:
            continue
        for key in TEX_KEYS:
            if not hasattr(mat, key):
                continue
            tex = getattr(mat, key)
            if tex is None:
                continue
            if isinstance(tex, Image.Image):
                setattr(mat, key, make_texture_square_pow2(tex))
            elif hasattr(tex, "image") and tex.image is not None:
                img = tex.image
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                tex.image = make_texture_square_pow2(img)
        if hasattr(mat, "image") and mat.image is not None:
            img = mat.image
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            mat.image = make_texture_square_pow2(img)
    return asset


def voxelize(glb_path, vxz_path):
    """Exact copy of data_toolkit/glb_to_vxz.glb_to_vxz, so what we verify IS what the
    builder will produce."""
    asset = trimesh.load(glb_path, force='scene')
    asset = preprocess_scene_textures(asset)
    aabb = asset.bounding_box.bounds
    center = (aabb[0] + aabb[1]) / 2
    scale = 0.99999 / (aabb[1] - aabb[0]).max()
    asset.apply_translation(-center)
    asset.apply_scale(scale)
    mesh = asset.to_mesh()
    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()

    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices, faces, grid_size=512, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False)
    vid = o_voxel.serialize.encode_seq(voxel_indices)
    mapping = torch.argsort(vid)
    voxel_indices = voxel_indices[mapping]
    dual_vertices = dual_vertices[mapping]
    intersected = intersected[mapping]

    voxel_indices_mat, attributes = o_voxel.convert.textured_mesh_to_volumetric_attr(
        asset, grid_size=512, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], timing=False)
    vid_mat = o_voxel.serialize.encode_seq(voxel_indices_mat)
    mapping_mat = torch.argsort(vid_mat)
    attributes = {k: v[mapping_mat] for k, v in attributes.items()}

    print("[raw attrs from textured_mesh_to_volumetric_attr] keys=", sorted(attributes.keys()))

    dual_vertices = dual_vertices * 512 - voxel_indices
    dual_vertices = (torch.clamp(dual_vertices, 0, 1) * 255).type(torch.uint8)
    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)
    attributes['dual_vertices'] = dual_vertices
    attributes['intersected'] = intersected
    o_voxel.io.write(vxz_path, voxel_indices, attributes)


def inspect(vxz_path, label):
    coords, data = o_voxel.io.read(vxz_path)
    print(f"\n===== {label}: {vxz_path} =====")
    print("n_surface_voxels:", coords.shape[0])
    print("attr keys:", sorted(data.keys()))
    out = {"label": label, "n_voxels": int(coords.shape[0]), "attrs": {}}
    for k in sorted(data.keys()):
        v = data[k]
        vf = v.float()
        info = dict(shape=list(v.shape), dtype=str(v.dtype),
                    min=float(vf.min()), max=float(vf.max()), mean=float(vf.mean()))
        print(f"  {k}: shape={info['shape']} dtype={info['dtype']} "
              f"min={info['min']:.4f} max={info['max']:.4f} mean={info['mean']:.4f}")
        out["attrs"][k] = info
    if "emissive" in data:
        em = data["emissive"].float()
        lum = em.mean(dim=-1) if em.ndim == 2 and em.shape[-1] == 3 else em.reshape(em.shape[0], -1).mean(dim=-1)
        # scale-agnostic: report distribution + fraction above a few thresholds
        print("  [emissive] per-voxel luminance: "
              f"min={lum.min():.4f} max={lum.max():.4f} mean={lum.mean():.4f} "
              f"p50={lum.median():.4f} p99={torch.quantile(lum, 0.99):.4f}")
        for thr in [1e-3, 0.01, 0.04, 0.1, 0.5]:
            frac = float((lum > thr).float().mean())
            print(f"    frac lum>{thr}: {frac:.4f}")
        out["emissive_lum"] = dict(min=float(lum.min()), max=float(lum.max()),
                                   mean=float(lum.mean()), p99=float(torch.quantile(lum, 0.99)),
                                   frac_gt_0p04=float((lum > 0.04).float().mean()))
    else:
        print("  !!! NO emissive attr — this is finding (a): bake broken / attr deleted")
        out["emissive_lum"] = None
    return out


if __name__ == "__main__":
    # argv: pairs of "sid:glb_path:label"
    results = []
    for spec in sys.argv[1:]:
        sid, glb, label, vxz = spec.split("||")
        print(f"\n######## voxelizing {label} sid={sid}")
        print("glb:", glb)
        voxelize(glb, vxz)
        results.append(inspect(vxz, f"{label} ({sid})"))
    print("\n\nSUMMARY_JSON " + json.dumps(results))
