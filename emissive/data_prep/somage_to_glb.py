"""
Convert our 'somage' assets → GLB so SegviGen's data_toolkit (glb_to_vxz, etc.)
works unchanged.

Source per asset (from the DiffusionNet pipeline):
  somage_original_mesh.npz : vert (V,3), face (F,3), repacked_uvs (F,3,2)
  somage.npz               : 512x512 maps — color, metal, rough, emission_color, ...

Two outputs:
  build_input_glb()          → textured GLB (base color / metallic-roughness) — the
                               INPUT appearance the model conditions on.
  build_emissive_target_glb() → solid white(emissive)/black(non-emissive) GLB — the
                               TARGET coloring (binary segmentation), per-face.

Per-face emissive labels come from labels_uv_74k/{sid}.npy if available; otherwise
we derive them by UV-sampling the emission_color map (threshold).

NOTE: assumes repacked_uvs in [0,1] with V flipped for image rows (standard glTF);
flip handled below. Validate on real data once the env is up.
"""
import os
import argparse
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial


# ── load ────────────────────────────────────────────────────────────────────

def load_somage(mesh_npz, somage_npz):
    m = np.load(mesh_npz)
    verts = m["vert"].astype(np.float32)            # (V,3)
    faces = m["face"].astype(np.int64)              # (F,3)
    uvs   = m["repacked_uvs"].astype(np.float32)    # (F,3,2) per-corner

    s = np.load(somage_npz)
    maps = {
        "color":          s["color"],               # (512,512,3) base color
        "metal":          s["metal"]    if "metal"   in s else None,
        "rough":          s["rough"]    if "rough"   in s else None,
        "emission_color": s["emission_color"] if "emission_color" in s else None,
    }
    return verts, faces, uvs, maps


def _to_img(arr):
    """np array (H,W,3) or (H,W) → uint8 RGB PIL image."""
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    return Image.fromarray(a)


def _unweld(verts, faces, uvs):
    """Per-corner UVs → unwelded mesh: 3F unique vertices, faces=arange, uv=(3F,2)."""
    v_un = verts[faces].reshape(-1, 3)              # (3F,3)
    f_un = np.arange(len(v_un)).reshape(-1, 3)      # (F,3)
    uv_un = uvs.reshape(-1, 2).copy()               # (3F,2)
    uv_un[:, 1] = 1.0 - uv_un[:, 1]                 # flip V for image-row convention
    return v_un, f_un, uv_un


# ── emissive per-face labels ──────────────────────────────────────────────────

def emissive_face_mask(sid, faces, uvs, maps, labels_dir=None, emis_thresh=0.04):
    """Return per-face bool (F,). Prefer labels_uv_74k; else threshold emission map."""
    if labels_dir:
        p = os.path.join(labels_dir, f"{sid}.npy")
        if os.path.exists(p):
            lab = np.load(p).astype(bool)
            if lab.shape[0] == faces.shape[0]:
                return lab
            # else fall through to emission-map derivation
    em = maps.get("emission_color")
    if em is None:
        return np.zeros(faces.shape[0], dtype=bool)
    em = np.asarray(em).astype(np.float32)
    if em.max() > 1.0:
        em = em / 255.0
    H, W = em.shape[:2]
    # sample emission at each face's UV centroid
    cuv = uvs.mean(axis=1)                          # (F,2) in [0,1]
    px = np.clip((cuv[:, 0] * (W - 1)).astype(int), 0, W - 1)
    py = np.clip(((1.0 - cuv[:, 1]) * (H - 1)).astype(int), 0, H - 1)
    val = em[py, px]
    bright = val.max(axis=-1) if val.ndim == 2 else val
    return bright > emis_thresh


# ── GLB builders ──────────────────────────────────────────────────────────────

def build_input_glb(verts, faces, uvs, maps, out_glb):
    v_un, f_un, uv_un = _unweld(verts, faces, uvs)
    base = _to_img(maps["color"])
    mat_kwargs = dict(baseColorTexture=base, metallicFactor=0.0, roughnessFactor=1.0)
    # pack metallic (B) + roughness (G) into a glTF metallicRoughness texture if present
    if maps.get("metal") is not None and maps.get("rough") is not None:
        met = np.asarray(maps["metal"]); rou = np.asarray(maps["rough"])
        met = met[..., 0] if met.ndim == 3 else met
        rou = rou[..., 0] if rou.ndim == 3 else rou
        if met.max() > 1: met = met / 255.0
        if rou.max() > 1: rou = rou / 255.0
        H, W = met.shape[:2]
        mr = np.zeros((H, W, 3), np.uint8)
        mr[..., 1] = (rou * 255).astype(np.uint8)   # G = roughness
        mr[..., 2] = (met * 255).astype(np.uint8)   # B = metallic
        mat_kwargs["metallicRoughnessTexture"] = Image.fromarray(mr)
    mat = PBRMaterial(**mat_kwargs)
    mesh = trimesh.Trimesh(vertices=v_un, faces=f_un, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv_un, material=mat)
    trimesh.Scene(mesh).export(out_glb)
    return out_glb


def _solid_pbr(mesh, rgb):
    r, g, b = [c / 255.0 for c in rgb]
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, vertex_colors=np.tile([*rgb, 255], (len(mesh.vertices), 1)).astype(np.uint8))
    mesh.visual.material = PBRMaterial(
        baseColorFactor=[r, g, b, 1.0], metallicFactor=0.0, roughnessFactor=1.0,
        emissiveFactor=[r, g, b])
    return mesh


def build_emissive_target_glb(verts, faces, emis_mask, out_glb):
    """Two solid submeshes: emissive→white, non-emissive→black."""
    scene = trimesh.Scene()
    for name, sel, rgb in [("emissive", emis_mask, (255, 255, 255)),
                           ("nonemissive", ~emis_mask, (0, 0, 0))]:
        if sel.sum() == 0:
            continue
        sub = trimesh.Trimesh(vertices=verts, faces=faces[sel], process=False)
        sub.remove_unreferenced_vertices()
        scene.add_geometry(_solid_pbr(sub, rgb), geom_name=name)
    scene.export(out_glb)
    return out_glb


def convert_one(sid, mesh_npz, somage_npz, out_dir, labels_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    verts, faces, uvs, maps = load_somage(mesh_npz, somage_npz)
    inp = build_input_glb(verts, faces, uvs, maps, os.path.join(out_dir, f"{sid}_input.glb"))
    emask = emissive_face_mask(sid, faces, uvs, maps, labels_dir=labels_dir)
    tgt = build_emissive_target_glb(verts, faces, emask, os.path.join(out_dir, f"{sid}_emissive.glb"))
    return inp, tgt, float(emask.mean())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sid", required=True)
    p.add_argument("--mesh_npz", required=True)
    p.add_argument("--somage_npz", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--labels_dir", default=None)
    a = p.parse_args()
    inp, tgt, frac = convert_one(a.sid, a.mesh_npz, a.somage_npz, a.out_dir, a.labels_dir)
    print(f"input={inp}\ntarget={tgt}\nemissive_face_frac={frac:.4f}")
