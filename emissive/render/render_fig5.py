"""Figure-5-style emission render: per-FACE UV-free mask transfer (as in
render_uvfree.py -- no chart, no divergence risk), but albedo now comes from
the material's REAL baseColorTexture sampled at each face's UV centroid, read
directly from the raw glTF/PNG bytes in Python (write side only, the same side
the mask transfer already reads, no Blender UV pass involved). This is the
paper's SegviGen convention, emission = mask x albedo, applied without the UV
atlas that caused the write/read divergence.

Falls back to the material's flat baseColorFactor only for a material with no
baseColorTexture, or a primitive with no TEXCOORD_0 at all.

Scene: reproduces handoff_fig7's PRIMARY treatment (images/emission_only_k0),
copied from its own recorded settings, not re-derived: dark_room(bg=0.012,
key=0.0), AgX view transform, exposure 0.0, bloom size 9 / threshold 1.0 /
mix -0.15, 256 samples. Camera: the same auto-solved place_camera() used
throughout this investigation (azimuth 38 / elevation 17 / lens 52 / margin
1.06), so GT and every draw of one shape share one camera.

Usage:
  PYTHONPATH=<xgutils>/src <venv>/bin/python render_fig5.py \
      --glb <path> --npz <coords+pred_bc+gt_e> --value pred|gt \
      --out <dir> --sid <sid> --tag <gt|rawseed3|emadraw3|emaseed4|emaseed5>
"""
import argparse
import base64
import io
import os
import sys

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, "/project/3dlg-hcvc/omages/xgutils/src")
from xgutils import bpyutil
import bpy

sys.path.insert(0, "/project/3dlg-hcvc/omages/yanxg_scratch/fig7/code")
import render_emissive as re_  # noqa: E402  (dark_room / add_bloom / place_camera / render)

sys.path.insert(0, "/3dlg-jupiter-project/lightgen/segvigen_emissive/code")
from pred_mask_to_asset import read_glb, primitives, buffer_bytes  # noqa: E402

GRID = 512


def srgb_to_linear(a):
    a = np.clip(a, 0.0, 1.0)
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def compute_frame(gltf, bins):
    prims = primitives(gltf, bins)
    allpos = np.concatenate([p["positions"] for p in prims], axis=0)
    lo, hi = allpos.min(0), allpos.max(0)
    centre = (lo + hi) / 2.0
    scale = 0.99999 / (hi - lo).max()
    return centre, scale, prims


def to_voxel(pts, centre, scale):
    return ((((pts - centre) * scale) + 0.5) * GRID) - 0.5


def material_albedo_const(gltf, mat_idx):
    """Flat RGB fallback, used only when a material has no baseColorTexture,
    or a primitive has no TEXCOORD_0 to sample one with."""
    mat = gltf["materials"][mat_idx]
    pbr = mat.get("pbrMetallicRoughness", {})
    if "baseColorFactor" in pbr:
        return np.array(pbr["baseColorFactor"][:3], dtype=np.float64)
    ext = mat.get("extensions", {}).get("KHR_materials_pbrSpecularGlossiness", {})
    if "diffuseFactor" in ext:
        return np.array(ext["diffuseFactor"][:3], dtype=np.float64)
    return np.array([1.0, 1.0, 1.0])


def decode_material_texture(gltf, bins, mat_idx):
    """(H,W,3) float64 LINEAR array for this material's baseColorTexture, or
    None. Raw glTF JSON + PNG decode; no Blender image load anywhere in this
    path, so no divergence with the write-side mask transfer."""
    mat = gltf["materials"][mat_idx]
    pbr = mat.get("pbrMetallicRoughness", {})
    bct = pbr.get("baseColorTexture")
    if bct is None:
        return None
    tex = gltf["textures"][bct["index"]]
    img_idx = tex.get("source")
    if img_idx is None:
        return None
    img_info = gltf["images"][img_idx]
    if "bufferView" in img_info:
        bv = gltf["bufferViews"][img_info["bufferView"]]
        raw = buffer_bytes(gltf, bins, bv.get("buffer", 0))
        start = bv.get("byteOffset", 0)
        img_bytes = raw[start:start + bv["byteLength"]]
    elif "uri" in img_info and img_info["uri"].startswith("data:"):
        img_bytes = base64.b64decode(img_info["uri"].split(",", 1)[1])
    else:
        print(f"  MAT {mat_idx}: external image URI, cannot embed-decode, "
              f"falling back to flat factor", flush=True)
        return None
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.asarray(im, dtype=np.float64) / 255.0
    return srgb_to_linear(arr)


def sample_texture_at_uv(tex, uv):
    """Nearest-neighbor sample at Nx2 UV in [0,1]. glTF UV origin is
    bottom-left, image row 0 is top -- same row-flip pred_mask_to_asset.py's
    rasterise_into() uses."""
    H, W = tex.shape[:2]
    u = np.mod(uv[:, 0], 1.0)
    v = np.mod(uv[:, 1], 1.0)
    px = np.clip((u * (W - 1)).round().astype(np.int64), 0, W - 1)
    py = np.clip(((1.0 - v) * (H - 1)).round().astype(np.int64), 0, H - 1)
    return tex[py, px]


def face_colors_for_material(gltf, prims, mat_idx, centre, scale, tree_lit, tol, tex_cache):
    """Per-face linear RGB = lit * albedo, albedo from the real texture at the
    face's own UV centroid where available, flat factor otherwise. Same
    nearest-LIT-voxel-within-tol convention as render_uvfree.py throughout."""
    tex = tex_cache[mat_idx]
    flat = material_albedo_const(gltf, mat_idx)
    colors = []
    for p in prims:
        if p["material"] != mat_idx:
            continue
        pos, faces = p["positions"], p["faces"]
        centroids = pos[faces].mean(axis=1)
        vox = to_voxel(centroids, centre, scale)
        if tree_lit is None:
            v = np.zeros(len(vox))
        else:
            d_lit, _ = tree_lit.query(vox, k=1)
            v = (d_lit <= tol).astype(np.float64)
        if tex is not None and p["uv"] is not None:
            uv_c = p["uv"][faces].mean(axis=1)
            albedo = sample_texture_at_uv(tex, uv_c)
        else:
            albedo = np.tile(flat[None, :], (len(faces), 1))
        rgb = v[:, None] * albedo
        colors.append(rgb)
    return np.concatenate(colors, axis=0) if colors else np.zeros((0, 3))


def build_uvfree_material(name, strength):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    attr = nt.nodes.new("ShaderNodeVertexColor")
    attr.layer_name = "emis_fig5"
    nt.links.new(attr.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = strength
    bsdf.inputs["Base Color"].default_value = (0, 0, 0, 1)
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.9
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--value", choices=["pred", "gt"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--tag", required=True,
                    help="output-name variant tag, e.g. gt/rawseed3/emadraw3/emaseed4/emaseed5")
    ap.add_argument("--azimuth", type=float, default=38.0)
    ap.add_argument("--elevation", type=float, default=17.0)
    ap.add_argument("--lens", type=float, default=52.0)
    ap.add_argument("--margin", type=float, default=1.06)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--strength", type=float, default=4.0)
    ap.add_argument("--bg", type=float, default=0.012)
    ap.add_argument("--bloom_size", type=float, default=9)
    ap.add_argument("--bloom_threshold", type=float, default=1.0)
    ap.add_argument("--bloom_mix", type=float, default=-0.15)
    ap.add_argument("--view_transform", default="AgX")
    ap.add_argument("--exposure", type=float, default=0.0)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=2.0)
    args = ap.parse_args()

    gltf, bins = read_glb(args.glb)
    centre, scale, prims = compute_frame(gltf, bins)

    z = np.load(args.npz)
    coords_vox = z["coords"].astype(np.float64)
    if args.value == "pred":
        lit = (z["pred_bc"] > args.thr)
    else:
        lit = z["gt_e"].astype(bool)
    print(f"LIT_VOXEL_FRAC {lit.mean():.6f} ({int(lit.sum())}/{len(lit)})", flush=True)
    tree_lit = cKDTree(coords_vox[lit]) if lit.sum() else None

    n_mats = len(gltf.get("materials", []))
    tex_cache = {}
    for mi in range(n_mats):
        tex_cache[mi] = decode_material_texture(gltf, bins, mi)
        print(f"MAT {mi}: texture={'yes ' + str(tex_cache[mi].shape[:2]) if tex_cache[mi] is not None else 'NO (flat factor)'}",
              flush=True)

    per_mat_colors = {}
    for mi in range(n_mats):
        c = face_colors_for_material(gltf, prims, mi, centre, scale, tree_lit, args.tol, tex_cache)
        if len(c):
            per_mat_colors[mi] = c
            print(f"MAT {mi} n_faces={len(c)} n_lit_faces={(c.max(axis=1) > 0).sum()}", flush=True)

    total_lit_faces = sum(int((c.max(axis=1) > 0).sum()) for c in per_mat_colors.values())
    print(f"TRANSFER_CHECK source_voxel_lit={int(lit.sum())} total_lit_faces={total_lit_faces}", flush=True)
    if lit.sum() > 0 and total_lit_faces == 0:
        raise SystemExit(
            f"REFUSING TO RENDER: source voxel mask has {int(lit.sum())} lit "
            f"voxels but zero faces transferred as lit.")

    bpyutil.load_blend(bpyutil.preset_glb)
    bpyutil.clear_collection("workbench")
    obj = bpyutil.load_glb(args.glb, import_shading=None)
    me = obj.data

    n_poly = len(me.polygons)
    n_raw_faces = sum(len(c) for c in per_mat_colors.values())
    print(f"CORRESPONDENCE_CHECK n_blender_polygons={n_poly} n_raw_gltf_faces={n_raw_faces}", flush=True)
    assert n_poly == n_raw_faces, (
        f"face count mismatch: Blender has {n_poly} polygons, raw glTF has {n_raw_faces}")

    col = me.color_attributes.new(name="emis_fig5", type="FLOAT_COLOR", domain="CORNER")
    per_mat_cursor = {mi: 0 for mi in per_mat_colors}
    corner_rgba = np.zeros((len(me.loops), 4), dtype=np.float32)
    corner_rgba[:, 3] = 1.0
    for p in me.polygons:
        mi = p.material_index
        if mi not in per_mat_colors:
            continue
        rgb = per_mat_colors[mi][per_mat_cursor[mi]]
        per_mat_cursor[mi] += 1
        for li in p.loop_indices:
            corner_rgba[li, :3] = rgb
    col.data.foreach_set("color", corner_rgba.ravel())

    for slot in obj.material_slots:
        if slot.material is None:
            continue
        mat = build_uvfree_material(f"fig5_{slot.material.name}", args.strength)
        slot.material = mat

    lo, hi = re_.drop_to_floor(obj)
    pos, cam_centre, dist = re_.place_camera(obj, args.azimuth, args.elevation, args.lens, args.margin)
    re_.dark_room(bg=args.bg, key=0.0)
    floor = bpy.data.objects.get("Floor")
    if floor is not None:
        floor.location.z = -0.004
    re_.add_bloom(size=args.bloom_size, threshold=args.bloom_threshold, mix=args.bloom_mix)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"fig5_{args.sid}_{args.tag}.png")
    re_.render(out_path, (args.res, args.res), args.samples, False,
              args.view_transform, args.exposure)
    print(f"FIG5_DONE {args.sid} {args.tag} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
