"""
Transfer a model's predicted emissive field from the 512-res voxel grid onto the ORIGINAL
asset GLB's UV space, writing one emissive mask PNG per MATERIAL.

Why this and not the predicted mesh: the model predicts WHERE a shape emits, not its
geometry or its albedo. Rendering the decoded remesh would change mesh, silhouette,
albedo and camera all at once, so a reader could not tell which difference produced the
picture. Substituting only the mask into the untouched asset makes the predicted panel
differ from the published oracle panel in exactly one thing.

Frame (from build_dataset_direct.merged_mesh_512_frame, which states it was verified
empirically against Dongchen's coords at grid 256 -- identity axes, no permutation):
    centre = bbox centre of the asset scene's MERGED geometry
    scale  = 0.99999 / max_extent            -> vertices land in [-0.5, 0.5]
    voxel  = (v_normalised + 0.5) * 512      -> cell index in [0, 511]

The frame is CHECKED, not assumed: every texel's surface point is queried against the
FULL predicted coordinate set (all surface voxels, not just the emissive ones) and the
median distance is reported per material. A correct frame puts that near zero. Querying
only the emissive voxels would confound a wrong frame with a small emissive region, which
is exactly the mistake that made the first version of this script look broken on a shape
whose prediction was in fact correct.

glTF is read directly rather than through trimesh's visual layer: trimesh drops UVs for
materials that carry no texture (48af42db's `Flame_0`, a KHR_materials_pbrSpecularGlossiness
emitter, has TEXCOORD_0 in the primitive but no texture, so `visual.uv` is None), and it
splits one material's primitives into separate geometries, which would write several
partial masks over the same shared atlas instead of one composited mask.

Every material gets a predicted mask, not only the ones that emit in the asset: the model
is free to predict emission on a material that was dark in the GT, and suppressing that
would flatter the prediction. Materials the model predicts nothing on get an all-black
emissive texture, which is the honest rendering of an empty prediction.

Runs on CPU (numpy + scipy + Pillow). No GPU, no trimesh.

  python code/pred_mask_to_asset.py --npz .../pred_voxels/<model>/<sid>.npz \
      --glb .../glb_src/<sid>.glb --out_dir .../pred_assets/<model> --sid <sid> --thr 0.5
"""
import os
import json
import base64
import struct
import argparse

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

GRID = 512

# glTF componentType -> (numpy dtype, bytes)
CTYPE = {5120: (np.int8, 1), 5121: (np.uint8, 1), 5122: (np.int16, 2),
         5123: (np.uint16, 2), 5125: (np.uint32, 4), 5126: (np.float32, 4)}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


# ------------------------------------------------------------------ glTF read
def read_glb(path):
    with open(path, "rb") as f:
        magic, version, _ = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67:
            raise ValueError(f"{path}: not a GLB (magic {magic:#x})")
        gltf, bins = None, b""
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            clen, ctype = struct.unpack("<II", hdr)
            data = f.read(clen)
            if ctype == 0x4E4F534A:
                gltf = json.loads(data.decode("utf-8"))
            elif ctype == 0x004E4942:
                bins = data
    if gltf is None:
        raise ValueError(f"{path}: no JSON chunk")
    return gltf, bins


def buffer_bytes(gltf, bins, idx):
    buf = gltf["buffers"][idx]
    uri = buf.get("uri")
    if uri is None:
        return bins
    if uri.startswith("data:"):
        return base64.b64decode(uri.split(",", 1)[1])
    raise ValueError(f"external buffer uri not supported: {uri[:60]}")


def read_accessor(gltf, bins, idx):
    acc = gltf["accessors"][idx]
    n = acc["count"]
    ncomp = NCOMP[acc["type"]]
    dt, csize = CTYPE[acc["componentType"]]
    if "bufferView" not in acc:
        return np.zeros((n, ncomp), dtype=dt).squeeze()
    bv = gltf["bufferViews"][acc["bufferView"]]
    raw = buffer_bytes(gltf, bins, bv.get("buffer", 0))
    start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride") or (ncomp * csize)
    if stride == ncomp * csize:
        end = start + n * ncomp * csize
        arr = np.frombuffer(raw[start:end], dtype=dt).reshape(n, ncomp)
    else:  # interleaved
        arr = np.empty((n, ncomp), dtype=dt)
        for i in range(n):
            o = start + i * stride
            arr[i] = np.frombuffer(raw[o:o + ncomp * csize], dtype=dt)
    return arr.astype(np.float64) if dt == np.float32 else arr


def node_transforms(gltf):
    """world matrix per node index, composing the scene graph from the roots."""
    nodes = gltf.get("nodes", [])
    out = {}

    def local(nd):
        if "matrix" in nd:
            return np.array(nd["matrix"], dtype=np.float64).reshape(4, 4).T
        M = np.eye(4)
        if "scale" in nd:
            M[:3, :3] = M[:3, :3] @ np.diag(nd["scale"])
        if "rotation" in nd:
            x, y, z, w = nd["rotation"]
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            M[:3, :3] = R @ M[:3, :3]
        if "translation" in nd:
            M[:3, 3] = nd["translation"]
        return M

    def walk(i, parent):
        M = parent @ local(nodes[i])
        out[i] = M
        for c in nodes[i].get("children", []):
            walk(c, M)

    scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
    for r in scene.get("nodes", list(range(len(nodes)))):
        walk(r, np.eye(4))
    return out


def primitives(gltf, bins):
    """[{material, positions(world), uv, faces}] for every primitive that has both
    POSITION and TEXCOORD_0."""
    tf = node_transforms(gltf)
    out = []
    for ni, nd in enumerate(gltf.get("nodes", [])):
        if "mesh" not in nd:
            continue
        M = tf.get(ni, np.eye(4))
        for pi, prim in enumerate(gltf["meshes"][nd["mesh"]]["primitives"]):
            at = prim.get("attributes", {})
            if "POSITION" not in at:
                continue
            pos = np.asarray(read_accessor(gltf, bins, at["POSITION"]), dtype=np.float64)
            pos = (M[:3, :3] @ pos.T).T + M[:3, 3]
            uvkey = "TEXCOORD_0" if "TEXCOORD_0" in at else None
            uv = np.asarray(read_accessor(gltf, bins, at[uvkey]), dtype=np.float64) if uvkey else None
            if "indices" in prim:
                idx = np.asarray(read_accessor(gltf, bins, prim["indices"])).reshape(-1)
            else:
                idx = np.arange(len(pos))
            out.append({"node": ni, "prim": pi, "material": prim.get("material"),
                        "positions": pos, "uv": uv, "faces": idx.reshape(-1, 3).astype(np.int64)})
    return out


# ------------------------------------------------------------------ transfer
def rasterise_into(uv, faces, positions, size, pos_buf, valid_buf):
    """Accumulate one primitive's surface positions into a shared (size,size,3) buffer,
    so every primitive sharing a material composites into that material's one atlas."""
    H = W = size
    px = uv[:, 0] * (W - 1)
    py = (1.0 - uv[:, 1]) * (H - 1)   # glTF UV origin is bottom-left, image row 0 is top
    for f in faces:
        i0, i1, i2 = f
        x0, x1, x2 = px[i0], px[i1], px[i2]
        y0, y1, y2 = py[i0], py[i1], py[i2]
        xlo = max(int(np.floor(min(x0, x1, x2))), 0); xhi = min(int(np.ceil(max(x0, x1, x2))), W - 1)
        ylo = max(int(np.floor(min(y0, y1, y2))), 0); yhi = min(int(np.ceil(max(y0, y1, y2))), H - 1)
        if xhi < xlo or yhi < ylo:
            continue
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(xlo, xhi + 1), np.arange(ylo, yhi + 1))
        l0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / den
        l1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / den
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        if not inside.any():
            continue
        p = (l0[..., None] * positions[i0] + l1[..., None] * positions[i1]
             + l2[..., None] * positions[i2])
        pos_buf[gy[inside], gx[inside]] = p[inside]
        valid_buf[gy[inside], gx[inside]] = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tex", type=int, default=1024)
    ap.add_argument("--tol", type=float, default=2.0,
                    help="max distance in voxel units from a texel's surface point to the "
                         "nearest PREDICTED-EMISSIVE voxel for that texel to count as lit")
    ap.add_argument("--survey", default=None,
                    help="paper_v3/material_survey.json. Supplies the BLENDER slot order, "
                         "which is NOT the glTF material index order (the headphone stand's "
                         "slot 0 is glTF material 10). The renderer indexes by slot, so the "
                         "npz has to be written in slot order or the masks land on the wrong "
                         "materials.")
    ap.add_argument("--source", choices=["pred", "gt"], default="pred",
                    help="'gt' runs the SAME UV path on the decoded GT mask instead of the "
                         "prediction. That is the round-trip check: if the GT mask pushed "
                         "through this resampler does not reproduce the asset's own emissive "
                         "coverage, the inversion is wrong and the predicted column would be "
                         "wrong in the same way, silently.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    z = np.load(args.npz)
    coords = z["coords"].astype(np.float64)
    pred_bc = z["pred_bc"].astype(np.float32)
    gt_e = z["gt_e"]
    lit = gt_e.astype(bool) if args.source == "gt" else (pred_bc > args.thr)
    print(f"[{args.source}] {int(lit.sum())}/{lit.size} voxels emissive"
          f"{'' if args.source == 'gt' else f' at thr {args.thr}'} "
          f"(frac {lit.mean():.4f}); GT frac {gt_e.mean():.4f}", flush=True)

    gltf, bins = read_glb(args.glb)
    prims = primitives(gltf, bins)
    allpos = np.concatenate([p["positions"] for p in prims], axis=0)
    lo, hi = allpos.min(0), allpos.max(0)
    centre = (lo + hi) / 2.0
    scale = 0.99999 / (hi - lo).max()
    print(f"[frame] centre={np.round(centre, 6).tolist()} scale={scale:.6f} "
          f"extent={np.round(hi - lo, 4).tolist()}", flush=True)

    tree_all = cKDTree(coords)                                  # frame check
    tree_lit = cKDTree(coords[lit]) if lit.sum() else None      # the mask itself

    def to_voxel(pts):
        """asset frame -> voxel CELL CENTRE coordinates. The -0.5 puts the query in the same
        convention as the stored integer cell indices (a cell's centre, not its corner);
        without it every query sits half a cell off in each axis."""
        return ((((pts - centre) * scale) + 0.5) * GRID) - 0.5

    # group primitives by material so one material -> one composited atlas
    by_mat = {}
    for p in prims:
        by_mat.setdefault(p["material"], []).append(p)

    stats = {"sid": args.sid, "npz": args.npz, "glb": args.glb, "thr": args.thr,
             "tex": args.tex, "tol_voxels": args.tol, "source": args.source,
             "pred_voxel_frac": float(lit.mean()), "gt_voxel_frac": float(gt_e.mean()),
             "empty_prediction": bool(lit.sum() == 0),
             "n_primitives": len(prims), "materials": []}

    mat_names = {i: m.get("name", f"material_{i}") for i, m in enumerate(gltf.get("materials", []))}

    # Blender slot order is NOT glTF material index order, and the renderer indexes by slot.
    slot_order = None
    if args.survey and os.path.exists(args.survey):
        sv = json.load(open(args.survey)).get(args.sid)
        if sv:
            slot_order = [m["material"] for m in sorted(sv["materials"], key=lambda m: m["slot"])]
            print(f"[slots] survey gives {len(slot_order)} slots: {slot_order}", flush=True)
    mask_by_name, uniform_by_name = {}, {}

    for mat, plist in sorted(by_mat.items(), key=lambda kv: (kv[0] is None, kv[0])):
        name = mat_names.get(mat, f"material_{mat}")
        pos_buf = np.zeros((args.tex, args.tex, 3), dtype=np.float64)
        valid = np.zeros((args.tex, args.tex), dtype=bool)
        n_uv = 0
        for p in plist:
            if p["uv"] is None or len(p["uv"]) != len(p["positions"]):
                continue
            n_uv += 1
            rasterise_into(p["uv"], p["faces"], p["positions"], args.tex, pos_buf, valid)

        rec = {"material_index": mat, "material_name": name,
               "n_primitives": len(plist), "n_primitives_with_uv": n_uv,
               "texel_coverage": float(valid.mean())}

        # A material carries a per-texel mask only if it has a usable parameterisation.
        # Two ways it can fail, and both occur among these eleven assets:
        #   - no TEXCOORD_0 at all (48af42db's `Flame_0`, which is that shape's ONLY emitter)
        #   - UVs present but collapsed to a point (three slots on the headphone stand)
        # Neither can localise emission within the material, so the honest granularity is a
        # single decision for the whole material: does the model fire it or not. That is also
        # exactly how the GT column renders these -- as uniform emitters -- so both columns
        # stay in the same representation.
        vertices = np.concatenate([p["positions"] for p in plist], axis=0)
        qv = to_voxel(vertices)
        dv_all, _ = tree_all.query(qv, k=1)
        if tree_lit is not None:
            dv_lit, _ = tree_lit.query(qv, k=1)
            vmask = (dv_lit <= args.tol)
        else:
            vmask = np.zeros(len(qv), dtype=bool)
        rec["lit_vertex_frac"] = float(vmask.mean())
        rec["frame_check_median_vertex_dist_voxels"] = float(np.median(dv_all))

        degenerate = valid.mean() < 1e-6 or n_uv == 0
        rec["uv_degenerate"] = bool(degenerate)

        mask = np.zeros((args.tex, args.tex), dtype=np.float32)
        if valid.any():
            q = to_voxel(pos_buf[valid])
            d_all, _ = tree_all.query(q, k=1)
            rec["frame_check_median_surface_dist_voxels"] = float(np.median(d_all))
            rec["frame_check_p90_surface_dist_voxels"] = float(np.percentile(d_all, 90))
            if tree_lit is not None:
                d_lit, _ = tree_lit.query(q, k=1)
                hit = d_lit <= args.tol
                mask[valid] = hit.astype(np.float32)
                rec["lit_texel_frac_of_covered"] = float(hit.mean())
            else:
                rec["lit_texel_frac_of_covered"] = 0.0
        else:
            rec["frame_check_median_surface_dist_voxels"] = None
            rec["lit_texel_frac_of_covered"] = 0.0

        if degenerate:
            # majority vote over the material's own vertices
            rec["uniform_value"] = float(vmask.mean() >= 0.5)
            uniform_by_name[name] = rec["uniform_value"]
            print(f"[uniform] mat{mat} {name[:24]:24} UV degenerate -> "
                  f"lit_vertex_frac={vmask.mean():.4f} -> uniform={rec['uniform_value']:.0f}",
                  flush=True)
        else:
            mask_by_name[name] = (mask > 0.5)
            uniform_by_name[name] = None

        # The renderer indexes masks by BLENDER SLOT, not by glTF material index. These
        # differ: the headphone stand's slot 0 is glTF material 10. Naming by glTF index
        # would put every mask on the wrong material and the panels would read as model
        # errors, so the filename carries the slot and the write is deferred until the slot
        # is known.
        rec["_mask"] = mask
        stats["materials"].append(rec)
        fc = rec["frame_check_median_surface_dist_voxels"]
        print(f"[mask] mat{mat} {name[:24]:24} prims={len(plist):3d}/{n_uv:<3d} "
              f"cov={rec['texel_coverage']:.3f} lit={rec['lit_texel_frac_of_covered']:.4f} "
              f"frame_med_dist={'n/a' if fc is None else f'{fc:.2f}'}", flush=True)

    fcs = [m["frame_check_median_surface_dist_voxels"] for m in stats["materials"]
           if m["frame_check_median_surface_dist_voxels"] is not None]
    stats["frame_check_worst_median_surface_dist_voxels"] = max(fcs) if fcs else None

    # ---- write per-slot mask PNGs + the renderer's npz, both indexed by BLENDER SLOT ----
    rec_by_name = {r["material_name"]: r for r in stats["materials"]}  # detail list, renamed at write time
    if slot_order is not None:
        uniform_json = {}
        for slot, mname in enumerate(slot_order):
            r = rec_by_name.get(mname)
            if r is None:
                continue
            if r.get("uv_degenerate"):
                # no parameterisation: a scalar is the whole answer this material can give
                uniform_json[str(slot)] = float(r["uniform_value"])
                r["slot"] = slot
                continue
            img = np.zeros((args.tex, args.tex, 4), dtype=np.uint8)
            img[..., 0] = img[..., 1] = img[..., 2] = (r["_mask"] * 255).astype(np.uint8)
            img[..., 3] = 255
            png = os.path.join(args.out_dir, f"{args.sid}__mat{slot}__emis.png")
            Image.fromarray(img).save(png)
            r["slot"] = slot
            r["mask_png"] = png
        stats["uniform"] = uniform_json
        n_slots_written = sum(1 for r in stats["materials"] if r.get("mask_png"))
        print(f"SLOT_PNGS_WRITTEN {n_slots_written} textures, "
              f"{len(uniform_json)} uniforms: {uniform_json}", flush=True)

        payload, uniforms, missing = {}, [], []
        for slot, mname in enumerate(slot_order):
            if mname in mask_by_name:
                payload[f"slot_{slot}"] = mask_by_name[mname]
                uniforms.append(np.nan)
            elif mname in uniform_by_name and uniform_by_name[mname] is not None:
                uniforms.append(uniform_by_name[mname])
            else:
                missing.append((slot, mname))
                uniforms.append(0.0)
        stats["slot_map"] = [{"slot": i, "material": m,
                              "carrier": ("texture" if f"slot_{i}" in payload
                                          else "uniform" if not np.isnan(uniforms[i]) else "none")}
                             for i, m in enumerate(slot_order)]
        stats["slots_unmatched"] = [{"slot": s, "material": m} for s, m in missing]
        if missing:
            # a name in the survey that no glTF material matched means the slot mapping is
            # broken; the renderer would then paint the wrong material and never know
            print(f"SLOT_MISMATCH {missing}", flush=True)
        npz_path = os.path.join(args.out_dir, f"{args.sid}.npz")
        np.savez_compressed(
            npz_path,
            materials=np.array(slot_order, dtype=object),
            uniform=np.array(uniforms, dtype=np.float32),
            meta=json.dumps({"sid": args.sid, "source": args.source, "npz": args.npz,
                             "thr": args.thr, "res": args.tex, "tol_voxels": args.tol,
                             "pred_voxel_frac": float(lit.mean()),
                             "gt_voxel_frac": float(gt_e.mean()),
                             "empty": bool(lit.sum() == 0)}),
            **payload)
        stats["renderer_npz"] = npz_path
        print(f"NPZ_WRITTEN {npz_path} slots={len(slot_order)} "
              f"textures={len(payload)} uniforms={int(np.sum(~np.isnan(uniforms)))}", flush=True)

    if slot_order is None:
        raise SystemExit("--survey is required: without the Blender slot order the masks "
                         "cannot be named correctly and the renderer would apply them to "
                         "the wrong materials")

    sp = os.path.join(args.out_dir, f"{args.sid}__stats.json")
    for r in stats["materials"]:
        r.pop("_mask", None)          # the array itself is on disk as a PNG
    # The renderer verifies the slot keying by NAME, and reads `materials` as one name per
    # slot in the asset's own material order. My per-material records are keyed differently
    # and are not in slot order, so they move aside and `materials` becomes exactly the
    # list the renderer's guard expects. Without this the guard sees None at every slot.
    stats["materials_detail"] = stats.pop("materials")
    stats["materials"] = list(slot_order)
    json.dump(stats, open(sp, "w"), indent=1)
    print(f"FRAME_CHECK worst_median_surface_dist_voxels="
          f"{stats['frame_check_worst_median_surface_dist_voxels']}", flush=True)
    print(f"PRED_COVERAGE sid={args.sid} source={args.source} voxel_frac={lit.mean():.6f} "
          f"empty={bool(lit.sum() == 0)}", flush=True)
    print(f"[done] wrote {sp}", flush=True)


if __name__ == "__main__":
    main()
