"""
Render seg-colored GLB meshes (from seg_to_mesh.py) into an HTML contact sheet —
the paper-style mesh view of the full part-segmentation (no voxel cubes, no coarsening).

  /localhome/xya120/studio/misc/lightgen/lightgen_repo/.venv/bin/python render_seg_mesh.py \
      --glb_dir .../seg_mesh_canon10 --albedo_dir .../canon10_glb --out .../seg_mesh_html
"""
import os, sys, glob, argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, "/localhome/xya120/studio/misc/lightgen/lightgen_repo")
from xgutils import bpyutil
from xgutils.miscutil import preset_glb

# same high-contrast palette as the voxel "parts" panel (render_seg.py)
PALETTE = np.array([
    [0.12,0.47,0.71],[1.00,0.50,0.05],[0.17,0.63,0.17],[0.84,0.15,0.16],
    [0.58,0.40,0.74],[0.55,0.34,0.29],[0.89,0.47,0.76],[0.50,0.50,0.50],
    [0.74,0.74,0.13],[0.09,0.75,0.81],[0.68,0.78,0.91],[1.00,0.73,0.47],
    [0.60,0.87,0.54],[1.00,0.60,0.59],[0.77,0.69,0.84],[0.77,0.61,0.58],
    [0.97,0.71,0.82],[0.78,0.78,0.78],[0.86,0.86,0.55],[0.62,0.85,0.90],
], np.float32)


def mesh_part_colors(verts, npz_path, use_palette=True):
    """Color each mesh vertex by its nearest seg voxel. Uses the VERIFIED seg_covers
    npz (coords + labels + raw seg_rgb) — recolored by part label with the palette so
    parts are crisp (matches the voxel 'parts' panel)."""
    d = np.load(npz_path)
    vox = d["coords"].astype(np.float64)                 # 0..511 grid
    if use_palette:
        labels = d["labels"].astype(np.int64)
        vox_col = PALETTE[labels % len(PALETTE)]
    else:
        vox_col = d["seg_rgb"].astype(np.float32) / 255.0
    # bbox-align mesh verts to the voxel grid extent, then nearest-voxel lookup
    vlo, vhi = verts.min(0), verts.max(0)
    xlo, xhi = vox.min(0), vox.max(0)
    verts_vox = (verts - vlo) / np.maximum(vhi - vlo, 1e-9) * (xhi - xlo) + xlo
    _, idx = cKDTree(vox).query(verts_vox, k=1)
    return vox_col[idx]


def _save(img, path):
    from PIL import Image
    img = np.asarray(img, np.float32)
    if img.ndim == 3 and img.shape[2] == 4:
        rgb, a = img[..., :3], img[..., 3:4]
        img = rgb * a + 0.07 * (1 - a)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def render_seg_mesh(glb, npz, path, res=460):
    """Render the mesh geometry from `glb`, colored by part label from `npz` — via the
    same render_mesh(vert_color=...) path that worked for the voxel cubes (bypasses the
    GLB-material vertex-color wiring that came out monochrome)."""
    m = trimesh.load(glb, force="mesh", process=False)
    verts = np.asarray(m.vertices, np.float64)
    faces = np.asarray(m.faces, np.int64)
    vcol = mesh_part_colors(verts, npz).astype(np.float32)
    bpyutil.load_blend(preset_glb); bpyutil.clear_collection("workbench")
    img = bpyutil.render_mesh(verts.astype(np.float32), faces.astype(np.int32),
                              vert_color=vcol, resolution=(res, res), samples=32,
                              shadow_catcher=False, camera_position=(0, -2.6, 1.4),
                              camera_up=(0, 0, 1))
    _save(img, path)


def render_albedo(glb, path, res=460):
    bpyutil.load_blend(preset_glb); bpyutil.clear_collection("workbench")
    obj = bpyutil.load_glb(glb, import_shading=None)
    img = bpyutil.render_scene(obj=obj, resolution=(res, res), samples=32,
                               camera_position=(0, -2.6, 1.4), camera_up=(0, 0, 1),
                               shadow_catcher=False)
    bpyutil.purge_obj(obj); _save(img, path)


def main():
    import subprocess
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb_dir", required=True)       # mesh geometry GLBs (from seg_to_mesh)
    ap.add_argument("--npz_dir", required=True)        # verified seg_covers npz (coords+labels)
    ap.add_argument("--albedo_dir", default=None)     # dataset .../<sid>/glb/<sid>_input.glb
    ap.add_argument("--out", required=True)
    ap.add_argument("--sid", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.sid:
        seg_glb = os.path.join(a.glb_dir, f"{a.sid}.glb")
        npz = os.path.join(a.npz_dir, f"{a.sid}.npz")
        render_seg_mesh(seg_glb, npz, os.path.join(a.out, f"{a.sid}_segmesh.png"))
        if a.albedo_dir:
            ag = os.path.join(a.albedo_dir, a.sid, "glb", f"{a.sid}_input.glb")
            if os.path.exists(ag):
                try:
                    render_albedo(ag, os.path.join(a.out, f"{a.sid}_albedo.png"))
                except Exception as e:
                    print(f"  [albedo skip] {repr(e)[:100]}", flush=True)
        return

    glbs = sorted(glob.glob(os.path.join(a.glb_dir, "*.glb")))
    print(f"{len(glbs)} meshes", flush=True)
    cards = []
    for i, g in enumerate(glbs):
        sid = os.path.splitext(os.path.basename(g))[0]
        if not os.path.exists(os.path.join(a.out, f"{sid}_segmesh.png")):
            cmd = [sys.executable, os.path.abspath(__file__), "--sid", sid,
                   "--glb_dir", a.glb_dir, "--npz_dir", a.npz_dir, "--out", a.out]
            if a.albedo_dir:
                cmd += ["--albedo_dir", a.albedo_dir]
            try:
                subprocess.run(cmd, timeout=200)
            except subprocess.TimeoutExpired:
                pass
        ok = os.path.exists(os.path.join(a.out, f"{sid}_segmesh.png"))
        print(f"  [{i+1}/{len(glbs)}] {sid} {'ok' if ok else 'FAILED'}", flush=True)
        if ok:
            cards.append(sid)

    rows = []
    for sid in cards:
        alb = f"{sid}_albedo.png" if os.path.exists(os.path.join(a.out, f"{sid}_albedo.png")) else None
        albcell = f'<td><img src="{alb}"><div class=cap>input appearance</div></td>' if alb else '<td class=na>—</td>'
        rows.append(f"""<tr><td class=sid>{sid}</td>{albcell}
          <td><img src="{sid}_segmesh.png"><div class=cap>part-seg on mesh</div></td></tr>""")
    html = f"""<!DOCTYPE html><html><head><meta charset=utf-8><title>full segmentation (mesh)</title><style>
    body{{background:#0e1117;color:#d8dde6;font-family:-apple-system,Segoe UI,sans-serif;padding:20px}}
    h1{{font-size:19px}} .meta{{color:#8b949e;font-size:13px;margin-bottom:14px}}
    table{{border-collapse:collapse}} td{{padding:6px;text-align:center;border-bottom:1px solid #21262d;vertical-align:top}}
    img{{width:300px;height:300px;border-radius:6px;background:#000}} .cap{{font-size:11px;color:#8b949e;margin-top:3px}}
    .sid{{font-family:ui-monospace,monospace;font-size:11px;color:#9aa4b2;max-width:150px;word-break:break-all;text-align:left}}
    .na{{color:#555}}</style></head><body>
    <h1>SegviGen full part-segmentation on the mesh surface (pretrained, zero-shot) — canonical overfit_split_10</h1>
    <div class=meta>{len(cards)} shapes · res-512 decoded mesh, vertex-colored by the model's part coloring (paper-style; no voxel coarsening)</div>
    <table>{''.join(rows)}</table></body></html>"""
    open(os.path.join(a.out, "index.html"), "w").write(html)
    print(f"\nwrote {a.out}/index.html", flush=True)


if __name__ == "__main__":
    main()
