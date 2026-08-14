"""
Render the PRETRAINED full part-segmentation as an HTML contact sheet:
  [ input albedo | raw model part-coloring | crisp recolored parts ]
so we can SEE how SegviGen decomposes each object into parts.

Input: per-sample npz from seg_covers_emissive.py --dump_vis, each with
  coords (N,3 int), seg_rgb (N,3 uint8 = model's part colors), labels (N int part id),
  gt_e (N bool, unused here).

Runs LOCALLY (bpy venv), one sample per subprocess (bpy crashes after a few in-proc
renders), with adaptive voxel coarsening (decoded voxels are 0.5-5M @ res512).

  /localhome/xya120/studio/misc/lightgen/lightgen_repo/.venv/bin/python render_seg.py \
      --vis_dir .../seg_vis_overfit_10 --glb_dir .../overfit10_glb --out .../seg_html
"""
import os, sys, glob, argparse
import numpy as np

sys.path.insert(0, "/localhome/xya120/studio/misc/lightgen/lightgen_repo")
from xgutils import bpyutil
from xgutils.miscutil import preset_glb

# high-contrast categorical palette (tab20-ish), RGB 0..1
PALETTE = np.array([
    [0.12,0.47,0.71],[1.00,0.50,0.05],[0.17,0.63,0.17],[0.84,0.15,0.16],
    [0.58,0.40,0.74],[0.55,0.34,0.29],[0.89,0.47,0.76],[0.50,0.50,0.50],
    [0.74,0.74,0.13],[0.09,0.75,0.81],[0.68,0.78,0.91],[1.00,0.73,0.47],
    [0.60,0.87,0.54],[1.00,0.60,0.59],[0.77,0.69,0.84],[0.77,0.61,0.58],
    [0.97,0.71,0.82],[0.78,0.78,0.78],[0.86,0.86,0.55],[0.62,0.85,0.90],
], np.float32)

_CUBE_V = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], np.float32) - 0.5
_CUBE_F = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[1,5,6],[1,6,2],
                    [2,6,7],[2,7,3],[3,7,4],[3,4,0]], np.int32)


def coarsen(coords, rgb, labels, factor=8, max_cells=15000):
    """Downsample to a coarse grid; per cell: mean raw color + majority part label."""
    c64 = coords.astype(np.int64)
    while True:
        cc = c64 // factor
        uniq, inv = np.unique(cc, axis=0, return_inverse=True)
        if len(uniq) <= max_cells or factor >= 64:
            break
        factor *= 2
    n = len(uniq)
    # mean raw color per cell
    sums = np.zeros((n, 3)); cnt = np.bincount(inv, minlength=n)
    for k in range(3):
        sums[:, k] = np.bincount(inv, weights=rgb[:, k], minlength=n)
    mean_rgb = sums / cnt[:, None]
    # majority label per cell
    maxl = int(labels.max()) + 1
    votes = np.zeros((n, maxl))
    np.add.at(votes, (inv, labels), 1)
    maj = votes.argmax(1)
    return uniq, mean_rgb.astype(np.float32), maj.astype(np.int64)


def voxel_mesh(coords, color):
    coords = coords.astype(np.float32)
    span = coords.max(0) - coords.min(0)
    s = float(span.max()) or 1.0
    centers = (coords - coords.min(0) - span / 2.0) / s
    cube = _CUBE_V * (1.0 / s) * 0.95
    V = (centers[:, None, :] + cube[None, :, :]).reshape(-1, 3)
    F = (_CUBE_F[None] + (np.arange(len(coords)) * 8)[:, None, None]).reshape(-1, 3)
    C = np.repeat(color, 8, axis=0)
    return V.astype(np.float32), F.astype(np.int32), C.astype(np.float32)


def render_voxels(coords, color, path, res=440):
    V, F, C = voxel_mesh(coords, color)
    bpyutil.load_blend(preset_glb)
    bpyutil.clear_collection("workbench")
    img = bpyutil.render_mesh(V, F, vert_color=C, resolution=(res, res), samples=24,
                              shadow_catcher=False, camera_position=(0, -2.6, 1.4),
                              camera_up=(0, 0, 1))
    _save(img, path)


def render_albedo(glb_path, path, res=440):
    try:
        bpyutil.load_blend(preset_glb); bpyutil.clear_collection("workbench")
        obj = bpyutil.load_glb(glb_path, import_shading=None)
        img = bpyutil.render_scene(obj=obj, resolution=(res, res), samples=32,
                                   camera_position=(0, -2.6, 1.4), camera_up=(0, 0, 1),
                                   shadow_catcher=False)
        bpyutil.purge_obj(obj); _save(img, path); return True
    except Exception as e:
        print(f"  [albedo skip] {repr(e)[:100]}", flush=True); return False


def _save(img, path):
    from PIL import Image
    img = np.asarray(img, np.float32)
    if img.ndim == 3 and img.shape[2] == 4:
        rgb, a = img[..., :3], img[..., 3:4]
        img = rgb * a + 0.07 * (1 - a)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def render_one(npz, out, glb_dir):
    sid = os.path.splitext(os.path.basename(npz))[0]
    d = np.load(npz)
    coords = d["coords"]; rgb = d["seg_rgb"].astype(np.float32) / 255.0; labels = d["labels"].astype(np.int64)
    cc, craw, clab = coarsen(coords, rgb, labels)
    render_voxels(cc, craw, os.path.join(out, f"{sid}_segraw.png"))
    render_voxels(cc, PALETTE[clab % len(PALETTE)], os.path.join(out, f"{sid}_parts.png"))
    if glb_dir:
        g = os.path.join(glb_dir, sid, "glb", f"{sid}_input.glb")
        if os.path.exists(g):
            render_albedo(g, os.path.join(out, f"{sid}_albedo.png"))


def main():
    import subprocess
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis_dir", required=True)
    ap.add_argument("--glb_dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sid", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.sid:
        render_one(os.path.join(a.vis_dir, f"{a.sid}.npz"), a.out, a.glb_dir)
        return

    npzs = sorted(glob.glob(os.path.join(a.vis_dir, "*.npz")))
    print(f"{len(npzs)} samples", flush=True)
    cards = []
    for i, p in enumerate(npzs):
        sid = os.path.splitext(os.path.basename(p))[0]
        if not os.path.exists(os.path.join(a.out, f"{sid}_parts.png")):
            cmd = [sys.executable, os.path.abspath(__file__), "--sid", sid,
                   "--vis_dir", a.vis_dir, "--out", a.out]
            if a.glb_dir:
                cmd += ["--glb_dir", a.glb_dir]
            try:
                subprocess.run(cmd, timeout=180)
            except subprocess.TimeoutExpired:
                pass
        ok = os.path.exists(os.path.join(a.out, f"{sid}_parts.png"))
        nparts = int(np.load(p)["labels"].max()) + 1
        print(f"  [{i+1}/{len(npzs)}] {sid} {'ok' if ok else 'FAILED'} ({nparts} parts)", flush=True)
        if ok:
            cards.append((sid, nparts))

    rows = []
    for sid, nparts in cards:
        alb = f"{sid}_albedo.png" if os.path.exists(os.path.join(a.out, f"{sid}_albedo.png")) else None
        albcell = f'<td><img src="{alb}"><div class=cap>input albedo</div></td>' if alb else '<td class=na>—</td>'
        rows.append(f"""<tr><td class=sid>{sid}<div class=cap>{nparts} parts</div></td>{albcell}
          <td><img src="{sid}_segraw.png"><div class=cap>model part-coloring</div></td>
          <td><img src="{sid}_parts.png"><div class=cap>parts (recolored)</div></td></tr>""")
    html = f"""<!DOCTYPE html><html><head><meta charset=utf-8><title>full segmentation</title><style>
    body{{background:#0e1117;color:#d8dde6;font-family:-apple-system,Segoe UI,sans-serif;padding:20px}}
    h1{{font-size:19px}} .meta{{color:#8b949e;font-size:13px;margin-bottom:14px}}
    table{{border-collapse:collapse}} td{{padding:6px;text-align:center;border-bottom:1px solid #21262d;vertical-align:top}}
    img{{width:250px;height:250px;border-radius:6px;background:#000}} .cap{{font-size:11px;color:#8b949e;margin-top:3px}}
    .sid{{font-family:ui-monospace,monospace;font-size:11px;color:#9aa4b2;max-width:150px;word-break:break-all;text-align:left}}
    .na{{color:#555}}</style></head><body>
    <h1>SegviGen full part-segmentation (pretrained, zero-shot) — overfit_10</h1>
    <div class=meta>{len(cards)} shapes · left→right: input appearance, the model's raw part-coloring, and parts recolored with a high-contrast palette for legibility</div>
    <table>{''.join(rows)}</table></body></html>"""
    open(os.path.join(a.out, "index.html"), "w").write(html)
    print(f"\nwrote {a.out}/index.html", flush=True)


if __name__ == "__main__":
    main()
