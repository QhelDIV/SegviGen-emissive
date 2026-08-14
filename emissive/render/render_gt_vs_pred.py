"""
High-res comparison: input appearance | GT parts | predicted parts, all on the ORIGINAL
GT mesh. GT parts from gt_parts_extract.py (somage component/submesh); predicted parts
from the pretrained full_seg run (seg_covers npz), mapped onto the same GT mesh.

  render_gt_vs_pred.py --gt_dir gt_parts_canon10 --pred_dir seg_vis_canon10 \
      --albedo_dir canon10_glb --out gt_vs_pred_html --res 900 [--gt_key gt_comp|gt_submesh]
"""
import os, sys, glob, argparse
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "/localhome/xya120/studio/misc/lightgen/lightgen_repo")
from xgutils import bpyutil
from xgutils.miscutil import preset_glb

PALETTE = np.array([
    [0.12,0.47,0.71],[1.00,0.50,0.05],[0.17,0.63,0.17],[0.84,0.15,0.16],
    [0.58,0.40,0.74],[0.55,0.34,0.29],[0.89,0.47,0.76],[0.50,0.50,0.50],
    [0.74,0.74,0.13],[0.09,0.75,0.81],[0.68,0.78,0.91],[1.00,0.73,0.47],
    [0.60,0.87,0.54],[1.00,0.60,0.59],[0.77,0.69,0.84],[0.77,0.61,0.58],
    [0.97,0.71,0.82],[0.78,0.78,0.78],[0.86,0.86,0.55],[0.62,0.85,0.90],
], np.float32)


def relabel_by_size(labels):
    """Relabel so the largest part = 0, next = 1, ... → biggest parts get the most
    distinct palette colors; tiny fragments wrap around."""
    u, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
    order = np.argsort(-cnt)
    rank = np.empty_like(order); rank[order] = np.arange(len(order))
    return rank[inv]


def palette_colors(labels):
    return PALETTE[relabel_by_size(labels) % len(PALETTE)]


def _save(img, path):
    from PIL import Image
    img = np.asarray(img, np.float32)
    if img.ndim == 3 and img.shape[2] == 4:
        rgb, a = img[..., :3], img[..., 3:4]
        img = rgb * a + 0.07 * (1 - a)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def render_mesh_colored(verts, faces, vcol, path, res):
    bpyutil.load_blend(preset_glb); bpyutil.clear_collection("workbench")
    img = bpyutil.render_mesh(verts.astype(np.float32), faces.astype(np.int32),
                              vert_color=vcol.astype(np.float32), resolution=(res, res),
                              samples=32, shadow_catcher=False,
                              camera_position=(0, -2.6, 1.4), camera_up=(0, 0, 1))
    _save(img, path)


def render_albedo(glb, path, res):
    bpyutil.load_blend(preset_glb); bpyutil.clear_collection("workbench")
    obj = bpyutil.load_glb(glb, import_shading=None)
    img = bpyutil.render_scene(obj=obj, resolution=(res, res), samples=32,
                               camera_position=(0, -2.6, 1.4), camera_up=(0, 0, 1),
                               shadow_catcher=False)
    bpyutil.purge_obj(obj); _save(img, path)


def render_one(sid, a):
    g = np.load(os.path.join(a.gt_dir, f"{sid}.npz"))
    verts = g["verts"].astype(np.float64); faces = g["faces"].astype(np.int64)
    # GT parts
    render_mesh_colored(verts, faces, palette_colors(g[a.gt_key]),
                        os.path.join(a.out, f"{sid}_gtparts.png"), a.res)
    # predicted parts: map GT verts -> nearest predicted seg voxel -> label
    pred = np.load(os.path.join(a.pred_dir, f"{sid}.npz"))
    vox = pred["coords"].astype(np.float64); plab = pred["labels"].astype(np.int64)
    vlo, vhi = verts.min(0), verts.max(0); xlo, xhi = vox.min(0), vox.max(0)
    verts_v = (verts - vlo) / np.maximum(vhi - vlo, 1e-9) * (xhi - xlo) + xlo
    _, idx = cKDTree(vox).query(verts_v, k=1)
    render_mesh_colored(verts, faces, palette_colors(plab[idx]),
                        os.path.join(a.out, f"{sid}_predparts.png"), a.res)
    # appearance
    if a.albedo_dir:
        ag = os.path.join(a.albedo_dir, sid, "glb", f"{sid}_input.glb")
        if os.path.exists(ag):
            try:
                render_albedo(ag, os.path.join(a.out, f"{sid}_albedo.png"), a.res)
            except Exception as e:
                print(f"  [albedo skip] {repr(e)[:90]}", flush=True)


def main():
    import subprocess
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--albedo_dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=512,
                    help="thumbnail render size; detail is available via the click-to-load 3D GLB")
    ap.add_argument("--gt_key", default="gt_comp", choices=["gt_comp", "gt_submesh"])
    ap.add_argument("--sid", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.sid:
        render_one(a.sid, a); return

    sids = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(a.gt_dir, "*.npz")))]
    print(f"{len(sids)} shapes @ {a.res}px, GT={a.gt_key}", flush=True)
    cards = []
    for i, sid in enumerate(sids):
        if not os.path.exists(os.path.join(a.out, f"{sid}_predparts.png")):
            cmd = [sys.executable, os.path.abspath(__file__), "--sid", sid, "--gt_dir", a.gt_dir,
                   "--pred_dir", a.pred_dir, "--out", a.out, "--res", str(a.res), "--gt_key", a.gt_key]
            if a.albedo_dir:
                cmd += ["--albedo_dir", a.albedo_dir]
            try:
                subprocess.run(cmd, timeout=300)
            except subprocess.TimeoutExpired:
                pass
        ok = os.path.exists(os.path.join(a.out, f"{sid}_predparts.png"))
        print(f"  [{i+1}/{len(sids)}] {sid} {'ok' if ok else 'FAILED'}", flush=True)
        if ok:
            cards.append(sid)

    rows = []
    for sid in cards:
        alb = f'<td><img src="{sid}_albedo.png"><div class=cap>input appearance</div></td>' \
              if os.path.exists(os.path.join(a.out, f"{sid}_albedo.png")) else '<td class=na>—</td>'
        rows.append(f"""<tr><td class=sid>{sid}</td>{alb}
          <td><img src="{sid}_gtparts.png"><div class=cap>GT parts ({a.gt_key})</div></td>
          <td><img src="{sid}_predparts.png"><div class=cap>predicted parts (full_seg)</div></td></tr>""")
    html = f"""<!DOCTYPE html><html><head><meta charset=utf-8><title>GT vs predicted parts</title><style>
    body{{background:#0e1117;color:#d8dde6;font-family:-apple-system,Segoe UI,sans-serif;padding:20px}}
    h1{{font-size:19px}} .meta{{color:#8b949e;font-size:13px;margin-bottom:14px}}
    table{{border-collapse:collapse}} td{{padding:6px;text-align:center;border-bottom:1px solid #21262d;vertical-align:top}}
    img{{width:340px;height:340px;border-radius:6px;background:#000}} .cap{{font-size:11px;color:#8b949e;margin-top:3px}}
    .sid{{font-family:ui-monospace,monospace;font-size:11px;color:#9aa4b2;max-width:140px;word-break:break-all;text-align:left}}
    .na{{color:#555}}</style></head><body>
    <h1>GT parts vs predicted full-segmentation — on the original GT mesh ({a.res}px), canonical overfit_split_10</h1>
    <div class=meta>{len(cards)} shapes · colors = distinct parts (palette by part size) · GT from somage {a.gt_key}; predicted from pretrained full_seg (zero-shot)</div>
    <table>{''.join(rows)}</table></body></html>"""
    open(os.path.join(a.out, "index.html"), "w").write(html)
    print(f"\nwrote {a.out}/index.html", flush=True)


if __name__ == "__main__":
    main()
