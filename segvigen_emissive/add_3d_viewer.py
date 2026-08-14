"""
Add interactive 3D (click a thumbnail → load the GLB in an in-browser viewer) to the
GT-vs-predicted page, WITHOUT re-rendering. Exports per-panel vertex-colored GLBs from
the existing npz (GT submesh parts, predicted parts) + copies the input appearance GLB,
then rewrites index.html with a <model-viewer> lightbox.

  python add_3d_viewer.py --gt_dir vis_data/gt_parts_canon10 --pred_dir vis_data/seg_vis_canon10 \
      --albedo_dir vis_data/canon10_glb --html_dir vis_data/gt_vs_pred_html --gt_key gt_submesh
"""
import os, sys, glob, json, shutil, argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree

PALETTE = np.array([
    [0.12,0.47,0.71],[1.00,0.50,0.05],[0.17,0.63,0.17],[0.84,0.15,0.16],
    [0.58,0.40,0.74],[0.55,0.34,0.29],[0.89,0.47,0.76],[0.50,0.50,0.50],
    [0.74,0.74,0.13],[0.09,0.75,0.81],[0.68,0.78,0.91],[1.00,0.73,0.47],
    [0.60,0.87,0.54],[1.00,0.60,0.59],[0.77,0.69,0.84],[0.77,0.61,0.58],
    [0.97,0.71,0.82],[0.78,0.78,0.78],[0.86,0.86,0.55],[0.62,0.85,0.90],
], np.float32)


def relabel_by_size(labels):
    u, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
    order = np.argsort(-cnt); rank = np.empty_like(order); rank[order] = np.arange(len(order))
    return rank[inv]


def palette_u8(labels):
    c = (PALETTE[relabel_by_size(labels) % len(PALETTE)] * 255).astype(np.uint8)
    return np.concatenate([c, np.full((len(c), 1), 255, np.uint8)], 1)


def export_glb(verts, faces, vcol, path):
    # Plain ColorVisuals → glTF COLOR_0 vertex colors (three.js/model-viewer renders these).
    # NOTE: do NOT assign m.visual.material here — it converts to TextureVisuals and DROPS
    # the per-vertex colors (verified: the mesh then renders as one flat color).
    m = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vcol, process=False)
    m.export(path)
    return os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--albedo_dir", default=None)
    ap.add_argument("--html_dir", required=True)
    ap.add_argument("--gt_key", default="gt_submesh")
    ap.add_argument("--res", type=int, default=900)
    a = ap.parse_args()

    sids = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(a.gt_dir, "*.npz")))]
    rows = []
    for sid in sids:
        if not os.path.exists(os.path.join(a.html_dir, f"{sid}_predparts.png")):
            continue
        g = np.load(os.path.join(a.gt_dir, f"{sid}.npz"))
        verts = g["verts"].astype(np.float64); faces = g["faces"].astype(np.int64)
        # GT parts GLB
        export_glb(verts, faces, palette_u8(g[a.gt_key]), os.path.join(a.html_dir, f"{sid}_gtparts.glb"))
        # predicted parts GLB (map GT verts -> nearest pred voxel)
        pred = np.load(os.path.join(a.pred_dir, f"{sid}.npz"))
        vox = pred["coords"].astype(np.float64); plab = pred["labels"].astype(np.int64)
        vlo, vhi = verts.min(0), verts.max(0); xlo, xhi = vox.min(0), vox.max(0)
        vv = (verts - vlo) / np.maximum(vhi - vlo, 1e-9) * (xhi - xlo) + xlo
        _, idx = cKDTree(vox).query(vv, k=1)
        export_glb(verts, faces, palette_u8(plab[idx]), os.path.join(a.html_dir, f"{sid}_predparts.glb"))
        # appearance GLB (copy the textured input glb)
        has_alb = False
        if a.albedo_dir:
            ag = os.path.join(a.albedo_dir, sid, "glb", f"{sid}_input.glb")
            if os.path.exists(ag):
                shutil.copy(ag, os.path.join(a.html_dir, f"{sid}_albedo.glb")); has_alb = True
        rows.append((sid, has_alb))
        print(f"[ok] {sid}", flush=True)

    # rewrite index.html with model-viewer lightbox
    def cell(sid, kind, cap, has_glb):
        img = f"{sid}_{kind}.png"; glb = f"{sid}_{kind}.glb"
        attr = f'class=clk data-glb="{glb}" data-title="{sid} · {cap}"' if has_glb else "class=noglb"
        return (f'<td><img src="{img}" {attr}>'
                f'<div class=cap>{cap}{" · 🔍 3D" if has_glb else ""}</div></td>')
    trs = []
    for sid, has_alb in rows:
        trs.append(f"""<tr><td class=sid>{sid}</td>
          {cell(sid,'albedo','input appearance',has_alb)}
          {cell(sid,'gtparts',f'GT parts ({a.gt_key})',True)}
          {cell(sid,'predparts','predicted parts (full_seg)',True)}</tr>""")
    html = f"""<!DOCTYPE html><html><head><meta charset=utf-8><title>GT vs predicted parts (3D)</title>
<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
<style>
body{{background:#0e1117;color:#d8dde6;font-family:-apple-system,Segoe UI,sans-serif;padding:20px}}
h1{{font-size:19px}} .meta{{color:#8b949e;font-size:13px;margin-bottom:14px}}
table{{border-collapse:collapse}} td{{padding:6px;text-align:center;border-bottom:1px solid #21262d;vertical-align:top}}
img{{width:330px;height:330px;border-radius:6px;background:#000;object-fit:contain}} .cap{{font-size:11px;color:#8b949e;margin-top:3px}}
.sid{{font-family:ui-monospace,monospace;font-size:11px;color:#9aa4b2;max-width:130px;word-break:break-all;text-align:left}}
img.clk{{cursor:pointer;transition:outline .1s}} img.clk:hover{{outline:2px solid #58a6ff}}
#mw{{position:fixed;inset:0;background:#000d;display:none;z-index:50;flex-direction:column;align-items:center;justify-content:center}}
#mw model-viewer{{width:min(90vw,900px);height:min(85vh,900px);background:#11161d;border-radius:10px}}
#mwbar{{color:#d8dde6;font-size:14px;margin:8px;display:flex;gap:14px;align-items:center}}
#mwbar button{{background:#1f6feb;color:#fff;border:0;border-radius:6px;padding:6px 14px;cursor:pointer}}
</style></head><body>
<h1>GT parts vs predicted full-segmentation — on the GT mesh ({a.res}px) · canonical overfit_split_10</h1>
<div class=meta>Click any image marked <b>🔍 3D</b> to open an interactive orbit/zoom 3D view (loads the GLB on demand). Colors = distinct parts.</div>
<table><tr><th>shape</th><th>appearance</th><th>GT parts</th><th>predicted parts</th></tr>{''.join(trs)}</table>
<div id=mw>
  <div id=mwbar><span id=mwt></span><button onclick="document.getElementById('mw').style.display='none'">✕ close</button></div>
  <model-viewer id=mv camera-controls auto-rotate exposure="1.1" shadow-intensity="0" interaction-prompt="none"></model-viewer>
</div>
<script>
document.querySelectorAll('img.clk').forEach(im=>im.onclick=()=>{{
  document.getElementById('mv').src=im.dataset.glb;
  document.getElementById('mwt').textContent=im.dataset.title;
  document.getElementById('mw').style.display='flex';
}});
document.getElementById('mw').onclick=e=>{{if(e.target.id==='mw')e.currentTarget.style.display='none'}};
</script></body></html>"""
    open(os.path.join(a.html_dir, "index.html"), "w").write(html)
    print(f"wrote {a.html_dir}/index.html with 3D viewer ({len(rows)} shapes)")


if __name__ == "__main__":
    main()
