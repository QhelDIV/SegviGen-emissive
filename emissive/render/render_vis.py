"""
Render a visual contact sheet for the emissive eval: per val sample, show
  [ input albedo GLB | GT emissive voxels | predicted emissive voxels ]
so we can SEE failure modes (muted colors, emissive-heavy blindness) that the
scalar IoU hides.

Input: a dir of per-sample npz dumped by eval_emissive.py --dump_vis, each with
  coords (N,3 int voxel xyz), pred_bc (N, soft emissive), gt_e (N, bool GT).
Optionally the dataset dir (for input.glb albedo context).

Runs LOCALLY with the bpy venv (no GPU needed):
  /localhome/xya120/studio/misc/lightgen/lightgen_repo/.venv/bin/python render_vis.py \
      --vis_dir .../eval_vis_v3ep8 --dataset .../dataset --split val \
      --thresh 0.3 --out .../vis_v3ep8

Produces <out>/<sid>_{gt,pred}.png (+ _albedo.png if GLB ok) and <out>/index.html.
"""
import os, sys, argparse, glob
import numpy as np

sys.path.insert(0, "/localhome/xya120/studio/misc/lightgen/lightgen_repo")
from xgutils import bpyutil
from xgutils.miscutil import preset_glb

EMIS = np.array([1.0, 0.63, 0.16], np.float32)   # warm orange
NON  = np.array([0.23, 0.23, 0.23], np.float32)  # dark grey

# unit cube corners / triangles (12 tris)
_CUBE_V = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], np.float32) - 0.5
_CUBE_F = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[1,5,6],[1,6,2],
                    [2,6,7],[2,7,3],[3,7,4],[3,4,0]], np.int32)


def coarsen(coords, mask, factor=8, max_cells=15000):
    """Downsample to a coarse grid (decoded voxels are 0.5-5M @ res512 → too many cubes).
    Adaptive: increase the factor until occupied cells <= max_cells, so even the biggest
    samples render reliably (bpy gets unstable with very large meshes).
    Returns coarse coords + majority-vote emissive mask per occupied cell."""
    c64 = coords.astype(np.int64)
    while True:
        cc = c64 // factor
        uniq, inv = np.unique(cc, axis=0, return_inverse=True)
        if len(uniq) <= max_cells or factor >= 64:
            break
        factor *= 2
    frac = np.bincount(inv, weights=mask.astype(np.float64), minlength=len(uniq)) \
         / np.bincount(inv, minlength=len(uniq))
    return uniq, frac > 0.5


def voxel_mesh(coords, emis_mask):
    """coords (N,3) int, emis_mask (N,) bool → exploded cube mesh + per-vert color."""
    coords = coords.astype(np.float32)
    span = coords.max(0) - coords.min(0)
    s = float(span.max()) or 1.0
    centers = (coords - coords.min(0) - span / 2.0) / s          # → roughly [-0.5,0.5]
    cube = _CUBE_V * (1.0 / s) * 0.95                            # cube ~= one voxel cell
    V = (centers[:, None, :] + cube[None, :, :]).reshape(-1, 3)  # (8N,3)
    F = (_CUBE_F[None] + (np.arange(len(coords)) * 8)[:, None, None]).reshape(-1, 3)
    col = np.where(emis_mask[:, None], EMIS, NON)                # (N,3)
    C = np.repeat(col, 8, axis=0)                                # (8N,3)
    return V.astype(np.float32), F.astype(np.int32), C.astype(np.float32)


def render_voxels(coords, mask, path, res=420, factor=8):
    coords, mask = coarsen(coords, mask, factor)
    V, F, C = voxel_mesh(coords, mask)
    bpyutil.load_blend(preset_glb)
    bpyutil.clear_collection("workbench")
    img = bpyutil.render_mesh(V, F, vert_color=C, resolution=(res, res), samples=24,
                              shadow_catcher=False, camera_position=(0, -2.6, 1.4),
                              camera_up=(0, 0, 1))
    _save(img, path)


def render_albedo(glb_path, path, res=420):
    try:
        bpyutil.load_blend(preset_glb)
        bpyutil.clear_collection("workbench")
        obj = bpyutil.load_glb(glb_path, import_shading=None)
        img = bpyutil.render_scene(obj=obj, resolution=(res, res), samples=32,
                                   camera_position=(0, -2.6, 1.4), camera_up=(0, 0, 1),
                                   shadow_catcher=False)
        bpyutil.purge_obj(obj)
        _save(img, path)
        return True
    except Exception as e:
        print(f"  [albedo skip] {os.path.basename(glb_path)}: {repr(e)[:120]}", flush=True)
        return False


def _save(img, path):
    from PIL import Image
    img = np.asarray(img, np.float32)
    if img.ndim == 3 and img.shape[2] == 4:
        rgb, a = img[..., :3], img[..., 3:4]
        img = rgb * a + 0.07 * (1 - a)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def render_one(npz, out, thresh, dataset, split):
    """Worker: render the 3 panels for ONE sample (own process → bpy crash is isolated)."""
    sid = os.path.splitext(os.path.basename(npz))[0]
    d = np.load(npz)
    coords, pred_bc, gt_e = d["coords"], d["pred_bc"].astype(np.float32), d["gt_e"].astype(bool)
    pred_e = pred_bc > thresh
    render_voxels(coords, gt_e,   os.path.join(out, f"{sid}_gt.png"))
    render_voxels(coords, pred_e, os.path.join(out, f"{sid}_pred.png"))
    if dataset:
        g = os.path.join(dataset, split, sid, "glb", f"{sid}_input.glb")
        if os.path.exists(g):
            render_albedo(g, os.path.join(out, f"{sid}_albedo.png"))


def main():
    import subprocess
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis_dir", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sid", default=None, help="internal worker: render one npz then exit")
    ap.add_argument("--meta_extra", default="",
                    help="extra text appended to the header (e.g. train IoU for overfit diagnosis)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # worker mode: one sample per process — bpy in this build crashes after a few
    # renders in-process, so isolation is required for a full batch.
    if a.sid:
        render_one(os.path.join(a.vis_dir, f"{a.sid}.npz"), a.out, a.thresh, a.dataset, a.split)
        return

    npzs = sorted(glob.glob(os.path.join(a.vis_dir, "*.npz")))
    print(f"{len(npzs)} samples; threshold {a.thresh}", flush=True)
    for i, p in enumerate(npzs):
        sid = os.path.splitext(os.path.basename(p))[0]
        if os.path.exists(os.path.join(a.out, f"{sid}_pred.png")):
            print(f"  [{i+1}/{len(npzs)}] {sid} (cached)", flush=True); continue
        cmd = [sys.executable, os.path.abspath(__file__), "--sid", sid, "--vis_dir", a.vis_dir,
               "--out", a.out, "--thresh", str(a.thresh), "--split", a.split]
        if a.dataset:
            cmd += ["--dataset", a.dataset]
        try:
            subprocess.run(cmd, timeout=180)
        except subprocess.TimeoutExpired:
            pass
        ok = os.path.exists(os.path.join(a.out, f"{sid}_pred.png"))
        print(f"  [{i+1}/{len(npzs)}] {sid} {'ok' if ok else 'FAILED'}", flush=True)

    # build cards from whatever rendered + recompute IoU from npz (cheap, no bpy)
    cards = []
    for p in npzs:
        sid = os.path.splitext(os.path.basename(p))[0]
        if not os.path.exists(os.path.join(a.out, f"{sid}_pred.png")):
            continue
        d = np.load(p)
        gt_e = d["gt_e"].astype(bool); pred_e = d["pred_bc"].astype(np.float32) > a.thresh
        union = (pred_e | gt_e).sum()
        iou = (pred_e & gt_e).sum() / union if union > 0 else 1.0
        alb = f"{sid}_albedo.png" if os.path.exists(os.path.join(a.out, f"{sid}_albedo.png")) else None
        cards.append((sid, float(iou), float(gt_e.mean()), float(pred_e.mean()), alb))

    cards.sort(key=lambda c: c[1])  # worst IoU first — failures up top for self-check
    miou = np.mean([c[1] for c in cards]) if cards else 0.0
    rows = []
    for sid, iou, gtf, prf, alb in cards:
        albcell = f'<td><img src="{alb}"><div class=cap>input albedo</div></td>' if alb else \
                  '<td class=na>(no albedo)</td>'
        rows.append(f"""<tr><td class=sid>{sid}<br><span class=iou>IoU {iou:.3f}</span>
          <div class=cap>gt {gtf:.2f} · pred {prf:.2f}</div></td>{albcell}
          <td><img src="{sid}_gt.png"><div class=cap>GT emissive</div></td>
          <td><img src="{sid}_pred.png"><div class=cap>predicted</div></td></tr>""")
    html = f"""<!DOCTYPE html><html><head><meta charset=utf-8><title>emissive vis</title><style>
    body{{background:#0e1117;color:#d8dde6;font-family:-apple-system,Segoe UI,sans-serif;padding:20px}}
    h1{{font-size:19px}} .meta{{color:#8b949e;font-size:13px;margin-bottom:14px}}
    table{{border-collapse:collapse}} td{{padding:6px;text-align:center;border-bottom:1px solid #21262d;vertical-align:top}}
    img{{width:240px;height:240px;border-radius:6px;background:#000}} .cap{{font-size:11px;color:#8b949e;margin-top:3px}}
    .sid{{font-family:ui-monospace,monospace;font-size:11px;color:#9aa4b2;max-width:150px;word-break:break-all;text-align:left}}
    .iou{{font-size:15px;font-weight:700;color:#fff}} .na{{color:#555;font-size:12px}}
    </style></head><body><h1>Emissive eval — GT vs predicted (worst IoU first)</h1>
    <div class=meta>{len(cards)} val samples · <b>val mean IoU@{a.thresh} = {miou:.4f}</b>{(' · ' + a.meta_extra) if a.meta_extra else ''} · DiffusionNet baseline 0.259 ·
    orange = emissive, grey = non-emissive · voxel-space (matches the IoU metric)</div>
    <table>{''.join(rows)}</table></body></html>"""
    open(os.path.join(a.out, "index.html"), "w").write(html)
    print(f"\nwrote {a.out}/index.html  (mean IoU@{a.thresh} = {miou:.4f})", flush=True)


if __name__ == "__main__":
    main()
