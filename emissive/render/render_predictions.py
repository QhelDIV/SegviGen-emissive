"""
Render the fine-tune's predicted white/black voxel map for the 8 finetune_examples sids,
from eval_emissive.py's --dump_vis npz ({coords int16 @512-res, pred_bc float16 continuous
predicted base color in [0,1]-ish, gt_e bool GT emissive}).

Thresholds pred_bc PER-VOXEL first (matching the eval's actual IoU metric), then coarsens
by majority vote per coarse cell for a legible cube render (same coarsen() approach as
render_seg.py). One sample per subprocess (bpy state corrupts after a few in-proc renders).

ALIGNMENT (owner feedback: appearance/GT/pred panels within a row must share orientation,
scale, and position, not each self-normalize to its own bbox): coords here are raw
512-res voxel indices straight from o_voxel, in the SAME "raw mesh" frame as the GLB's
own vertices (never touched by any glTF import). glb_to_vxz.py centers each GLB on its
own raw bounding box and voxelizes onto a grid_size=512 grid spanning aabb
[-0.5,-0.5,-0.5]..[0.5,0.5,0.5]; bpyutil.normalize_mesh() (used for the appearance/GT GLB
renders, via render_finetune_examples.py) centers on that SAME raw bbox and scales to
max-extent 1.9999 -- exactly 2x glb_to_vxz's 0.99999. So a raw voxel index maps into the
identical frame the GLB panels render in via:
    aligned = (index + 0.5) / 256.0 - 1.0
(derived: glb_to_vxz cell center = -0.5 + (index+0.5)/512; bpyutil frame = that * 2).
voxel_mesh() below places coarse cells using this formula instead of self-fitting to their
own bbox, so flipping between appearance/GT/pred columns looks like a texture change on a
fixed object, not a different zoom/pose.

  /localhome/xya120/studio/misc/lightgen/lightgen_repo/.venv/bin/python render_predictions.py \
      --dump_dir vis_data/finetune_examples --npz_dir dump_w5ema --tag w5ema --thr 0.2
"""
import os, sys, glob, argparse
import numpy as np

sys.path.insert(0, "/localhome/xya120/studio/misc/lightgen/lightgen_repo")
from xgutils import bpyutil
from xgutils.miscutil import preset_glb

WHITE = np.array([0.96, 0.97, 0.98], np.float32)
BLACK = np.array([0.10, 0.11, 0.13], np.float32)

_CUBE_V = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], np.float32) - 0.5
_CUBE_F = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[1,5,6],[1,6,2],
                    [2,6,7],[2,7,3],[3,7,4],[3,4,0]], np.int32)


def coarsen_binary(coords, is_white, factor=4, max_cells=15000):
    """Downsample to a coarse grid; per cell: majority vote of the per-voxel binary
    prediction (mirrors render_seg.py's coarsen(), specialized to a 2-class vote).
    Returns (uniq, maj_white, factor) -- factor is needed by voxel_mesh() to place cells
    in the aligned frame (a coarse cell's size/position depends on how many raw voxels
    it groups)."""
    c64 = coords.astype(np.int64)
    while True:
        cc = c64 // factor
        uniq, inv = np.unique(cc, axis=0, return_inverse=True)
        if len(uniq) <= max_cells or factor >= 64:
            break
        factor *= 2
    n = len(uniq)
    cnt = np.bincount(inv, minlength=n)
    white_cnt = np.bincount(inv, weights=is_white.astype(np.float64), minlength=n)
    maj_white = (white_cnt / cnt) > 0.5
    return uniq, maj_white, factor


def voxel_mesh(uniq, factor, color, pad=0.92):
    """Place coarse voxel cells in the SAME normalized frame bpyutil uses for the
    appearance/GT GLB renders (see module docstring for the derivation) -- NOT a
    self-fit-to-own-bbox normalization, so this aligns with the other panels in the row."""
    uniq = uniq.astype(np.float32)
    centers = (uniq * factor + factor / 2.0) / 256.0 - 1.0
    cell = (factor / 256.0) * pad
    cube = _CUBE_V * cell
    V = (centers[:, None, :] + cube[None, :, :]).reshape(-1, 3)
    F = (_CUBE_F[None] + (np.arange(len(uniq)) * 8)[:, None, None]).reshape(-1, 3)
    C = np.repeat(color, 8, axis=0)
    return V.astype(np.float32), F.astype(np.int32), C.astype(np.float32)


def _save(img, path):
    from PIL import Image
    img = np.asarray(img, np.float32)
    if img.ndim == 3 and img.shape[2] == 4:
        rgb, a = img[..., :3], img[..., 3:4]
        img = rgb * a + 0.07 * (1 - a)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def render_one(npz_path, out_path, thr, res=512):
    d = np.load(npz_path)
    coords = d["coords"]
    is_white = d["pred_bc"].astype(np.float32) > thr
    cc, maj_white, factor = coarsen_binary(coords, is_white)
    color = np.where(maj_white[:, None], WHITE[None], BLACK[None]).astype(np.float32)
    V, F, C = voxel_mesh(cc, factor, color)
    bpyutil.load_blend(preset_glb)
    bpyutil.clear_collection("workbench")
    img = bpyutil.render_mesh(V, F, vert_color=C, resolution=(res, res), samples=24,
                              shadow_catcher=False, camera_position=(0, -2.6, 1.4),
                              camera_up=(0, 0, 1))
    _save(img, out_path)


def main():
    import subprocess
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True, help="vis_data/finetune_examples (per-sid subdirs)")
    ap.add_argument("--npz_dir", required=True, help="local dir with <sid>.npz dumps")
    ap.add_argument("--tag", required=True, help="e.g. w5ema / w1ema -> writes render_pred_<tag>.png")
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--sid", default=None)
    a = ap.parse_args()

    if a.sid:
        npz = os.path.join(a.npz_dir, f"{a.sid}.npz")
        out = os.path.join(a.dump_dir, a.sid, f"render_pred_{a.tag}.png")
        render_one(npz, out, a.thr, a.res)
        print(f"[ok] {a.sid}", flush=True)
        return

    sids = sorted(d for d in os.listdir(a.dump_dir) if os.path.isdir(os.path.join(a.dump_dir, d)))
    print(f"{len(sids)} sids, tag={a.tag} thr={a.thr}", flush=True)
    for i, sid in enumerate(sids):
        npz = os.path.join(a.npz_dir, f"{sid}.npz")
        out = os.path.join(a.dump_dir, sid, f"render_pred_{a.tag}.png")
        if not os.path.exists(npz):
            print(f"  [{i+1}/{len(sids)}] {sid} MISSING npz", flush=True)
            continue
        if os.path.exists(out):
            continue
        cmd = [sys.executable, os.path.abspath(__file__), "--sid", sid, "--dump_dir", a.dump_dir,
               "--npz_dir", a.npz_dir, "--tag", a.tag, "--thr", str(a.thr), "--res", str(a.res)]
        try:
            subprocess.run(cmd, timeout=180)
        except subprocess.TimeoutExpired:
            print(f"  [{i+1}/{len(sids)}] {sid} TIMEOUT", flush=True)
            continue
        ok = os.path.exists(out)
        print(f"  [{i+1}/{len(sids)}] {sid} {'ok' if ok else 'FAILED'}", flush=True)


if __name__ == "__main__":
    main()
