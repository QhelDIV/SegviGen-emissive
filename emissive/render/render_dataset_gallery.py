"""
Render appearance + GT-target thumbnails for the 48-sample dataset gallery (12 per
emissive_frac bucket, random seed 42, from train_1k — see compute_dataset_stats.py).
Same 3/4 camera + light backdrop as render_section_a.py, at 320px (small enough to keep
render time + page weight sane for ~96 renders). One subprocess per render (bpy state
corrupts after a few in-process renders).

Applies fix_glb_upright() (see glb_orient_fix.py) to every loaded GLB: confirmed via a
decisive test against 4 shapes' `somage_original_mesh.npz` (the true dataset/somage frame,
rendered directly with no GLB import at all) that plain bpyutil.load_glb double-rotates
every exported GLB relative to that true frame — Dongchen's somage-creation step already
applies the glTF Y-up->Z-up conversion once (via its own bpyutil.load_glb call) before
baking `somage_original_mesh.npz`, so our later GLB export + reimport applies it AGAIN.
This is systematic, not per-asset — a source asset that is genuinely authored sideways
(confirmed on one gallery shape, a dynamite bundle) will still render sideways after the
fix, because that IS the frame the model and the rest of the data pipeline see.

  /localhome/xya120/studio/misc/lightgen/lightgen_repo/.venv/bin/python render_dataset_gallery.py
"""
import os, sys, glob, argparse
import numpy as np

sys.path.insert(0, "/localhome/xya120/studio/misc/lightgen/lightgen_repo")
from xgutils import bpyutil
from xgutils.miscutil import preset_glb
from glb_orient_fix import fix_glb_upright

DIR = "/local-scratch2/xya120/studio/misc/lightgen/segvigen_emissive/vis_data/dataset_gallery"
CAM_POS = (-1.9, -1.9, 1.6)
CAM_UP = (0, 0, 1)
BG = 0.94
RES = 320


def _save(img, path, bg=BG):
    from PIL import Image
    img = np.asarray(img, np.float32)
    if img.ndim == 3 and img.shape[2] == 4:
        rgb, a = img[..., :3], img[..., 3:4]
        img = rgb * a + bg * (1 - a)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def render_glb(glb_path, out_path, res=RES):
    bpyutil.load_blend(preset_glb)
    bpyutil.clear_collection("workbench")
    obj = bpyutil.load_glb(glb_path, import_shading=None)
    fix_glb_upright(obj)
    img = bpyutil.render_scene(obj=obj, resolution=(res, res), samples=32,
                               camera_position=CAM_POS, camera_up=CAM_UP,
                               shadow_catcher=True)
    bpyutil.purge_obj(obj)
    _save(img, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default=None)
    ap.add_argument("--kind", default=None, choices=["input", "emissive"])
    a = ap.parse_args()

    if a.sid:
        glb = os.path.join(DIR, a.sid, "glb", f"{a.sid}_{a.kind}.glb")
        out = os.path.join(DIR, a.sid, f"render_{a.kind}.png")
        render_glb(glb, out)
        print(f"[ok] {a.sid} {a.kind}", flush=True)
        return

    import subprocess
    sids = sorted(d for d in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, d)))
    print(f"{len(sids)} sids", flush=True)
    n_fail = 0
    for i, sid in enumerate(sids):
        for kind in ["input", "emissive"]:
            out = os.path.join(DIR, sid, f"render_{kind}.png")
            if os.path.exists(out):
                continue
            cmd = [sys.executable, os.path.abspath(__file__), "--sid", sid, "--kind", kind]
            try:
                subprocess.run(cmd, timeout=120)
            except subprocess.TimeoutExpired:
                print(f"  [{i+1}/{len(sids)}] {sid} {kind} TIMEOUT", flush=True)
                n_fail += 1
                continue
            ok = os.path.exists(out)
            if not ok:
                n_fail += 1
            print(f"  [{i+1}/{len(sids)}] {sid} {kind} {'ok' if ok else 'FAILED'}", flush=True)
    print(f"\n{n_fail} failures", flush=True)


if __name__ == "__main__":
    main()
