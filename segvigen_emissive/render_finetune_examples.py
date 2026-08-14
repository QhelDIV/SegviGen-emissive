"""
Render appearance + GT-emissive-target GLBs for the 8 hand-picked finetune_examples sids
(one sample per subprocess — bpy state corrupts after a few in-process renders).

  /localhome/xya120/studio/misc/lightgen/lightgen_repo/.venv/bin/python render_finetune_examples.py \
      --dir vis_data/finetune_examples
"""
import os, sys, glob, argparse
import numpy as np

sys.path.insert(0, "/localhome/xya120/studio/misc/lightgen/lightgen_repo")
from xgutils import bpyutil
from xgutils.miscutil import preset_glb
from glb_orient_fix import fix_glb_upright


def _save(img, path):
    from PIL import Image
    img = np.asarray(img, np.float32)
    if img.ndim == 3 and img.shape[2] == 4:
        rgb, a = img[..., :3], img[..., 3:4]
        img = rgb * a + 0.07 * (1 - a)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def render_glb(glb_path, out_path, res=512):
    bpyutil.load_blend(preset_glb)
    bpyutil.clear_collection("workbench")
    obj = bpyutil.load_glb(glb_path, import_shading=None)
    fix_glb_upright(obj)
    img = bpyutil.render_scene(obj=obj, resolution=(res, res), samples=32,
                               camera_position=(0, -2.6, 1.4), camera_up=(0, 0, 1),
                               shadow_catcher=False)
    bpyutil.purge_obj(obj)
    _save(img, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sid", default=None)
    ap.add_argument("--kind", default=None, choices=["input", "emissive"])
    ap.add_argument("--res", type=int, default=512)
    a = ap.parse_args()

    if a.sid:
        glb = os.path.join(a.dir, a.sid, "glb", f"{a.sid}_{a.kind}.glb")
        out = os.path.join(a.dir, a.sid, f"render_{a.kind}.png")
        render_glb(glb, out, a.res)
        print(f"[ok] {a.sid} {a.kind}", flush=True)
        return

    import subprocess
    sids = sorted(d for d in os.listdir(a.dir) if os.path.isdir(os.path.join(a.dir, d)))
    print(f"{len(sids)} sids", flush=True)
    for i, sid in enumerate(sids):
        for kind in ["input", "emissive"]:
            out = os.path.join(a.dir, sid, f"render_{kind}.png")
            if os.path.exists(out):
                continue
            cmd = [sys.executable, os.path.abspath(__file__), "--sid", sid, "--kind", kind,
                   "--dir", a.dir, "--res", str(a.res)]
            try:
                r = subprocess.run(cmd, timeout=180)
            except subprocess.TimeoutExpired:
                print(f"  [{sid} {kind}] TIMEOUT", flush=True)
                continue
            ok = os.path.exists(out)
            print(f"  [{i+1}/{len(sids)}] {sid} {kind} {'ok' if ok else 'FAILED'}", flush=True)


if __name__ == "__main__":
    main()
