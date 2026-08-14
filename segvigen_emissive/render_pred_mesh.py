"""
Render a paper-style predicted-emissive GLB (from code/make_pred_glb.py -> slat_to_glb)
into a PNG, using the SAME recipe render_finetune_examples.render_glb uses for the
appearance/GT panels (load_glb + render_scene, camera (0,-2.6,1.4)) so pred/GT/appearance
share orientation, scale, and camera within a results_2k row.

Orientation: the appearance/GT GLBs are somage-exported (double-rotated) and need
fix_glb_upright. The slat_to_glb pred GLB may or may not — pass --both on one shape to
render fix / nofix side by side and pick whichever matches render_emissive.png, then use
that mode (--fix / --nofix) for the rest.

  lightgen_repo/.venv/bin/python render_pred_mesh.py --glb pred.glb --out out.png --both
  lightgen_repo/.venv/bin/python render_pred_mesh.py --glb pred.glb --out out.png --fix
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, "/localhome/xya120/studio/misc/lightgen/lightgen_repo")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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


def render_glb(glb_path, out_path, fix=True, res=512):
    bpyutil.load_blend(preset_glb)
    bpyutil.clear_collection("workbench")
    obj = bpyutil.load_glb(glb_path, import_shading=None)
    if fix:
        fix_glb_upright(obj)
    img = bpyutil.render_scene(obj=obj, resolution=(res, res), samples=32,
                               camera_position=(0, -2.6, 1.4), camera_up=(0, 0, 1),
                               shadow_catcher=False)
    bpyutil.purge_obj(obj)
    _save(img, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=512)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fix", action="store_true", help="apply fix_glb_upright (default)")
    g.add_argument("--nofix", action="store_true", help="do NOT apply fix_glb_upright")
    g.add_argument("--both", action="store_true", help="render <out>_fix.png and <out>_nofix.png")
    a = ap.parse_args()

    if a.both:
        base, ext = os.path.splitext(a.out)
        render_glb(a.glb, base + "_fix.png", fix=True, res=a.res)
        render_glb(a.glb, base + "_nofix.png", fix=False, res=a.res)
        print(f"[ok] wrote {base}_fix.png and {base}_nofix.png", flush=True)
    else:
        fix = not a.nofix
        render_glb(a.glb, a.out, fix=fix, res=a.res)
        print(f"[ok] wrote {a.out} (fix={fix})", flush=True)


if __name__ == "__main__":
    main()
