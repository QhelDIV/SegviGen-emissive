"""
Render the 5 "how a training sample is made" hero images for finetune_examples_html
Section A, using d5fb4f19d4164612b165caac5471555c (a recognizable green/yellow fish with
a small black body region and a few white emissive markings — eye, top fin, belly patch)
at a consistent 3/4 camera angle, composited over a LIGHT background so shapes read
against the page's dark theme (Section A only; the B+C table is untouched).

Produces, in vis_data/finetune_examples_html/:
  secA_1_source.png   — artist source (input.glb, textured appearance)
  secA_2_target.png   — the emissive.glb target (white=emissive/black=not), same camera
  secA_3_voxel.png     — dense 512-res surface voxels, colored white/black by GT emissive
                         (from the coords+gt_e npz dump — the actual per-voxel GT, not a mockup)
  secA_4_latent.png    — sparse 32-res latent blocks (coords // 16, actual block count),
                         majority-colored white/black — the same shape, now a handful of tokens
  secA_5_photo.png     — img.png (bg-removed cond photo) composited onto the same light backdrop

  python render_section_a.py
"""
import os, sys
import numpy as np

sys.path.insert(0, "/localhome/xya120/studio/misc/lightgen/lightgen_repo")
from xgutils import bpyutil
from xgutils.miscutil import preset_glb
from glb_orient_fix import fix_glb_upright

SID = "d5fb4f19d4164612b165caac5471555c"
EX_DIR = "/local-scratch2/xya120/studio/misc/lightgen/segvigen_emissive/vis_data/finetune_examples"
OUT_DIR = "/local-scratch2/xya120/studio/misc/lightgen/segvigen_emissive/vis_data/finetune_examples_html"
NPZ = "/local-scratch2/xya120/studio/misc/lightgen/segvigen_emissive/dump_w5ema/d5fb4f19d4164612b165caac5471555c.npz"

# CAM_POS re-picked after fixing the GLB up-axis bug (see glb_orient_fix.py): the previous
# camera (-1.5,-2.1,1.0) was empirically tuned against the WRONG (90 deg off) orientation,
# so it had to be redone once the fix was applied. This one is low-elevation/side-biased
# per the same "flat/wide shape" rule, and shows eye + tail + belly marking clearly with
# the fish now genuinely upright (dorsal ridge up, belly down) rather than lying on its side.
CAM_POS = (-2.0, -2.0, 0.8)
CAM_UP = (0, 0, 1)
BG = 0.94   # light neutral backdrop (vs. the near-black 0.07 used elsewhere on this page)
RES = 480

WHITE = np.array([0.97, 0.98, 0.99], np.float32)
BLACK = np.array([0.09, 0.10, 0.12], np.float32)

_CUBE_V = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], np.float32) - 0.5
_CUBE_F = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[1,5,6],[1,6,2],
                    [2,6,7],[2,7,3],[3,7,4],[3,4,0]], np.int32)


def _save(img, path, bg=BG):
    from PIL import Image
    img = np.asarray(img, np.float32)
    if img.ndim == 3 and img.shape[2] == 4:
        rgb, a = img[..., :3], img[..., 3:4]
        img = rgb * a + bg * (1 - a)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def render_glb_appearance(path, out):
    bpyutil.load_blend(preset_glb)
    bpyutil.clear_collection("workbench")
    obj = bpyutil.load_glb(path, import_shading=None)
    fix_glb_upright(obj)
    img = bpyutil.render_scene(obj=obj, resolution=(RES, RES), samples=48,
                               camera_position=CAM_POS, camera_up=CAM_UP,
                               shadow_catcher=True)
    bpyutil.purge_obj(obj)
    _save(img, out)


def voxel_mesh(coords, color, pad=0.95):
    coords = coords.astype(np.float32)
    span = coords.max(0) - coords.min(0)
    s = float(span.max()) or 1.0
    centers = (coords - coords.min(0) - span / 2.0) / s
    cube = _CUBE_V * (1.0 / s) * pad
    V = (centers[:, None, :] + cube[None, :, :]).reshape(-1, 3)
    F = (_CUBE_F[None] + (np.arange(len(coords)) * 8)[:, None, None]).reshape(-1, 3)
    C = np.repeat(color, 8, axis=0)
    return V.astype(np.float32), F.astype(np.int32), C.astype(np.float32)


def coarsen_majority(coords, is_white, factor, max_cells):
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
    return uniq, maj_white


def render_cubes(coords, maj_white, out, pad=0.95):
    color = np.where(maj_white[:, None], WHITE[None], BLACK[None]).astype(np.float32)
    V, F, C = voxel_mesh(coords, color, pad=pad)
    bpyutil.load_blend(preset_glb)
    bpyutil.clear_collection("workbench")
    img = bpyutil.render_mesh(V, F, vert_color=C, resolution=(RES, RES), samples=32,
                              shadow_catcher=True, camera_position=CAM_POS, camera_up=CAM_UP)
    _save(img, out)


def stage1():
    render_glb_appearance(os.path.join(EX_DIR, SID, "glb", f"{SID}_input.glb"),
                          os.path.join(OUT_DIR, "secA_1_source.png"))


def stage2():
    render_glb_appearance(os.path.join(EX_DIR, SID, "glb", f"{SID}_emissive.glb"),
                          os.path.join(OUT_DIR, "secA_2_target.png"))


def stage3():
    d = np.load(NPZ)
    coords = d["coords"]
    gt_e = d["gt_e"]
    print(f"  dense surface voxels: {len(coords)} (white frac {gt_e.mean():.3f})", flush=True)
    cc, maj = coarsen_majority(coords, gt_e, factor=2, max_cells=18000)
    render_cubes(cc, maj, os.path.join(OUT_DIR, "secA_3_voxel.png"), pad=0.96)
    print(f"  stage 3: {len(cc)} coarse cells", flush=True)


def stage4():
    d = np.load(NPZ)
    coords = d["coords"]
    gt_e = d["gt_e"]
    block = coords.astype(np.int64) // 16
    uniq, inv = np.unique(block, axis=0, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(uniq))
    white_cnt = np.bincount(inv, weights=gt_e.astype(np.float64), minlength=len(uniq))
    maj_white = (white_cnt / cnt) > 0.5
    print(f"  sparse latent tokens: {len(uniq)} (majority-white: {maj_white.sum()})", flush=True)
    render_cubes(uniq, maj_white, os.path.join(OUT_DIR, "secA_4_latent.png"), pad=0.85)
    with open(os.path.join(OUT_DIR, "secA_stats.txt"), "w") as f:
        f.write(f"sid={SID}\ndense_voxels={len(coords)}\nlatent_tokens={len(uniq)}\n"
                f"gt_white_frac={gt_e.mean():.4f}\n")


def stage5():
    from PIL import Image
    photo = np.asarray(Image.open(os.path.join(EX_DIR, SID, "img.png")).convert("RGBA"), np.float32) / 255.0
    _save(photo, os.path.join(OUT_DIR, "secA_5_photo.png"))


STAGES = {"1": stage1, "2": stage2, "3": stage3, "4": stage4, "5": stage5}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if len(sys.argv) > 1:
        STAGES[sys.argv[1]]()
        print(f"[ok] stage {sys.argv[1]}", flush=True)
        return
    import subprocess
    for k in ["1", "2", "3", "4", "5"]:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), k], timeout=180)
        print(f"[{'ok' if r.returncode == 0 else 'FAILED'}] stage {k}", flush=True)


if __name__ == "__main__":
    main()
