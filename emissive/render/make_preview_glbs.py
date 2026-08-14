"""
Produce LIGHTWEIGHT preview GLBs for the in-browser model-viewer lightbox on
results_2k_v1's Paper-style mesh view.

v2 (2026-07-08, owner feedback): keep the baked TEXTURE (crisp, mesh-density-independent
boundary) instead of baking to per-vertex COLOR_0 (which INTERPOLATES across each boundary
triangle -> smeared edge). And lift the non-emissive color off pure black to a shadeable
dark gray so its geometry/form catches light in model-viewer (pure (0,0,0) albedo reflects
nothing, so no lighting can reveal shape). White (emissive) is kept white. This is a
PREVIEW-GLB presentation choice only — the metric/thresholding is unaffected.

Modes:
  --mode tex   : keep the GLB's baseColorTexture; optionally downscale to --texsize and
                 lift dark texels to gray (--gray). matte PBR (metallic 0, roughness 1) so
                 the gray form shades. Use for pred (4096^2 -> 2048^2) and GT (tiny UV atlas,
                 no downscale needed) — both white/black-on-mesh.
  --mode copy  : re-export preserving the visual as-is (appearance PBR — no black issue).

Orientation (model-viewer = standard glTF viewer, +Y up, NO Blender Y-up->Z-up):
  --rot none : slat_to_glb pred GLB (already Y-up).
  --rot xn90 : somage-exported GT/appearance GLB (stores Z-up in a Y-up container).

  lightgen_repo/.venv/bin/python make_preview_glbs.py --src pred.glb --out p.glb \
      --mode tex --texsize 2048 --gray 0.18 --rot none
"""
import os, sys, argparse
import numpy as np
import trimesh
from PIL import Image


def _rot_matrix(rot):
    if rot == "none":
        return None
    a = np.pi / 2 if rot == "xp90" else -np.pi / 2
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], np.float64)


def load_mesh(path):
    m = trimesh.load(path, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = m.to_mesh()
    return m


def lift_dark_to_gray(img, gray):
    """Lift the darkest texels to a shadeable gray floor (per channel max with gray*255),
    leaving white/AA texels untouched -> no pure black, boundary stays crisp."""
    a = np.array(img.convert("RGBA"))
    floor = int(round(gray * 255))
    a[..., :3] = np.maximum(a[..., :3], floor)
    return Image.fromarray(a, "RGBA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["tex", "copy"], default="tex")
    ap.add_argument("--rot", choices=["none", "xp90", "xn90"], default="none")
    ap.add_argument("--texsize", type=int, default=0,
                    help="downscale baseColorTexture to this square size (0 = leave as-is)")
    ap.add_argument("--gray", type=float, default=-1.0,
                    help=">=0: lift dark texels to this gray level (0..1); <0: no remap")
    a = ap.parse_args()

    m = load_mesh(a.src)
    R = _rot_matrix(a.rot)
    if R is not None:
        m.apply_transform(R)

    if a.mode == "tex":
        mat = m.visual.material
        img = getattr(mat, "baseColorTexture", None)
        if img is None:
            print("[warn] no baseColorTexture; exporting as-is", flush=True)
        else:
            if a.texsize and max(img.size) > a.texsize:
                img = img.resize((a.texsize, a.texsize), Image.LANCZOS)
            if a.gray >= 0:
                img = lift_dark_to_gray(img, a.gray)
            mat.baseColorTexture = img
            # matte so the gray non-emissive form shades under IBL (no shiny speculars)
            try:
                mat.metallicFactor = 0.0
                mat.roughnessFactor = 1.0
                mat.metallicRoughnessTexture = None
                mat.emissiveTexture = None
                mat.emissiveFactor = [0, 0, 0]
            except Exception as e:
                print(f"[warn] material tweak: {e!r}", flush=True)
            m.visual.material = mat

    m.export(a.out)
    sz = os.path.getsize(a.out)
    print(f"[ok] {os.path.basename(a.out)}  {sz/1e6:.2f}MB  (mode={a.mode}, rot={a.rot}, "
          f"texsize={a.texsize or 'orig'}, gray={a.gray})", flush=True)


if __name__ == "__main__":
    main()
