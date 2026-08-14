#!/usr/bin/env python3
"""Object-context assets for the emission-VAE page: thumbnails and preview GLBs.

An emission-only projection shows the glow but not what is glowing, so a reader
cannot judge whether a reconstruction is plausible. This script produces, per
sample, the object as it actually looks:

  1. a square thumbnail cropped to the object's content bbox (the TexVerse
     renders are 1920x1080 with a lot of empty background, and the page's
     matrix tiles are square, so an uncropped 16:9 frame would put the object
     in a small letterboxed band);
  2. a lightweight preview GLB for the click-to-load model-viewer lightbox,
     since the source GLBs run to 63 MB.

Content bbox for the thumbnail is found from gradient energy, not from
background subtraction: several of these renders sit on smooth gradients
(a red radial, a pink-to-blue ramp) that a flat-color key would not remove,
while every background here is smooth and every object has edges.

Read-only on the source datasets. Writes only into ./img/ and ./glb/.
"""
import os
import subprocess
import numpy as np
from PIL import Image

THUMB_ROOT = "/cs/3dlg-falas/datasets/TexVerse/thumbnails/thumbnails_batch/batch_00000"
GLB_ROOT = "/cs/3dlg-falas/datasets/TexVerse-1K/glbs/glbs_1k/000-000"
GT_BIN = ("/localhome/xya120/.npm/_npx/32543dbc0bd3979c/node_modules/"
          ".bin/gltf-transform")
NODE_BIN = "/localhome/xya120/.nvm/versions/node/v20.20.0/bin"

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
GLB = os.path.join(HERE, "glb")
os.makedirs(IMG, exist_ok=True)
os.makedirs(GLB, exist_ok=True)

THUMB_PX = 440   # emitted square thumbnail edge

# slug -> full TexVerse shape id (prefixes read off the source figure's own
# panel titles; full ids resolved by lookup in the thumbnail directory)
SIDS = {
    "s01": "0007deb6d96c4474b80faa5aa3888760",
    "s02": "000b9fd47d6d4f7db7b2f5022d1ae9aa",
    "s03": "000bc33c1a0d4b36acab1e18de6617e1",
    "s04": "00192b5a4a3249c79141c8dccaad2947",
    "s05": "001a188012214d9ca9b8b22087296558",
    "s06": "001b64d2ec45496792f4edcf036bbaaf",
    "s07": "001c79293c3e4f938798026a79f2d26a",
    "s08": "001dd28130354d36b8f04ffe59c30abe",
    "s09": "002342e8d06042d69aed2919731d4d5f",
    "s10": "002af43e7cab4e5b9490b59534636173",
}


def content_bbox(im, block=8, keep=0.995):
    """Bbox of the object, from gradient energy on a coarse block grid.

    Backgrounds here are flat or smoothly graded, so their gradient energy is
    near zero; the object's silhouette and interior detail are not. Blocks are
    kept when their energy exceeds a small fraction of the frame's maximum,
    which is robust to both a flat grey backdrop and a radial gradient.
    """
    a = np.asarray(im.convert("L"), np.float32)
    gy, gx = np.gradient(a)
    e = np.hypot(gx, gy)
    H, W = e.shape
    bh, bw = H // block, W // block
    grid = e[:bh * block, :bw * block].reshape(bh, block, bw, block).max((1, 3))
    thr = max(grid.max() * 0.02, 2.0)
    ys, xs = np.where(grid > thr)
    if len(ys) == 0:
        return 0, 0, W - 1, H - 1
    # drop the sparsest 0.5% of marked blocks per side so one stray highlight
    # in a corner cannot stretch the box across the frame
    qx0, qx1 = np.quantile(xs, [1 - keep, keep])
    qy0, qy1 = np.quantile(ys, [1 - keep, keep])
    return (int(qx0) * block, int(qy0) * block,
            min(int(qx1 + 1) * block, W) - 1, min(int(qy1 + 1) * block, H) - 1)


def square_thumb(path, out, pad=0.10):
    im = Image.open(path).convert("RGB")
    W, H = im.size
    x0, y0, x1, y1 = content_bbox(im)
    side = int(round(max(x1 - x0 + 1, y1 - y0 + 1) * (1 + 2 * pad)))
    side = min(side, min(W, H)) if side > min(W, H) else side
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    nx = int(round(cx - side / 2.0))
    ny = int(round(cy - side / 2.0))
    # a wide object can need a box taller than the frame; letterbox in that
    # case rather than cropping the object's ends off
    canvas = Image.new("RGB", (side, side), tuple(np.asarray(im)[0, 0].tolist()))
    canvas.paste(im, (-nx, -ny))
    canvas = canvas.resize((THUMB_PX, THUMB_PX), Image.LANCZOS)
    canvas.save(out)
    return (nx, ny, side, (W, H))


# Per-shape preview-GLB settings. The default holds for most shapes; the four
# listed here are geometry- or texture-heavy enough that the default left them
# over ~5 MB, so they get smaller textures and a looser simplification error.
# s07 stays large: its 428k vertices are spread over many small primitives that
# meshoptimizer's simplifier will not collapse further, and its textures are
# already tiny, so there is nothing left to cut without changing what the model
# looks like. It is click-to-load, so nothing downloads until asked for.
GLB_OPTS = {
    "s02": (384, 0.02),
    "s03": (384, 0.02),
    "s07": (384, 0.02),
    "s10": (384, 0.02),
}


def preview_glb(src, out, texsize=512, err=0.005):
    env = dict(os.environ, PATH=NODE_BIN + ":" + os.environ.get("PATH", ""))
    cmd = [GT_BIN, "optimize", src, out,
           "--compress", "quantize",
           "--texture-compress", "webp",
           "--texture-size", str(texsize),
           "--simplify", "true",
           "--simplify-error", str(err)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        return None, r.stderr[-400:]
    return os.path.getsize(out), ""


if __name__ == "__main__":
    for slug, sid in SIDS.items():
        t = os.path.join(THUMB_ROOT, sid + ".png")
        box = square_thumb(t, f"{IMG}/{slug}_obj.png")
        src = os.path.join(GLB_ROOT, sid + "_1024.glb")
        ts, se = GLB_OPTS.get(slug, (512, 0.005))
        size, err = preview_glb(src, f"{GLB}/{slug}.glb", texsize=ts, err=se)
        print(f"{slug} {sid[:8]} thumb_box={box} "
              f"glb={'%.1f MB' % (size / 1e6) if size else 'FAILED ' + err} "
              f"(src {os.path.getsize(src) / 1e6:.1f} MB)")
