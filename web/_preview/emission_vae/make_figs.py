#!/usr/bin/env python3
"""Derive legible panels for the emission-VAE page from Dongchen's source PNGs.

Two problems make the source figures unreadable at a glance:

1. The emissive region occupies a tiny fraction of each 256-voxel projection
   panel, so a reader sees an almost entirely black frame (xgpage design law
   D9: sparse canvases get cropped to their content bbox before sizing).
2. Every failure mode lives in the dark range. At native exposure "zero" and
   "nearly zero but wrong" are the same pixel, so the leak the caption talks
   about is invisible.

This script therefore, for each sample:
  - locates the content bbox from the ground-truth panel unioned with the
    BRIGHT part of the reconstruction (so a reconstruction that puts its
    content somewhere else stays inside the frame), pads it, squares it, and
    crops EVERY panel of that sample to the SAME box so the panels stay
    registered;
  - upscales with nearest-neighbor, which keeps one source pixel one visible
    block rather than smearing it;
  - emits a native-exposure panel and a cube-root tone-curve panel
    (display value v -> 255*(v/255)**(1/3)), so near-black-but-nonzero content
    becomes visible;
  - emits a two-color leak map: white where ground truth is nonzero, accent
    where the reconstruction is nonzero and the ground truth is exactly zero.

The overfit figure is a matplotlib `hot` colormap render (verified: its
generator is data_toolkit/vis_emission_vae.py in the same code snapshot), so
its panels are first inverted through the hot lookup table back to the scalar
field before any tone curve is applied. The ten-sample figure is direct RGB,
so the tone curve is applied per channel.

Read-only on the sources. Writes only into ./img/.
"""
import os
import numpy as np
from PIL import Image
import matplotlib

SRC = "/cs/3dlg-jupiter-project/lightgen/trellis2_bw/code_snapshot"
HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
os.makedirs(IMG, exist_ok=True)

OUT_PX = 440          # emitted panel edge in device pixels (displays at 220)
ACCENT = (214, 106, 63)   # leak color, the v2 terracotta accent
TARGET = (236, 236, 232)  # ground-truth support color


# ---------------------------------------------------------------- exposure --
def tone(a):
    """Cube-root tone curve on 8-bit display values, applied per channel."""
    return np.clip((a / 255.0) ** (1.0 / 3.0) * 255.0, 0, 255).astype(np.uint8)


def upscale(a, out=OUT_PX):
    im = Image.fromarray(a.astype(np.uint8))
    return im.resize((out, out), Image.NEAREST)


def square_pad(x0, y0, x1, y1, lim, pad_frac=0.22, min_side=56):
    """Pad a bbox, square it, clamp it to the panel."""
    w, h = x1 - x0 + 1, y1 - y0 + 1
    side = max(w, h)
    side = int(round(side * (1 + 2 * pad_frac)))
    side = max(side, min_side)
    side = min(side, lim)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    nx0 = int(round(cx - side / 2.0))
    ny0 = int(round(cy - side / 2.0))
    nx0 = max(0, min(nx0, lim - side))
    ny0 = max(0, min(ny0, lim - side))
    return nx0, ny0, nx0 + side, ny0 + side


def leak_map(gt, rec):
    """White where GT is nonzero, accent where only the reconstruction is."""
    g = gt.max(2) if gt.ndim == 3 else gt
    r = rec.max(2) if rec.ndim == 3 else rec
    out = np.zeros((*g.shape, 3), np.uint8)
    out[(r > 0) & (g == 0)] = ACCENT
    out[g > 0] = TARGET
    return out


# ------------------------------------------------------- ten-sample figure --
TEN = os.path.join(SRC, "emission_vae_10sample_vis.png")
TEN_COLS = [34, 428, 821]           # GT | VAE recon | |GT-VAE| x3
TEN_ROWS = [191, 509, 827, 1146, 1464, 1782, 2100, 2418, 2736, 3054]
TEN_SZ, TEN_INSET = 253, 3          # panel edge, axes-spine inset

SAMPLES = [
    # (index, shape-id prefix, L1 from the panel title, slug, group)
    (0, "0007deb6d96c44", "0.0066", "s01", "success"),
    (1, "000b9fd47d6d4f", "0.0277", "s02", "vanish"),
    (2, "000bc33c1a0d4b", "0.0092", "s03", "vanish"),
    (3, "00192b5a4a3249", "0.5031", "s04", "collapse"),
    (4, "001a188012214d", "0.0154", "s05", "collapse"),
    (5, "001b64d2ec4549", "0.1247", "s06", "collapse"),
    (6, "001c79293c3e4f", "0.0082", "s07", "vanish"),
    (7, "001dd28130354d", "0.0153", "s08", "vanish"),
    (8, "002342e8d06042", "1.3856", "s09", "worst"),
    (9, "002af43e7cab4e", "0.0477", "s10", "vanish"),
]


def ten_sample_panels():
    src = np.array(Image.open(TEN).convert("RGB")).astype(np.int16)
    lim = TEN_SZ - 2 * TEN_INSET
    report = []
    for idx, sha, l1, slug, group in SAMPLES:
        rt = TEN_ROWS[idx] + TEN_INSET
        cut = [src[rt:rt + lim, c + TEN_INSET:c + TEN_INSET + lim] for c in TEN_COLS]
        gt, rec = cut[0], cut[1]
        mask = (gt.max(2) >= 8) | (rec.max(2) >= 32)
        ys, xs = np.where(mask)
        x0, y0, x1, y1 = square_pad(xs.min(), ys.min(), xs.max(), ys.max(), lim)
        g = gt[y0:y1, x0:x1]
        r = rec[y0:y1, x0:x1]
        upscale(g).save(f"{IMG}/{slug}_gt.png")
        upscale(r).save(f"{IMG}/{slug}_rec.png")
        upscale(tone(r)).save(f"{IMG}/{slug}_rec_boost.png")
        upscale(tone(g)).save(f"{IMG}/{slug}_gt_boost.png")
        upscale(leak_map(g, r)).save(f"{IMG}/{slug}_leak.png")
        report.append(dict(slug=slug, sha=sha, l1=l1, group=group,
                           crop=(x0, y0, x1 - x0),
                           gt_max=int(g.max()), rec_max=int(r.max()),
                           gt_nz=float((g.max(2) > 0).mean()),
                           rec_nz=float((r.max(2) > 0).mean())))
    return report


# ---------------------------------------------------------- overfit figure --
OVF = os.path.join(SRC, "emission_vae_overfit_vis.png")
OVF_COLS = [272, 735, 1197, 1660]   # Input (GT) | Reconstructed | |Error| | Overlay
OVF_ROWS = [164, 585, 1007]         # XY | XZ | YZ
OVF_SZ, OVF_INSET = 314, 3
HOT = matplotlib.colormaps["hot"](np.linspace(0, 1, 1024))[:, :3] * 255.0


def inv_hot(patch):
    """Invert the `hot` colormap back to its scalar field (0..1 of the row vmax).

    The generator renders both the GT and the reconstruction with
    imshow(cmap="hot", vmin=0, vmax=shared_row_max), and `hot` is injective,
    so nearest-neighbor lookup in the colormap table recovers the scalar the
    panel was drawn from. Working on the scalar rather than on the rendered
    RGB lets the tone curve act on the quantity itself.
    """
    h, w, _ = patch.shape
    d = ((patch.reshape(-1, 1, 3) - HOT.reshape(1, -1, 3)) ** 2).sum(-1)
    idx = d.argmin(1)
    return (idx / 1023.0).reshape(h, w).astype(np.float64)


def hot_render(t):
    return (matplotlib.colormaps["hot"](np.clip(t, 0, 1))[:, :, :3] * 255).astype(np.uint8)


def overfit_panels():
    src = np.array(Image.open(OVF).convert("RGB")).astype(np.float32)
    lim = OVF_SZ - 2 * OVF_INSET
    views = ["xy", "xz", "yz"]
    report = []
    for ri, name in enumerate(views):
        rt = OVF_ROWS[ri] + OVF_INSET
        gt = inv_hot(src[rt:rt + lim, OVF_COLS[0] + OVF_INSET:OVF_COLS[0] + OVF_INSET + lim])
        rc = inv_hot(src[rt:rt + lim, OVF_COLS[1] + OVF_INSET:OVF_COLS[1] + OVF_INSET + lim])
        mask = (gt > 0) | (rc > 0)
        ys, xs = np.where(mask)
        x0, y0, x1, y1 = square_pad(xs.min(), ys.min(), xs.max(), ys.max(), lim,
                                    pad_frac=0.04)
        g, r = gt[y0:y1, x0:x1], rc[y0:y1, x0:x1]
        upscale(hot_render(g)).save(f"{IMG}/ovf_{name}_gt.png")
        upscale(hot_render(r)).save(f"{IMG}/ovf_{name}_rec.png")
        upscale(hot_render(g ** (1 / 3.0))).save(f"{IMG}/ovf_{name}_gt_boost.png")
        upscale(hot_render(r ** (1 / 3.0))).save(f"{IMG}/ovf_{name}_rec_boost.png")
        upscale(leak_map((g * 255).astype(np.uint8), (r * 255).astype(np.uint8))
                ).save(f"{IMG}/ovf_{name}_leak.png")
        report.append(dict(view=name, crop=(x0, y0, x1 - x0),
                           gt_nz=float((g > 0).mean()), rec_nz=float((r > 0).mean()),
                           gt_max=float(g.max()), rec_max=float(r.max())))
    return report


# ------------------------------------------------------------ e2e figure ----
E2E = os.path.join(SRC, "emission_vae_optB_test.png")
E2E_COLS = [134, 314, 494, 674, 854]
E2E_ROWS = [66, 254, 442]
E2E_SZ = 156


def e2e_probe():
    """Locate the panel grid of the end-to-end figure (geometry differs)."""
    a = np.array(Image.open(E2E).convert("RGB")).astype(np.int16)
    nw = a.sum(2) < 700

    def runs(v, thr):
        out, s = [], None
        for i, x in enumerate(v):
            if x > thr and s is None:
                s = i
            elif x <= thr and s is not None:
                out.append((s, i - 1))
                s = None
        if s is not None:
            out.append((s, len(v) - 1))
        return out
    cols = [r for r in runs(nw.sum(0), 60) if r[1] - r[0] > 60]
    rows = [r for r in runs(nw.sum(1), 60) if r[1] - r[0] > 60]
    return cols, rows


def e2e_panels(cols, rows, inset=3):
    src = np.array(Image.open(E2E).convert("RGB")).astype(np.int16)
    out = []
    names = ["xy", "xz", "yz"]
    size = min(c[1] - c[0] + 1 for c in cols[:3])
    lim = size - 2 * inset
    for ri, (rt, _) in enumerate(rows[:3]):
        gt = src[rt + inset:rt + inset + lim, cols[0][0] + inset:cols[0][0] + inset + lim]
        rc = src[rt + inset:rt + inset + lim, cols[1][0] + inset:cols[1][0] + inset + lim]
        mask = (gt.max(2) >= 8) | (rc.max(2) >= 8)
        ys, xs = np.where(mask)
        x0, y0, x1, y1 = square_pad(xs.min(), ys.min(), xs.max(), ys.max(), lim,
                                    pad_frac=0.04)
        g, r = gt[y0:y1, x0:x1], rc[y0:y1, x0:x1]
        upscale(g).save(f"{IMG}/e2e_{names[ri]}_gt.png")
        upscale(tone(g)).save(f"{IMG}/e2e_{names[ri]}_gt_boost.png")
        upscale(r).save(f"{IMG}/e2e_{names[ri]}_rec.png")
        upscale(tone(r)).save(f"{IMG}/e2e_{names[ri]}_rec_boost.png")
        upscale(leak_map(g, r)).save(f"{IMG}/e2e_{names[ri]}_leak.png")
        out.append(dict(view=names[ri], gt_nz=float((g.max(2) > 0).mean()),
                        rec_nz=float((r.max(2) > 0).mean()),
                        gt_max=int(g.max()), rec_max=int(r.max())))
    return out


def e2e_latent(cols, rows, inset=3):
    """Crop the input projection and the emission-latent panel of the same view.

    The latent panel is drawn at the latent grid's own resolution, so its
    block size is the visual statement of the 16x downsampling.
    """
    src = np.array(Image.open(E2E).convert("RGB")).astype(np.uint8)
    size = cols[0][1] - cols[0][0] + 1
    lim = size - 2 * inset
    rt = rows[0][0] + inset
    gt = src[rt:rt + lim, cols[0][0] + inset:cols[0][0] + inset + lim]
    lat = src[rt:rt + lim, cols[3][0] + inset:cols[3][0] + inset + lim]
    Image.fromarray(gt).resize((OUT_PX, OUT_PX), Image.NEAREST).save(
        f"{IMG}/res_input_xy.png")
    Image.fromarray(lat).resize((OUT_PX, OUT_PX), Image.NEAREST).save(
        f"{IMG}/res_latent_xy.png")
    return lim


# ------------------------------------------------------------ whole sources -
def copy_sources():
    for name, cap in [("emission_vae_10sample_vis", 1100),
                      ("emission_vae_overfit_vis", 1400),
                      ("emission_vae_optB_test", 1500)]:
        im = Image.open(os.path.join(SRC, name + ".png")).convert("RGB")
        im.thumbnail((cap, 10000), Image.LANCZOS)
        im.save(f"{IMG}/src_{name}.png")
        print("source copy", name, im.size)


if __name__ == "__main__":
    r1 = ten_sample_panels()
    for r in r1:
        print("{slug} {sha} L1={l1} {group:9s} crop={crop} "
              "gt_max={gt_max:3d} rec_max={rec_max:3d} "
              "gt_nz={gt_nz:.3f} rec_nz={rec_nz:.3f}".format(**r))
    r2 = overfit_panels()
    for r in r2:
        print("ovf {view} crop={crop} gt_nz={gt_nz:.4f} rec_nz={rec_nz:.4f} "
              "gt_max={gt_max:.3f} rec_max={rec_max:.3f}".format(**r))
    c, w = e2e_probe()
    print("e2e cols", c, "rows", w)
    print("e2e latent panel edge", e2e_latent(c, w))
    r3 = e2e_panels(c, w)
    for r in r3:
        print("e2e {view} gt_nz={gt_nz:.4f} rec_nz={rec_nz:.4f} "
              "gt_max={gt_max} rec_max={rec_max}".format(**r))
    copy_sources()
