#!/usr/bin/env python3
"""Re-grade a glare-free linear render through the Glare node, many settings.

Bloom is a POST-PROCESS: Cycles renders a linear image, the compositor runs on
it, and only then does the view transform map it to display. So a bloom sweep
does not need one render per parameter set. Render each shape once with the
Glare node bypassed, save the linear result as OpenEXR, and push that through a
compositor graph per setting: seconds a cell instead of minutes.

The compositor needs a render to run, so the scene here is empty (no objects, no
lights, one sample) and the Render Layers node is replaced by an Image node
reading the EXR. Cycles produces a blank frame in a fraction of a second and the
compositor does the actual work.

VERIFY MODE (--verify) re-grades at the settings the shipped renders used and
diffs against them, so the claim "this reproduces the in-render result" is
checked rather than assumed.

Run on a CPU node with the shared venv.
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np

import bpy  # noqa: E402


def blank_scene(width, height, view_transform, exposure):
    """An empty Cycles scene at the EXR's resolution: the compositor needs a
    render to run, and a scene with nothing in it produces one in a fraction of
    a second, leaving the Glare node to do all the work."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = view_transform
    scene.view_settings.look = "None"
    scene.view_settings.exposure = exposure
    scene.render.use_compositing = True
    scene.use_nodes = True
    return scene


def grade(exr_path, out_path, *, size, threshold, mix, view_transform, exposure,
          bloom=True, fmt="PNG"):
    """One graded PNG from one linear EXR."""
    img = bpy.data.images.load(exr_path, check_existing=False)
    w, h = img.size
    scene = blank_scene(w, h, view_transform, exposure)
    # read_factory_settings dropped the datablock, so load it into the new file
    img = bpy.data.images.load(exr_path, check_existing=False)
    img.colorspace_settings.name = "Linear Rec.709"

    nt = scene.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    src = nt.nodes.new("CompositorNodeImage")
    src.image = img
    comp = nt.nodes.new("CompositorNodeComposite")
    if bloom:
        glare = nt.nodes.new("CompositorNodeGlare")
        glare.glare_type = "FOG_GLOW"
        glare.quality = "HIGH"
        glare.size = size
        glare.threshold = threshold
        glare.mix = mix
        nt.links.new(src.outputs["Image"], glare.inputs["Image"])
        nt.links.new(glare.outputs["Image"], comp.inputs["Image"])
    else:
        nt.links.new(src.outputs["Image"], comp.inputs["Image"])
    if fmt == "OPEN_EXR":
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    bpy.data.images.remove(img)


def read(path):
    """RGB float array. PIL/matplotlib cannot decode OpenEXR, so linear EXRs go
    through Blender's own image loader instead."""
    if path.lower().endswith(".exr"):
        img = bpy.data.images.load(path, check_existing=False)
        w, h = img.size
        buf = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
        bpy.data.images.remove(img)
        return buf.reshape(h, w, 4)[::-1, :, :3].astype(np.float64)
    import matplotlib.image as mpimg
    return mpimg.imread(path)[..., :3].astype(np.float64)


def read_u8(path):
    """The 8-bit RGB values, as integers, exactly as stored."""
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB")).astype(np.int16)


# The cut, in whole 8-bit levels. THREE, not two, and the choice is forced.
#
# The inputs are 8-bit, so the lift is quantized to whole levels and a large
# share of every halo lands on exactly two: 4.96 percent of the candles frame in
# one cell. A cut placed AT that value is decided by floating-point residue
# rather than by the image, and the same PNGs then yield 0.056 or 0.080 for the
# same cell depending only on whether the reader hands back float32 or float64.
# Three levels is the first cut with an unpopulated boundary, so it is stable.
#
# This is also what every number already published from this script means. The
# original code compared floats against 2/255, whose realized behaviour is this
# cut (verified: agrees with the integer form to 0.0066 across all 50 cells
# measured, and to under 0.001 on the shipped ones). Its docstring claimed "more
# than one 8-bit step", which was wrong and which I repeated in two reports.
# Rewriting it in integers changes no published value; it removes the ambiguity
# and makes the code say what it does. Credit to sweep-page, who found it.
LIFT_STEPS = 3


def bloom_stats(bloomed_path, plain_path):
    """How much of the frame the bloom actually touches, and by how much.

    Measured against the SAME image with the Glare node bypassed, so it is the
    bloom being measured and not the scene. A pixel counts as bloomed when the
    glare lifts its brightest channel by at least LIFT_STEPS whole 8-bit levels.

    Display-referred: computed after the view transform and the exposure lift,
    so it answers how much of the VISIBLE frame the bloom touches, not how much
    energy the glare added. Shares are not comparable across tone settings.
    """
    a, b = read_u8(bloomed_path), read_u8(plain_path)
    lift_steps = a.max(axis=-1) - b.max(axis=-1)
    touched = lift_steps >= LIFT_STEPS
    lift = lift_steps / 255.0
    return {"bloomed_px_frac": float(touched.mean()),
            "lift_steps_cut": LIFT_STEPS,
            "mean_lift": float(lift.mean()),
            "mean_lift_where_touched": float(lift[touched].mean()) if touched.any() else 0.0,
            "max_lift": float(lift.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exr_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--view_transform", default="Filmic")
    ap.add_argument("--exposure", type=float, default=1.5)
    ap.add_argument("--verify", default=None,
                    help="directory of shipped PNGs to diff the current "
                         "settings against")
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--mix", type=float, default=-0.15)
    ap.add_argument("--mechanism", type=int, default=0)
    ap.add_argument("--combos", default=None,
                    help="explicit \"thr,mix,size;...\" cells")
    args = ap.parse_args()

    if args.mechanism:
        # Does exposure change WHICH pixels bloom? Blender applies exposure in
        # the display transform, downstream of the compositor, so the glare's
        # own (scene-linear) output should be bit-identical at two exposures.
        # Written as EXR to compare before any tone mapping.
        exr = sorted(glob.glob(os.path.join(args.exr_dir, "*_box.exr")))[0]
        outs = []
        for e in (0.0, 1.5):
            o = os.path.join(args.out, f"mech_e{e:g}.exr")
            grade(exr, o, size=args.size, threshold=args.threshold, mix=args.mix,
                  view_transform=args.view_transform, exposure=e, fmt="OPEN_EXR")
            outs.append(o)
        a, b = read(outs[0]), read(outs[1])
        same = float(np.abs(a - b).max())
        print(f"MECHANISM glare output max|diff| between exposure 0 and +1.5: "
              f"{same:.6g}  ->  "
              f"{'identical: exposure does NOT reach the compositor' if same < 1e-6 else 'DIFFERENT: exposure does reach the compositor'}",
              flush=True)
        os._exit(0)

    exrs = sorted(glob.glob(os.path.join(args.exr_dir, "*_box.exr")))
    assert exrs, f"no EXRs in {args.exr_dir}"
    os.makedirs(args.out, exist_ok=True)
    report = {}

    for exr in exrs:
        sid = os.path.basename(exr).split("_box")[0]
        # the no-bloom baseline, for the bloomed-pixel measure
        plain = os.path.join(args.out, f"{sid}_none.png")
        grade(exr, plain, size=0, threshold=0, mix=0, bloom=False,
              view_transform=args.view_transform, exposure=args.exposure)

        if args.verify:
            cur = os.path.join(args.out, f"{sid}_verify.png")
            grade(exr, cur, size=args.size, threshold=args.threshold,
                  mix=args.mix, view_transform=args.view_transform,
                  exposure=args.exposure)
            ship = os.path.join(args.verify, f"{sid}_box.png")
            if os.path.exists(ship):
                d = np.abs(read(cur) - read(ship))
                report[f"{sid}/verify"] = {
                    "mean_abs_diff_255": float(d.mean() * 255),
                    "p99_abs_diff_255": float(np.percentile(d, 99) * 255),
                    "max_abs_diff_255": float(d.max() * 255)}
                print(f"VERIFY {sid[:8]} mean={d.mean()*255:.3f}/255 "
                      f"p99={np.percentile(d,99)*255:.3f} max={d.max()*255:.3f}",
                      flush=True)
            continue

        # one axis at a time from the current setting, or explicit combos for
        # the refinement pass once an axis has been identified
        if args.combos:
            cells = [dict(zip(("threshold", "mix", "size"),
                              (float(a), float(b), float(c))))
                     for a, b, c in (t.split(",") for t in args.combos.split(";"))]
            axes = [("combo", cells)]
        else:
            axes = [("threshold", [1.0, 1.5, 2.5]),
                    ("mix", [-0.15, -0.45, -0.70]),
                    ("size", [9, 7, 5])]
        base = {"threshold": args.threshold, "mix": args.mix, "size": args.size}
        seen = set()
        for name, values in axes:
            for v in values:
                cfg = dict(base)
                if name == "combo":
                    cfg.update(v)
                else:
                    cfg[name] = v
                key = (cfg["threshold"], cfg["mix"], cfg["size"])
                tag = f"t{cfg['threshold']:g}_m{cfg['mix']:g}_s{cfg['size']:g}"
                if key in seen:
                    continue
                seen.add(key)
                out = os.path.join(args.out, f"{sid}_{tag}.png")
                grade(exr, out, size=int(cfg["size"]),
                      threshold=cfg["threshold"], mix=cfg["mix"],
                      view_transform=args.view_transform,
                      exposure=args.exposure)
                st = bloom_stats(out, plain)
                report[f"{sid}/{tag}"] = st
                print(f"{sid[:8]} {tag:22s} bloomed={st['bloomed_px_frac']:.3f} "
                      f"lift={st['mean_lift']:.4f}", flush=True)

    with open(os.path.join(args.out, "bloom_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("SWEEP_DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
