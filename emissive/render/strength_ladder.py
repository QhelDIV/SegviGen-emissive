#!/usr/bin/env python3
"""Render one shape across a ladder of emission strengths, and contact-sheet it.

Emission strength changes light transport, so every rung is a real render and
not a post-process. Everything else is pinned and identical across the ladder,
so strength is the only variable.

WHAT TO EXPECT, so a correct result is not mistaken for a bug: the bloom grows
as strength rises. The Glare node's threshold is 1.0 in linear space, and raising
strength moves emitters further above it. That is the compositor behaving as
designed, not a bloom setting that needs retuning.

WHAT THE LADDER DOES NOT SHOW: our masks are exactly binary, so a surface the
model did not select stays exactly black at every rung. Measured across 99 mask
files from three checkpoints, 34,603,008 pixels, the only values present are 0
and 255. Amplification therefore cannot make this formulation leak, which is a
property of predicting a mask rather than a continuous texture.

Run: see README.md, "Strength ladder".
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RENDERER = os.path.join(HERE, "render_emissive.py")


def rung_dir(out, strength):
    # a filename-safe label that still sorts and reads correctly: 0.5 -> s0p5
    return os.path.join(out, "s" + str(strength).replace(".", "p"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--glb_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", required=True,
                    help="one sid; the ladder is a per-shape figure")
    ap.add_argument("--strengths", default="0,1,4,8,16",
                    help="comma separated, in order (default 0,1,4,8,16)")
    ap.add_argument("--pred_masks", default=None,
                    help="render the model's prediction band instead of ground "
                         "truth; run twice, once with and once without, to get "
                         "the two bands the comparison needs")
    ap.add_argument("--camera_json", default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--samples", type=int, default=256)
    # the key-lit preset, pinned rather than defaulted: see README "Presets"
    ap.add_argument("--key", type=float, default=8.0)
    ap.add_argument("--bg", type=float, default=0.012)
    ap.add_argument("--view_transform", default="AgX")
    ap.add_argument("--exposure", type=float, default=0.0)
    ap.add_argument("--bloom_size", type=int, default=9)
    ap.add_argument("--bloom_threshold", type=float, default=1.0)
    ap.add_argument("--bloom_mix", type=float, default=-0.15)
    ap.add_argument("--sheet", default=None,
                    help="contact sheet path (default <out>/ladder.png)")
    args = ap.parse_args()

    strengths = [float(x) for x in args.strengths.split(",")]
    os.makedirs(args.out, exist_ok=True)
    made = []
    for st in strengths:
        d = rung_dir(args.out, st)
        os.makedirs(d, exist_ok=True)
        cmd = [args.python, RENDERER,
               "--manifest", args.manifest, "--glb_dir", args.glb_dir,
               "--out", d, "--only", args.only, "--mode", "method",
               "--res", str(args.res), "--samples", str(args.samples),
               "--samples_lit", "96",
               "--key", str(args.key), "--bg", str(args.bg),
               "--view_transform", args.view_transform,
               "--exposure", str(args.exposure),
               "--bloom", "1", "--bloom_size", str(args.bloom_size),
               "--bloom_threshold", str(args.bloom_threshold),
               "--bloom_mix", str(args.bloom_mix),
               "--emit_strength", str(st),
               "--export_glb", "0", "--overwrite", "1"]
        if args.pred_masks:
            cmd += ["--pred_masks", args.pred_masks]
        if args.camera_json:
            cmd += ["--camera_json", args.camera_json]
        print(f"=== strength {st} -> {d}", flush=True)
        r = subprocess.run(cmd)
        # the renderer's own exit status is now meaningful, but a rung that
        # produced no file is the failure that matters, so check the file
        panel = os.path.join(d, f"{args.only}_glow.png")
        if r.returncode != 0 or not os.path.exists(panel):
            sys.exit(f"strength {st} produced no panel at {panel} "
                     f"(exit {r.returncode})")
        made.append((st, panel))

    sheet = args.sheet or os.path.join(args.out, "ladder.png")
    contact_sheet(made, sheet, args.only)
    print(f"LADDER_DONE {sheet}", flush=True)


def contact_sheet(made, path, sid):
    from PIL import Image, ImageDraw
    W, PAD, TOP = 300, 6, 26
    sheet = Image.new("RGB", (len(made) * (W + PAD), W + TOP), (250, 249, 245))
    d = ImageDraw.Draw(sheet)
    for i, (st, p) in enumerate(made):
        x = i * (W + PAD)
        d.text((x + 4, 6), f"strength {st:g}", fill=(20, 20, 20))
        sheet.paste(Image.open(p).convert("RGB").resize((W, W)), (x, TOP))
    sheet.save(path)
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump({"sid": sid, "rungs": [{"strength": s, "panel": p}
                                         for s, p in made]}, f, indent=1)


if __name__ == "__main__":
    main()
