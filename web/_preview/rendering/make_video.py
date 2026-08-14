#!/usr/bin/env python3
"""Turn the box-render frame sweep into a short looping video for the page.

The renders are a one-way ramp of 33 emission strengths, eased with a
smoothstep so the turnarounds are smooth. This script ping-pongs them into 64
frames (0..32 then 31..1) and encodes at 24 fps, which is 2.67 seconds a loop.
Rendering only the upward half halves the cluster cost and guarantees the loop
is exactly symmetric, which a separately rendered downward half would not be.

Three outputs, because one format cannot serve every browser: VP9 in WebM,
H.264 in MP4 as the fallback, and a poster still (the brightest frame) that
shows while the video loads and stands in wherever video is blocked.

Run (after the frames are on local disk):
    .venv_console/bin/python web/_preview/rendering/make_video.py
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
# The rendered stills live OUTSIDE this page directory on purpose. Everything
# under the page dir is treated as a page asset and gets copied into the
# published site and into every immutable version snapshot; the 33 frames are
# a 21 MB build input that no page references, and keeping them here once put
# 42 MB of them on the web server, twice.
FRAMES = os.environ.get(
    "SWEEP_FRAMES",
    "/project/3dlg-hcvc/omages/yanxg_scratch/render_doc/frames")
SEQ = os.path.join(HERE, "_seq")               # the 64-frame ping-pong sequence
SID = "51a60b164e874bf891597d9c6c1941af"
FPS = 24


def load_plan():
    with open(os.path.join(FRAMES, "frame_plan.json")) as f:
        return json.load(f)


def frame_path(i):
    return os.path.join(FRAMES, f"{SID}_box_f{i:03d}.png")


def check_frames(plan):
    missing = [f["i"] for f in plan["frames"] if not os.path.exists(frame_path(f["i"]))]
    if missing:
        sys.exit(f"MISSING FRAMES: {missing}. Re-render those indices before encoding.")
    return len(plan["frames"])


# What every frame must have been rendered with for the clip to be a box render
# rather than something that merely looks like one. Each render writes these
# into its own sidecar, so this is read back rather than assumed.
BOX_SETTINGS = {"samples": 1024, "max_bounces": 32, "diffuse_bounces": 16,
                "wall": 0.8, "view_transform": "Filmic", "exposure": 1.5}


def check_settings(plan):
    """Every frame's sidecar agrees with the box render, and every frame shares
    one camera. A frame rendered under different settings would change the clip
    mid-loop and read as a flicker rather than as a settings error."""
    problems, cameras = [], set()
    for f in plan["frames"]:
        side = frame_path(f["i"])[:-4] + ".json"
        if not os.path.exists(side):
            problems.append(f"frame {f['i']}: no sidecar")
            continue
        with open(side) as fh:
            cfg = json.load(fh)
        for k, want in BOX_SETTINGS.items():
            if cfg.get(k) != want:
                problems.append(f"frame {f['i']}: {k}={cfg.get(k)!r}, expected {want!r}")
        cameras.add(tuple(round(c, 6) for c in cfg["camera"]))
    if len(cameras) > 1:
        problems.append(f"{len(cameras)} distinct cameras across the frames; "
                        "the sweep must hold one viewpoint")
    if problems:
        sys.exit("FRAME SETTINGS CHECK FAILED:\n  " + "\n  ".join(problems))
    return cameras.pop() if cameras else None


def build_sequence(plan):
    """0..N-1 then N-2..1, a ping-pong with no repeated endpoint, so the loop
    has no stutter at either turnaround."""
    n = len(plan["frames"])
    order = list(range(n)) + list(range(n - 2, 0, -1))
    if os.path.isdir(SEQ):
        shutil.rmtree(SEQ)
    os.makedirs(SEQ)
    for k, i in enumerate(order):
        os.link(frame_path(i), os.path.join(SEQ, f"f{k:04d}.png"))
    return order


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"FFMPEG FAILED: {' '.join(cmd)}\n{r.stderr[-2000:]}")


def encode(nframes):
    pat = os.path.join(SEQ, "f%04d.png")
    webm = os.path.join(IMG, "sweep.webm")
    mp4 = os.path.join(IMG, "sweep.mp4")
    # VP9: -b:v 0 makes -crf a true quality target rather than a cap.
    run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", pat,
         "-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-row-mt", "1",
         "-pix_fmt", "yuv420p", "-an", webm])
    # H.264 needs yuv420p and even dimensions for the widest player support;
    # faststart puts the index first so playback can begin before the full
    # file arrives.
    run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", pat,
         "-c:v", "libx264", "-crf", "20", "-preset", "slow",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", mp4])
    return webm, mp4


def poster(plan):
    """The brightest frame: what a reader should see if the video never plays."""
    top = max(plan["frames"], key=lambda f: f["strength"])["i"]
    dst = os.path.join(IMG, "sweep_poster.png")
    shutil.copy2(frame_path(top), dst)
    return dst, top


def probe(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate,codec_name",
         "-of", "json", path], capture_output=True, text=True)
    s = json.loads(r.stdout)["streams"][0]
    return {"codec": s["codec_name"], "w": s["width"], "h": s["height"],
            "fps": s["r_frame_rate"], "frames": int(s["nb_read_frames"])}


def main():
    plan = load_plan()
    n = check_frames(plan)
    camera = check_settings(plan)
    order = build_sequence(plan)
    webm, mp4 = encode(len(order))
    post, top = poster(plan)
    info = {"rendered_frames": n, "video_frames": len(order), "fps": FPS,
            "camera": camera, "box_settings": BOX_SETTINGS,
            "seconds": round(len(order) / FPS, 2),
            "poster_frame": top, "max_strength": plan["smax"],
            "easing": plan["easing"],
            "webm": probe(webm), "mp4": probe(mp4),
            "webm_bytes": os.path.getsize(webm), "mp4_bytes": os.path.getsize(mp4)}
    with open(os.path.join(IMG, "sweep_video.json"), "w") as f:
        json.dump(info, f, indent=1)
    shutil.rmtree(SEQ)
    for k, v in info.items():
        print(f"  {k}: {v}")
    if info["webm"]["frames"] != len(order) or info["mp4"]["frames"] != len(order):
        sys.exit("FRAME COUNT MISMATCH between the sequence and an encoded file.")
    print("VIDEO OK")


if __name__ == "__main__":
    main()
