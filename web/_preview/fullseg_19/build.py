#!/usr/bin/env python3
"""Build the fullseg_19 page: the ORIGINAL pretrained SegviGen full-segmentation
model (full_seg.ckpt, no emissive fine-tuning), run zero-shot on 19 raw shapes not
seen anywhere else on this page's data path.

Two groups: fig7 (Dongchen's 11 figure-7 qualitative shapes) and val96 (8 shapes
from the fbv1_repro validation set). No ground-truth part labels exist for either
group, so this is a qualitative-only gallery (no IoU/quant section): what the
pretrained model decomposes each shape into, nothing more.

Pipeline (3 stages, all on solar, see code/ in the scratch dir):
  A. render_cond.py   (CPU, bpy)      -- one 512x512 conditioning/input image per shape
  B. infer_fullseg.py (GPU, trellis2) -- glb -> vxz -> shape+tex slats -> flow sample
                                          -> decode -> slat_to_glb (the model's own
                                          part-coloring, baked straight into the
                                          exported mesh -- same convention verified
                                          against the reference fullseg_canon10_mesh
                                          page's cake example, no extra palette
                                          recoloring applied or needed)
  C. render_seg_views.py (CPU, bpy)   -- 2 camera views of each pred_glb

Run: /local-scratch2/xya120/studio/misc/lightgen/.venv_console/bin/python \
        web/_preview/fullseg_19/build.py
"""
import hashlib
import html
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WEB)
sys.path.insert(0, os.path.join(REPO, "tools"))

import xgpage as lp                      # noqa: E402
import workspace_zone as wz              # noqa: E402
from xgpage.publish import publish_assets  # noqa: E402

SITE_ROOT = "/projects/omages/yanxg/lightgen"
SITE_ASSETS = f"{SITE_ROOT}/assets"
FAVICON = f"{SITE_ROOT}/assets/images/favicon.png"
PAGE_DATE = "2026-08-09"

BASE = "/project/3dlg-hcvc/omages/yanxg_scratch/fullseg19"
IMG_DIR = os.path.join(HERE, "img")
os.makedirs(IMG_DIR, exist_ok=True)

SHAPES = json.load(open(os.path.join(BASE, "shapes.json")))
MANIFEST = json.load(open(os.path.join(BASE, "manifest.json")))

GROUP_LABEL = {"fig7": "Figure-7 group (11 shapes)", "val96": "Val-96 group (8 shapes)"}
GROUP_ORDER = ["fig7", "val96"]

OUTLINE = [
    ("verdict", "What the panels show"),
    ("gallery", "19 shapes, pretrained zero-shot full segmentation"),
    ("parts", "Part-count distribution"),
    ("provenance", "How this was produced"),
]


def copy_img(src_path, dest_name):
    dst = os.path.join(IMG_DIR, dest_name)
    shutil.copy2(src_path, dst)
    return dst


def rel_img(abspath):
    return "img/" + os.path.relpath(abspath, IMG_DIR)


def img_ref(abspath):
    h = hashlib.md5(open(abspath, "rb").read()).hexdigest()[:8]
    return f"{rel_img(abspath)}?v={h}"


def stage_images():
    """Copy the per-shape input/seg-view-a/seg-view-b panels into this page's own
    img/. Any shape missing a manifest ok=True or its seg_img files is dropped
    from the gallery and reported, not silently skipped."""
    out = {}
    missing = []
    for rec in SHAPES:
        sid, group = rec["sid"], rec["group"]
        m = MANIFEST.get(sid)
        if not m or not m.get("ok"):
            missing.append((sid, group, "stage B (inference) not ok"))
            continue
        cond_src = os.path.join(BASE, "cond_img", f"{sid}.png")
        va_src = os.path.join(BASE, "seg_img", f"{sid}_va.png")
        vb_src = os.path.join(BASE, "seg_img", f"{sid}_vb.png")
        missing_files = [p for p in (cond_src, va_src, vb_src) if not os.path.exists(p)]
        if missing_files:
            missing.append((sid, group, f"missing render(s): {missing_files}"))
            continue
        out[sid] = {
            "input": copy_img(cond_src, f"{sid}_input.png"),
            "seg_a": copy_img(va_src, f"{sid}_seg_a.png"),
            "seg_b": copy_img(vb_src, f"{sid}_seg_b.png"),
            "n_parts": m["n_parts"],
            "n_voxels": m["n_voxels"],
            "seconds": m["seconds"],
        }
    return out, missing


def sec_verdict(imgs, missing):
    n_ok = len(imgs)
    n_total = len(SHAPES)
    parts_vals = [v["n_parts"] for v in imgs.values()]
    n_single = sum(1 for p in parts_vals if p == 1)
    body = lp.callout(
        "<b>This is the ORIGINAL pretrained full_seg checkpoint, unmodified, run "
        "zero-shot on 19 raw shapes it has never seen.</b> No ground-truth part "
        "labels exist for either group, so there is no accuracy number on this "
        "page: it shows what the model decomposes each shape into, for visual "
        "inspection only. The part-coloring shown is the model's own raw output "
        "(the tex_slat this checkpoint samples IS the segmentation signal, baked "
        "straight into the exported mesh by slat_to_glb), matching the convention "
        "already verified on the reference fullseg_canon10_mesh page.",
        title="Read this before the gallery")
    if n_single:
        body += lp.callout(
            f"<b>{n_single} of {n_ok} shapes came back with n_parts=1</b> (a single "
            "quantized color cluster covering the whole object). Checked visually "
            "per shape in the gallery below rather than assumed to be a bug or a "
            "genuinely single-part object case by case.",
            warn=True, title="Single-part shapes")
    body += lp.callout(
        "<b>Many SEG panels show the object tumbled relative to INPUT, even though "
        "both use the identical nominal camera transform.</b> Checked directly on "
        "several shapes: the upright Mjolnir-style hammer, the standing robot, and "
        "the standing character all come back lying on their side in the seg view, "
        "while an elongated lightsaber and the jack-o'-lantern keep a recognizable "
        "orientation. The predicted mesh (<code>pred_glb</code>, from "
        "<code>slat_to_glb</code>) is decoded in the model's own coordinate frame "
        "via <code>process_glb_to_vxz</code>, which does not always share the "
        "original GLB's up-axis/orientation convention -- applying camera A/B "
        "(tuned to the original GLB) to that frame can show the object from an "
        "arbitrary, non-canonical angle. This is a camera/frame artifact of this "
        "gallery's rendering, not a defect in the segmentation itself: part colors "
        "and counts are computed directly in the model's own voxel space, "
        "unaffected by which way the render camera happens to be pointing. Not "
        "fixed here (would need reverse-engineering trellis2's own normalization "
        "convention); read each row's two SEG views together rather than expecting "
        "either to match INPUT's framing.",
        warn=True, title="Camera/orientation mismatch between INPUT and SEG views")
    if missing:
        rows = "".join(f"<li><code>{sid}</code> ({group}): {why}</li>" for sid, group, why in missing)
        body += lp.callout(f"<b>{len(missing)} of {n_total} shapes are missing from the "
                           f"gallery below.</b><ul>{rows}</ul>", warn=True, title="Incomplete")
    return lp.section_v2("verdict", 1, f"{n_ok}/{n_total} shapes, pretrained zero-shot full segmentation", body)


def _matrix_rows(imgs, group):
    rows = []
    for rec in SHAPES:
        if rec["group"] != group or rec["sid"] not in imgs:
            continue
        sid = rec["sid"]
        e = imgs[sid]
        label = f"{sid[:8]}… · {e['n_parts']} parts"
        rows.append((label, [img_ref(e["input"]), img_ref(e["seg_a"]), img_ref(e["seg_b"])]))
    return rows


def sec_gallery(imgs):
    body = ""
    for group in GROUP_ORDER:
        rows = _matrix_rows(imgs, group)
        if not rows:
            continue
        body += lp.prose(f"<b>{GROUP_LABEL[group]}</b>")
        body += lp.method_matrix(
            columns=["INPUT", "SEG VIEW A", "SEG VIEW B"],
            rows=rows,
            caption_html=(
                "Input: the same rendered/DINOv3-conditioning image fed to the "
                "model (camera A, transforms_v0.json). SEG VIEW A/B: the model's "
                "own predicted part-coloring, baked into the decoded res-512 mesh "
                "by <code>slat_to_glb</code>, rendered from two angles "
                "(transforms_v0/v1.json, ~140&deg; apart) so occluded parts are "
                "visible from at least one view. Row label part count = distinct "
                "quantized color clusters after merging any cluster below 1% of "
                "voxels (same convention as <code>seg_covers_emissive.py</code>)."),
            native_px=384, content="photo", page_inner=820, id=f"gallery-{group}")
    return lp.section_v2("gallery", 2, "Pretrained full_seg, zero-shot, no fine-tuning", body)


def sec_parts(imgs):
    rows = sorted(
        ({"label": f"{sid[:8]}… ({SHAPES_BY_SID[sid]['group']})", "value": e["n_parts"],
          "display": f"{e['n_parts']} parts, {e['n_voxels']:,} voxels"}
         for sid, e in imgs.items()), key=lambda r: -r["value"])
    chart = lp.hbar_chart(rows, title="Part count per shape (quantized color clusters)", label_w=180)
    body = lp.prose(
        f"Range: {min(r['value'] for r in rows)}–{max(r['value'] for r in rows)} parts. "
        "Part count is a quantization artifact of the model's own continuous color "
        "output (6 levels per channel, clusters below 1% of voxels merged into the "
        "largest neighbor), not a count the model predicts directly.")
    body += chart
    return lp.section_v2("parts", 3, "Part-count distribution across the 19 shapes", body)


def sec_provenance(imgs, missing):
    param_rows = "".join(
        f'<tr><td style="text-align:left">{html.escape(k)}</td><td>{html.escape(v)}</td></tr>'
        for k, v in [
            ("checkpoint", "fenghora/SegviGen full_seg.ckpt (pretrained, no fine-tuning)"),
            ("sampling steps", "12 (the pipeline's own default; a --steps 25 flag in "
             "infer_fullseg.py is dead code and was NOT what actually ran)"),
            ("conditioning", "real image (stage A cond render), not zero_cond"),
            ("cameras", "transforms_v0.json (input + seg view A), "
             "transforms_v1.json (seg view B, ~140° turned)"),
            ("render engine", "Cycles, 512px, CPU-only"),
        ])
    param_table = lp.results_table(["parameter", "value"], param_rows)

    job_rows = "".join(
        f'<tr><td style="text-align:left">{html.escape(stage)}</td><td>{jid}</td><td>{res}</td></tr>'
        for stage, jid, res in [
            ("stage A: cond render, 19 shapes", "242175", "CPU, 64 cpus/task, debug partition"),
            ("stage B: GPU full_seg inference, 19 shapes", "242177", "GPU A40 x1 (cs-venus-13), 8 cpus"),
            ("stage C: seg render, 19 shapes x 2 views", "242192", "CPU, debug partition, node scheduler-picked"),
            ("tiny-INPUT fix: diagnose + re-render 1 shape", "242193 (diag), 242194 (fix)", "CPU, 4 cpus, debug partition"),
        ])
    job_table = lp.results_table(["stage", "solar job id", "resources"], job_rows)

    body = lp.prose(
        "Environment: conda <code>trellis2</code> env (stage B), cwd = "
        "<code>segvigen_emissive/code/SegviGen</code> so the repo's own relative "
        "imports resolve; <code>HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache</code>. "
        "A real bug was found and fixed in <code>infer_fullseg.py</code> during this "
        "run: <code>BiRefNet</code>/<code>DinoV3FeatureExtractor</code> are plain "
        "classes (not <code>nn.Module</code>), so their <code>.to(device)</code> "
        "mutates in place and returns <code>None</code> -- the original chained "
        "<code>X(...).to(device)</code> silently rebound the variable to "
        "<code>None</code> and crashed inference with a <code>TypeError</code>. Fixed "
        "by un-chaining construction and the device move, matching the upstream "
        "<code>inference_full.inference()</code> call pattern.")
    body += lp.callout(
        "<b>The INPUT panel for the top-hatted robot (34170054845344...) originally "
        "rendered as a 0.01%-of-pixels speck.</b> Diagnosed by enumerating every mesh "
        "object bpy sees after importing the raw source GLB (job 242193): the scene "
        "carries an unparented, unrelated 'Icosphere' mesh with a world bbox extent of "
        "~2 units, while the robot's own geometry (all nodes under its "
        "<code>NewRobot_*</code> hierarchy) sits entirely within extent ~0.001-0.02 -- "
        "roughly 100x smaller. <code>BpyRenderer.normalize_scene()</code> unions ALL "
        "mesh bboxes with no outlier rejection, so that one stray sphere, not the "
        "robot, decided the auto-scale. Fixed (job 242194) by subclassing "
        "<code>BpyRenderer</code> to exclude any mesh literally named "
        "<code>Icosphere</code> from the bbox before normalizing, then re-rendering "
        "only this one shape's input with the same camera/transform as every other "
        "panel; confirmed by eye against the panel's own SEG views (comparable scale) "
        "before republishing. No other shape showed this symptom (every other INPUT "
        "panel covers 0.85%-63% of pixels), so the fix was scoped to this one shape, "
        "not a change to the shared render_cond.py.",
        warn=True, title="Fixed: tiny INPUT panel (stray Icosphere in the source GLB)")
    body += lp.prose("Parameters:") + param_table
    body += lp.prose("Solar job ids and resources, account <code>3dlg-hcvc-lab</code>:")
    body += job_table
    if missing:
        body += lp.prose(f"{len(missing)} shape(s) incomplete at build time; see the verdict section above.")

    apx = lp.appendix("Sources", [
        "Scratch dir: /project/3dlg-hcvc/omages/yanxg_scratch/fullseg19/ "
        "(shapes.json, transforms_v0/v1.json, cond_img/, pred_glb/, seg_img/, "
        "manifest.json, code/).",
        "Stage A: code/render_cond.py, wrapping SegviGen's own "
        "data_toolkit/bpy_render.render_from_transforms.",
        "Stage B: code/infer_fullseg.py, following the repo's documented full-"
        "segmentation usage (README.md / inference_full.py main()): "
        "process_glb_to_vxz -> vxz_to_latent_slat -> real image condition "
        "(DINOv3) -> tex_slat_sample_single -> tex_decoder -> slat_to_glb.",
        "Stage C: code/render_seg_views.py, same render_from_transforms utility, "
        "camera A (same as input) and camera B (~140° turned).",
        "Part count: quantize_parts() in infer_fullseg.py, same convention as "
        "seg_covers_emissive.py's quantize_parts (6 color levels/channel, "
        "clusters below 1% of voxels merged).",
        "GLB sources: fig7 group from /project/3dlg-hcvc/omages/yanxg_scratch/"
        "fig7/glb_src/, val96 group from /project/3dlg-hcvc/omages/yanxg_scratch/"
        "fbv1_repro/glb/.",
    ])
    return lp.section_v2("provenance", 4, "Commands, parameters, job ids", body), apx


def main():
    global SHAPES_BY_SID
    SHAPES_BY_SID = {r["sid"]: r for r in SHAPES}
    imgs, missing = stage_images()

    n_ok = len(imgs)
    parts_vals = [v["n_parts"] for v in imgs.values()]
    stats = [
        (str(n_ok), "shapes rendered"),
        (str(len(SHAPES) - n_ok), "missing"),
        (f"{min(parts_vals) if parts_vals else 0}–{max(parts_vals) if parts_vals else 0}", "part-count range"),
    ]
    hero = lp.hero_header(
        "SegviGen · original full-segmentation checkpoint",
        "19 shapes, pretrained zero-shot, no fine-tuning",
        dek_html=(
            "The ORIGINAL pretrained SegviGen full_seg checkpoint (not the emissive "
            "fine-tune) run on 11 fig7 shapes + 8 val96 shapes, none of which have "
            "ground-truth part labels. Qualitative gallery only: what the model "
            "decomposes each shape into."),
        stats=stats,
        toc=[(i, lab) for i, lab in OUTLINE])

    body = [sec_verdict(imgs, missing), sec_gallery(imgs), sec_parts(imgs)]
    prov_body, apx = sec_provenance(imgs, missing)
    body.append(prov_body)

    page_html = lp.page(
        title="Original SegviGen full-segmentation, 19 shapes (fullseg_19)",
        header_html=hero,
        body_sections=body + [apx],
        assets_rel=SITE_ASSETS,
        assets_dir=os.path.join(WEB, "assets"),
        theme="v3",
        tree_html=wz.tree_html(active_href=None),
        nav_title="fullseg_19",
        outline_entries=[{"id": i, "label": lab} for i, lab in OUTLINE],
        version_slot=lp.v3_version_slot(date=PAGE_DATE),
        needs_katex=False,
        extra_head=f'<link rel="icon" href="{FAVICON}">',
    )

    violations = wz.console_links_in(page_html)
    if violations:
        sys.exit(f"ZONE-LINK GUARD FAILED: page links to the console: {violations}")

    out = os.path.join(HERE, "index.html")
    with open(out, "w") as f:
        f.write(page_html)
    print(f"wrote {out} ({len(page_html)} bytes, {n_ok}/{len(SHAPES)} shapes)")
    print("  zone-link guard: clean")

    publish_assets(os.path.join(WEB, "assets"))
    print("assets published")


if __name__ == "__main__":
    main()
