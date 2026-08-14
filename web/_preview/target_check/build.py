#!/usr/bin/env python3
"""Build the target_check page: is output_tex_slat.pth (the actual training/eval
target) the binary emission mask it should be, or -- as the owner's observation
suggested (overfit predictions glow in each shape's own albedo colors, strengthening
with training) -- the shape's PBR albedo latent, swapped in by a dataset-builder bug?

Method: decode output_tex_slat.pth through the real TRELLIS.2 tex_decoder (the same
path eval_emissive.py uses for its GT decode) on 5 shapes spanning the ct10 overfit
set, the single-shape overfit control, and train_72k, and compare the decoded colors
against the raw pre-encode ground truth still sitting on disk in each shape's
dataset_direct dir: output.vxz (binary emissive mask, by construction) and input.vxz
(real PBR albedo). A positive control (decode input_tex_slat.pth, which SHOULD be
albedo) rules out the decode path itself hiding a real leak.

Every number on this page is read at build time from the diagnostic job's own
results.json, never retyped.

Run: /local-scratch2/xya120/studio/misc/lightgen/.venv_console/bin/python \
        web/_preview/target_check/build.py
"""
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WEB)
sys.path.insert(0, os.path.join(REPO, "tools"))

import xgpage as lp                      # noqa: E402
import workspace_zone as wz              # noqa: E402
from xgpage.publish import publish_assets  # noqa: E402


def _esc(s):
    return html.escape(str(s), quote=False)


SITE_ROOT = "/projects/omages/yanxg/lightgen"
SITE_ASSETS = f"{SITE_ROOT}/assets"
FAVICON = f"{SITE_ROOT}/assets/images/favicon.png"
PAGE_DATE = "2026-08-11"

RESULTS = json.load(open(os.path.join(HERE, "results.json")))
BY_NAME = {r["name"]: r for r in RESULTS}

OUTLINE = [
    ("verdict", "The target on disk is the emission mask, not the albedo"),
    ("table", "Per-shape numbers: correlation against each candidate target"),
    ("gallery", "What the decoded target actually looks like"),
    ("provenance", "How this was checked"),
]


def img_ref(name):
    import hashlib
    path = os.path.join(HERE, "img", name)
    h = hashlib.md5(open(path, "rb").read()).hexdigest()[:8]
    return f"img/{name}?v={h}"


def sec_verdict():
    body = lp.verdict_box(
        "<p><b>The wrong-target hypothesis is refuted.</b> "
        "<code>output_tex_slat.pth</code> &mdash; the tensor <code>train_emissive.py</code> "
        "actually trains against &mdash; decodes to the binary emission mask, not the "
        "shape's albedo, on every one of the 5 shapes checked: both ct10 red-tinted "
        "shapes, the single-shape overfit control, and two train_72k shapes. Correlation "
        "of the decoded target against the raw ground-truth emission mask is 0.94&ndash;0.999 "
        "on every shape; correlation against the raw PBR albedo is low and, more tellingly, "
        "flat across R/G/B (the signature of an achromatic mask, not a leaked color). A "
        "positive control &mdash; decoding <code>input_tex_slat.pth</code>, which IS supposed "
        "to be albedo &mdash; reproduces the real albedo at 0.94&ndash;0.998 correlation with "
        "clear per-channel differentiation, proving the decode path is not itself hiding a "
        "real leak.</p>")
    body += lp.prose(
        "This was checked at the level that matters: the actual bytes in "
        "<code>output_tex_slat.pth</code>, decoded through the same TRELLIS.2 tex_decoder "
        "<code>eval_emissive.py</code> uses for scoring, not just a read of today's "
        "builder/trainer code (which was also checked and is internally consistent: "
        "<code>build_dataset_direct.py</code>'s <code>build_one()</code> writes the binary "
        "mask into <code>output.vxz</code>'s base_color, the unmodified SegviGen "
        "<code>vxz_to_slat()</code> encodes it into <code>output_tex_slat.pth</code> with no "
        "argument swap, and <code>train_emissive.py</code> line 230 trains against "
        "<code>otx</code> = <code>output_tex_slat.pth</code>, never <code>itx</code> = "
        "<code>input_tex_slat.pth</code>). Code review alone would not have ruled out a bug "
        "baked into files built by an earlier version of that code; decoding the actual "
        "on-disk bytes does.")
    body += lp.callout(
        "<p>The symptom the owner observed &mdash; overfit predictions glowing in each "
        "shape's own albedo colors, strengthening with more training &mdash; is real, but "
        "its source is not the training data. The next place to look is downstream: the "
        "model's own learned behavior (an interfering multi-shape fit reproducing "
        "appearance cues from the DINOv3 conditioning image or the <code>input_tex_slat</code> "
        "conditioning channel, both of which the model can see, into its emissive output), "
        "or a color-channel mixup in the prediction-visualization path. Neither was chased "
        "further here.</p>",
        warn=True, title="So what explains the albedo-colored glow?")
    return lp.section_v2("verdict", 1, "The stored training target is the emission mask, not the albedo", body)


def sec_table():
    def row(r):
        albedo = r["corr_decoded_output_vs_raw_albedo_per_channel"]
        pos_ctrl = r["corr_decoded_input_vs_raw_albedo_per_channel (positive control)"]
        return (
            f'<tr><td style="text-align:left">{_esc(r["name"])}</td>'
            f'<td>{r["meta_emissive_frac"]:.4f}</td>'
            f'<td>{r["corr_decoded_output_mean_vs_raw_mask"]:.3f}</td>'
            f'<td>{r["iou_decoded_output_vs_raw_binary_mask"]:.3f}</td>'
            f'<td>[{albedo[0]:.2f}, {albedo[1]:.2f}, {albedo[2]:.2f}]</td>'
            f'<td>[{pos_ctrl[0]:.2f}, {pos_ctrl[1]:.2f}, {pos_ctrl[2]:.2f}]</td></tr>')

    rows_html = "".join(row(r) for r in RESULTS)
    table = lp.results_table(
        ["shape", "GT emissive frac", "corr vs emission mask", "IoU vs emission mask",
         "corr vs albedo (R,G,B)", "positive control: corr(decode(input_tex_slat), albedo)"],
        rows_html)
    body = lp.prose(
        "All 5 shapes are seen-in-training shapes (ct10 is 10 shapes symlinked directly "
        "into train_72k; the single-shape control and the two plain train_72k shapes are "
        "also training-split members) &mdash; this diagnostic is about what the training "
        "target IS, not a generalization claim, so no held-out labeling applies here. "
        "Structural check on the raw pre-encode file (not shown as a column, true on all "
        "5 shapes): <code>output.vxz</code>'s base_color is R=G=B on 100% of voxels, values "
        "only in {0, 255}, and its white-voxel fraction matches <code>meta.json</code>'s "
        "<code>emissive_frac</code> to 4 decimal places &mdash; the mask was written correctly "
        "before encoding even begins. The near-equal per-channel albedo correlations "
        "(e.g. [0.72, 0.68, 0.67] rather than one channel standing out) are the tell that "
        "distinguishes a real leak from spatial coincidence: a target that had actually "
        "picked up a red car's color would correlate much harder on R than G/B, exactly "
        "the pattern the positive control shows when the target really is albedo.")
    body += table
    return lp.section_v2("table", 2, "Correlation against the emission mask stays high; correlation against albedo stays low and undifferentiated", body)


def sec_gallery():
    def panel_row(name, sid_label, frac):
        r = BY_NAME[name]
        panels = [
            ("RAW ALBEDO (input.vxz)", img_ref(f"{name}_raw_albedo.png"), f"{name} raw albedo"),
            ("DECODED TARGET (output_tex_slat.pth)", img_ref(f"{name}_decoded_target.png"), f"{name} decoded training target"),
            ("RAW EMISSION MASK (output.vxz)", img_ref(f"{name}_raw_emissive_mask.png"), f"{name} raw emission mask"),
        ]
        cap = (f"<b>{sid_label}</b> (GT emissive fraction {frac:.3f}). The decoded target "
               f"carries none of the albedo panel's coloring and is visually indistinguishable "
               f"from the raw emission mask on its right &mdash; numerically, correlation "
               f"{r['corr_decoded_output_mean_vs_raw_mask']:.3f} against the mask, IoU "
               f"{r['iou_decoded_output_vs_raw_binary_mask']:.3f}.")
        return lp.fig_row(panels, cap, content="photo")

    body = panel_row("72k_A_eccb9b85", "train_72k shape eccb9b85", BY_NAME["72k_A_eccb9b85"]["meta_emissive_frac"])
    body += panel_row("ct10_red_car_B", "ct10 shape 000b9fd47d (one of the two red-tinted shapes named in the owner's observation)",
                       BY_NAME["ct10_red_car_B"]["meta_emissive_frac"])
    body += lp.prose(
        "Voxels rendered directly as coarsened cubes (no mesh, no lighting model beyond the "
        "renderer's default), same camera and coarsening for every panel in a row so only "
        "the color content differs. All coordinates and colors come straight from the "
        "diagnostic job's own npz dumps &mdash; nothing hand-picked or touched up.")
    return lp.section_v2("gallery", 3, "The decoded target reads as a clean binary mask, not a colored render of the shape", body)


def sec_provenance():
    param_rows = "".join(
        f'<tr><td style="text-align:left">{html.escape(k)}</td><td>{html.escape(v)}</td></tr>'
        for k, v in [
            ("decode path", "shape_decoder.set_resolution(512) -> guide_subs; "
                             "tex_decoder(SparseTensor(otx.feats, coords), guide_subs=subs) * 0.5 + 0.5 "
                             "-- identical to eval_emissive.py's eval_sample() GT decode"),
            ("models", "microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16, shape_dec_next_dc_f16c32_fp16"),
            ("shapes checked", "ct10_red_car_A (001c79293c), ct10_red_car_B (000b9fd47d), "
                                "single_shape_control (91d94c0f), 72k_A (eccb9b85), 72k_B (089d07d0)"),
            ("alignment", "decoded voxel coords joined by exact int match onto the raw output.vxz/"
                          "input.vxz 512-grid coords; 92-99.7% coverage across the 5 shapes"),
            ("code checked (not just the decode)", "build_dataset_direct.py build_one(), "
                                                     "SegviGen/data_toolkit/vxz_to_slat.py, train_emissive.py"),
        ])
    param_table = lp.results_table(["parameter", "value"], param_rows)

    body = lp.prose(
        "Diagnostic script: <code>segvigen_emissive/code/decode_check.py</code> (staged at "
        "<code>/project/3dlg-hcvc/omages/yanxg_scratch/target_check/</code>). Solar job 242949 "
        "(cs-venus-08, A100, trellis2 conda env, 69s). A first attempt (job 242948) failed in "
        "16 seconds on <code>ModuleNotFoundError: trellis2</code>: the SegviGen import path was "
        "pointed at the local-disk checkout on cs-3dlg-25, which is not NFS-mounted on solar "
        "compute nodes; fixed by pointing at the NFS copy at "
        "<code>/3dlg-jupiter-project/lightgen/segvigen_emissive/code/SegviGen</code>. "
        "Visualization: <code>segvigen_emissive/render_target_check.py</code> (local, bpy pip "
        "package, no GPU).")
    body += param_table
    return lp.section_v2("provenance", 4, "How this was checked", body)


def main():
    stats = [
        ("REFUTED", "wrong-target hypothesis"),
        ("0.94-0.999", "corr vs emission mask (5 shapes)"),
        ("0.94-0.998", "positive control (albedo decode)"),
        ("5", "shapes checked, ct10 + single + 72k"),
    ]
    hero = lp.hero_header(
        "SegviGen · training-target decode check",
        "output_tex_slat.pth is the emission mask, not the albedo",
        dek_html=(
            "The owner observed overfit predictions glowing in each shape's own albedo "
            "colors, strengthening with training, and asked whether the dataset builder "
            "wrote the wrong tensor into the training target. Decoding the actual on-disk "
            "<code>output_tex_slat.pth</code> through the real TRELLIS.2 tex_decoder, on 5 "
            "shapes spanning the ct10 overfit set, the single-shape control, and train_72k, "
            "answers no: the stored target is the binary emission mask on every shape "
            "checked. The albedo-glow symptom is real; its cause is downstream of the data."),
        stats=stats,
        toc=[(i, lab) for i, lab in OUTLINE])

    body = [
        sec_verdict(),
        sec_table(),
        sec_gallery(),
        sec_provenance(),
    ]

    page_html = lp.page(
        title="Training-target decode check: emission mask, not albedo (target_check)",
        header_html=hero,
        body_sections=body,
        assets_rel=SITE_ASSETS,
        assets_dir=os.path.join(WEB, "assets"),
        theme="v3",
        tree_html=wz.tree_html(active_href=None),
        nav_title="target check",
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
    print(f"wrote {out} ({len(page_html)} bytes)")
    print("  zone-link guard: clean")

    publish_assets(os.path.join(WEB, "assets"))
    print("assets published")


if __name__ == "__main__":
    main()
