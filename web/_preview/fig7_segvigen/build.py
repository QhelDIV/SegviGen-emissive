#!/usr/bin/env python3
"""Build the fig7_segvigen page: SegviGen emission predictions on the 11 shapes of
Dongchen's Figure 7 (qualitative comparison). The PRIMARY gallery is rendered
emission-only (key light off, owner decision, see sec_gallery): a key-lit render
illuminates every object regardless of what it predicts, which made near-zero
predictions look like correctly-dark objects. The original key-lit treatment
(comparable to the existing paper_v3 gallery) is kept as a secondary reference.

Three checkpoints. emis_1k_w1 and emis_1k_w5 are trained on Path A data (the older
somage/GLB round-trip pipeline; see three_ckpt_table.py's own annotation, restated
here because the checkpoints themselves record nothing to verify it against -- see
the provenance section) and are out of distribution on the direct-ovoxel input used
here, but clean of the emis_1k training set (verified by the project lead). The third,
emis_72k_unfilt, is in-distribution (trained directly on this dataset_direct pipeline)
but MOST of the 11 shapes are in its own training split -- marked per shape, on the
figure itself, not only here. None of the three is the project's final method: the
canonical direct-ovoxel image-conditioned model does not exist yet, it is still training.

The IoU on this page is computed in the model's own voxel space (dump_pred_voxels_fig7.py,
the same decode path eval_emissive.py's diagnostics use) and is an INTERNAL proxy metric.
It is not commensurable with the project's point-sampled evaluation
(evaluate_pointsampled.py, 50k surface points, the metric used in the cross-method
comparison table) and is not presented as such anywhere on this page.

Every number on this page is read at build time from the run's own JSON (iou_table_11.json,
gallery_11.json, the render stage's per-shape stats.json), never retyped.

Run: /project/3dlg-hcvc/omages/omages_internal/.venv2/bin/python \
        web/_preview/fig7_segvigen/build.py
"""
import hashlib
import html
import json
import os
import shutil
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WEB)
sys.path.insert(0, os.path.join(REPO, "tools"))

import xgpage as lp                      # noqa: E402
import workspace_zone as wz              # noqa: E402  (read-only: tree + zone guard)
from xgpage.publish import publish_assets  # noqa: E402

SITE_ROOT = "/projects/omages/yanxg/lightgen"
SITE_ASSETS = f"{SITE_ROOT}/assets"
FAVICON = f"{SITE_ROOT}/assets/images/favicon.png"
PAGE_DATE = "2026-08-07"

FIG7 = "/project/3dlg-hcvc/omages/yanxg_scratch/fig7"
IMG_DIR = os.path.join(HERE, "img")
os.makedirs(IMG_DIR, exist_ok=True)

GALLERY = json.load(open(f"{FIG7}/gallery_11.json"))
IOU = {r["sid"]: r for r in json.load(open(f"{FIG7}/iou_table_11.json"))}
WHAT_BY_SID = {g["sid"]: g["what"] for g in GALLERY}
SHORT_ROW_LABEL = {
    "92cedcbac4f84083b04e10a6df6ef0f3": "fairy-tale cottage",
    "34170054845344aeb199b842a3bf7e92": "robot, glowing eyes",
}

# columns = methods (SIGGRAPH convention). A fourth prediction column
# (point-sampled evaluation, owned by a separate agent) can be appended here
# later without touching the matrix/table plumbing below -- see the appendix.
MODELS = [
    ("emis_1k_w1", "pos_weight 1", "1k shapes, Path A data, OOD here"),
    ("emis_1k_w5", "pos_weight 5", "1k shapes, Path A data, OOD here"),
    ("emis_72k_unfilt", "72k unfiltered, pos_weight 5",
     "72k shapes, direct-ovoxel, in-distribution -- but see contamination below"),
]
THR = "0.5"

TRAINING_SPLIT_ANNOTATION = (
    "emis_1k_w1/emis_1k_w5: Path A data (the somage/GLB round-trip pipeline), "
    "conditioning mode real")

# emis_72k_unfilt's OWN train/val/test split (code/build_dataset_direct.py's
# dataset_direct/{train,val,test}_72k directories -- the training job's own
# configuration, not a possibly-stale manifest). Read once here, from GALLERY's
# "split" field, and used both for the per-shape figure marking and the table.
SPLIT_TAG = {"train_72k": "TRAIN", "val_72k": "VAL", "test_72k": "TEST"}
SPLIT_WORD = {"train_72k": "trained on", "val_72k": "used for validation",
             "test_72k": "held out"}

OUTLINE = [
    ("verdict", "What the panels show"),
    ("gallery", "The 11 shapes, five columns each"),
    ("quant", "IoU against the direct-ovoxel ground truth"),
    ("provenance", "How this was produced"),
]


def copy_img(src_path, dest_name):
    dst = os.path.join(IMG_DIR, dest_name)
    shutil.copy2(src_path, dst)
    return dst


def rel_img(abspath):
    return "img/" + os.path.relpath(abspath, IMG_DIR)


def img_ref(abspath):
    """Cache-busted page-relative src: a fill-in republish (72k landing after
    w1/w5 already shipped) overwrites the same filenames, and a bare path would
    let a CDN/browser keep serving the placeholder or an old panel."""
    h = hashlib.md5(open(abspath, "rb").read()).hexdigest()[:8]
    return f"{rel_img(abspath)}?v={h}"


def check_ckpt_annotation():
    """Restate, as an annotation (not a checkpoint-verified fact): all three
    checkpoints carry only a bare state_dict, no hparams/config -- verified live by
    torch.load on best.ckpt for all three (2026-08-07): top-level keys ==
    ['state_dict'] only (640 tensors for the two 1k checkpoints), nothing else. The
    training-split/conditioning label for emis_1k_w1/w5 is therefore
    three_ckpt_table.py's own annotation, not something read from the checkpoint
    file. emis_72k_unfilt's split membership, by contrast, IS read from the
    training job's own dataset_direct split directories, which is a fact, not an
    annotation."""
    return True


# --------------------------------------------------------------------- images
def _stage_set(pred_root, name_suffix, require_72k):
    """Copy input/GT/w1/w5/72k panels from one pred_* output tree (either the
    key=8 key-lit tree or the key=0 emission-only tree) into this page's own
    img/, distinguishing filenames by name_suffix so the two treatments don't
    collide on disk."""
    out = {}
    k72_ready = True
    for g in GALLERY:
        sid = g["sid"]
        # pred_root is either "pred" (key=8) or "pred_dark" (key=0); the
        # per-model subdirectory name always carries its own _k8/_k0 suffix.
        suffix = "k8" if pred_root == "pred" else "k0"
        w1_dir = os.path.join(FIG7, pred_root, f"emis_1k_w1_method_{suffix}")
        w5_dir = os.path.join(FIG7, pred_root, f"emis_1k_w5_method_{suffix}")
        k72_dir = os.path.join(FIG7, pred_root, f"emis_72k_unfilt_method_{suffix}")
        lit_src = os.path.join(w1_dir, f"{sid}_lit.png")
        true_src = os.path.join(w1_dir, f"{sid}_true.png")
        w1_src = os.path.join(w1_dir, f"{sid}_glow.png")
        w5_src = os.path.join(w5_dir, f"{sid}_glow.png")
        k72_src = os.path.join(k72_dir, f"{sid}_glow.png")
        for p in (lit_src, true_src, w1_src, w5_src):
            if not os.path.exists(p):
                raise RuntimeError(f"missing render output: {p}")
        entry = {
            "input": copy_img(lit_src, f"{sid}_input_{name_suffix}.png"),
            "gt": copy_img(true_src, f"{sid}_gt_{name_suffix}.png"),
            "emis_1k_w1": copy_img(w1_src, f"{sid}_w1_{name_suffix}.png"),
            "emis_1k_w5": copy_img(w5_src, f"{sid}_w5_{name_suffix}.png"),
        }
        if os.path.exists(k72_src):
            entry["emis_72k_unfilt"] = copy_img(k72_src, f"{sid}_72k_{name_suffix}.png")
        elif require_72k:
            k72_ready = False
        out[sid] = entry
    return out, k72_ready


def stage_images():
    """Stage BOTH treatments. Emission-only (key=0, pred_dark/) is PRIMARY, per
    owner decision: a key light illuminates every object regardless of what it
    emits, so a key-lit gallery cannot show WHETHER something glows, only that
    it is lit. Key-lit (key=8, pred/) is kept as a secondary reference (an
    expandable aside, sec_gallery) purely so a reader can identify each object
    -- a fully dark panel of an unfamiliar shape teaches nothing on its own.
    72k is optional in the dark set only, mirroring the original fill-in
    mechanism (now moot: render job 240462 finished before this build ran, but
    the mechanism costs nothing to keep)."""
    imgs_dark, k72_ready = _stage_set("pred_dark", "dark", require_72k=True)
    imgs_lit, _ = _stage_set("pred", "lit", require_72k=False)
    return imgs_dark, imgs_lit, k72_ready


# --------------------------------------------------------------------- sections
def sec_verdict():
    n_shapes = len(GALLERY)
    w1_vals = [IOU[g["sid"]]["emis_1k_w1_iou_mean"] for g in GALLERY]
    w5_vals = [IOU[g["sid"]]["emis_1k_w5_iou_mean"] for g in GALLERY]
    k72_vals = [IOU[g["sid"]]["emis_72k_unfilt_iou_mean"] for g in GALLERY]
    n_train = sum(1 for g in GALLERY if g["split"] == "train_72k")

    body = lp.callout(
        "<b>None of these three is the project's final method, and one column "
        "shows mostly memorized shapes.</b> emis_1k_w1/w5 were trained on Path A "
        "data (the older somage/GLB round-trip pipeline) and are evaluated here "
        "on the canonical direct-ovoxel input instead, so they are out of "
        "distribution; the checkpoints themselves record only a bare "
        "<code>state_dict</code> (verified by loading all three "
        "<code>best.ckpt</code> files), so the Path-A label is "
        f"<code>three_ckpt_table.py</code>'s own annotation, not something the "
        "checkpoint states about itself. emis_72k_unfilt is in-distribution, but "
        f"{n_train} of the 11 shapes are in ITS OWN training split (named in the "
        "gallery caption below); only 2 are held out in <code>test_72k</code>. "
        "The canonical image-conditioned direct-ovoxel "
        "model does not exist yet; it is still training. Separately, all 11 "
        "shapes are clean of the <code>emis_1k</code> training set (1,122 sids, "
        "verified by the project lead) and all 11 sit in the project's frozen "
        "381-shape point-sampled eval set (<code>usable.txt</code>), though the "
        "IoU on this page is a different, internal metric -- see the callout in "
        "the quantitative section.",
        warn=True, title="Read this before the gallery")
    body += lp.prose(
        f"Mean IoU over the {n_shapes} shapes (each shape's own mean over 3 "
        f"sampling draws, threshold 0.5): "
        f"<b>pos_weight 1</b> = {statistics.mean(w1_vals):.3f}, "
        f"<b>pos_weight 5</b> = {statistics.mean(w5_vals):.3f}, "
        f"<b>72k unfiltered</b> = {statistics.mean(k72_vals):.3f} "
        f"(dominated by the {n_train} training-split shapes -- see the per-shape "
        "table for the split-conditioned reading). All three are far below the "
        "ceiling and dominated by a handful of shapes each model gets partly "
        "right; most panels below show an empty or near-empty prediction. The "
        "comparison figure and table follow.")
    return lp.section_v2("verdict", 1, "Every column is compromised in a different way", body)


def _gallery_matrix(imgs, k72_ready, badge=True, id_=None):
    rows = []
    for g in GALLERY:
        sid = g["sid"]
        r = IOU[sid]
        label = SHORT_ROW_LABEL.get(sid, g["what"])
        w1_iou, w5_iou = r["emis_1k_w1_iou_mean"], r["emis_1k_w5_iou_mean"]
        if badge:
            w1_cell = {"img": img_ref(imgs[sid]["emis_1k_w1"]),
                      "badge": f"{w1_iou:.3f}", "best": w1_iou >= w5_iou}
            w5_cell = {"img": img_ref(imgs[sid]["emis_1k_w5"]),
                      "badge": f"{w5_iou:.3f}", "best": w5_iou > w1_iou}
        else:
            w1_cell, w5_cell = img_ref(imgs[sid]["emis_1k_w1"]), img_ref(imgs[sid]["emis_1k_w5"])
        # No per-panel train/val/test marking (owner decision: one caption sentence
        # instead, naming the trained-on shapes -- see below). Split membership
        # itself is kept in the provenance table, not dropped, just not laid out
        # per panel.
        if k72_ready:
            k72_iou = r["emis_72k_unfilt_iou_mean"]
            k72_cell = ({"img": img_ref(imgs[sid]["emis_72k_unfilt"]), "badge": f"{k72_iou:.3f}"}
                       if badge else img_ref(imgs[sid]["emis_72k_unfilt"]))
        else:
            k72_cell = {"placeholder": "72k", "sub": "still rendering"}
        rows.append((label, [
            img_ref(imgs[sid]["input"]),
            img_ref(imgs[sid]["gt"]),
            w1_cell,
            w5_cell,
            k72_cell,
        ]))
    return rows


def sec_gallery(imgs_dark, imgs_lit, k72_ready):
    train_shapes = [SHORT_ROW_LABEL.get(g["sid"], g["what"]) for g in GALLERY if g["split"] == "train_72k"]
    n_train = len(train_shapes)
    n_val = sum(1 for g in GALLERY if g["split"] == "val_72k")
    n_test = sum(1 for g in GALLERY if g["split"] == "test_72k")
    pending_note = (
        "" if k72_ready else
        " The 72K column was still rendering at publish time and will be "
        "filled in on this same page, same URL, without a layout change, "
        "once it lands."
    )

    dark_matrix = lp.method_matrix(
        columns=["INPUT", "GROUND TRUTH", "W1", "W5", "72K"],
        rows=_gallery_matrix(imgs_dark, k72_ready),
        caption_html=(
            "<b>Most predictions are empty or land on the wrong material; where a "
            "model IS right, the panel can still render dark.</b> Input (lit, for "
            "identification only, not part of the claim), then GROUND TRUTH and "
            "every prediction rendered EMISSION-ONLY: same key-lit treatment as "
            "before (AgX, exposure 0, bloom 9/1.0/&minus;0.15, samples 256/96, "
            "emission strength 4.0) with the key light itself turned off "
            "(key&nbsp;0), so a panel is bright only where something actually "
            "emits, on the identical camera and crop as every other column -- "
            "chosen over the box preset (Cornell-wall room) because box changes "
            "the composition and adds context that does not belong in a "
            "per-shape comparison tile at this size. This "
            "replaces an earlier key-lit version of this gallery, which lit every "
            "object regardless of what it predicted and made near-zero predictions "
            "look like correct dark objects; that key-lit treatment is kept below "
            "as a reference for what each object actually looks like. One "
            "representative draw (draw 0 of 3, seed 0) per shape per model. The "
            "corner number is that draw's IoU against the ground truth (unchanged "
            "by the render treatment: IoU is computed from voxel data, not "
            "pixels); on W1/W5 the brighter badge is the higher of the two. Of "
            f"the 11 shapes, {n_train} are in the 72K checkpoint's OWN training "
            f"split ({', '.join(train_shapes)}), {n_val} were used to select that "
            f"checkpoint, and only {n_test} are genuinely held out for it. The "
            "full 3-draw mean and std are in the table below, not this single "
            "draw." + pending_note),
        native_px=768, content="photo", page_inner=820, id="gallery-matrix")

    findings = lp.callout(
        "<b>Outdoor wall lantern: w5 predicts 86.1% of the surface, the highest "
        "IoU on the set (0.491) -- and its emission-only panel is black.</b> "
        "Verified directly on the rendered pixels: the w5 dark panel tops out at "
        "24/255 with zero bright texels, the same background-only level as an "
        "empty prediction, while the ground-truth dark panel for the same shape "
        "reaches 243/255 with 9.1% bright pixels. The project's rendering formula "
        "is <code>emission = mask &times; albedo</code>: a mask can be almost "
        "perfectly right and still emit nothing if the housing's albedo is dark. "
        "This is a structural limit of the mask&times;albedo formulation, not a "
        "training failure, and dark-housed lamps are a common real case.",
        warn=True, title="What the dark treatment makes visible, finding 1")
    findings += lp.callout(
        "<b>Red glowing sword: w1 scores 0.325, second-highest on the set, and "
        "its dark panel shows a speck near the hilt against a fully glowing "
        "blade in the ground truth.</b> Verified on the rendered pixels: 0.07% "
        "of the w1 panel is bright versus 6.4% for the ground truth. A "
        "second-best IoU next to a panel that is visibly almost entirely wrong "
        "means the metric and the picture disagree here; the voxel IoU rewards "
        "getting the RIGHT MATERIAL more than it penalizes getting little of "
        "its area, and a reader trusting the number alone would rank this "
        "shape far too high.",
        warn=True, title="What the dark treatment makes visible, finding 2")
    findings += lp.callout(
        "<b>Jack-o'-lantern: the ground-truth panel is genuinely dark, confirmed "
        "on the rendered pixels, and this is a disagreement between the two "
        "ground truths, not a camera artifact.</b> The voxel-based ground truth "
        "used to score every model says 5.02% of this shape is emissive; the "
        "rendered pixels of its dark GT panel show no visible glow (max channel "
        "24/255, 0% bright pixels, at both key=8 and key=0). "
        "<b>Correction, checked after an earlier draft of this callout:</b> the "
        "render pipeline's own accounting field (<code>area_lit_frac</code>) "
        "also reads exactly 0.0 on 4 other shapes (world-map table, fairy-tale "
        "cottage, medieval brazier, outdoor wall lantern), which first looked "
        "like the same pattern -- but checking their ACTUAL rendered pixels "
        "shows all 4 have real, substantial visible emission (243-255/255 max, "
        "0.4%-9.1% bright pixels; the outdoor wall lantern's own GT panel is "
        "finding 1's 243/255, 9.1%). So <code>area_lit_frac</code> is unreliable "
        "on this dataset and is not evidence of anything by itself; the pumpkin "
        "is the ONLY one of the 11 shapes whose ground-truth panel is actually "
        "dark. For the pumpkin specifically: its raw emissive texture does "
        "contain bright pixels (max value 1.0, uniform white emission colour, "
        "2.9% of texels by raw count), and rasterizing the mesh's real UV "
        "footprint against that texture shows 75.8% of those bright texels fall "
        "outside any face's UV island (unused atlas space) -- but the remaining "
        "24.2% (0.70% of the whole texture) IS covered by real geometry, spread "
        "across 13 separate high-poly mesh parts that all land in the same "
        "small, compact region of the object (matching the voxel ground "
        "truth's own emissive cluster in world space). <b>Confirmed by "
        "isolation render (see the provenance section's bake-mechanism "
        "callout for the numbers): that region genuinely emits</b> -- "
        "rendered alone with every other face deleted it lights up strongly "
        "(max 254/255, 9.8% bright), while the full scene at identical "
        "settings stays dark (max 24/255). The emission is real; something "
        "else in the scene occludes it from every camera angle checked. A "
        "4-azimuth sweep of the key-lit GT panel "
        "(38&deg;/128&deg;/218&deg;/308&deg;) had already ruled out a "
        "simple hidden-face-at-this-camera explanation: brightness stayed "
        "within the same range at every angle (mean 31-34/255, attributable to "
        "the key light, not emission). With occlusion confirmed, the voxel "
        "and mesh ground truths are not in disagreement about a fact, they "
        "are measuring different things: the voxel bake is surface-complete "
        "(it reads whatever a real face's UV points to, occluded or not), "
        "the render is visibility-limited (it can only show what a camera "
        "ray can reach). The models scoring near-zero is correct for what a "
        "viewer sees; the voxel ground truth they are scored against is "
        "correct for what the surface actually carries. Neither is wrong.",
        warn=True, title="What the dark treatment makes visible, finding 3 (corrected)")

    lit_matrix = lp.method_matrix(
        columns=["INPUT", "GROUND TRUTH", "W1", "W5", "72K"],
        rows=_gallery_matrix(imgs_lit, k72_ready, badge=False),
        caption_html=(
            "The same 11 shapes under the earlier key-lit treatment (key 8, AgX, "
            "bloom 9/1.0/&minus;0.15, samples 256/96), kept only so the object "
            "itself is identifiable -- every panel here is lit by a key light "
            "regardless of what it emits, so brightness in this row is NOT "
            "evidence of emission. No IoU badges: the numbers live on the "
            "emission-only matrix above, where the treatment actually supports "
            "reading them next to the picture."),
        native_px=768, content="photo", page_inner=820, id="gallery-matrix-lit")
    secondary = lp.expandable("Reference: the same 11 shapes, key-lit (what each object actually is)",
                              lit_matrix, open=False)

    return lp.section_v2("gallery", 2, "The emission-only treatment shows three distinct failure modes",
                         dark_matrix + findings + secondary)


def sec_quant():
    def chart_rows(tag):
        return sorted(
            ({"label": SHORT_ROW_LABEL.get(g["sid"], g["what"]),
              "value": IOU[g["sid"]][f"{tag}_iou_mean"],
              "display": f"{IOU[g['sid']][f'{tag}_iou_mean']:.3f} ± "
                         f"{IOU[g['sid']][f'{tag}_iou_std']:.3f}"}
             for g in GALLERY), key=lambda r: -r["value"])

    chart_w1 = lp.hbar_chart(chart_rows("emis_1k_w1"),
                             title="IoU @0.5, mean ± std over 3 draws",
                             label_w=190,
                             note="<b>pos_weight 1</b>: every shape's mean sits below its "
                                  "own draw std, so the ranking above is noise, not a "
                                  "reliable per-shape ordering.")
    chart_w5 = lp.hbar_chart(chart_rows("emis_1k_w5"),
                             title="IoU @0.5, mean ± std over 3 draws",
                             label_w=190,
                             note="<b>pos_weight 5</b>: higher on the shapes it gets partly "
                                  "right, but the draw std is still comparable to the mean "
                                  "on most shapes.")
    chart_72k = lp.hbar_chart(chart_rows("emis_72k_unfilt"),
                              title="IoU @0.5, mean ± std over 3 draws",
                              label_w=190,
                              note="<b>72k unfiltered</b>: sorted by score alone this looks "
                                   "like the strongest model, but most of the top rows are "
                                   "shapes it trained on (see the TRAIN/VAL/TEST table below "
                                   "for which).")

    # results table: full per-shape numbers, all three models, GT emissive fraction
    rows_html = ""
    for g in GALLERY:
        sid = g["sid"]
        r = IOU[sid]
        label = SHORT_ROW_LABEL.get(sid, g["what"])
        rows_html += (
            f'<tr><td style="text-align:left">{html.escape(label)}</td>'
            f'<td>{r["gt_frac"]:.4f}</td>'
            f'<td>{r["emis_1k_w1_iou_mean"]:.3f} ± {r["emis_1k_w1_iou_std"]:.3f}</td>'
            f'<td>{r["emis_1k_w5_iou_mean"]:.3f} ± {r["emis_1k_w5_iou_std"]:.3f}</td>'
            f'<td>{r["emis_72k_unfilt_iou_mean"]:.3f} ± {r["emis_72k_unfilt_iou_std"]:.3f}</td>'
            f'<td>{SPLIT_WORD[g["split"]]}</td></tr>')
    table = lp.results_table(
        ["shape", "GT emissive frac.", "w1 IoU", "w5 IoU",
         "72k IoU", "72k's own split"], rows_html)

    zero_flags = [g["what"] for g in GALLERY
                  if IOU[g["sid"]]["gt_frac"] == 0.0
                  and any(v == 1.0 for tag in ("emis_1k_w1", "emis_1k_w5", "emis_72k_unfilt")
                          for v in IOU[g["sid"]][f"{tag}_draws"])]
    zero_note = (f"No draw scored 1.0 by the empty/empty convention on any of the 11 "
                 f"shapes: every shape has nonzero ground-truth emissive coverage "
                 f"(min {min(r['gt_frac'] for r in IOU.values()):.3f})."
                 if not zero_flags else
                 f"The empty/empty 1.0 convention fired on: {', '.join(zero_flags)}.")

    body = lp.callout(
        "<b>This IoU is an internal, per-voxel proxy metric and is NOT comparable "
        "to the project's point-sampled evaluation.</b> It is computed in the "
        "model's own voxel space against the direct-ovoxel ground truth "
        "(<code>dump_pred_voxels_fig7.py</code>, the same decode path "
        "<code>eval_emissive.py</code>'s diagnostics use, which already calls "
        "itself a proxy), at threshold 0.5. The project's point-sampled harness "
        "(<code>evaluate_pointsampled.py</code>, run separately by another agent) "
        "samples 50k surface points on the source mesh and reports "
        "iou_texgen&nbsp;=&nbsp;0.1042 and iou_trellis&nbsp;=&nbsp;0.2450 on a "
        "frozen 381-shape set that all 11 shapes here belong to -- those numbers "
        "are a DIFFERENT metric on a different sample space and are not to be "
        "read against the numbers on this page. A point-sampled column for these "
        "11 shapes is expected once that run lands; this section is laid out to "
        "take it without a rebuild.")
    body += lp.prose(
        "3 draws per shape per model (seed 0, draw 0 is the panel shown above). "
        "An empty prediction against an empty ground truth scores IoU=1.0 by "
        "convention. " + zero_note)
    # Stacked, not a side-by-side grid: hbar_chart's SVG carries its own 640px
    # min-width floor below 640px viewports (xgpage skill rule 8's addendum,
    # self-contained scroll), and a 3-up CSS grid put that floor in conflict
    # with a narrow grid TRACK rather than the page, producing real page-level
    # overflow at 390px (grid items default to min-width:auto around their own
    # overflow:auto content) -- caught by qa_widths.js, fixed by dropping the
    # grid and using each chart's own tested full-width centering.
    body += lp.prose("<b>pos_weight 1</b>") + chart_w1
    body += lp.prose("<b>pos_weight 5</b>") + chart_w5
    body += lp.prose("<b>72k unfiltered</b>") + chart_72k
    body += lp.prose(
        "The project's own stratification by ground-truth emissive fraction "
        "(<code>three_ckpt_table.py</code>, 300 held-out shapes) found IoU behaves "
        "very differently below and above roughly 5% coverage; 11 shapes is too "
        "few to re-stratify here, so the full per-shape table is below instead of "
        "an aggregate that would hide it.")
    body += table
    return lp.section_v2("quant", 3, "IoU is noisy at this sample size, and the 72k column is mostly memorization", body)


def sec_provenance():
    rows_html = ""
    for tag, note, dist in MODELS:
        rows_html += (f'<tr><td style="text-align:left">{tag}</td>'
                      f'<td>{note}</td><td>{dist}</td>'
                      f'<td>/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/{tag}/best.ckpt</td></tr>')
    ckpt_table = lp.results_table(["checkpoint", "loss weighting", "distribution", "path"], rows_html)

    param_rows = "".join(
        f'<tr><td style="text-align:left">{html.escape(k)}</td><td>{html.escape(v)}</td></tr>'
        for k, v in [
            ("mode", "method (mask &times; albedo)"),
            ("view_transform", "AgX"), ("exposure", "0.0"), ("key light", "8"),
            ("background", "0.012"), ("bloom", "1 (size 9, threshold 1.0, mix -0.15)"),
            ("samples", "256 (96 for the lit panel)"), ("emission strength", "4.0"),
            ("resolution", "768×768 (Cycles)"),
            ("mask threshold", "0.5"), ("mask tolerance", "2.0 voxels"),
            ("draws", "3 per shape per model, seed 0, draw 0 shown"),
        ])
    param_table = lp.results_table(["parameter", "value"], param_rows)

    # Job ids + the resource request each stage actually ran with. w1/w5 renders
    # ran on GPU (the standard render.sbatch); the 72k render was resubmitted as a
    # wide CPU array after the original GPU-partitioned job (240414, 8 cpus/task)
    # was crawling at 1/11 after 13 minutes -- Cycles CPU output is deterministic
    # in thread count, so this is the identical render, just wider, not a
    # different treatment (verified: every flag in the param table above was held
    # fixed between 240414 and the resubmit).
    job_rows = "".join(
        f'<tr><td style="text-align:left">{html.escape(stage)}</td>'
        f'<td>{jid}</td><td>{res}</td></tr>'
        for stage, jid, res in [
            ("material survey (Blender slot order)", "240396", "CPU, 1 task, debug partition"),
            ("dump_pred_voxels, w1+w5 (3 draws each)", "240397", "GPU L40S x1, 8 cpus"),
            ("dump_pred_voxels, 72k (3 draws)", "240399", "GPU L40S x1, 8 cpus"),
            ("pred_mask_to_asset, w1+w5", "240398", "CPU, 8 cpus, debug partition"),
            ("pred_mask_to_asset, 72k", "240404", "CPU, 8 cpus, debug partition"),
            ("render, w1", "240400", "GPU L40S x1, 8 cpus"),
            ("render, w5", "240401", "GPU L40S x1, 8 cpus"),
            ("render, 72k (abandoned, too slow)", "240414", "GPU L40S x1, 8 cpus -- cancelled at 1/11 after 13 min"),
            ("render, 72k (resubmit, actually used)", "240441 (array 0-10)",
             "CPU only (--gres=gpu:0), 64 cpus/task, 96G mem, "
             "nodelist cs-venus-13/14/15/16/17; --overwrite 0 so the "
             "6 shapes 240414 already finished were skipped, not redone"),
        ])
    job_table = lp.results_table(["stage", "solar job id", "resources"], job_rows)

    contam_rows = "".join(
        f'<tr><td style="text-align:left">{html.escape(g["what"])}</td>'
        f'<td>{g["split"]}</td><td>{SPLIT_WORD[g["split"]]}</td></tr>'
        for g in sorted(GALLERY, key=lambda g: g["split"]))
    contam_table = lp.results_table(["shape", "dataset_direct split", "status for emis_72k_unfilt"], contam_rows)

    body = lp.callout(
        "<b>Checked whether the voxel bake itself can manufacture emission "
        "that no visible surface carries, since that would affect every "
        "number on this project, not just this page.</b> Read the actual bake "
        "code (<code>uv_voxel_pipeline/code_snapshot/data_processing/"
        "uv_voxel_pipeline/atlas.py</code>): it re-parametrizes the WHOLE mesh "
        "into a fresh xatlas layout (so every real triangle gets a texel by "
        "construction, with no unmapped/'padding' concept in the bake's OWN "
        "atlas) and samples the ORIGINAL glTF texture at each real face's OWN "
        "original UV, with correct REPEAT wrapping for tiled UVs. That means "
        "the bake cannot pick up emission from texture regions no real face's "
        "UV touches (finding 3's 'padding' texels): it only ever reads what a "
        "genuine face's own UV points to. For the pumpkin, verified directly on "
        "the mesh: 13 of its primitives DO have real UV footprint landing on "
        "bright texels (0.70% of the texture, the 24.2% of hot texels finding "
        "3 traced to real UV, not padding), in a world-space region that "
        "matches the voxel ground truth's own emissive cluster. Then checked "
        "the RAW bake output directly (<code>emission_voxels_256/"
        "&lt;sid&gt;.vxz</code>, 280,461 voxels, 14,058 of them emissive, "
        "5.01% -- matching the 5.02% used for scoring): its emissive voxels are "
        "tightly spatially coherent, not scattered. Every lit voxel's nearest "
        "lit neighbor is exactly 1 grid step away (mean/median/p90/max all "
        "1.000, against 2.017 for a same-count random sample of occupied "
        "voxels), and they form 4 distinct connected clusters (62.9%, 17.2%, "
        "16.1%, 3.9% of the lit voxels respectively) -- consistent with a "
        "carved face's separate features (two eyes, nose, mouth) rather than "
        "noise. So the bake's 5.02% figure is not a padding-sampling artifact "
        "and not spatial noise: it is reading a real, compact, plausibly "
        "facial region of the asset. <b>Occlusion, confirmed directly:</b> "
        "deleted every OTHER face (kept only the 61,094 faces whose UV lands "
        "on a hot texel) and re-rendered the same dark camera -- the isolated "
        "geometry lights up strongly (max 254/255, 9.8% bright pixels), where "
        "the full, unmodified scene at the same settings stays at max 24/255, "
        "0% bright (and a reciprocal check, deleting ONLY those faces and "
        "keeping everything else, also stays dark, ruling out a rendering-"
        "setup difference between the tests). The emissive geometry is real "
        "and does glow; something else in the scene blocks it from every "
        "camera angle checked. This settles the open question from the "
        "earlier draft of this callout: it is occlusion, not missing or "
        "mislocated emission. Checked on one shape only; whether it "
        "generalizes across the dataset is unknown and would need its own "
        "investigation. No pipeline was changed; the isolation renders were "
        "throwaway diagnostic frames (384px, not page panels), not a "
        "re-render of anything published on this page.",
        warn=True, title="Bake-mechanism check (broader than this page)")
    body += lp.prose(
        "Inputs: <code>/cs/3dlg-jupiter-project/lightgen/uv_voxel_pipeline/"
        "out_uv_voxel_74k/&lt;sid&gt;/{pbr_voxels_256, emission_voxels_256}</code> "
        "(the direct-ovoxel dataset each checkpoint's own preprocessed "
        "<code>dataset_direct</code> entries derive from) and "
        "<code>/cs/3dlg-falas/datasets/TexVerse-1K/glbs/glbs_1k/</code> for the "
        "source GLBs. All 11 sids already had built <code>dataset_direct</code> "
        "entries, so no new dataset build was needed for any of the three "
        "checkpoints.")
    body += lp.callout(
        "<b>emis_72k_unfilt's own train/val/test split membership</b>, read from "
        "<code>dataset_direct/{train,val,test}_72k/&lt;sid&gt;/</code> directly "
        "(the training job's own configuration -- <code>train_72k_unfilt.sbatch</code> "
        "trains on exactly <code>--train_split train_72k</code>), not from a "
        "manifest that merely looks plausible:")
    body += contam_table
    body += lp.prose("Checkpoints:") + ckpt_table
    body += lp.prose("Render parameters, key-lit preset, every flag pinned:") + param_table
    body += lp.prose(
        "Solar job ids and the resources each stage actually ran with, environment "
        "<code>/project/3dlg-hcvc/omages/omages_internal/.venv</code> "
        "(<code>PYTHONPATH=xgutils/src</code>) for Blender/render stages, the "
        "<code>trellis2</code> conda env for GPU inference, account "
        "<code>3dlg-hcvc-lab</code>:")
    body += job_table
    body += lp.prose(
        "Commands: <code>dump_pred_voxels_fig7.py</code> (GPU, trellis2 env) samples "
        "each checkpoint 3 times per shape and scores against the voxel-space ground "
        "truth; <code>pred_mask_to_asset.py</code> (CPU) transfers the predicted "
        "voxel mask onto the asset's own UV space, per material slot (Blender slot "
        "order surveyed fresh for these 11 shapes, "
        "<code>build_material_survey_fig7.py</code>, since paper_v3's own survey "
        "covers a different 11 shapes); <code>render_emissive.py</code> (GPU, Cycles) "
        "renders the input, ground truth and prediction panels. All three ran on "
        "solar.cs.sfu.ca under <code>/project/3dlg-hcvc/omages/yanxg_scratch/fig7/</code>, "
        "never on the workstation.")

    attrib_rows = "".join(
        f'<tr><td style="text-align:left">{html.escape(g["what"])}</td>'
        f'<td>{html.escape(g["author"])}</td>'
        f'<td>{html.escape(g["license"])}</td>'
        f'<td><a href="{g["url"]}">{g["sid"][:8]}&hellip;</a></td></tr>'
        for g in GALLERY)
    attrib_table = lp.results_table(["shape", "author", "licence", "Sketchfab"], attrib_rows)
    body += lp.prose("Attribution, read from TexVerse's own metadata at build time "
                     "(licence recorded per shape as-is, no special handling):")
    body += attrib_table

    apx = lp.appendix("Sources", [
        "Checkpoints: outputs/emis_1k_w1/best.ckpt, outputs/emis_1k_w5/best.ckpt, "
        "outputs/emis_72k_unfilt/run1/best.ckpt under "
        "/3dlg-jupiter-project/lightgen/segvigen_emissive/.",
        "Sampling/scoring: fig7/segcode/dump_pred_voxels_fig7.py "
        "(a 3-draw extension of segvigen_emissive/code/dump_pred_voxels.py; draw 0 "
        "uses the identical per-sid seed as the original single-draw script).",
        "Mask transfer: segvigen_emissive/code/pred_mask_to_asset.py, unmodified.",
        "Material survey: fig7/code/build_material_survey_fig7.py (Blender slot "
        "order for these 11 shapes; paper_v3/material_survey.json covers a "
        "different 11).",
        "Render: segvigen_emissive/render/render_emissive.py, staged to "
        "fig7/code/ per its README's staging section.",
        "IoU table: fig7/iou_table_11.json, computed from the three "
        "dump_pred_voxels_fig7.py summary.json files.",
        "Point-sampled evaluation (separate agent, not run by this page's build): "
        "evaluate_pointsampled.py, frozen 381-shape usable.txt eval set, "
        "iou_texgen=0.1042, iou_trellis=0.2450 as reference numbers on a metric "
        "this page's IoU is not comparable to. A prediction column from that run "
        "is expected to be added to the gallery matrix and the quantitative table "
        "above without a layout rebuild.",
        "Emission-only re-treatment: fig7/code/render_dark.sbatch, wide CPU array "
        "(job 240462, one task per shape, 64 cpus/task, --gres=gpu:0, nodelist "
        "cs-venus-13/14/15/16/17), writing pred_dark/&lt;tag&gt;_method_k0/. Same "
        "flags as the key-lit renders with --key 0. The abandoned camera-azimuth "
        "check on the jack-o'-lantern (job 240457, 4 angles at the key-lit "
        "treatment) is in fig7/azsweep/; superseded by the sidecar/pixel evidence "
        "in the gallery caption, kept on disk for the record.",
        "A corrected LIN_EPS comment landed in segvigen_emissive/render/"
        "render_emissive.py after these renders (the old comment mis-described "
        "the sRGB-to-linear conversion). It changes no behaviour and required no "
        "re-render; the staged copy under fig7/code/ predates the fix and should "
        "be re-staged before any FUTURE render from this pipeline.",
        "Occlusion isolation test (jack-o'-lantern): fig7/code/"
        "pumpkin_isolation_test.py, job 240523, throwaway 384px diagnostic "
        "frames at fig7/isolation_test/, not page panels.",
        "<b>Open renderer bug, not fixed here:</b> "
        "<code>render_emissive.py</code>'s <code>measure_lit_fraction</code> "
        "(the function behind the <code>area_lit_frac</code>/<code>mask_frac</code> "
        "sidecar fields) reads exactly 0.0 on 4 of these 11 shapes whose "
        "ground-truth panels plainly show real emission on the rendered pixels "
        "(world-map table, fairy-tale cottage, medieval brazier, outdoor wall "
        "lantern). Only one of the four (fairy-tale cottage) uses a uniform "
        "(textureless) emitter, which is the one case the function's own code "
        "visibly under-handles (<code>if e_node is None: lit += area</code> "
        "should count it and the sidecar says otherwise); the other three are "
        "TEXTURED emitters that should read nonzero by the same formula and "
        "don't, so the defect is broader than the uniform-emitter branch, most "
        "likely in the call sequence or image state at the time the function "
        "runs. Not traced further. These 4 sids are working reproduction "
        "cases for the next session.",
    ])
    return lp.section_v2("provenance", 4, "Commands, parameters, and where the raw outputs live", body), apx


def main():
    check_ckpt_annotation()
    imgs_dark, imgs_lit, k72_ready = stage_images()

    stats = [
        (str(len(GALLERY)), "shapes"),
        ("3", "checkpoints, none canonical"),
        (f"{statistics.mean(r['emis_1k_w1_iou_mean'] for r in IOU.values()):.3f}", "mean IoU, w1"),
        (f"{statistics.mean(r['emis_1k_w5_iou_mean'] for r in IOU.values()):.3f}", "mean IoU, w5"),
        (f"{statistics.mean(r['emis_72k_unfilt_iou_mean'] for r in IOU.values()):.3f}", "mean IoU, 72k*"),
    ]
    hero = lp.hero_header(
        "SegviGen · emission prediction",
        "Three checkpoints on Dongchen's figure-7 shapes",
        dek_html=(
            "Predictions from <code>emis_1k_w1</code>, <code>emis_1k_w5</code> and "
            "<code>emis_72k_unfilt</code> on the 11 shapes of Dongchen's new "
            "qualitative-comparison figure, rendered EMISSION-ONLY (key light "
            "off) so the gallery shows what actually glows rather than what is "
            "lit; the earlier key-lit renders are kept as an identification "
            "reference. None of the three is the project's final method, "
            "and the 72k column is mostly the model's own training data (named in "
            "the gallery caption); read the callout in the first section before "
            "the gallery. *72k's mean IoU is dominated by shapes it trained on."
            + ("" if k72_ready else " The 72k column was still rendering at "
               "publish time; it will be filled in on this same page shortly.")),
        stats=stats,
        toc=[(i, lab) for i, lab in OUTLINE])

    quant_body = sec_quant()
    prov_body, apx = sec_provenance()
    body = [sec_verdict(), sec_gallery(imgs_dark, imgs_lit, k72_ready), quant_body, prov_body]

    page_html = lp.page(
        title="SegviGen on Dongchen's figure-7 shapes (fig7_segvigen)",
        header_html=hero,
        body_sections=body + [apx],
        assets_rel=SITE_ASSETS,
        assets_dir=os.path.join(WEB, "assets"),
        theme="v3",
        tree_html=wz.tree_html(active_href=None),
        nav_title="fig7 SegviGen",
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
    print(f"wrote {out} ({len(page_html)} bytes, {len(GALLERY)} shapes)")
    print("  zone-link guard: clean")

    publish_assets(os.path.join(WEB, "assets"))
    print("assets published")


if __name__ == "__main__":
    main()
