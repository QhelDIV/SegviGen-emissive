#!/usr/bin/env python3
"""Fixed-bake galleries: epoch-8 checkpoint predictions for the fig7_11 (val,
held out) and showcase_12 (train, seen in training) shape sets, rendered
through the fixed bake (write and read UV agree by construction, no atlas to
diverge on; see the mask_debug page for the mechanism). GT and one picked
prediction per shape, plus the full sampling-density story: the original
10-draw pick table and, for the 10 shapes whose closest-to-GT draw was still
far off, the outcome of resampling those shapes with 16 fresh seeds, kept as a separate table.

Every image here is a real box render from the fixed pipeline. Two panels
(cyberpunk ATM, cartoonish green ghost) needed a fallback draw after
verification caught their closest-to-GT pick transferring to nothing
visible (a genuine near-miss against the write-side voxel tolerance, not a
rendering bug); both are labeled "(fallback)" and explained in their own
section below.

Run: /project/3dlg-hcvc/omages/omages_internal/.venv2/bin/python \
        web/_preview/fixedbake_galleries/build.py [--publish]
"""
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WEB)
sys.path.insert(0, os.path.join(REPO, "tools"))
import workspace_zone as wz  # noqa: E402

import xgpage as lp  # noqa: E402
from xgpage.publish import publish_assets  # noqa: E402

sys.path.insert(0, HERE)
from chart_and_strips import distribution_chart, draw_strip_html  # noqa: E402
from threshold_grid import threshold_grid_html  # noqa: E402

SITE_ROOT = "/projects/omages/yanxg/lightgen"
SITE_ASSETS = f"{SITE_ROOT}/assets"
FAVICON = f"{SITE_ROOT}/assets/images/favicon.png"

PAGE_SLUG = "fixedbake_galleries"
PAGE_HREF = f"{wz.WORKSPACE_URL}/{PAGE_SLUG}/index.html"
PUBLISH_DIR = os.path.join(str(wz.WORKSPACE_DIR), PAGE_SLUG)
PAGE_DATE = "2026-08-11"

IMG = os.path.join(HERE, "img")
DBG = "/project/3dlg-hcvc/omages/yanxg_scratch/mask_debug"

MANIFEST = json.load(open(f"{DBG}/gallery_final_manifest.json"))
RESCUE = json.load(open(f"{DBG}/rescue_repick.json"))
BY_SID = {r["sid"]: r for r in MANIFEST}

FIG7_SIDS = [r["sid"] for r in MANIFEST if r["set"] == "fig7_11"]
OURS_SIDS = [r["sid"] for r in MANIFEST if r["set"] == "ours12"]

DRAWS = json.load(open(f"{DBG}/draws_manifest.json")) if os.path.exists(f"{DBG}/draws_manifest.json") else {}

CARVED_LANTERN_FAMILY = {
    "064e4156b5c345c796cc00d3fa2e2243",  # jack-o'-lantern
    "48af42db48c44cd9bfab32bbb057a39c",  # pumpkin
    "75da9a74403946dda954f08a067e8ad5",  # outdoor wall lantern
}


def check_assets():
    problems = []
    for r in MANIFEST:
        for suffix in ("gt", "pred"):
            p = os.path.join(IMG, f"{r['sid']}_{suffix}.png")
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                problems.append(p)
    for sid in ("e1fbc7943362485489dba5a951ebc4b1", "b74fc2533d5345629f2c3ce2c8ab340a"):
        p = os.path.join(IMG, f"{sid}_pred_black_before.png")
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            problems.append(p)
    from threshold_grid import COL_LABELS
    for row in THRESHOLD_ROWS:
        for col_tag, _ in COL_LABELS:
            p = os.path.join(IMG, f'thrgrid_{row["tag"]}_{col_tag}.png')
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                problems.append(p)
    return problems


def src(name):
    h = hashlib.md5(open(os.path.join(IMG, name), "rb").read()).hexdigest()[:8]
    return f"img/{name}?v={h}"


def tree_html():
    entries = wz.tree_entries()
    for group in entries:
        for leaf in group.get("children", []):
            leaf["active"] = leaf["href"] == PAGE_HREF
    return lp.v3_tree(entries, title="Lightgen", subtitle="research workspace",
                      tree_src=wz.TREE_JSON_URL)


def short_badge(r):
    paper = " (fallback)" not in r["pick_label"] and "(paper pick)" in r["pick_label"]
    label = r["pick_label"].replace(" (fallback)", "").replace(" (paper pick)", "")
    label = (label.replace("rescue_raw seed", "resampled raw s")
                  .replace("rescue_ema seed", "resampled ema s")
                  .replace("draw", "d"))
    return label + " · paper" if paper else label


# ------------------------------------------------------------------ sections
def sec_stats():
    n_total = len(MANIFEST)
    n_orig_nondegenerate = sum(1 for r in MANIFEST if r["pick_frac"] > 0)
    diffs_before = [abs(BY_SID_ORIG_DIFF[r["sid"]]) for r in MANIFEST]
    n_rescued = len(RESCUE)
    n_improved = sum(1 for v in RESCUE.values() if v["improved"])
    body = lp.prose(
        "Each shape ran the full pipeline: 10 prediction draws (5 raw-weight, "
        "5 EMA-weight), the draw whose predicted-lit fraction lands closest "
        "to the shape's own ground truth picked for display, and both "
        "rendered through the fixed bake in the box setup. The numbers "
        "below say how often that worked, and the galleries show every "
        "result, ground truth beside prediction.")
    body += lp.prose(
        f"{n_total} shapes total: 11 from fig7_11 (val, held out from the "
        f"epoch-8 checkpoint's training) and 12 from showcase_12 (train, "
        f"seen during training, labeled seen-in-training throughout). Every "
        f"shape produced at least one non-degenerate draw (nonzero predicted "
        f"voxels) among its original 10 (5 raw-weight, 5 EMA-weight). "
        f"{n_rescued} shapes whose closest-to-GT draw was still far off "
        f"(|frac diff| &gt; 0.3) were sampled again with 16 fresh seeds; {n_improved} of "
        f"those {n_rescued} improved, 2 of them (street lamp, wall fixtures) "
        f"to an essentially exact match.")
    body += lp.prose(
        "Note: the three shapes shared with the mask_debug page (glowing "
        "warhammer, red glowing sword, robot with glowing eyes) display the "
        "paper's own picked draws here, not this gallery's own closest-to-GT "
        "pick, so the two pages show the same shape the same way. Those "
        "picks came from an earlier fresh-seed resampling and an eye check done "
        "for the paper figure directly; their full draw tables are on the "
        "mask_debug page, not repeated in the tables below.")
    body += lp.callout(
        "<b>The fullbright mode has a shape-family signature, not scattered "
        "bad luck.</b> The three shapes that never left the fullbright "
        "extreme, in the original 10 draws or the 16 fresh-seed draws "
        "(26 draws total each), are the jack-o'-lantern, the pumpkin, and "
        "the outdoor wall lantern: every one a carved or glass-shelled "
        "light-container shape. This corroborates the epoch-8 evaluation's "
        "own carved-lantern finding (blanket collapse surviving zero "
        "conditioning across two preprocessing paths), now reproduced on a "
        "fresh sample. Ordinary objects reach the middle mode reliably with "
        "enough sampling; this one semantic family does not. The wall "
        "lantern gets one honest caveat: its own ground truth is 56% lit, "
        "so fullbright is genuinely the closest mode there, not purely a "
        "collapse artifact.")
    return lp.section_v2("stats", None, "Sampling finds a usable draw for every shape except one family", body)


def sec_gallery(sids, title, split_note):
    body = lp.prose(
        f"{title}, epoch-8 checkpoint, fixed bake. Each shape: ground truth "
        f"and the displayed draw side by side, then every draw tried for "
        f"that shape, in order, badged with weight/seed and its own "
        f"frac@0.5. The displayed draw is outlined in the strip; it is the "
        f"draw closest to GT by the stated rule, presented inside the full "
        f"set of draws, not a claim that it is correct. What to look for in "
        f"each strip: two modes, near-black and near-fullbright, with few "
        f"thumbnails landing in between.")
    for sid in sids:
        r = BY_SID[sid]
        d = DRAWS.get(sid)
        pair = lp.fig_row(
            [("ground truth", src(f"{sid}_gt.png")),
             (f"displayed: {short_badge(r)}", src(f"{sid}_pred.png"))],
            native_px=384, content="photo", key=f"pair-{sid}")
        strip_html = ""
        if d:
            strip_html = draw_strip_html(d, src)
        else:
            strip_html = '<p style="opacity:0.6;font-size:13px">Draw strip pending render.</p>'
        body += (f'<div style="margin-top:22px;padding-top:18px;'
                f'border-top:1px solid var(--dv-grid,#d8d6d0)">'
                f'<h3 style="margin:0 0 4px 0">{r["caption"]}</h3>'
                f'<div style="font-size:13px;opacity:0.7;margin-bottom:8px">'
                f'GT frac {r["gt_frac"]:.4f}</div>'
                f'{pair}{strip_html}</div>')
    return lp.section_v2(title.lower().replace(" ", "-"), None,
                         f"{title} ({split_note})", body)


THRESHOLD_ROWS = [
    dict(tag="robot_seed4", caption="robot, EMA seed 4 (bimodal)",
         max=1.0547, mean=0.2883, frac05=0.2886),
    dict(tag="robot_seed5", caption="robot, EMA seed 5 (diffuse)",
         max=0.9644, mean=0.1019, frac05=0.0340),
    dict(tag="cottage_draw1", caption="fairy-tale cottage, raw draw 1",
         max=1.0684, mean=0.2257, frac05=0.2648),
    dict(tag="brazier_rs6", caption="medieval brazier, resampled raw seed 6",
         max=1.1250, mean=0.6323, frac05=0.6324),
    dict(tag="weapon_ema4", caption="sci-fi weapon, EMA draw 4",
         max=1.0928, mean=0.2939, frac05=0.2658),
]


def sec_threshold_grid():
    grid = threshold_grid_html(THRESHOLD_ROWS, src)
    body = lp.prose(
        "The pick rule throughout this page cuts a draw's confidence field "
        "at frac@0.5. Five draws, chosen to span the range from confident "
        "to diffuse, show what that single cut is standing in for: GT, then "
        "the same draw binarized at five thresholds (0.01 through 0.9), "
        "then a continuous rendering with no threshold at all, the raw "
        "confidence written straight into the emissive texture so brightness "
        "tracks certainty voxel by voxel.")
    body += f'<div style="margin-top:8px">{grid}</div>'
    body += lp.callout(
        "<b>Threshold-stability is a readout of confidence shape, not of "
        "correctness, and the stablest draw here is also the worst miss.</b> "
        "The robot's two EMA seeds are the clean contrast: seed 4 is "
        "bimodal, with values reaching past 1.0, and its lit fraction "
        "barely moves across the whole sweep (0.290 at threshold 0.01, "
        "still 0.287 at 0.9); seed 5 is diffuse, and its lit fraction "
        "collapses steadily as the cut rises (0.697 &rarr; 0.286 &rarr; "
        "0.129 &rarr; 0.034 at 0.01/0.1/0.3/0.5) and nearly vanishes at 0.9 "
        "(0.000013, a single surviving voxel visible as one lit pixel, not "
        "literally zero). Against this shape's own GT fraction (0.080), "
        "stable seed 4 over-predicts by 3.6&times; while unstable seed 5 "
        "lands closer to the truth, at 0.43&times; GT. The brazier pushes "
        "the same point further: it is the most threshold-stable draw in "
        "the set (0.665 down to only 0.617 across the whole sweep) and "
        "also the worst miss by far, 8.1&times; its own GT fraction (0.632 "
        "predicted against 0.078 true); the cottage and weapon rows "
        "over-predict by 22&times; and 5.3&times; respectively, cottage "
        "declining gradually from 0.01 to 0.5 (0.350 &rarr; 0.265) before a "
        "sharp cliff to 0.006 at 0.9, and weapon showing a thin skin of "
        "near-zero noise that the 0.01 cut alone removes (0.999 &rarr; "
        "0.269) before flattening out from 0.1 to 0.9. The continuous "
        "column makes the mechanism "
        "visible directly: seed 5's panel is a scatter of faint, spatially "
        "spread speckle, not a shape with a soft edge, which is why no "
        "single threshold recovers a clean silhouette from it. This "
        "connects to the calibration finding on the epoch-8 evaluation "
        "page: EMA weights read as confident there because their "
        "draw-to-draw variance is near zero, not because they are more "
        "accurate (mean absolute error 0.541 versus raw's 0.312 on the "
        "familiar-8 set). Threshold-stability is that same signature "
        "measured a different way: a draw that barely moves under the cut "
        "reads as trustworthy, but here the single most stable draw in the "
        "set (the brazier) is also its single largest miss among the "
        "stable rows, and the largest miss overall (the cottage, 22&times; "
        "GT) is only middlingly stable, not the flattest curve in the "
        "sweep. Stability and accuracy do not move together across these "
        "five. frac@0.5 alone cannot separate a confident correct draw "
        "from a confidently wrong one; stability has to be checked against "
        "ground truth, not read off the sweep by itself.")
    return lp.section_v2("thresholdgrid", None,
                         "A confident draw survives thresholding; a diffuse draw shatters into dots",
                         body)


def sec_distribution():
    order = FIG7_SIDS + OURS_SIDS
    chart = distribution_chart(DRAWS, order)
    n_draws_total = sum(len(d["draws"]) for d in DRAWS.values())
    body = lp.prose(
        f"Every draw run for every shape, {n_draws_total} points total: "
        f"frac@0.5 (share of voxels predicted emissive) on the x-axis, one "
        f"row per shape. The tick marks ground truth; the ringed point is "
        f"the displayed draw. Reading across a row is the whole finding: "
        f"most shapes cluster their draws at the two ends (empty, "
        f"fullbright) with few in between, and the three carved-lantern "
        f"shapes (top of the val block, mid showcase_12 block) never "
        f"produce a point away from the fullbright end at all.")
    body += f'<div style="margin-top:8px;overflow-x:auto">{chart}</div>'
    return lp.section_v2("distribution", None,
                         "Every draw, not just the pick", body)



def plain_label(label):
    return (str(label).replace("rescue_raw seed", "resampled raw seed")
                      .replace("rescue_ema seed", "resampled ema seed"))

def sec_pick_table():
    rows_html = ""
    for r in MANIFEST:
        diff = abs(r["pick_frac"] - r["gt_frac"])
        rows_html += (
            f'<tr><td style="text-align:left">{r["caption"]}</td>'
            f'<td>{r["set"]}</td><td>{r["split"]}</td>'
            f'<td>{r["gt_frac"]:.4f}</td>'
            f'<td style="text-align:left">{plain_label(r["pick_label"])}</td>'
            f'<td>{r["pick_frac"]:.4f}</td><td>{diff:.4f}</td></tr>')
    table = lp.results_table(
        ["shape", "set", "split", "GT frac", "picked draw", "picked frac", "|diff|"],
        rows_html)
    body = lp.prose(
        "Pick rule: for every shape, measure frac@0.5 (share of predicted-"
        "emissive voxels) on every one of 10 draws (5 raw-weight, 5 "
        "EMA-weight, epoch-8 checkpoint, real conditioning), then pick "
        "whichever nonzero draw lands closest to that shape's own "
        "ground-truth fraction. This is the FINAL pick shown in the "
        "galleries above (resampled draws where resampling improved on this "
        "one; see the resampling table below).")
    body += table
    return lp.section_v2("picktable", None, "The 10-draw pick table: reference", body)


def sec_rescue_table():
    rows_html = ""

    for sid, v in RESCUE.items():
        arrow = "improved" if v["improved"] else "no change"
        rows_html += (
            f'<tr><td style="text-align:left">{v["caption"]}</td>'
            f'<td>{v["split"]}</td>'
            f'<td>{plain_label(v["old_pick"])}</td><td>{v["old_diff"]:.4f}</td>'
            f'<td>{plain_label(v["new_pick"])}</td><td>{v["new_diff"]:.4f}</td>'
            f'<td style="text-align:left">{arrow}</td></tr>')
    table = lp.results_table(
        ["shape", "split", "old pick", "old |diff|", "resampling pick", "resampling |diff|", "outcome"],
        rows_html)
    body = lp.prose(
        "The 10 shapes whose original best pick was still far from GT "
        "(|diff| &gt; 0.3) got 16 fresh seeds (8 raw-weight, 8 EMA-weight, "
        "seed bases offset so none overlap the original 10 draws), then the "
        "same pick rule applied over all 26 draws together. 9 of 10 "
        "improved; the outdoor wall lantern did not, fullbright in every "
        "one of its 26 draws, see the statistics section above.")
    body += table
    return lp.section_v2("rescuetable", None, "Resampling outcome (16 fresh seeds): reference", body)


def sec_fallback_note():
    body = lp.fig_row(
        [("ATM: closest-to-GT pick, transferred to nothing", src("e1fbc7943362485489dba5a951ebc4b1_pred_black_before.png")),
         ("ATM: fallback pick (shown above)", src("e1fbc7943362485489dba5a951ebc4b1_pred.png")),
         ("ghost: closest-to-GT pick, transferred to nothing", src("b74fc2533d5345629f2c3ce2c8ab340a_pred_black_before.png")),
         ("ghost: fallback pick (shown above)", src("b74fc2533d5345629f2c3ce2c8ab340a_pred.png"))],
        caption_html=(
            "<b>Two panels needed a fallback draw, and the reason is a real "
            "finding, not a rendering defect.</b> The closest-to-GT pick for "
            "the cyberpunk ATM (432 lit voxels out of 933,817) and the "
            "cartoonish green ghost (2,129 lit voxels out of 252,245) were "
            "both so sparse that every one of their lit voxels fell outside "
            "the established nearest-lit-voxel tolerance of every mesh face "
            "(or landed on a face whose UV footprint rasterizes to zero "
            "texels): the transfer produced a genuinely empty panel from a "
            "genuinely nonzero prediction. Verification caught both before "
            "publishing; a zero-transfer warning is now wired into the "
            "rebake script so this cannot pass silently again. Both panels "
            "shown in the galleries above use the next-best nonzero draw "
            "instead, labeled (fallback); they read as less accurate "
            "against GT than the picked-but-invisible draw would have "
            "claimed to be, which is the honest tradeoff."),
        native_px=768, content="photo")
    return lp.section_v2("fallback", None, "Two fallback panels: what happened and why", body)


def sec_provenance():
    body = lp.prose(
        "Checkpoint: emis_72kv2_cond_pw1b epoch 8 (raw and EMA weights), "
        "real conditioning. Dumps: dump_pred_alldraws.py, one GPU job per "
        "set for the original 10 draws, one job for the 16-seed resampling. "
        "Mask bake: bpy_rebake.py, rasterizing against Blender's own "
        "imported UV (not the raw glTF UV pred_mask_to_asset.py used), so "
        "write and read agree by construction; see the mask_debug page for "
        "the full mechanism and the two node-transform bugs found and fixed "
        "along the way. Render: render_emissive_closest.py, box preset "
        "(Filmic, exposure 1.5, wall albedo 0.80, 1024 samples). GT and "
        "prediction share one camera per shape (auto-solved, azimuth 38, "
        "elevation 17, lens 52, margin 1.06). Full job ids and the "
        "differential-diagnosis discussion are in the jobs board, "
        "fixedbake_galleries and mask_debug entries, gallery-runner, "
        "2026-08-11.")
    return lp.section_v2("provenance", None, "Provenance", body)


# ----------------------------------------------------------------- the build
def build(publish=False):
    global BY_SID_ORIG_DIFF
    assets_dir = os.path.join(WEB, "assets")
    problems = check_assets()
    if problems:
        sys.exit("ASSET CHECK FAILED:\n  " + "\n  ".join(problems))

    # "before" diff for the statistics prose: the pick as originally chosen
    # from the 10-draw pass, before any resampling substitution
    fig7_orig = json.load(open(f"{DBG}/fig7_11_picks.json"))
    ours_orig = json.load(open(f"{DBG}/ours12_picks.json"))
    orig_all = {**fig7_orig, **ours_orig}
    BY_SID_ORIG_DIFF = {sid: v["abs_diff"] for sid, v in orig_all.items()}

    hero = lp.hero_header(
        f"lightgen &middot; fixed-bake galleries &middot; {PAGE_DATE}",
        "Fixed-Bake Galleries: fig7_11 and showcase_12",
        dek_html=(
            "The render path was fixed and validated on three hand-picked "
            "shapes (the mask-bake debug page). That left the real question "
            "open: across representative shape sets, how often does the "
            "epoch-8 checkpoint produce a usable emission prediction, and "
            "what does it take to find one? This page answers with all 23 "
            "shapes of both example sets: every ordinary shape reaches a "
            "plausible prediction with enough sampling, but usable draws "
            "are a minority, and one semantic family (carved light "
            "containers) never leaves the fullbright extreme."),
        toc=[("stats", "The answer in numbers"), ("distribution", "Every draw"),
             ("fig7-11-val", "fig7_11 (val)"),
             ("showcase-12-train", "showcase_12 (train)"),
             ("thresholdgrid", "Threshold sweep"),
             ("fallback", "Fallback panels"),
             ("picktable", "10-draw pick table"),
             ("rescuetable", "Resampling outcome (16 fresh seeds)"),
             ("provenance", "Provenance")],
    )

    page_html = lp.page(
        title="Fixed-Bake Galleries: fig7_11 and showcase_12",
        header_html=hero,
        body_sections=[sec_stats(), sec_distribution(),
                       sec_gallery(FIG7_SIDS, "fig7_11", "val, held out"),
                       sec_gallery(OURS_SIDS, "showcase_12", "train, seen-in-training"),
                       sec_threshold_grid(),
                       sec_fallback_note(),
                       sec_pick_table(), sec_rescue_table(), sec_provenance()],
        assets_rel=SITE_ASSETS,
        assets_dir=assets_dir,
        theme="v3",
        tree_html=tree_html(),
        nav_title="Fixed-bake galleries",
        version_slot=lp.v3_version_slot(date=PAGE_DATE),
        needs_katex=False,
        extra_head=f'<link rel="icon" href="{FAVICON}">',
        outline_entries=[
            {"id": "stats", "label": "The answer in numbers"},
            {"id": "distribution", "label": "Every draw"},
            {"id": "fig7-11-val", "label": "fig7_11 (val)"},
            {"id": "showcase-12-train", "label": "showcase_12 (train)"},
            {"id": "thresholdgrid", "label": "Threshold sweep"},
            {"id": "fallback", "label": "Fallback panels"},
            {"id": "picktable", "label": "10-draw pick table"},
            {"id": "rescuetable", "label": "Resampling outcome (16 fresh seeds)"},
            {"id": "provenance", "label": "Provenance"},
        ],
    )

    violations = wz.console_links_in(page_html)
    if violations:
        sys.exit(f"ZONE-LINK GUARD FAILED: page links to the console: {violations}")

    out_path = os.path.join(HERE, "index.html")
    with open(out_path, "w") as f:
        f.write(page_html)
    print(f"wrote {out_path}")
    print(f"  {len(MANIFEST)*2} images checked, all present")
    print("  zone-link guard: clean")

    publish_assets(assets_dir)
    print(f"assets published -> {assets_dir}")

    if publish:
        os.makedirs(PUBLISH_DIR, exist_ok=True)
        shutil.copytree(IMG, os.path.join(PUBLISH_DIR, "img"), dirs_exist_ok=True)
        shutil.copy2(out_path, os.path.join(PUBLISH_DIR, "index.html"))
        wz.write_tree_json()
        for p in [PUBLISH_DIR, *[os.path.join(dp, f)
                                 for dp, _dn, fn in os.walk(PUBLISH_DIR)
                                 for f in fn]]:
            try:
                os.chmod(p, os.stat(p).st_mode | (0o005 if os.path.isdir(p) else 0o004))
            except OSError:
                pass
        print(f"published -> {PUBLISH_DIR}")
        print(f"tree.json refreshed -> {wz.TREE_JSON}")


if __name__ == "__main__":
    build(publish="--publish" in sys.argv)
