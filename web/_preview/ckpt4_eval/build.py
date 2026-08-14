#!/usr/bin/env python3
"""Build the ckpt4_eval page: the first honest read on the 72k conditioned checkpoint
(outputs/emis_72kv2_cond_pw1b, epoch 4 of a still-running job), evaluated with the
K-draw averaged protocol (no single-draw numbers, per the project's own re-score-swing
finding: the same checkpoint re-scored with K=1 can swing 0.096->0.128).

Three populations, checked for what they actually are before being reported:
- val_72k (a 96-shape subset, true held-out for this checkpoint's train_72k split)
- the historical val_96 set (mostly THIS checkpoint's own training data: 105 of its
  111 shapes are inside the current train_72k split; kept only for continuity with
  the old oracle/honest-old reference numbers, labeled contaminated everywhere)
- the 8 familiar fbv1 shapes (also inside train_72k for this checkpoint; qualitative
  examples, not a generalization claim, labeled seen-in-training on every panel)

Every number is read at build time from the eval stage's own JSON files, never retyped.

Run: /project/3dlg-hcvc/omages/omages_internal/.venv2/bin/python \
        web/_preview/ckpt4_eval/build.py
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
PAGE_DATE = "2026-08-10"

WORK = "/project/3dlg-hcvc/omages/yanxg_scratch/ckpt4_eval"
IMG_DIR = os.path.join(HERE, "img")
os.makedirs(IMG_DIR, exist_ok=True)

# reference points, cited from where they were measured, not retyped from memory
ORACLE_ALL = 0.3953
ORACLE_NONZERO = 0.2195      # dataset/oracle_val96.json, n=111, old Path A pipeline
OLD_HONEST = 0.154           # first (best) of a declining sequence 0.154/0.130/0.113/0.102
VAE_CEILING = 0.96           # FACTSHEET_diagnostics.md round-trip ceiling
TEN_SHAPE_CEILING = 0.317    # this project's own ct10 pos_weight-1, 400-epoch best (context, not a target)

FBV1_SIDS = ["0414e54cda324108a7a51615f5cfd376", "10b7ad59f3bc4851a86d7f165ecd4c16",
             "a82965cbfbe3470eae134efdccf15011", "bbeccdb222e74d99812cd2bd892222a8",
             "d5fb4f19d4164612b165caac5471555c", "e9e31994a53d4fa68308f745c682a0b9",
             "f52e9b616c0a4075a70e5eb844f07bb3", "f65a020ba69c47e2a66f635ee0e6f8c2"]
SHORT = {s: s[:8] for s in FBV1_SIDS}

OUTLINE = [
    ("verdict", "What the first checkpoint reaches"),
    ("table", "The averaged numbers, all three populations"),
    ("gallery", "Ground truth vs prediction, the 8 familiar shapes"),
    ("provenance", "How this was produced"),
]


def load_eval(tag):
    return json.load(open(os.path.join(WORK, "eval", f"{tag}.json")))


def copy_img(src, dst_name):
    import shutil
    dst = os.path.join(IMG_DIR, dst_name)
    shutil.copy2(src, dst)
    return dst


def rel_img(abspath):
    return "img/" + os.path.relpath(abspath, IMG_DIR)


def img_ref(abspath):
    import hashlib
    h = hashlib.md5(open(abspath, "rb").read()).hexdigest()[:8]
    return f"{rel_img(abspath)}?v={h}"


# --------------------------------------------------------------------- sections
def sec_verdict(evals):
    # the best true-held-out number, across raw/ema x real/zero
    held_out = {k: v for k, v in evals.items() if k.startswith("val72kok_")}
    best_key = max(held_out, key=lambda k: held_out[k]["nonzero"]["iou_at_5"])
    best = held_out[best_key]["nonzero"]["iou_at_5"]
    best_std = held_out[best_key].get("draw_std", 0.0) or 0.0
    label = best_key.replace("val72kok_", "").replace("_", " cond ")

    fbv1_best_key = max((k for k in evals if k.startswith("fbv18_")),
                         key=lambda k: evals[k]["nonzero"]["iou_at_5"])
    fbv1_best = evals[fbv1_best_key]["nonzero"]["iou_at_5"]

    body = lp.verdict_box(
        f"<p><b>This is epoch 4 of a still-training run, not a finished model.</b> "
        f"On the true held-out set (96 shapes, checked to have zero overlap with this "
        f"checkpoint's own training split), the best of the four weight/conditioning "
        f"combinations ({label}) reaches {best:.3f} mean IoU (std {best_std:.3f} across "
        f"5 draws), threshold 0.5, gated to shapes with real ground-truth coverage. That "
        f"already passes the old honest number of {OLD_HONEST:.3f} from the previous "
        f"pipeline, but sits well below the {ORACLE_NONZERO:.3f} zero-shot upper bound "
        f"and far below the {VAE_CEILING:.2f} VAE round-trip ceiling that would mean the "
        "representation itself is not the limit. On the 8 familiar shapes (all inside "
        f"this checkpoint's own training split), the best draw reaches {fbv1_best:.3f} "
        "&mdash; higher, consistent with early memorization signal on shapes the model "
        "has actually seen, though at epoch 4 that signal is still modest.</p>")
    body += lp.prose(
        f"<b>Reference points, and where each one came from:</b> the zero-shot oracle "
        f"({ORACLE_NONZERO:.3f} nonzero, {ORACLE_ALL:.3f} over all shapes) is the best "
        "any per-part-color assignment could score with no learning at all, measured on "
        "the old val_96 set (<code>dataset/oracle_val96.json</code>, n=111). The old "
        f"honest number ({OLD_HONEST:.3f}) is the first, best point of a run that then "
        "declined over its own training. Both are from the earlier Path A pipeline, not "
        "this one, and are cited for scale, not as a matched comparison. "
        f"The {TEN_SHAPE_CEILING:.3f} figure is this project's own ten-shape overfit "
        "ceiling (pos_weight 1, 400 epochs) from a separate diagnostic: even a model "
        "trying to memorize just ten shapes on today's code tops out well below the "
        f"{VAE_CEILING:.2f} representational ceiling, so a first-epoch generalization "
        "number in the same rough band is not itself surprising.")
    return lp.section_v2("verdict", 1, "An early checkpoint, already past the old honest number, still far from the ceiling", body)


def sec_table(evals):
    def row(tag, pop_label, contamination):
        d = evals[tag]
        nz = d["nonzero"]
        std = d.get("draw_std", 0.0) or 0.0
        return (f'<tr><td style="text-align:left">{pop_label}</td>'
                f'<td>{d["n"]}</td><td>{d["n_nonzero"]}</td>'
                f'<td>{nz["iou_at_5"]:.3f} &plusmn; {std:.3f}</td>'
                f'<td>{nz["best_iou"]:.3f}</td><td>{contamination}</td></tr>')

    rows_html = ""
    pops = [
        ("val72kok_raw_real", "held-out, raw weights, real cond", "clean"),
        ("val72kok_raw_zero", "held-out, raw weights, zero cond", "clean"),
        ("val72kok_ema_real", "held-out, EMA weights, real cond", "clean"),
        ("val72kok_ema_zero", "held-out, EMA weights, zero cond", "clean"),
        ("val96legacy_raw_real", "historical val_96, raw weights, real cond", "mostly seen in training"),
        ("val96legacy_raw_zero", "historical val_96, raw weights, zero cond", "mostly seen in training"),
        ("val96legacy_ema_real", "historical val_96, EMA weights, real cond", "mostly seen in training"),
        ("val96legacy_ema_zero", "historical val_96, EMA weights, zero cond", "mostly seen in training"),
        ("fbv18_raw_real", "8 familiar shapes, raw weights, real cond", "seen in training"),
        ("fbv18_raw_zero", "8 familiar shapes, raw weights, zero cond", "seen in training"),
        ("fbv18_ema_real", "8 familiar shapes, EMA weights, real cond", "seen in training"),
        ("fbv18_ema_zero", "8 familiar shapes, EMA weights, zero cond", "seen in training"),
    ]
    for tag, label, contam in pops:
        rows_html += row(tag, label, contam)
    table = lp.results_table(
        ["configuration", "n shapes", "n with GT coverage", "mean IoU@0.5 (nonzero) &plusmn; draw std",
         "best-threshold IoU (nonzero)", "training-set overlap"],
        rows_html)
    body = lp.prose(
        "5 draws per shape, mean and standard deviation across draws (the project's own "
        "K-draw protocol; a single draw of the same checkpoint has been seen to swing by "
        "0.03 or more on its own). IoU is gated to shapes with real ground-truth coverage "
        "(nonzero), the timidity-proof aggregate a checkpoint cannot win by predicting "
        "all-black on empty-glow shapes. The held-out set is 96 shapes checked to have no "
        "overlap with this checkpoint's train_72k split. The historical val_96 set and the "
        "8 familiar shapes are both mostly or entirely inside that training split for THIS "
        "checkpoint; kept for continuity with earlier reference numbers and for "
        "identifiable qualitative examples, never as a held-out reading.")
    body += table
    return lp.section_v2("table", 2, "The averaged numbers, all three populations, both weight sets, both conditioning modes", body)


def sec_gallery():
    rows = []
    for sid in FBV1_SIDS:
        lit = os.path.join(WORK, "out", "lit", f"{sid}_lit.png")
        gt = os.path.join(WORK, "out", "gt_box", f"{sid}_box.png")
        pred = os.path.join(WORK, "out", "ema_real_box", f"{sid}_box.png")
        for p in (lit, gt, pred):
            if not os.path.exists(p):
                raise RuntimeError(f"missing render output: {p}")
        row = {
            "lit": img_ref(copy_img(lit, f"{sid}_lit.png")),
            "gt": img_ref(copy_img(gt, f"{sid}_gt_box.png")),
            "pred": img_ref(copy_img(pred, f"{sid}_pred_box.png")),
        }
        rows.append((SHORT[sid], [{"img": row["lit"]}, {"img": row["gt"]}, {"img": row["pred"]}]))

    grid = lp.grid_figure(
        row_labels=[r[0] for r in rows],
        col_labels=["IDENTIFICATION", "GROUND TRUTH", "PREDICTION"],
        cells=[r[1] for r in rows],
        id="ckpt4-gallery",
        caption=(
            "<b>All 8 familiar shapes are inside this checkpoint's own training split "
            "&mdash; every row here is a seen-in-training example, not a generalization "
            "claim.</b> Identification panel is a bright studio render for telling shapes "
            "apart; ground truth and prediction are both the project's ratified box "
            "render (dark room, no external light, emission only), EMA weights, real "
            "conditioning &mdash; the same treatment for both so only the content "
            "differs. One representative draw per shape; the averaged numbers this "
            "checkpoint actually reaches are in the table above, not this single draw."))
    return lp.section_v2("gallery", 3, "8 familiar shapes, all seen in training: ground truth against prediction", grid)


def sec_provenance():
    param_rows = "".join(
        f'<tr><td style="text-align:left">{html.escape(k)}</td><td>{html.escape(v)}</td></tr>'
        for k, v in [
            ("checkpoint", "outputs/emis_72kv2_cond_pw1b/epoch_0004.ckpt (+ _ema), job 242125 still training"),
            ("dataset", "dataset_direct (v2 split)"),
            ("held-out subset", "96 shapes from val_72k, checked to have no overlap with train_72k"),
            ("historical val_96 subset", "111-shape legacy list, resolved into dataset_direct wherever each "
                                          "shape currently lives (105 in train_72k, 2 in test_72k, 4 not found)"),
            ("familiar shapes", "the same 8 sids used in the earlier fbv1 report page"),
            ("draws", "5 per shape, mean and std reported"),
            ("sampling steps", "12"),
            ("threshold", "0.5, plus a best-of-sweep column"),
            ("render, ground truth and prediction", "box render (RENDERING.md setup 1): mode box, 768px, "
                                                      "1024 samples, Filmic, exposure 1.5, bloom size 7/thr 1.0/"
                                                      "mix -0.45, emission strength 4.0"),
            ("render, identification panel", "key-lit render (RENDERING.md setup 2): key 8, AgX, exposure 0, "
                                              "bloom size 9/thr 1.0/mix -0.15, samples 256"),
        ])
    param_table = lp.results_table(["parameter", "value"], param_rows)

    job_rows = "".join(
        f'<tr><td style="text-align:left">{html.escape(stage)}</td><td>{jid}</td><td>{res}</td></tr>'
        for stage, jid, res in [
            ("averaged eval, 12 configurations", "242421-242499 (several retried, see below)",
             "GPU a40x1 each, 8 cpus, debug partition"),
            ("prediction dump for the render figure (EMA, real cond, 3 draws)", "242433",
             "GPU a40x1, 8 cpus"),
            ("mask transfer, sharded array", "242434 (30/32 tasks; 2 retried as 242455)",
             "CPU, 8 cpus/task, 16-task array"),
            ("render (box + key-lit)", "242466", "CPU, 64 cpus, long partition, no dependency chain"),
        ])
    job_table = lp.results_table(["stage", "solar job id", "resources"], job_rows)

    body = lp.prose(
        "The held-out eval crashed twice, always on real-conditioning configurations, "
        "always with the same file reported missing, even though the same file loads "
        "cleanly with a normal-shaped tensor from the login node and from a fresh probe "
        "job right after each crash. Checked directly rather than assumed: a scan of the "
        "first 150 shapes in the split, loading every conditioning file rather than just "
        "checking it exists, found nothing wrong with any of them. Zero-conditioning runs "
        "never touch that file and never failed, on any node. The failures kept landing on "
        "whichever node the job happened to be scheduled to (cs-venus-13 twice, then "
        "cs-venus-07 once, each time on a different shape), which points at an "
        "intermittent per-node access problem rather than bad data. Practical fix: "
        "excluded each node as it showed the problem and reran; the fourth attempt, "
        "excluding cs-venus-05/07/09/13/15/19, completed cleanly.")
    body += param_table
    body += lp.prose("Job ids and resources:")
    body += job_table
    body += lp.prose(
        "No git commit, no workstation compute; all GPU/CPU work ran on solar (account "
        "3dlg-hcvc-lab). The four standard exclusions "
        "(cs-venus-05/09/15/19) plus cs-venus-07 and cs-venus-13, the two nodes that "
        "showed the conditioning-file access problem during this run.")
    return lp.section_v2("provenance", 4, "Commands, parameters, and where the raw outputs live", body)


def main():
    tags = ["val72kok_raw_real", "val72kok_raw_zero", "val72kok_ema_real", "val72kok_ema_zero",
            "val96legacy_raw_real", "val96legacy_raw_zero", "val96legacy_ema_real", "val96legacy_ema_zero",
            "fbv18_raw_real", "fbv18_raw_zero", "fbv18_ema_real", "fbv18_ema_zero"]
    evals = {t: load_eval(t) for t in tags}

    held_out = {k: v for k, v in evals.items() if k.startswith("val72kok_")}
    best_key = max(held_out, key=lambda k: held_out[k]["nonzero"]["iou_at_5"])
    best = held_out[best_key]["nonzero"]["iou_at_5"]

    stats = [
        ("epoch 4", "checkpoint (still training)"),
        (f"{best:.3f}", "best held-out IoU (nonzero)"),
        (f"{OLD_HONEST:.3f}", "old honest number"),
        (f"{ORACLE_NONZERO:.3f}", "zero-shot oracle"),
        (f"{VAE_CEILING:.2f}", "VAE round-trip ceiling"),
    ]
    hero = lp.hero_header(
        "SegviGen · first 72k conditioned checkpoint",
        "The honest first read: past the old number, nowhere near the ceiling",
        dek_html=(
            "The first checkpoint from the full 72k conditioned training run "
            "(<code>emis_72kv2_cond_pw1b</code>, epoch 4 of a run still in progress) "
            "evaluated with the project's averaged, 5-draw protocol on three "
            "populations: a 96-shape subset of val_72k checked to be genuinely held "
            "out for this checkpoint, the historical val_96 set (mostly this "
            "checkpoint's own training data, kept only for continuity with the old "
            "reference numbers), and 8 familiar shapes (also inside training, kept "
            "for identifiable qualitative examples). Both raw and EMA weights, both "
            "real and zero conditioning."),
        stats=stats,
        toc=[(i, lab) for i, lab in OUTLINE])

    body = [
        sec_verdict(evals),
        sec_table(evals),
        sec_gallery(),
        sec_provenance(),
    ]

    page_html = lp.page(
        title="First 72k conditioned checkpoint: averaged eval (ckpt4_eval)",
        header_html=hero,
        body_sections=body,
        assets_rel=SITE_ASSETS,
        assets_dir=os.path.join(WEB, "assets"),
        theme="v3",
        tree_html=wz.tree_html(active_href=None),
        nav_title="ckpt4 eval",
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
