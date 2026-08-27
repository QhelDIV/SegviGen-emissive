#!/usr/bin/env python3
"""Build the live_eval REDESIGN MOCKUP: same real data as the live page, a
different layout. This is review material for the owner, not a live page.

Board: live_eval_redesign. Does not touch web/_preview/live_eval/ or its
build.py; reads the same evaluation store on falas but writes to its own
directory so the every-ten-minutes loop over the real page cannot collide
with it.

Two changes from the live page, both owner asks (2026-08-26): (1) concise —
the IoU curve and the visual wall are the whole page, everything else
demoted; (2) an epoch slider over the wall, so the same ten shapes' draws
can be scrubbed across checkpoints instead of stacked one page-section per
checkpoint.

Run:
  /cs/3dlg-falas/project/omages/lightgen/segvigen_emissive/xgconsole/.venv_console/bin/python build_mock.py
"""
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WEB)
sys.path.insert(0, os.path.join(REPO, "tools"))

import xgpage as lp                        # noqa: E402
import workspace_zone as wz                # noqa: E402
from xgpage.publish import publish_assets  # noqa: E402

SITE_ROOT = "/projects/omages/yanxg/lightgen"
FAVICON = f"{SITE_ROOT}/assets/images/favicon.png"

# A FROZEN snapshot, not the live store. Found live while building this
# mockup (2026-08-26 20:5x): liveeval-builder is mid-migration on
# outputs/live_eval/ (records_legacy_quick/, bridged/, shards/ all appeared
# within minutes; records/ transiently held one file, then began rewriting
# older checkpoints to "status": "failed" stubs pending re-sharded re-eval).
# None of that is a bug and none of it is this page's data to referee, but a
# design mockup has no business racing an in-flight schema migration, so this
# reads a one-time copy taken from records_legacy_quick/ + img/ at 20:54 into
# _snapshot/ (see INCORPORATION.md: the real build.py should keep reading the
# live store once it settles).
STORE = os.path.join(HERE, "_snapshot")
IMG_LOCAL = os.path.join(HERE, "img")
N_WALLS = 5              # how many checkpoints the slider carries (mockup: all we have)
RUN_NAME = "segvigen_256_bw_6gpu"
STEPS_PER_EPOCH = 755
FULL_EVERY = 5
VAL_SPLIT_N = 387
DRAWS = 5

OUTLINE = [
    ("curve", "IoU across training"),
    ("wall", "Scrub the same ten shapes"),
    ("detail", "Numbers, method, provenance"),
]

INK = "var(--ink, #333)"
MUT = "var(--ink-3, #777)"
ACC = "var(--accent-ink, #b4552d)"
BLUE = "var(--blue, #4E7FD0)"
GREEN = "var(--green, #4a8f5c)"
LINE = "var(--line, #ccc)"


# ------------------------------------------------------------------ data (read-only)
def load_records():
    """Reads the frozen snapshot's records/ (see the STORE comment above)."""
    recs = []
    for p in sorted(glob.glob(os.path.join(STORE, "records", "step*.json"))):
        try:
            with open(p) as f:
                r = json.load(f)
        except (ValueError, OSError):
            continue
        r.setdefault("tier", "quick")
        recs.append(r)
    recs.sort(key=lambda r: (r["step"], r["tier"]))
    return recs


def is_ok(r):
    return r.get("status") == "ok" and r.get("mean_iou") is not None


def epoch_of(step):
    return step / STEPS_PER_EPOCH


def epoch_label(step):
    e = epoch_of(step)
    return f"{e:.0f}" if step % STEPS_PER_EPOCH == 0 else f"{e:.1f}"


def split_by_gt(rec):
    ok = [s for s in rec["per_shape"] if s["status"] == "ok"]
    return ([s for s in ok if s.get("gt_frac", 0) > 0],
            [s for s in ok if s.get("gt_frac", 0) == 0])


def emissive_mean(rec):
    emissive, empty = split_by_gt(rec)
    if not empty:
        return rec["mean_iou"], len(emissive), 0
    if not emissive:
        return None, 0, len(empty)
    return (sum(s["iou_mean"] for s in emissive) / len(emissive),
            len(emissive), len(empty))


# ------------------------------------------------------------------ assets
def copy_images(quick_shown, sids):
    os.makedirs(IMG_LOCAL, exist_ok=True)
    copied = 0
    for sid in sids:
        for name in ("geom", "gt"):
            src = os.path.join(STORE, "img", f"{sid}_{name}.png")
            dst = os.path.join(IMG_LOCAL, f"{sid}_{name}.png")
            if os.path.exists(src) and (not os.path.exists(dst)
                                        or os.path.getmtime(src) > os.path.getmtime(dst)):
                shutil.copy2(src, dst)
                copied += 1
    for r in quick_shown:
        sub = f"step{r['step']:07d}"
        src_dir = os.path.join(STORE, "img", sub)
        dst_dir = os.path.join(IMG_LOCAL, sub)
        if not os.path.isdir(src_dir):
            continue
        os.makedirs(dst_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            if not f.endswith(".png"):
                continue
            src, dst = os.path.join(src_dir, f), os.path.join(dst_dir, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1
    return copied


# ------------------------------------------------------------------ curve (unchanged
# visual logic from live_eval/build.py's curve_figure, trimmed caption only)
def curve_figure(quick, full):
    W, H = 900, 380
    L, R, T, B = 56, 30, 40, 44
    all_steps = [r["step"] for r in quick] + [r["step"] for r in full]
    x_lo, x_hi = min(all_steps), max(all_steps)
    if x_hi == x_lo:
        x_lo, x_hi = x_lo - STEPS_PER_EPOCH, x_hi + STEPS_PER_EPOCH

    per_shape = {}
    for r in quick:
        for sh in r["per_shape"]:
            if sh["status"] == "ok":
                per_shape.setdefault(sh["sid"], []).append((r["step"], sh["iou_mean"]))
    y_vals = ([v for pts in per_shape.values() for _, v in pts]
              + [r["mean_iou"] for r in quick]
              + [emissive_mean(r)[0] for r in full if emissive_mean(r)[0] is not None])
    y_hi = max(0.05, max(y_vals) * 1.15) if y_vals else 0.05

    def px(x):
        return L + (x - x_lo) / (x_hi - x_lo) * (W - L - R)

    def py(y):
        return T + (y_hi - y) / y_hi * (H - T - B)

    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Emissive-voxel IoU against '
         'optimizer step">']
    for i in range(5):
        gy = y_hi * i / 4
        s.append(f'<line x1="{L}" y1="{py(gy):.1f}" x2="{W-R}" y2="{py(gy):.1f}" '
                 f'stroke="{LINE}" stroke-width="1"/>')
        s.append(f'<text x="{L-8}" y="{py(gy)+4:.1f}" text-anchor="end" fill="{MUT}" '
                 f'font-size="11">{gy:.3f}</text>')

    for r in quick:
        s.append(f'<text x="{px(r["step"]):.1f}" y="{H-B+16}" text-anchor="middle" '
                 f'fill="{MUT}" font-size="10">{r["step"]}</text>')
        s.append(f'<text x="{px(r["step"]):.1f}" y="{H-B+29}" text-anchor="middle" '
                 f'fill="{MUT}" font-size="9">ep {epoch_label(r["step"])}</text>')

    for sid, pts in per_shape.items():
        pts.sort()
        if len(pts) == 1:
            s.append(f'<circle cx="{px(pts[0][0]):.1f}" cy="{py(pts[0][1]):.1f}" r="2.5" '
                     f'fill="{MUT}" opacity="0.55"/>')
            continue
        d = " ".join(("M" if i == 0 else "L") + f"{px(a):.1f},{py(b):.1f}"
                     for i, (a, b) in enumerate(pts))
        s.append(f'<path d="{d}" fill="none" stroke="{MUT}" stroke-width="1" opacity="0.45"/>')

    def series(recs, colour, width, radius, square=False):
        pts = [(r["step"], r["mean_iou"]) for r in recs]
        if len(pts) > 1:
            d = " ".join(("M" if i == 0 else "L") + f"{px(a):.1f},{py(b):.1f}"
                         for i, (a, b) in enumerate(pts))
            s.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}"/>')
        for a, b in pts:
            if square:
                s.append(f'<rect x="{px(a)-radius:.1f}" y="{py(b)-radius:.1f}" '
                         f'width="{radius*2}" height="{radius*2}" fill="{colour}"/>')
            else:
                s.append(f'<circle cx="{px(a):.1f}" cy="{py(b):.1f}" r="{radius}" fill="{colour}"/>')

    series(quick, ACC, 2.4, 3)
    series([dict(r, mean_iou=emissive_mean(r)[0]) for r in full if emissive_mean(r)[0] is not None],
           BLUE, 2.6, 5, square=True)

    legend = [(ACC, "ten-shape screen, every epoch", "line")]
    if full:
        legend.append((BLUE, f"full validation, {emissive_mean(full[-1])[1]} emitting shapes", "square"))
    lx, row = L, 0
    for colour, label, kind in legend:
        w = 30 + len(label) * 5.7
        if lx + w > W - R and row == 0:
            row, lx = 1, L
        yy = 14 + row * 17
        if kind == "square":
            s.append(f'<rect x="{lx:.0f}" y="{yy-4}" width="8" height="8" fill="{colour}"/>')
        else:
            s.append(f'<line x1="{lx:.0f}" y1="{yy}" x2="{lx+18:.0f}" y2="{yy}" '
                     f'stroke="{colour}" stroke-width="2.4"/>')
        s.append(f'<text x="{lx+24:.0f}" y="{yy+4}" font-size="11" fill="{MUT}">{label}</text>')
        lx += w
    s.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-4}" text-anchor="middle" fill="{MUT}" '
             'font-size="11">optimizer step</text>')
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------------ the slider wall
def build_wall_and_index(quick):
    """Fixed 10-shape x (geom, GT, 5 draws) grid, plus the JSON index the JS
    scrubber reads. Row order is fixed by IoU at the LATEST checkpoint (never
    reorders while scrubbing — that would defeat the point of watching one
    shape move). Static <img> defaults are the LATEST checkpoint's images, so
    a reader with no JS sees a complete, correct wall."""
    latest = quick[-1]
    order = sorted([s["sid"] for s in latest["per_shape"] if s["status"] == "ok"],
                   key=lambda sid: -next(s["iou_mean"] for s in latest["per_shape"] if s["sid"] == sid))

    shapes_idx = []
    for sid in order:
        s = next(s for s in latest["per_shape"] if s["sid"] == sid)
        shapes_idx.append({"sid": sid, "geom": f"img/{sid}_geom.png", "gt": f"img/{sid}_gt.png",
                           "gt_frac": round(s["gt_frac"], 4)})

    ckpts_idx = []
    for r in quick:
        per_shape = {}
        for s in r["per_shape"]:
            if s["status"] == "ok":
                per_shape[s["sid"]] = {"iou_mean": round(s["iou_mean"], 4),
                                       "iou_per_draw": [round(v, 4) for v in s["iou_per_draw"]]}
        ckpts_idx.append({"step": r["step"], "epoch": epoch_label(r["step"]),
                          "iou": round(r["mean_iou"], 4), "n_scored": r["n_scored"],
                          "per_shape": per_shape})

    index = {"run": RUN_NAME, "draws": DRAWS, "img_dir_tmpl": "img/step%07d",
             "shapes": shapes_idx, "checkpoints": ckpts_idx, "default_idx": len(ckpts_idx) - 1}

    latest_dir = f"img/step{latest['step']:07d}"
    header = ('<div class="gf-cell gf-corner"></div>'
              '<div class="gf-cell gf-colhead">shape</div>'
              '<div class="gf-cell gf-colhead">emissive ground truth</div>'
              + "".join(f'<div class="gf-cell gf-colhead">draw {k}</div>' for k in range(DRAWS)))
    rows = []
    for sid in order:
        s = next(s for s in latest["per_shape"] if s["sid"] == sid)
        row = [f'<div class="gf-cell gf-rowhead" data-sw-rowlabel="{sid}">{sid[:8]}  '
               f'IoU {s["iou_mean"]:.3f}</div>',
               f'<div class="gf-cell gf-imgcell"><img loading="lazy" src="img/{sid}_geom.png" '
               f'alt="{sid} geometry"></div>',
               f'<div class="gf-cell gf-imgcell"><img loading="lazy" src="img/{sid}_gt.png" '
               f'alt="{sid} ground-truth emission"><div class="gf-cap">'
               f'{s["gt_frac"] * 100:.1f}% of voxels</div></div>']
        for k in range(DRAWS):
            iou_k = s["iou_per_draw"][k]
            row.append(
                f'<div class="gf-cell gf-imgcell" data-sw-cell data-sw-sid="{sid}" data-sw-draw="{k}">'
                f'<img loading="lazy" src="{latest_dir}/{sid}_d{k}.png" alt="{sid} draw {k}">'
                f'<div class="gf-cap" data-sw-cap>IoU {iou_k:.3f}</div></div>')
        rows.append("".join(row))
    grid = (f'<div class="grid-figure-scroll"><div class="grid-figure sw-grid" '
            f'style="grid-template-columns: auto repeat({2 + DRAWS}, 1fr);" '
            f'data-sw-grid>{header}{"".join(rows)}</div></div>')

    latest_label = f"step {latest['step']} &middot; epoch {epoch_label(latest['step'])} &middot; screen IoU {latest['mean_iou']:.3f}"
    ticks = "".join(f'<span>ep {epoch_label(r["step"])}</span>' for r in quick)
    controls = f'''<div class="sw-controls" data-sw-root data-src="ckpts.json">
      <button class="sw-btn" type="button" data-sw-prev aria-label="Previous checkpoint">&larr; prev</button>
      <input type="range" class="sw-range" data-sw-range min="0" max="{len(quick) - 1}"
             step="1" value="{len(quick) - 1}" aria-label="Checkpoint, step over training">
      <button class="sw-btn" type="button" data-sw-next aria-label="Next checkpoint">next &rarr;</button>
      <span class="sw-label" data-sw-label>{latest_label}</span>
    </div>
    <div class="sw-ticks" aria-hidden="true">{ticks}</div>
    <noscript><p class="sw-noscript">Shown: the newest evaluated checkpoint. Enable JavaScript to
      scrub across earlier ones.</p></noscript>'''

    return controls + grid, index


# ------------------------------------------------------------------ sections
def sec_curve(quick, full):
    latest = quick[-1]
    cap = (f'<b>Screen IoU is {latest["mean_iou"]:.3f} at epoch {epoch_label(latest["step"])} '
           f'({len(quick)} epochs screened so far).</b> Faint lines: the ten screening shapes. '
           'Accent line: their mean, every epoch. Blue squares: full-validation runs, every '
           f'{FULL_EVERY}th epoch, scored only on shapes whose ground truth actually emits.')
    body = lp.fig_html(curve_figure(quick, full), cap, key="curve", max_px=900)
    return lp.section_v2("curve", 1, "IoU across training", body)


def sec_wall(quick):
    inner, index = build_wall_and_index(quick)
    cap = ('<b>Same ten shapes, same seeds, every checkpoint.</b> Scrub the slider to watch '
           'one checkpoint’s draws become the next’s. Rows are ordered by IoU at the '
           'newest checkpoint and stay in that order while you scrub.')
    body = f'<figure class="sw-figure">{inner}<figcaption>{lp.kicker("")}{cap}</figcaption></figure>'
    return lp.section_v2("wall", 2, "Scrub the same ten shapes across training", body), index


def sec_detail(recs, quick, full):
    """Everything the live page currently spends most of its length on, folded
    into one collapsed zone: still complete, no longer competing with the two
    things that matter."""
    stages = (lp.flow_stage(1, "training run", "finishes an epoch")
              + lp.flow_arrow()
              + lp.flow_stage(2, "watcher", "reads the step, copies it out")
              + lp.flow_arrow()
              + lp.flow_stage(3, "one GPU job", "10 shapes, or all 387", highlight=True)
              + lp.flow_arrow()
              + lp.flow_stage(4, "this page", "rebuilt from every record"))
    method = lp.prose(
        "Ten held-out shapes, five fixed-seed draws, screened every epoch; the whole "
        f"validation split ({VAL_SPLIT_N} shapes) every {FULL_EVERY}th epoch. Full "
        "validation IoU, not the screen and not validation loss, is what a checkpoint is "
        "chosen by. Panels are painted from decoded voxels, not rendered, so the loop can "
        "keep this cadence.") + lp.flow_wrap(stages)

    rows = []
    for r in sorted(recs, key=lambda r: -r["step"]):
        tier = "screen" if r["tier"] == "quick" else "full validation"
        if is_ok(r):
            rows.append(f"<tr><td>{r['step']}</td><td>{epoch_label(r['step'])}</td>"
                        f"<td>{tier}</td><td>{r['mean_iou']:.3f}</td><td>{r['n_scored']}</td>"
                        f"<td>{r['evaluated_at']}</td></tr>")
        else:
            rows.append(f"<tr><td>{r['step']}</td><td>{epoch_label(r['step'])}</td>"
                        f"<td>{tier}</td><td colspan='3'>not evaluated: "
                        f"{r.get('reason', 'unknown')}</td></tr>")
    table = lp.results_table(["step", "epoch", "tier", "mean IoU", "shapes", "evaluated"],
                             "".join(rows))

    caveats = lp.prose(
        "Ten shapes cannot separate neighbouring epochs; a move of a few thousandths between "
        "them is not progress. Some validation shapes have no emissive ground truth at all and "
        "score a perfect 1.0 when the model paints nothing, so they are excluded from the "
        "selection number. IoU here is voxel-space against this project's own decoder and is "
        "not comparable with the team's shared evaluator.")

    prov = lp.appendix("Provenance", [
        f"<b>Training run:</b> <code>{RUN_NAME}</code>, {STEPS_PER_EPOCH} steps per epoch.",
        "<b>Screening shapes:</b> the first ten of the owner's hand-picked 56, held-out split.",
        f"<b>Full tier:</b> all of <code>dataset_direct/val_72k</code> ({VAL_SPLIT_N} shapes).",
        f"<b>This build:</b> {len(recs)} record(s), {len(quick)} screens, {len(full)} full runs.",
    ])

    body = (lp.expandable("How an epoch becomes a row here", method)
           + lp.expandable("Every checkpoint's numbers", table)
           + lp.expandable("What this does and does not say", caveats)
           + lp.expandable("Provenance", prov))
    return lp.section_v2("detail", 3, "Numbers, method, provenance", body)


DESIGN_NOTES = lp.callout(
    "<p class='t'>Design notes for review (remove before this ships)</p>"
    "<p>Rebuilt for the owner's ask: nobody has time to read the full text, so the curve and "
    "the visual wall are now the whole page. Method, per-checkpoint numbers, caveats, and "
    "provenance still exist, complete, folded into one collapsed zone at the bottom instead of "
    "four full sections between the two things that matter.</p>"
    "<p>The wall is now one fixed grid with a checkpoint slider instead of one grid per "
    "checkpoint stacked down the page. Scrubbing is justified under D10 (interaction earns its "
    "keep only when inspection IS the finding): watching one shape's prediction move across "
    "training is the finding, and no static stacking at this density carries it. A reader "
    "without JavaScript still sees a complete, correct wall &mdash; the newest checkpoint, "
    "statically, exactly as it renders in the current live page.</p>"
    "<p>This is a MOCKUP built from the real evaluation store (5 checkpoints on disk right "
    "now, same data the live page reads). It is not linked from the workspace tree and does "
    "not touch <code>live_eval/</code>. Incorporation spec for whoever wires this into the "
    "real build.py: <code>INCORPORATION.md</code> next to this page.</p>",
    warn=True)


def main():
    recs = load_records()
    quick = [r for r in recs if r["tier"] == "quick" and is_ok(r)][-N_WALLS:]
    full = [r for r in recs if r["tier"] == "full_val" and is_ok(r)]
    if not quick:
        sys.exit("no quick-tier records on disk; the mockup needs at least one to build")

    sids = []
    for r in quick:
        for s in r["per_shape"]:
            if s["sid"] not in sids:
                sids.append(s["sid"])
    n_copied = copy_images(quick, sids)

    latest = quick[-1]
    n_ckpts = len({r["step"] for r in quick} | {r["step"] for r in full})
    stats = [
        (f"{full[-1]['mean_iou']:.3f}" if full else "not yet run", "latest full-validation IoU"),
        (f"step {latest['step']} (epoch {epoch_label(latest['step'])})", "newest checkpoint"),
        (str(n_ckpts), "checkpoints evaluated"),
    ]
    hero = lp.hero_header(
        f"lightgen · live evaluation, redesign mockup · {RUN_NAME}",
        "Every Epoch, Ten Shapes, Five Draws",
        dek_html="The curve and the visual wall are the whole page; everything else is one "
                "collapsed section at the bottom.",
        stats=stats,
        toc=[(i, lab) for i, lab in OUTLINE])

    sec_wall_html, ckpt_index = sec_wall(quick)
    body_sections = [DESIGN_NOTES, sec_curve(quick, full), sec_wall_html,
                     sec_detail(recs, quick, full)]

    page_html = lp.page(
        title="Live Checkpoint Evaluation (redesign mockup)",
        header_html=hero,
        body_sections=body_sections,
        assets_rel=f"{SITE_ROOT}/assets",
        assets_dir=os.path.join(WEB, "assets"),
        theme="v3",
        tree_html=wz.tree_html(active_href=None),
        nav_title="live eval (mockup)",
        outline_entries=[{"id": i, "label": lab} for i, lab in OUTLINE],
        needs_katex=False,
        extra_head=(f'<link rel="icon" href="{FAVICON}">'
                    '<link rel="stylesheet" href="slider_view.css">'),
        extra_body_end='<script src="slider_view.js"></script>',
    )

    violations = wz.console_links_in(page_html)
    if violations:
        sys.exit(f"ZONE-LINK GUARD FAILED: page links to the console: {violations}")

    with open(os.path.join(HERE, "index.html"), "w") as f:
        f.write(page_html)
    with open(os.path.join(HERE, "ckpts.json"), "w") as f:
        json.dump(ckpt_index, f, indent=1)
    print(f"wrote index.html ({len(page_html)} bytes) + ckpts.json "
          f"({len(quick)} checkpoints, {len(full)} full runs); {n_copied} thumbnail(s) copied")
    publish_assets(os.path.join(WEB, "assets"))


if __name__ == "__main__":
    main()
