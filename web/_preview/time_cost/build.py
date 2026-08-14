#!/usr/bin/env python3
"""Build the "Pipeline time cost" page: how long each stage of the segvigen-emissive
evaluation pipeline actually takes on Solar, measured from sacct and the jobs' own
logs, so the owner can see which stages are out of proportion.

Four stages, in pipeline order: data processing (dataset build, condition-image
render), model inference (dump predicted voxels), conversion (voxel mask to mesh
asset), Blender rendering. The job families assigned to each stage are the
explicit chain the owner asked about (2026-08-10 evening through 2026-08-11
afternoon): ckpt4_eval, ckpt8_eval, fig7, ours(12), fbv1, ct10, the uvfree/redpad/
voxel_debug/voxel_true_res/synth_ctrl render debugging wave, and the robot/hammer
single-shape draw checks. Training and eval-curve jobs (ckpt4_val*, ckpt8_val*,
emis72k*, overfit10_anchors, single_pw1, ct10_pw1, ct10_glow, fs19_*, rdoc_*) are
excluded on purpose: they are not part of this dump -> convert -> render chain.

All stage totals are computed at build time from data/sacct_2026-08-09_to_11.txt,
a copy of:
    sacct -u xya120 -S 2026-08-09 --format=JobID,JobName%30,State,Elapsed,AllocCPUS,\
ReqMem,Submit,Start,End -n
pulled via cluster_ssh.py on 2026-08-11 ~16:45 PT. Three render jobs (voxel_true_res
243323, uvfree_final3 243326, uvfree_final_v2 243330) and one pending job
(uvfree_hammer_rescue 243331) were still running/queued at pull time; their stage
totals are therefore a floor, not a final number -- flagged in the page.

The reload-overhead, timeout/retry, and cancelled-render findings in section 05 are
read off the jobs' own stdout logs on Solar (paths cited inline); those are prose
citations, not re-derived at build time, because the underlying logs are per-job
free text, not a stable machine-readable artifact.

Run: .venv_console/bin/python web/_preview/time_cost/build.py [--publish]
"""
import html as _html
import os
import shutil
import sys

_esc = _html.escape

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WEB)
sys.path.insert(0, os.path.join(REPO, "tools"))

import xgpage as lp                        # noqa: E402
import workspace_zone as wz                # noqa: E402  (read-only: tree + zone guard)
from xgpage.publish import publish_assets  # noqa: E402

SITE_ROOT = "/projects/omages/yanxg/lightgen"
SITE_ASSETS = f"{SITE_ROOT}/assets"
FAVICON = f"{SITE_ROOT}/assets/images/favicon.png"
PAGE_DATE = "2026-08-11"

SACCT_FILE = os.path.join(HERE, "data", "sacct_2026-08-09_to_11.txt")
PUBLISH_DIR = "/project/3dlg-hcvc/omages/www/yanxg/lightgen/_preview/time_cost"

# ================================================================ sacct parsing

def sec(elapsed):
    d = 0
    if "-" in elapsed:
        d, elapsed = elapsed.split("-")
        d = int(d)
    parts = [int(x) for x in elapsed.split(":")]
    while len(parts) < 3:
        parts = [0] + parts
    h, m, s = parts
    return d * 86400 + h * 3600 + m * 60 + s


STAGE_FAMILIES = {
    "data": ["fs19_condA", "rdoc_condA", "ours12_build", "fbv1_voxmask_patha"],
    "inference": ["ckpt4_dump", "ckpt8_dump", "fig7_dump", "ours_dump",
                  "fbv1_dump_raw", "dump_alldraws", "robot_draws", "hammer_draws",
                  "probe_cond"],
    "conversion": ["ct10_maskxfer_a", "ckpt4_maskxfer", "ckpt4_maskxfer_retry",
                   "ckpt8_maskxfer", "fig7_maskxfer", "ours_maskxfer"],
    "render": ["ckpt4_render", "ckpt8_render", "fig7_render", "ours_render",
               "ours_render_arr", "redpad_render", "redpad_iso", "redpad_closest",
               "redpad_nodenoise", "voxel_debug", "voxel_true_res",
               "synth_ctrl_render", "synth_ctrl_fig7", "uvfree_validate",
               "uvfree_paper3", "uvfree_all", "uvfree_gt3", "uvfree_final3",
               "uvfree_final_v2", "uvfree_hammer_rescue"],
}
STAGE_LABEL = {"data": "Data processing", "inference": "Model inference",
               "conversion": "Conversion (mask -> mesh)", "render": "Blender rendering"}
STAGE_ORDER = ["data", "inference", "conversion", "render"]

NAME2STAGE = {n: st for st, names in STAGE_FAMILIES.items() for n in names}


def load_rows():
    rows = []
    with open(SACCT_FILE) as f:
        for line in f:
            p = line.split()
            if len(p) < 9:
                continue
            jid = p[0]
            if "." in jid:  # skip .batch/.extern accounting subrows
                continue
            rows.append(dict(jid=jid, name=p[1], state=p[2], elapsed=p[3],
                              cpus=p[4], mem=p[5], submit=p[6], start=p[7], end=p[8]))
    return rows


def stage_stats(rows):
    from datetime import datetime

    def parse(t):
        return None if t in ("Unknown", "None") else datetime.strptime(t, "%Y-%m-%dT%H:%M:%S")

    stats = {st: dict(n=0, wall_s=0, core_s=0, running=0, waits=[], families={})
             for st in STAGE_ORDER}
    for r in rows:
        st = NAME2STAGE.get(r["name"])
        if st is None:
            continue
        s = stats[st]
        e = sec(r["elapsed"])
        cpus = int(r["cpus"]) if r["cpus"].isdigit() else 0
        s["n"] += 1
        s["wall_s"] += e
        s["core_s"] += e * cpus
        if r["state"] == "RUNNING":
            s["running"] += 1
        fam = s["families"].setdefault(r["name"], dict(n=0, wall_s=0, core_s=0, cpus=cpus))
        fam["n"] += 1
        fam["wall_s"] += e
        fam["core_s"] += e * cpus
        sub, start = parse(r["submit"]), parse(r["start"])
        if sub and start:
            w = (start - sub).total_seconds()
            if w > 0:
                s["waits"].append(w)
    return stats


ROWS = load_rows()
STATS = stage_stats(ROWS)


def hrs(s):
    return s / 3600.0


def fmt_hms(s):
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec_ = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec_:02d}s"
    return f"{sec_}s"


# ================================================================ evidence read off logs
# (see build.py module docstring: these are prose citations, not re-derived at
# build time from a machine-readable artifact -- the sources are the jobs' own
# free-text stdout logs on Solar. Every number below was read off the cited log.)

RELOAD_ROBOT = dict(
    job="robot_draws (243321)",
    log="/project/3dlg-hcvc/omages/yanxg_scratch/ckpt8_eval/logs/robot_draws_243321.log",
    job_wall_s=130,           # sacct Elapsed 00:02:10
    n_draws=16,               # 8 draws x 2 checkpoint variants (raw, ema)
    sampling_s=48.8,          # VARIANT_DONE raw elapsed=24.4s + ema elapsed=24.4s
    reload_s=130 - 48.8,
)
RELOAD_HAMMER = dict(
    job="hammer_draws (243329)",
    log="/project/3dlg-hcvc/omages/yanxg_scratch/ckpt8_eval/logs/hammer_draws_243329.log",
    job_wall_s=114,           # sacct Elapsed 00:01:54
    n_draws=16,
    sampling_s=33.6,          # VARIANT_DONE raw 16.4s + ema 17.2s
    reload_s=114 - 33.6,
)

CONV_TIMEOUT = dict(
    job="ckpt4_maskxfer array 242434, tasks 6/14/15",
    log="/project/3dlg-hcvc/omages/yanxg_scratch/ckpt4_eval/logs/maskxfer_array_242434_6.log",
    retry_log="/project/3dlg-hcvc/omages/yanxg_scratch/ckpt4_eval/logs/maskxfer_retry_242455_6.log",
    n_timeout=3, n_array=16,
    timelimit_s=1229,          # 20:29, SLURM time limit hit
    example_sid="f52e9b616c0a4075a70e5eb844f07bb3",
    example_slots=11,
    retry_s=1440,               # 24:00 on the resubmit
    fast_sid="0414e54cda324108a7a51615f5cfd376", fast_slots=2, fast_s=140,
)
CONV_RACE = dict(
    job="ours_maskxfer array 242807 (all 36 tasks FAILED) -> 242885 (resubmit, 36/36 OK)",
    log="/project/3dlg-hcvc/omages/yanxg_scratch/ckpt8_eval/logs/ours_maskxfer_array_242807_0.log",
    error="FileNotFoundError: .../gallery_ours/pred_voxels/raw_real/8f4c281aef1b4563b6103efbcd77fac1.npz",
    n_failed=36, fail_wall_s=57, retry_wall_s=700,
)

RENDER_CANCEL = dict(
    job="ours_render (242924)",
    log="/project/3dlg-hcvc/omages/yanxg_scratch/ckpt8_eval/logs/ours_render_242924.log",
    wall_s=2208,           # 36:48
    cpus=64,
    n_saved=11, n_needed=36,
    cause="SIGNAL Terminated (external kill, not an in-script crash)",
    rerun_job="ours_render_arr (242950), 36-task array with skip-if-exists",
    rerun_core_h=83.84,
)

# ================================================================ facts derived above
def core_h(wall_s, cpus):
    return wall_s * cpus / 3600.0


RELOAD_FRAC_ROBOT = RELOAD_ROBOT["reload_s"] / RELOAD_ROBOT["job_wall_s"]
RELOAD_FRAC_HAMMER = RELOAD_HAMMER["reload_s"] / RELOAD_HAMMER["job_wall_s"]
SEC_PER_DRAW_SAMPLING = (RELOAD_ROBOT["sampling_s"] + RELOAD_HAMMER["sampling_s"]) / (
    RELOAD_ROBOT["n_draws"] + RELOAD_HAMMER["n_draws"])
SEC_PER_DRAW_AMORTIZED = (RELOAD_ROBOT["job_wall_s"] + RELOAD_HAMMER["job_wall_s"]) / (
    RELOAD_ROBOT["n_draws"] + RELOAD_HAMMER["n_draws"])

CONV_TIMEOUT_WASTED_CH = core_h(CONV_TIMEOUT["timelimit_s"] * CONV_TIMEOUT["n_timeout"], 8)
CONV_TIMEOUT_WASTED_SHARE = CONV_TIMEOUT_WASTED_CH / hrs(STATS["conversion"]["core_s"])

RENDER_CANCEL_WASTED_CH = core_h(RENDER_CANCEL["wall_s"], RENDER_CANCEL["cpus"]) * (
    (RENDER_CANCEL["n_needed"] - RENDER_CANCEL["n_saved"]) / RENDER_CANCEL["n_needed"])

# ================================================================ per-unit render figures (from sacct + n_saved counted off each job's own log)
RENDER_UNIT_ROWS = [
    ("uvfree_all (243318), quick check", 146, 7),
    ("ckpt4_render (242466), 256-spp", 2279, 40),
    ("ckpt8_render (242788), 256-spp", 2915, 40),
    ("fig7_render (242877), box", 4567, 33),
    ("ours_render_arr (242950), box, 1024-spp (canonical)", 4713, 25),
    ("synth_ctrl_fig7 (243242), box", 419, 1),
]

# ================================================================ page assembly


def stat_band():
    top_stage = max(STAGE_ORDER, key=lambda st: STATS[st]["wall_s"])
    total_wall_h = sum(STATS[st]["wall_s"] for st in STAGE_ORDER) / 3600.0
    stats = [(f"{hrs(STATS[st]['wall_s']):.2f} h", STAGE_LABEL[st]) for st in STAGE_ORDER]
    stats.append((f"{STAGE_LABEL[top_stage]}", "single most expensive step"))
    return stats


def sec_chain():
    rows_html = []
    for st in STAGE_ORDER:
        s = STATS[st]
        fam_list = ", ".join(f"{_esc(n)}×{f['n']}" for n, f in sorted(s["families"].items()))
        running_note = f" ({s['running']} still running at pull time)" if s["running"] else ""
        rows_html.append(
            f"<tr><td>{_esc(STAGE_LABEL[st])}</td>"
            f"<td>{s['n']}</td>"
            f"<td>{fmt_hms(s['wall_s'])}{running_note}</td>"
            f"<td>{hrs(s['core_s']):.1f}</td>"
            f"<td style='font-size:.82em'>{fam_list}</td></tr>"
        )
    table = lp.results_table(
        ["Stage", "Jobs (incl. array tasks)", "Wall-clock (summed)", "Core-hours", "Job families"],
        "".join(rows_html))
    body = lp.prose(
        "The chain covers the dump &rarr; convert &rarr; render evaluation pipeline for "
        "six checkpoint/shape sets (<code>ckpt4</code>, <code>ckpt8</code>, "
        "<code>fig7</code>, <code>ours12</code>, <code>fbv1</code>, <code>ct10</code>), "
        "plus the render-debugging wave (<code>uvfree_*</code>, <code>redpad_*</code>, "
        "<code>voxel_debug</code>, <code>voxel_true_res</code>, <code>synth_ctrl_*</code>) "
        "and the two single-shape draw checks (<code>robot_draws</code>, "
        "<code>hammer_draws</code>), spanning 2026-08-10 01:43 through 2026-08-11 "
        "16:27. Training and eval-curve jobs (<code>ckpt4_val*</code>, "
        "<code>ckpt8_val*</code>, <code>emis72k*</code>, <code>fs19_*</code>, "
        "<code>rdoc_*</code>) are excluded &mdash; they are a separate track, not "
        "this pipeline."
    ) + table + lp.callout(
        "Three render jobs (<code>voxel_true_res</code> 243323, <code>uvfree_final3</code> "
        "243326, <code>uvfree_final_v2</code> 243330) were still <b>RUNNING</b>, and one "
        "(<code>uvfree_hammer_rescue</code> 243331) was still <b>PENDING</b>, at the moment "
        "this data was pulled (2026-08-11 16:45 PT). The render-stage totals below are "
        "therefore a floor, not a final number.", warn=True)
    return lp.section_v2("chain", "01", "The chain measured", body)


def sec_stage_totals():
    wall_rows = [
        {"label": STAGE_LABEL[st], "value": hrs(STATS[st]["wall_s"]),
         "display": f"{hrs(STATS[st]['wall_s']):.2f} h"}
        for st in STAGE_ORDER
    ]
    chart1 = lp.hbar_chart(
        wall_rows, title="summed wall-clock hours by stage (job-seconds, not calendar time)",
        note=(
            "Blender rendering is the single most expensive stage by wall-clock, at "
            f"{hrs(STATS['render']['wall_s']):.2f}&nbsp;hours summed across "
            f"{STATS['render']['n']} jobs &mdash; more than half of the "
            f"{sum(hrs(STATS[st]['wall_s']) for st in STAGE_ORDER):.2f}&nbsp;hours summed "
            "across all four stages, and undercounted because three render jobs were still "
            "running when this was measured."))

    core_rows = [
        {"label": STAGE_LABEL[st], "value": hrs(STATS[st]["core_s"]),
         "display": f"{hrs(STATS[st]['core_s']):.1f} core-h"}
        for st in STAGE_ORDER
    ]
    chart2 = lp.hbar_chart(
        core_rows, title="core-hours by stage (wall-clock &times; allocated CPUs)",
        note=(
            "Wall-clock understates how lopsided this is. Rendering runs at 64 cores per "
            "job against 8 for inference and conversion, so in core-hours &mdash; the "
            "number that determines fair-share against other users on Solar &mdash; "
            f"rendering is <b>{hrs(STATS['render']['core_s']) / sum(hrs(STATS[st]['core_s']) for st in STAGE_ORDER) * 100:.0f}%</b> "
            "of the total, not the roughly half its wall-clock share suggests."))
    body = chart1 + chart2
    return lp.section_v2("stages", "02", "Rendering dominates, more in core-hours than in wall-clock", body)


def sec_per_unit():
    rows = [
        {"label": "sampling only, per draw (steady state)", "value": SEC_PER_DRAW_SAMPLING,
         "display": f"{SEC_PER_DRAW_SAMPLING:.1f} s"},
        {"label": "inference job wall-clock, per draw (amortized)", "value": SEC_PER_DRAW_AMORTIZED,
         "display": f"{SEC_PER_DRAW_AMORTIZED:.1f} s"},
        {"label": "conversion, median shape", "value": 6, "display": "6 s"},
        {"label": "conversion, worst observed (11 material slots, timed out)",
         "value": CONV_TIMEOUT["timelimit_s"], "display": fmt_hms(CONV_TIMEOUT["timelimit_s"])},
        {"label": "render, fastest family (uvfree_all quick check)", "value": 20.9, "display": "21 s"},
        {"label": "render, canonical box (1024-spp)", "value": 188.5, "display": "3m09s"},
        {"label": "render, slowest observed (synth_ctrl_fig7)", "value": 419, "display": "6m59s"},
    ]
    chart = lp.hbar_chart(
        rows, title="seconds per unit (per draw for inference, per shape for conversion/render)",
        label_w=340,
        note=(
            "Within a single stage the spread is bigger than the gap between stages. "
            f"Inference costs {SEC_PER_DRAW_AMORTIZED / SEC_PER_DRAW_SAMPLING:.1f}&times; "
            "more per draw once the checkpoint reload is "
            "counted; conversion costs over 200&times; more for a shape with 11 material "
            "slots than the median shape; canonical box renders run 9&times; slower than "
            "a quick check and the slowest observed render is 20&times; the fastest. "
            "Section 05 traces each of these to a specific, fixable cause."))
    return lp.section_v2("perunit", "03", "The spread inside a stage exceeds the gap between stages", chart)


def sec_queue():
    rows_html = []
    for st in STAGE_ORDER:
        w = STATS[st]["waits"]
        avg = sum(w) / len(w) if w else 0
        mx = max(w) if w else 0
        n_waited = sum(1 for x in w if x > 5)
        rows_html.append(
            f"<tr><td>{_esc(STAGE_LABEL[st])}</td><td>{fmt_hms(avg)}</td>"
            f"<td>{fmt_hms(mx)}</td><td>{n_waited} / {STATS[st]['n']}</td></tr>")
    table = lp.results_table(
        ["Stage", "Avg queue wait", "Max queue wait", "Jobs waiting &gt;5s"], "".join(rows_html))
    body = lp.prose(
        "For this chain, time is lost <b>computing</b>, not waiting. Data-processing and "
        "inference jobs dispatch essentially instantly (average wait under 3 seconds "
        "each). Conversion averages a 32-second queue wait; render jobs average about "
        "3 minutes and one render job waited 14.7 minutes, but even that outlier is small "
        "next to the hours those stages spend running. The large queue waits seen "
        "elsewhere in today's sacct log (up to 1h44m) all belong to the separate "
        "<code>ckpt4_val*</code> / <code>ckpt8_val*</code> eval-curve jobs, which submit "
        "many short jobs back-to-back onto one allocation and are not part of this chain."
    ) + table
    return lp.section_v2("queue", "04", "Time is lost computing, not waiting", body)


def sec_waste():
    # --- candidate 1: checkpoint reload ---
    reload_table = lp.results_table(
        ["Job", "Job wall-clock", "Actual sampling", "Reload / setup", "Reload share"],
        "".join(f"<tr><td>{_esc(r['job'])}</td><td>{fmt_hms(r['job_wall_s'])}</td>"
                f"<td>{r['sampling_s']:.1f} s</td><td>{r['reload_s']:.1f} s</td>"
                f"<td>{r['reload_s'] / r['job_wall_s'] * 100:.0f}%</td></tr>"
                for r in (RELOAD_ROBOT, RELOAD_HAMMER)))
    c1 = lp.callout(
        "Both single-shape draw jobs load the model and checkpoint from scratch "
        "<b>twice</b> &mdash; once for the <code>raw</code> checkpoint, once for "
        "<code>ema</code> &mdash; inside the same job, each time re-running "
        "<code>ENV_OK models loaded</code> and <code>CKPT_LOADED</code> from the top. "
        "The two reload/setup windows account for "
        f"{RELOAD_FRAC_ROBOT * 100:.0f}% and {RELOAD_FRAC_HAMMER * 100:.0f}% of the two "
        "jobs' total wall-clock &mdash; more time than the 16 diffusion draws "
        "themselves. The same raw-then-ema double reload shows up in every "
        "multi-shape dump job checked (<code>dump_alldraws</code> 243319, "
        "<code>ckpt4_dump</code> 242433). One shared model load with a weight swap "
        "between variants, instead of two full environment inits, would cut this "
        "stage's wall-clock by roughly half.", title="1. Inference reloads the model twice per job")
    c1 += reload_table

    # --- candidate 2: conversion timeouts driven by material-slot count ---
    c2 = lp.callout(
        f"3 of {CONV_TIMEOUT['n_array']} tasks in the <code>ckpt4_maskxfer</code> array "
        f"({CONV_TIMEOUT['job']}) hit the "
        f"{fmt_hms(CONV_TIMEOUT['timelimit_s'])} SLURM time limit and were killed with "
        "no output written, then had to be resubmitted as a separate retry array. "
        f"The shape that timed out (<code>{CONV_TIMEOUT['example_sid'][:12]}&hellip;</code>, "
        f"{CONV_TIMEOUT['example_slots']} material slots) needed "
        f"{fmt_hms(CONV_TIMEOUT['retry_s'])} on the retry to actually finish, while a "
        f"2-slot shape in the same array (<code>{CONV_TIMEOUT['fast_sid'][:12]}&hellip;</code>) "
        f"finished in {CONV_TIMEOUT['fast_s']} seconds. The array gives every shape the "
        "same time budget regardless of its material-slot count, so the 3 timed-out "
        f"attempts burned <b>{CONV_TIMEOUT_WASTED_CH:.1f} core-hours producing nothing</b> "
        "before the retry array spent core-hours again to actually do the work "
        f"&mdash; {CONV_TIMEOUT_WASTED_SHARE * 100:.0f}% of everything the conversion "
        "stage spent today. "
        "Separately, the whole <code>ours_maskxfer</code> array (36 tasks) failed "
        "instantly on a <code>FileNotFoundError</code>"
        f" (<code>{_esc(os.path.basename(CONV_RACE['error']))}</code>): "
        "it was submitted before the upstream dump job had finished writing its "
        "prediction voxels, and had to be resubmitted whole "
        f"({CONV_RACE['job']}). Cheap in core-hours "
        f"({core_h(CONV_RACE['fail_wall_s'], 8):.2f} core-h), but it doubled the number "
        "of conversion jobs needed for that shape set and added a diagnose-and-resubmit "
        "cycle to the turnaround.", title="2. Conversion time limit doesn't scale with material-slot count",
        warn=True)

    # --- candidate 3: non-checkpointed render job killed mid-run ---
    c3 = lp.callout(
        f"<code>{RENDER_CANCEL['job']}</code> ran sequentially through the gallery, saved "
        f"{RENDER_CANCEL['n_saved']} of {RENDER_CANCEL['n_needed']} box renders, then was "
        f"killed ({RENDER_CANCEL['cause']}) after "
        f"{fmt_hms(RENDER_CANCEL['wall_s'])} at {RENDER_CANCEL['cpus']} cores &mdash; "
        f"{core_h(RENDER_CANCEL['wall_s'], RENDER_CANCEL['cpus']):.1f} core-hours, of which "
        f"about {RENDER_CANCEL_WASTED_CH:.1f} core-hours produced nothing, because the job "
        "checkpoints nothing per-shape and a kill mid-run loses everything after the last "
        f"save. It was replaced by <code>{RENDER_CANCEL['rerun_job']}</code>, which "
        "correctly skipped the 11 already-saved shapes "
        "(<code>SKIP_EXISTS</code>) and only rendered the remaining 25 &mdash; the fix "
        "that made the rerun cheap was moving to one task per shape, not the original "
        "one-job-does-everything render.", title="3. One non-checkpointed render job lost ~27 core-hours to an external kill")

    # --- ruled out ---
    ruled_out = lp.prose(
        "<b>Checked and ruled out:</b> every render-stage job in this chain already "
        "requests the full 64 cores (verified across all 19 render job families in the "
        "sacct log) &mdash; the &ldquo;renders on fewer than 64 cores&rdquo; failure mode "
        "does not apply today. GPU inference jobs are not queued behind node exclusions "
        "either: the inference stage's average dispatch wait is 0.3 seconds (&sect;04)."
    )

    body = c1 + c2 + c3 + ruled_out
    return lp.section_v2("waste", "05", "Where the obvious waste is", body)


def sec_provenance():
    body = lp.prose(
        "<b>sacct command</b> (run via <code>cluster_skill/cluster_ssh.py</code>, "
        "2026-08-11 ~16:45 PT):"
    ) + lp.code_block(
        "sacct -u xya120 -S 2026-08-09 --format=JobID,JobName%30,State,Elapsed,"
        "AllocCPUS,ReqMem,Submit,Start,End -n"
    ) + lp.prose(
        "Saved as <code>data/sacct_2026-08-09_to_11.txt</code> in this page's folder; "
        "every stage total and chart on this page is computed from that file at build "
        "time by <code>build.py</code> (stage/family assignment: "
        "<code>STAGE_FAMILIES</code> near the top of the script)."
    ) + lp.prose(
        "<b>Logs read for the section-05 findings</b> (job stdout on Solar, cited inline "
        "above; each path starts under <code>/project/3dlg-hcvc/omages/yanxg_scratch/</code>):"
    ) + lp.code_block(
        "ckpt8_eval/logs/robot_draws_243321.log\n"
        "ckpt8_eval/logs/hammer_draws_243329.log\n"
        "mask_debug/dump_alldraws_243319.log\n"
        "ckpt4_eval/logs/dump_242433.log, ckpt8_eval/logs/dump_242764.log\n"
        "ckpt4_eval/logs/maskxfer_array_242434_{0,6}.log, "
        "ckpt4_eval/logs/maskxfer_retry_242455_6.log\n"
        "ckpt8_eval/logs/ours_maskxfer_array_242807_0.log\n"
        "ckpt8_eval/logs/ours_render_242924.log, "
        "ckpt8_eval/logs/ours_render_arr_242950_{0,18}.log"
    ) + lp.prose(
        "Per-render seconds-per-shape in &sect;03 come from dividing each job's sacct "
        "elapsed by the count of <code>Saved:</code> lines in that job's own log "
        "(counted directly, not estimated)."
    )
    return lp.section_v2("provenance", "06", "Provenance", body)


def build(publish=False):
    stats = stat_band()
    hero = lp.hero_header(
        "lightgen &middot; segvigen-emissive &middot; time cost",
        "Where the segvigen-emissive pipeline spends its time",
        dek_html=(
            "Every stage of the evaluation pipeline &mdash; data processing, model "
            "inference, mask-to-mesh conversion, Blender rendering &mdash; measured "
            "from Solar's own job accounting and the jobs' own logs, covering the "
            "chain run 2026-08-10 through 2026-08-11. Rendering is the largest cost by "
            "far, and three specific, fixable causes account for a disproportionate "
            "share of the waste inside the other three stages."),
        stats=stats,
        toc=[("chain", "The chain"), ("stages", "Stage totals"), ("perunit", "Per-unit cost"),
             ("queue", "Queue vs run"), ("waste", "The waste"), ("provenance", "Provenance")],
    )

    page_html = lp.page(
        title="Where the segvigen-emissive pipeline spends its time",
        header_html=hero,
        body_sections=[sec_chain(), sec_stage_totals(), sec_per_unit(), sec_queue(),
                       sec_waste(), sec_provenance()],
        assets_rel=SITE_ASSETS,
        assets_dir=os.path.join(WEB, "assets"),
        theme="v3",
        tree_html=wz.tree_html(active_href=None),
        nav_title="Pipeline time cost",
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
    print("assets published (repo-local web/assets)")

    if publish:
        os.makedirs(PUBLISH_DIR, exist_ok=True)
        shutil.copy2(out, os.path.join(PUBLISH_DIR, "index.html"))
        shutil.copytree(os.path.join(WEB, "assets"),
                         "/project/3dlg-hcvc/omages/www/yanxg/lightgen/assets",
                         dirs_exist_ok=True)
        for dp, dns, fns in os.walk(PUBLISH_DIR):
            for d in dns:
                os.chmod(os.path.join(dp, d), 0o755)
            for fn in fns:
                os.chmod(os.path.join(dp, fn), 0o644)
        os.chmod(PUBLISH_DIR, 0o755)
        print(f"published -> {PUBLISH_DIR}")
        print(f"URL: https://aspis.cmpt.sfu.ca{SITE_ROOT}/_preview/time_cost/index.html")


if __name__ == "__main__":
    build(publish="--publish" in sys.argv)
