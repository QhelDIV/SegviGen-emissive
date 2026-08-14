"""Gather every draw's frac@0.5 and thumbnail path per shape, in draw order,
for the fixedbake_galleries full-sample extension."""
import glob
import json
import os
import shutil

import numpy as np

DBG = "/project/3dlg-hcvc/omages/yanxg_scratch/mask_debug"
THUMB_RENDERS = f"{DBG}/thumb_out/renders"
WEB_IMG = "/local-scratch2/xya120/studio/misc/lightgen/web/_preview/fixedbake_galleries/img"

FIG7_SIDS = ["064e4156b5c345c796cc00d3fa2e2243", "4e383188516c46a58e96b1b7fc2f16a7",
            "ac6c953289bb4f56836c00830a7bb111", "a05f414c2ffd4103a964a9be5ef2d157",
            "f6314284e0e84a14ba466613ae776110", "34170054845344aeb199b842a3bf7e92",
            "92cedcbac4f84083b04e10a6df6ef0f3", "1102d5a523c442829a8ce9930e4b692b",
            "75da9a74403946dda954f08a067e8ad5", "2b7ace1f2de04e98a5c8874866dc473b",
            "e1fbc7943362485489dba5a951ebc4b1"]
OURS_SIDS = ["8f4c281aef1b4563b6103efbcd77fac1", "b74fc2533d5345629f2c3ce2c8ab340a",
            "c1e3035d1ccb49df9c09aa86681faf30", "1e9c6545b4da42e0ba4e5dbcd2e0e8ff",
            "48af42db48c44cd9bfab32bbb057a39c", "51a60b164e874bf891597d9c6c1941af",
            "e5eecab2bc8649548b48b79e705d768e", "9418a924a50d44c186dd499006b62424",
            "619f0732286f4a4683412d7f1cae983b", "4e105e043a6447439e98e9831aed122e",
            "b7709a651d144134a5babce33223380a", "658ecf9f837246509b0b1c4aa81e9e5b"]


HAMMER = "4e383188516c46a58e96b1b7fc2f16a7"
ROBOT = "34170054845344aeb199b842a3bf7e92"
# the paper panels for these two shapes were picked from a SEPARATE fresh-seed
# dump (hammer_draws/, robot_draws/), not from fig7_11_alldraws at all; fold
# those extra draws into the strip too, or the paper pick would not even
# appear among "every draw tried" for its own shape.
YANXG_SCRATCH = "/project/3dlg-hcvc/omages/yanxg_scratch"
PAPER_EXTRA_DIRS = {
    HAMMER: f"{YANXG_SCRATCH}/hammer_draws",
    ROBOT: f"{YANXG_SCRATCH}/robot_draws",
}


def cases_for(sid, setname):
    base = f"{DBG}/fig7_11_alldraws" if setname == "fig7_11" else f"{DBG}/ours12_alldraws"
    cases = []
    for weight, wtag in (("raw_real", "raw"), ("ema_real", "ema")):
        for p in sorted(glob.glob(f"{base}/{weight}/{sid}__draw*.npz")):
            k = int(os.path.basename(p).split("__draw")[1].split(".npz")[0])
            cases.append((f"{wtag}d{k}", wtag, k, p))
    for weight, wtag in (("raw_real", "rescue_raw"), ("ema_real", "rescue_ema")):
        for p in sorted(glob.glob(f"{DBG}/rescue_alldraws/{weight}/{sid}__draw*.npz")):
            k = int(os.path.basename(p).split("__draw")[1].split(".npz")[0])
            cases.append((f"{wtag}s{k}", wtag, k, p))
    if sid in PAPER_EXTRA_DIRS:
        extra_dir = PAPER_EXTRA_DIRS[sid]
        for wtag in ("raw", "ema"):
            for seed in range(8):
                p = f"{extra_dir}/{wtag}_seed{seed}/{sid}.npz"
                if os.path.exists(p):
                    cases.append((f"paper_{wtag}s{seed}", f"paper_{wtag}", seed, p))
    return cases


final_manifest = json.load(open(f"{DBG}/gallery_final_manifest.json"))
by_sid = {r["sid"]: r for r in final_manifest}

out = {}
for setname, sids in (("fig7_11", FIG7_SIDS), ("ours12", OURS_SIDS)):
    for sid in sids:
        cases = cases_for(sid, setname)
        draws = []
        for tag, wtag, k, npz in cases:
            z = np.load(npz)
            frac = float((z["pred_bc"] > 0.5).mean())
            png = os.path.join(THUMB_RENDERS, sid, f"{sid}_box_{tag}.png")
            thumb_name = None
            if os.path.exists(png):
                thumb_name = f"{sid}_thumb_{tag}.png"
                shutil.copy2(png, os.path.join(WEB_IMG, thumb_name))
            draws.append({"tag": tag, "weight": wtag, "idx": k, "frac": frac,
                         "thumb": thumb_name})
        r = by_sid[sid]
        # is this shape's DISPLAYED pick among these draws? match by frac closeness
        # (paper-pick swaps for hammer/robot/saber use a draw outside this set's
        # tag scheme, handled separately below)
        picked_tag = None
        pick_label = r["pick_label"]
        tol = 1e-2 if "(paper pick)" in pick_label else 1e-6
        for d in draws:
            if abs(d["frac"] - r["pick_frac"]) < tol:
                picked_tag = d["tag"]
                break
        out[sid] = {"set": setname, "split": r["split"], "caption": r["caption"],
                    "gt_frac": r["gt_frac"], "pick_label": pick_label,
                    "pick_frac": r["pick_frac"], "picked_tag": picked_tag,
                    "draws": draws}
        n_missing = sum(1 for d in draws if d["thumb"] is None)
        print(f"{sid} ({r['caption']}): {len(draws)} draws, {n_missing} thumbs missing, "
              f"picked_tag={picked_tag}", flush=True)

json.dump(out, open(f"{DBG}/draws_manifest.json", "w"), indent=1)
print(f"DONE n_shapes={len(out)}", flush=True)
