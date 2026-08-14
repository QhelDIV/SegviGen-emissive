"""Copy final GT + pred panels into the web img dir, with clean names."""
import json
import os
import shutil

DBG = "/project/3dlg-hcvc/omages/yanxg_scratch/mask_debug"
IMG = "/local-scratch2/xya120/studio/misc/lightgen/web/_preview/fixedbake_galleries/img"

rows = json.load(open(f"{DBG}/gallery_verify.json"))

# overrides for the two fallback-fixed shapes
FALLBACK = {
    "e1fbc7943362485489dba5a951ebc4b1": {
        "jobname": "fig7_11_e1fbc7943362485489dba5a951ebc4b1_emadraw0",
        "pick_label": "ema draw0", "pick_frac": None, "fallback": True,
    },
    "b74fc2533d5345629f2c3ce2c8ab340a": {
        "jobname": "ours12_b74fc2533d5345629f2c3ce2c8ab340a_rawdraw0",
        "pick_label": "raw draw0", "pick_frac": None, "fallback": True,
    },
}

out_manifest = []
for row in rows:
    sid = row["sid"]
    gt_src = row["gt_png"]
    pred_src = row["pred_png"]
    pick_label = row["pick_label"]
    pick_frac = row["pick_frac"]
    fallback = False

    if sid in FALLBACK:
        jobname = FALLBACK[sid]["jobname"]
        pred_dir = f"{DBG}/gallery_out/renders/{jobname}"
        cands = [f for f in os.listdir(pred_dir) if f.endswith(".png")] if os.path.isdir(pred_dir) else []
        if cands:
            pred_src = os.path.join(pred_dir, cands[0])
            pick_label = FALLBACK[sid]["pick_label"]
            fallback = True

    gt_dst = os.path.join(IMG, f"{sid}_gt.png")
    pred_dst = os.path.join(IMG, f"{sid}_pred.png")
    if gt_src:
        shutil.copy2(gt_src, gt_dst)
    if pred_src:
        shutil.copy2(pred_src, pred_dst)
    out_manifest.append({**row, "pick_label": pick_label, "pred_src_used": pred_src,
                         "fallback_used": fallback})
    print(f"{sid}: gt<-{'OK' if gt_src else 'MISSING'} pred<-{'OK' if pred_src else 'MISSING'} "
          f"{'FALLBACK' if fallback else ''}")

json.dump(out_manifest, open(f"{DBG}/gallery_final_manifest.json", "w"), indent=1)
print(f"DONE n={len(out_manifest)}")
