"""Merge the original 10-draw pick with the 16 fresh-seed rescue draws for
the 10 POOR shapes, and report before/after."""
import glob
import json
import os

import numpy as np

DBG = "/project/3dlg-hcvc/omages/yanxg_scratch/mask_debug"

POOR = {
    "064e4156b5c345c796cc00d3fa2e2243": ("fig7_11", "jack-o'-lantern", "val"),
    "f6314284e0e84a14ba466613ae776110": ("fig7_11", "world-map table", "val"),
    "1102d5a523c442829a8ce9930e4b692b": ("fig7_11", "medieval brazier", "val"),
    "75da9a74403946dda954f08a067e8ad5": ("fig7_11", "outdoor wall lantern", "val"),
    "8f4c281aef1b4563b6103efbcd77fac1": ("ours12", "headphone stand", "train"),
    "c1e3035d1ccb49df9c09aa86681faf30": ("ours12", "turquoise sci-fi robot", "train"),
    "1e9c6545b4da42e0ba4e5dbcd2e0e8ff": ("ours12", "medieval wooden street lamp", "train"),
    "48af42db48c44cd9bfab32bbb057a39c": ("ours12", "pumpkin", "train"),
    "e5eecab2bc8649548b48b79e705d768e": ("ours12", "wall-mounted light fixtures", "train"),
    "b7709a651d144134a5babce33223380a": ("ours12", "Halloween animatronic", "train"),
}

ORIG = {"fig7_11": (f"{DBG}/fig7_11_alldraws/raw_real", f"{DBG}/fig7_11_alldraws/ema_real"),
        "ours12": (f"{DBG}/ours12_alldraws/raw_real", f"{DBG}/ours12_alldraws/ema_real")}
RESCUE = (f"{DBG}/rescue_alldraws/raw_real", f"{DBG}/rescue_alldraws/ema_real")

orig_picks = json.load(open(f"{DBG}/fig7_11_picks.json"))
orig_picks.update(json.load(open(f"{DBG}/ours12_picks.json")))


def load_draws(d, sid, tag):
    out = []
    for p in sorted(glob.glob(os.path.join(d, f"{sid}__draw*.npz"))):
        k = int(os.path.basename(p).split("__draw")[1].split(".npz")[0])
        z = np.load(p)
        frac = float((z["pred_bc"] > 0.5).mean())
        gt_frac = float(z["gt_e"].astype(bool).mean())
        out.append((tag, k, frac, gt_frac, p))
    return out


results = {}
for sid, (setname, caption, split) in POOR.items():
    raw_dir, ema_dir = ORIG[setname]
    r_raw = load_draws(raw_dir, sid, "raw")
    r_ema = load_draws(ema_dir, sid, "ema")
    q_raw = load_draws(RESCUE[0], sid, "rescue_raw")
    q_ema = load_draws(RESCUE[1], sid, "rescue_ema")
    all_draws = r_raw + r_ema + q_raw + q_ema
    gt_frac = all_draws[0][3]
    candidates = [(tag, k, f, p) for tag, k, f, _, p in all_draws if f > 0]
    tag, k, frac, npz_path = min(candidates, key=lambda c: abs(c[2] - gt_frac))
    old = orig_picks[sid]
    improved = abs(frac - gt_frac) < old["abs_diff"] - 1e-6
    results[sid] = {
        "caption": caption, "set": setname, "split": split,
        "gt_frac": gt_frac,
        "old_pick": f"{old['picked_weight']} draw{old['picked_draw']}",
        "old_frac": old["picked_frac"], "old_diff": old["abs_diff"],
        "new_pick": f"{tag} {'draw'+str(k) if not tag.startswith('rescue') else 'seed'+str(k)}",
        "new_frac": frac, "new_diff": abs(frac - gt_frac),
        "new_npz_path": npz_path, "new_tag": tag, "new_k": k,
        "improved": improved,
    }
    arrow = "IMPROVED" if improved else "no better than original pick"
    print(f"{sid} ({caption}, {split}): GT={gt_frac:.4f} | "
          f"OLD {old['picked_weight']} draw{old['picked_draw']} frac={old['picked_frac']:.4f} diff={old['abs_diff']:.4f} | "
          f"NEW {tag} k{k} frac={frac:.4f} diff={abs(frac-gt_frac):.4f} -> {arrow}", flush=True)

json.dump(results, open(f"{DBG}/rescue_repick.json", "w"), indent=1)
n_improved = sum(1 for v in results.values() if v["improved"])
print(f"RESCUE_SUMMARY n_improved={n_improved}/{len(results)}", flush=True)
