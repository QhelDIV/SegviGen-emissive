"""
Does the PRETRAINED full part-segmentation already carve out the emissive region as
its own part(s)?  If yes, "emissive prediction" reduces to: segment parts (zero-shot,
pretrained) → label each part emissive/not (1 bit per part), instead of the hard
per-voxel coloring fine-tune.

Experiment (no training): run the pretrained SegviGen full_seg.ckpt on a few assets,
decode the per-voxel part coloring, cluster colors into parts, then compare the parts
to the GT emissive mask. The headline number is the ORACLE PART-LABELING IoU:
  - label each part emissive iff >50% of its voxels are GT-emissive;
  - predicted-emissive = union of those parts;
  - IoU(predicted, GT) = best emissive IoU achievable from this part-seg + perfect
    per-part labels. High → "emissive = a part" works; low → parts ignore emissive.

Also reports: best single-part IoU, #parts, and dumps per-voxel npz for visualization.

Usage (GPU node, trellis2 env):
  python seg_covers_emissive.py --dataset .../dataset --split overfit_10 \
    --seg_ckpt .../full_seg.ckpt --steps 25 --dump_vis seg_vis_overfit10
"""
import os, sys, json, argparse
ROOT = os.path.dirname(os.path.abspath(__file__))
SEGVIGEN = os.path.join(ROOT, "SegviGen")
if os.path.isdir(SEGVIGEN):
    sys.path.insert(0, SEGVIGEN)   # legacy layout: script sits next to a separate SegviGen/ clone
else:
    SEGVIGEN = ROOT                # this script now lives inside the SegviGen repo root
    sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/3dlg-jupiter-project/lightgen/hf_cache")

import torch
import numpy as np
from collections import OrderedDict
import trellis2.modules.sparse as sp
from trellis2 import models
from inference_full import Gen3DSeg, Sampler
from huggingface_hub import hf_hub_download


def load_pipeline_args():
    pj = hf_hub_download(repo_id="microsoft/TRELLIS.2-4B", filename="pipeline.json")
    return json.load(open(pj))["args"]


def quantize_parts(rgb, levels=6, min_frac=0.01):
    """rgb (N,3) float in ~[0,1] → integer part label per voxel.
    Quantize each channel to `levels` bins; merge tiny clusters (<min_frac) into the
    nearest-by-population large cluster's id is overkill — instead relabel tiny ones to
    a shared 'misc' id so they don't inflate the part count."""
    q = np.clip((rgb * (levels - 1)).round(), 0, levels - 1).astype(np.int64)
    key = q[:, 0] * levels * levels + q[:, 1] * levels + q[:, 2]
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    big = cnt >= max(1, int(min_frac * len(rgb)))
    remap = np.where(big, np.arange(len(uniq)), -1)
    labels = remap[inv]                       # -1 for tiny clusters
    # give all tiny voxels one shared label
    if (labels < 0).any():
        labels[labels < 0] = labels.max() + 1
    # compactify
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def iou(a, b):
    inter = (a & b).sum(); union = (a | b).sum()
    return inter / union if union > 0 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="overfit_10")
    ap.add_argument("--seg_ckpt", required=True, help="pretrained full_seg.ckpt")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--levels", type=int, default=6, help="color quantization bins per channel")
    ap.add_argument("--zero_cond", action="store_true", default=False)
    ap.add_argument("--dump_vis", default=None)
    args = ap.parse_args()
    device = "cuda"
    COND_T, COND_D = 1024, 1024
    if args.dump_vis:
        os.makedirs(args.dump_vis, exist_ok=True)

    flow = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")
    gen = Gen3DSeg(flow).to(device)
    sd = torch.load(args.seg_ckpt, map_location=device)["state_dict"]
    sd = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in sd.items()])
    gen.load_state_dict(sd); gen.eval()
    tex_decoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16").cuda().eval()
    shape_decoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16").cuda().eval()

    pa = load_pipeline_args()
    sm = torch.tensor(pa["shape_slat_normalization"]["mean"])[None].to(device)
    ss = torch.tensor(pa["shape_slat_normalization"]["std"])[None].to(device)
    tm = torch.tensor(pa["tex_slat_normalization"]["mean"])[None].to(device)
    ts = torch.tensor(pa["tex_slat_normalization"]["std"])[None].to(device)
    sampler = Sampler()
    sp_params = pa["tex_slat_sampler"]["params"]; sp_params["steps"] = args.steps

    sdir = os.path.join(args.dataset, args.split)
    oracle_ious, best_part_ious, nparts_list, fracs = [], [], [], []
    for sid in sorted(os.listdir(sdir)):
        d = os.path.join(sdir, sid)
        if not os.path.exists(os.path.join(d, "output_tex_slat.pth")):
            continue
        shp = torch.load(os.path.join(d, "shape_slat.pth"), map_location=device)
        itx = torch.load(os.path.join(d, "input_tex_slat.pth"), map_location=device)
        otx = torch.load(os.path.join(d, "output_tex_slat.pth"), map_location=device)
        if args.zero_cond or not os.path.exists(os.path.join(d, "cond.pth")):
            cond = torch.zeros(1, COND_T, COND_D, device=device)
        else:
            cond = torch.load(os.path.join(d, "cond.pth"), map_location=device)["cond"]
        coords = shp["coords"].to(device)

        shp_n = sp.SparseTensor((shp["feats"].to(device) - sm) / ss, coords)
        itx_n = sp.SparseTensor((itx["feats"].to(device) - tm) / ts, coords)
        cond_dict = {"cond": cond, "neg_cond": torch.zeros_like(cond)}
        with torch.no_grad():
            shape_decoder.set_resolution(512)
            _, subs = shape_decoder(sp.SparseTensor(shp["feats"].to(device), coords), return_subs=True)
            noise = sp.SparseTensor(torch.randn_like(itx_n.feats), coords)
            out = sampler.sample(gen, noise, itx_n, shp_n, [coords.shape[0]], cond_dict, sp_params)
            out = out * ts + tm
            seg_vox = tex_decoder(out, guide_subs=subs) * 0.5 + 0.5            # part coloring (RGB)
            gt_vox  = tex_decoder(sp.SparseTensor(otx["feats"].to(device), coords), guide_subs=subs) * 0.5 + 0.5

        seg_rgb = seg_vox.feats[:, :3].float().cpu().numpy()
        gt_e = (gt_vox.feats[:, :3].mean(-1) > 0.5).cpu().numpy()
        vox_coords = seg_vox.coords[:, 1:].cpu().numpy().astype(np.int16)

        labels = quantize_parts(seg_rgb, levels=args.levels)
        nparts = labels.max() + 1
        # oracle: label part emissive iff majority emissive
        pred_e = np.zeros_like(gt_e)
        best_single = 0.0
        for p in range(nparts):
            m = labels == p
            if m.sum() == 0:
                continue
            if gt_e[m].mean() > 0.5:
                pred_e |= m
            bi = iou(m, gt_e)
            best_single = max(best_single, bi)
        oiou = iou(pred_e, gt_e)
        oracle_ious.append(oiou); best_part_ious.append(best_single)
        nparts_list.append(nparts); fracs.append(float(gt_e.mean()))
        print(f"{sid} gt_frac={gt_e.mean():.3f} nparts={nparts:3d} "
              f"oracle_IoU={oiou:.3f} best_single_part_IoU={best_single:.3f}", flush=True)

        if args.dump_vis:
            np.savez_compressed(os.path.join(args.dump_vis, f"{sid}.npz"),
                                coords=vox_coords,
                                seg_rgb=(seg_rgb * 255).astype(np.uint8),
                                labels=labels.astype(np.int16),
                                gt_e=gt_e)

    print(f"\n=== {args.split} ({len(fracs)} samples) — does pretrained part-seg cover emissive? ===", flush=True)
    print(f"  mean ORACLE part-labeling IoU = {np.mean(oracle_ious):.4f}  "
          f"(upper bound for 'segment → label parts'; DiffusionNet per-voxel ~0.259)", flush=True)
    print(f"  mean best-single-part IoU     = {np.mean(best_part_ious):.4f}", flush=True)
    print(f"  mean #parts                   = {np.mean(nparts_list):.1f}", flush=True)
    json.dump({"oracle_iou": float(np.mean(oracle_ious)),
               "best_single_part_iou": float(np.mean(best_part_ious)),
               "mean_nparts": float(np.mean(nparts_list)),
               "per_sample": [{"oracle": o, "best_single": b, "nparts": int(n), "gt_frac": f}
                              for o, b, n, f in zip(oracle_ious, best_part_ious, nparts_list, fracs)]},
              open(os.path.join(args.dataset, f"seg_covers_{args.split}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
