"""
Standalone fine-tune of Trellis.2's slat_flow_imgshape2tex flow for BINARY EMISSIVE
segmentation, framed as colorization (white=emissive / black=non).

Reuses SegviGen's Gen3DSeg wrapper + the inference flow-matching convention:
    x_t = t*noise + (1-t)*data,   v_target = noise - data,   model sees t*1000
(verified against inference_full.py Sampler). Inputs already carry appearance
(input_tex_slat = PBR latent) + shape (shape_slat) + DINOv3 cond — so no architecture
change; we just retarget the output_tex_slat to the emissive coloring.

Init from a SegviGen checkpoint via --init_ckpt (default full_seg — verified as what
every real training run to date has actually warm-started from; the old docstring/help
text said interactive_seg but that was never true, see --init_ckpt help).

Usage (GPU node, trellis2 env):
  python train_emissive.py --dataset .../dataset --out_dir .../outputs/emis_pilot \
      --epochs 300 --lr 1e-5 --n_per_epoch 0 --cond zero
"""
import os, sys, json, argparse, glob
ROOT = os.path.dirname(os.path.abspath(__file__))
SEGVIGEN = os.path.join(ROOT, "SegviGen")
if os.path.isdir(SEGVIGEN):
    sys.path.insert(0, SEGVIGEN)   # legacy layout: script sits next to a separate SegviGen/ clone
else:
    SEGVIGEN = ROOT                # this script now lives inside the SegviGen repo root
    sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/3dlg-jupiter-project/lightgen/hf_cache")

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import trellis2.modules.sparse as sp
from trellis2 import models
from inference_full import Gen3DSeg            # reuse the exact wrapper
from huggingface_hub import hf_hub_download
from eval_emissive import load_eval_models, evaluate_split, THRS


def load_norm_stats(device):
    pj = hf_hub_download(repo_id="microsoft/TRELLIS.2-4B", filename="pipeline.json")
    args = json.load(open(pj))["args"]
    def mk(d): return (torch.tensor(d["mean"])[None].to(device), torch.tensor(d["std"])[None].to(device))
    sm, ss = mk(args["shape_slat_normalization"])
    tm, ts = mk(args["tex_slat_normalization"])
    return sm, ss, tm, ts


def resolve_init_ckpt(spec):
    """spec is 'full_seg', 'interactive_seg', or an explicit ckpt path. The named forms
    resolve via the same fenghora/SegviGen HF repo that outputs/*/last.ckpt were all
    warm-started from (see sbatch history — every real run passed the full_seg.ckpt path
    under HF_HOME's hub cache; this just does the equivalent hf_hub_download lookup)."""
    if spec in ("full_seg", "interactive_seg"):
        return hf_hub_download(repo_id="fenghora/SegviGen", filename=f"{spec}.ckpt")
    return spec


# DINOv3-L cond: (1, num_patch_tokens=1024 @512px, cond_channels=1024). For the
# zero-cond ablation (DINOv3 gated) we feed zeros of this shape consistently.
COND_T, COND_D = 1024, 1024


class EmisDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root, split, cond_mode, require_mask=True):
        assert cond_mode in ("real", "zero")
        self.cond_mode = cond_mode
        self.require_mask = require_mask
        self.dirs = []
        self.fracs = []   # per-sample emissive fraction (for class-imbalance oversampling)
        sdir = os.path.join(dataset_root, split)
        core = ["shape_slat.pth", "input_tex_slat.pth", "output_tex_slat.pth"]
        for sid in sorted(os.listdir(sdir)):
            d = os.path.join(sdir, sid)
            if not all(os.path.exists(os.path.join(d, f)) for f in core):
                continue
            # No silent fallback: a sample missing cond.pth under --cond real, or
            # missing emis_mask.pth while pos_weight is active, is a hard error — it
            # means the dataset build is incomplete, not something to quietly skip.
            if cond_mode == "real" and not os.path.exists(os.path.join(d, "cond.pth")):
                raise RuntimeError(f"--cond real but cond.pth missing for {d} "
                                   f"(run build_dataset.py --real_cond, or use --cond zero)")
            if require_mask and not os.path.exists(os.path.join(d, "emis_mask.pth")):
                raise RuntimeError(f"emis_mask.pth missing for {d} "
                                   f"(run make_emis_mask.py, or pass --pos_weight 1.0 to disable weighting)")
            self.dirs.append(d)
            mp = os.path.join(d, "meta.json")
            fr = json.load(open(mp)).get("emissive_frac", 0.0) if os.path.exists(mp) else 0.0
            self.fracs.append(float(fr))

    def __len__(self): return len(self.dirs)

    def __getitem__(self, i):
        d = self.dirs[i]
        shp = torch.load(os.path.join(d, "shape_slat.pth"), map_location="cpu")
        itx = torch.load(os.path.join(d, "input_tex_slat.pth"), map_location="cpu")
        otx = torch.load(os.path.join(d, "output_tex_slat.pth"), map_location="cpu")
        if self.cond_mode == "zero":
            cond = torch.zeros(1, COND_T, COND_D)
        else:
            cond = torch.load(os.path.join(d, "cond.pth"), map_location="cpu")["cond"]
        mask = torch.load(os.path.join(d, "emis_mask.pth"), map_location="cpu") if self.require_mask else None
        return shp, itx, otx, cond, mask


def ema_update(ema_state, model, decay):
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                ema_state[k].mul_(decay).add_(v.detach(), alpha=1 - decay)
            else:
                ema_state[k].copy_(v)


def save_ckpt(state_dict, path):
    torch.save({"state_dict": OrderedDict([(f"gen3dseg.{k}", v) for k, v in state_dict.items()])}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--init_ckpt", default="full_seg",
                    help="{full_seg,interactive_seg} to resolve via HF hub, or an explicit ckpt path "
                         "to warm-start from. NOTE: every real training run to date used full_seg "
                         "(the old --segvigen_ckpt docstring saying interactive_seg was never true).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n_per_epoch", type=int, default=0, help="0 = all")
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="val", help="split for --val_quick quick-val tracking")
    ap.add_argument("--emis_oversample", action="store_true", default=False,
                    help="weight per-epoch sampling by (emissive_frac+0.1)**oversample_pow to fight class imbalance")
    ap.add_argument("--oversample_pow", type=float, default=1.0,
                    help="sharpen --emis_oversample weights: (emissive_frac+0.1)**P; P>1 = sharper")
    ap.add_argument("--cond", required=True, choices=["real", "zero"],
                    help="explicit — no silent zero-cond fallback")
    ap.add_argument("--pos_weight", type=float, default=5.0,
                    help="per-voxel flow-loss weight w=1+(pos_weight-1)*emis_mask, mean-normalized per "
                         "sample so lr semantics are unchanged. Requires emis_mask.pth (make_emis_mask.py) "
                         "for every training sample UNLESS pos_weight==1.0 (fully off, old unweighted MSE).")
    ap.add_argument("--ema", type=float, default=0.999,
                    help="EMA decay for a shadow copy of the flow weights, saved alongside the regular "
                         "ckpt as epoch_XXXX_ema.ckpt. 0 = off (no EMA file written).")
    ap.add_argument("--val_quick", type=int, default=8,
                    help="after each save, run a quick N-sample val IoU (12-step sampling) on "
                         "--val_split and track best.ckpt + train_curve.json. 0 = off.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    require_mask = args.pos_weight != 1.0

    # model: flow + Gen3DSeg wrapper, warm-started from a SegviGen ckpt
    init_ckpt = resolve_init_ckpt(args.init_ckpt)
    print(f"[init] warm-starting from {init_ckpt}", flush=True)
    flow = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")
    # gradient checkpointing — activations dominate memory for the 1.3B sparse DiT; this
    # lets the full fine-tune fit on a 44GB GPU (l40s/a40).
    n_ckpt = 0
    for m in flow.modules():
        if hasattr(m, "use_checkpoint"):
            m.use_checkpoint = True; n_ckpt += 1
    print(f"[mem] enabled gradient checkpointing on {n_ckpt} modules", flush=True)
    gen = Gen3DSeg(flow).to(device)
    sd = torch.load(init_ckpt, map_location=device)["state_dict"]
    sd = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in sd.items()])
    gen.load_state_dict(sd)
    gen.train()

    ema_state = None
    if args.ema > 0:
        ema_state = {k: v.detach().clone() for k, v in gen.state_dict().items()}
        print(f"[ema] tracking shadow weights, decay={args.ema}", flush=True)

    eval_models = None
    if args.val_quick > 0:
        eval_models = load_eval_models(device)
        print(f"[val_quick] loaded eval decoders for {args.val_quick}-sample quick-val on '{args.val_split}'", flush=True)

    sm, ss, tm, ts = load_norm_stats(device)
    opt = torch.optim.AdamW(gen.parameters(), lr=args.lr, weight_decay=0.0)

    ds = EmisDataset(args.dataset, args.train_split, cond_mode=args.cond, require_mask=require_mask)
    print(f"[data] {len(ds)} samples from '{args.train_split}' (cond={args.cond}, "
          f"oversample={args.emis_oversample}, pos_weight={args.pos_weight})", flush=True)
    samp_w = torch.tensor([(f + 0.1) ** args.oversample_pow for f in ds.fracs]) if args.emis_oversample else None
    log = []
    curve = []
    best_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        n_draw = args.n_per_epoch or len(ds)
        if samp_w is not None:
            idxs = torch.multinomial(samp_w, n_draw, replacement=True).tolist()
        else:
            idxs = torch.randperm(len(ds)).tolist()[:n_draw]
        ep_loss = 0.0
        for j in idxs:
            shp, itx, otx, cond, mask = ds[j]
            coords = shp["coords"].to(device)
            shp_f = (shp["feats"].to(device) - sm) / ss
            itx_f = (itx["feats"].to(device) - tm) / ts
            data  = (otx["feats"].to(device) - tm) / ts
            cond  = cond.to(device)

            noise = torch.randn_like(data)
            t = torch.rand(1, device=device)
            x_t = t * noise + (1 - t) * data
            v_target = noise - data

            x_t_st = sp.SparseTensor(x_t, coords)
            itx_st = sp.SparseTensor(itx_f, coords)
            shp_st = sp.SparseTensor(shp_f, coords)
            t_model = (t * 1000).expand(1)

            v_pred = gen(x_t_st, itx_st, shp_st, t_model, cond, [coords.shape[0]])
            if args.pos_weight != 1.0:
                w = 1 + (args.pos_weight - 1) * mask.to(device)
                w = w / w.mean().clamp(min=1e-8)
                loss = (w[:, None] * (v_pred.feats - v_target) ** 2).mean()
            else:
                loss = nn.functional.mse_loss(v_pred.feats, v_target)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt.step()
            if ema_state is not None:
                ema_update(ema_state, gen, args.ema)
            ep_loss += loss.item()

        ep_loss /= max(1, len(idxs))
        log.append({"epoch": epoch, "loss": ep_loss})
        print(f"epoch {epoch:4d} | flow_loss {ep_loss:.5f}", flush=True)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_ckpt(gen.state_dict(), os.path.join(args.out_dir, "last.ckpt"))
            # ALSO keep per-epoch ckpts (pilot overwrote ep25 which was better than ep50 →
            # overfitting; keep history to find the sweet spot / early-stop).
            ep_path = os.path.join(args.out_dir, f"epoch_{epoch:04d}.ckpt")
            save_ckpt(gen.state_dict(), ep_path)
            if ema_state is not None:
                save_ckpt(ema_state, os.path.join(args.out_dir, f"epoch_{epoch:04d}_ema.ckpt"))
            json.dump(log, open(os.path.join(args.out_dir, "log.json"), "w"), indent=2)

            val_iou = None
            per_sample = None
            if eval_models is not None:
                gen.eval()
                result = evaluate_split(gen, eval_models, args.dataset, args.val_split, args.cond,
                                        device=device, steps=12, thrs=THRS, n=args.val_quick, verbose=False)
                gen.train()
                val_iou = result["best_iou"]
                per_sample = result["per_sample"]
                per_sample_s = " ".join(f"{p['sid'][:8]}={p['best_iou']:.3f}" for p in per_sample)
                print(f"[val_quick] epoch {epoch:4d} | {args.val_split}[:{args.val_quick}] "
                      f"best IoU {val_iou:.4f} @thr={result['best_thr']} | per-sample: {per_sample_s}", flush=True)
                if val_iou > best_iou:
                    best_iou = val_iou
                    best_link = os.path.join(args.out_dir, "best.ckpt")
                    if os.path.islink(best_link) or os.path.exists(best_link):
                        os.remove(best_link)
                    os.symlink(os.path.basename(ep_path), best_link)
                    print(f"[val_quick] new best ({val_iou:.4f}) → best.ckpt -> {os.path.basename(ep_path)}", flush=True)
            curve.append({"epoch": epoch, "train_loss": ep_loss, "val_iou": val_iou, "per_sample": per_sample})
            json.dump(curve, open(os.path.join(args.out_dir, "train_curve.json"), "w"), indent=2)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
