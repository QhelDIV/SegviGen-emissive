"""
Decode ONE emissive prediction to a paper-style GLB mesh (official remesh+bake), so the
2k model's prediction can be shown as a smooth surface instead of coarse voxel cubes.

Reuses eval_emissive.load_eval_models (tex/shape decoders + norm stats + sampler) and the
same sampling path eval_emissive.eval_sample uses (cond real, K independent draws collapse
to one fixed-seed draw here), then runs inference_full.slat_to_glb (o_voxel remesh at
res-512 + 4096^2 base-color bake) on the decoded tex voxels — the official path, identical
to inference_full.inference()'s final two steps.

The predicted base_color is white where emissive, black where not (the fine-tune target is
GT white/black emissive coloring). --thr>=0 hard-thresholds the baked base_color to pure
white/black (crisp, matches the @thr IoU headline metric and GT's look); --thr<0 keeps the
raw continuous prediction. metallic/roughness/alpha are left as decoded.

Runs on a GPU node in the trellis2 env (see emissive/slurm/make_pred_glb.sbatch). Mesh = ONE
representative fixed-seed draw; the captioned IoU on the page is the K=4 average.

  python emissive/eval/make_pred_glb.py --dataset .../dataset --split val_96 \
      --ckpt .../outputs/emis_2k_w5/best.ckpt --sid e9e3... --out .../pred.glb --thr 0.5
"""
import os, sys, argparse
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(ROOT, "inference_full.py")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError(f"could not locate SegviGen repo root (inference_full.py) above {__file__}")
    ROOT = parent   # walk up: this script now lives nested under emissive/eval/, not repo root
SEGVIGEN = ROOT
sys.path.insert(0, SEGVIGEN)
os.environ.setdefault("HF_HOME", "/3dlg-jupiter-project/lightgen/hf_cache")

import json
import torch
import numpy as np
from collections import OrderedDict
import trellis2.modules.sparse as sp
from trellis2 import models
from inference_full import Gen3DSeg, Sampler, slat_to_glb
from eval_emissive import load_eval_models, COND_T, COND_D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val_96")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--out", required=True, help="output glb path")
    ap.add_argument("--cond", default="real", choices=["real", "zero"])
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5,
                    help=">=0: hard-threshold baked base_color to white/black; <0: raw continuous")
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[load] flow model + ckpt {args.ckpt}", flush=True)
    flow = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")
    gen = Gen3DSeg(flow).to(device)
    sd = torch.load(args.ckpt, map_location=device)["state_dict"]
    sd = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in sd.items()])
    gen.load_state_dict(sd); gen.eval()

    md = load_eval_models(device)
    tex_decoder, shape_decoder, sampler = md["tex_decoder"], md["shape_decoder"], md["sampler"]
    sm, ss, tm, ts = md["sm"], md["ss"], md["tm"], md["ts"]
    sp_params = dict(md["pipeline_args"]["tex_slat_sampler"]["params"]); sp_params["steps"] = args.steps

    d = os.path.join(args.dataset, args.split, args.sid)
    shp = torch.load(os.path.join(d, "shape_slat.pth"), map_location=device)
    itx = torch.load(os.path.join(d, "input_tex_slat.pth"), map_location=device)
    coords = shp["coords"].to(device)
    if args.cond == "zero":
        cond = torch.zeros(1, COND_T, COND_D, device=device)
    else:
        cond = torch.load(os.path.join(d, "cond.pth"), map_location=device)["cond"]
    cond_dict = {"cond": cond, "neg_cond": torch.zeros_like(cond)}

    shp_n = sp.SparseTensor((shp["feats"].to(device) - sm) / ss, coords)
    itx_n = sp.SparseTensor((itx["feats"].to(device) - tm) / ts, coords)

    with torch.no_grad():
        shape_decoder.set_resolution(512)
        # meshes = decoded shape surface (res-512); subs = upsampling structure the tex
        # decoder needs. eval_emissive discards `meshes` (IoU only) — we keep it for slat_to_glb.
        meshes, subs = shape_decoder(sp.SparseTensor(shp["feats"].to(device), coords), return_subs=True)
        print(f"[decode] shape mesh: {len(meshes)} mesh(es)", flush=True)
        noise = sp.SparseTensor(torch.randn_like(itx_n.feats), coords)
        out = sampler.sample(gen, noise, itx_n, shp_n, [coords.shape[0]], cond_dict, sp_params)
        out = out * ts + tm
        tex_voxels = tex_decoder(out, guide_subs=subs) * 0.5 + 0.5  # base color in [0,1]-ish

    bc = tex_voxels.feats[:, :3].mean(-1)
    print(f"[pred] base_color mean={bc.mean().item():.3f}  frac>0.5={ (bc>0.5).float().mean().item():.3f}", flush=True)
    if args.thr >= 0:
        new_feats = tex_voxels.feats.clone()
        white = (bc > args.thr).float()[:, None]
        new_feats[:, 0:3] = white  # 1->(1,1,1) white emissive, 0->(0,0,0) black
        tex_voxels = tex_voxels.replace(new_feats)
        print(f"[thr] hard-thresholded base_color at {args.thr}", flush=True)

    print(f"[glb] slat_to_glb (remesh 512 + bake 4096) ...", flush=True)
    glb = slat_to_glb(meshes, tex_voxels)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    glb.export(args.out)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
