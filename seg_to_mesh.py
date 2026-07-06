"""
Render the full part-segmentation on the ACTUAL MESH SURFACE (paper-style), not as
coarsened voxel cubes. Per shape: sample the pretrained full_seg flow, decode the seg
coloring to 512^3 voxels AND get the decoded surface mesh, then color each mesh vertex
by its nearest seg-voxel and export a vertex-colored GLB.

This avoids the two artifacts of the voxel-cube view: (1) the adaptive coarsening that
merged parts, (2) blocky cubes. The mesh is the res-512 decode, so it's full fidelity.

Usage (GPU node, trellis2 env):
  python seg_to_mesh.py --dataset .../dataset --split canon_overfit10 \
     --seg_ckpt .../full_seg.ckpt --out_glb_dir seg_mesh_canon10 --steps 25
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
import trimesh
from collections import OrderedDict
from scipy.spatial import cKDTree
import trellis2.modules.sparse as sp
from trellis2 import models
from inference_full import Gen3DSeg, Sampler
from huggingface_hub import hf_hub_download


def load_pipeline_args():
    pj = hf_hub_download(repo_id="microsoft/TRELLIS.2-4B", filename="pipeline.json")
    return json.load(open(pj))["args"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="canon_overfit10")
    ap.add_argument("--seg_ckpt", required=True)
    ap.add_argument("--out_glb_dir", required=True)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--zero_cond", action="store_true", default=False)
    args = ap.parse_args()
    device = "cuda"
    COND_T, COND_D = 1024, 1024
    os.makedirs(args.out_glb_dir, exist_ok=True)

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
    for sid in sorted(os.listdir(sdir)):
        d = os.path.join(sdir, sid)
        if not os.path.exists(os.path.join(d, "shape_slat.pth")):
            continue
        shp = torch.load(os.path.join(d, "shape_slat.pth"), map_location=device)
        itx = torch.load(os.path.join(d, "input_tex_slat.pth"), map_location=device)
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
            meshes, subs = shape_decoder(sp.SparseTensor(shp["feats"].to(device), coords), return_subs=True)
            noise = sp.SparseTensor(torch.randn_like(itx_n.feats), coords)
            out = sampler.sample(gen, noise, itx_n, shp_n, [coords.shape[0]], cond_dict, sp_params)
            out = out * ts + tm
            seg_vox = tex_decoder(out, guide_subs=subs) * 0.5 + 0.5

        # decoded mesh (res-512). m.vertices/faces may be CUDA torch tensors → to numpy.
        m = meshes[0]
        def _np(x):
            return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        verts = _np(m.vertices).astype(np.float64)
        faces = _np(m.faces).astype(np.int64)
        seg_rgb = np.clip(seg_vox.feats[:, :3].float().cpu().numpy(), 0, 1)
        vox = seg_vox.coords[:, 1:].cpu().numpy().astype(np.float64)  # voxel grid coords 0..511

        # map mesh verts → nearest seg voxel. Frame-robust: bbox-align mesh verts to the
        # voxel grid extent (mesh & voxels are the same object decoded at the same res, so
        # their bounding boxes correspond) — avoids assuming world-vs-voxel units.
        vlo, vhi = verts.min(0), verts.max(0)
        xlo, xhi = vox.min(0), vox.max(0)
        scale = (xhi - xlo) / np.maximum(vhi - vlo, 1e-9)
        verts_vox = (verts - vlo) * scale + xlo
        tree = cKDTree(vox)
        _, idx = tree.query(verts_vox, k=1)
        vcol = (seg_rgb[idx] * 255).astype(np.uint8)
        vcol = np.concatenate([vcol, np.full((len(vcol), 1), 255, np.uint8)], 1)

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vcol, process=False)
        out_path = os.path.join(args.out_glb_dir, f"{sid}.glb")
        mesh.export(out_path)
        print(f"[ok] {sid}  verts={len(verts)} faces={len(faces)} segvox={len(vox)} -> {out_path}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
