"""
Agent-facing inference CLI: raw GLB in -> binary emissive-mask prediction out.

Not a training/eval script — no ground truth is required or used. Input can be ANY .glb
(any source, not just our "somage" assets); output is a predicted per-voxel emissive
mask + a decoded mesh, for a downstream agent (or a person) to consume without touching
any of the eval_emissive.py / build_dataset.py internals directly.

Pipeline (every stage below REUSES existing code; see the attribution comment at each
import/lift — nothing here is a silent fork of eval_emissive.py's or make_pred_glb.py's
logic):
  1. glb_to_vxz            (data_toolkit/glb_to_vxz.py, upstream, unmodified)
  2. vxz_to_latent_slat x2 (data_toolkit/vxz_to_slat.py, upstream, unmodified) with
     shape_encoder + tex_encoder, then intersected onto a shared coords set — see
     `_common_coords_2` below, a 2-slat analogue of vxz_to_slat.get_common_coords (that
     helper hardcodes exactly 4 slats for the train-time shape/input_tex/output_tex/
     foreground case; inference has no GT output slat, so we only ever intersect 2).
  3. cond:
       --zero_cond   -> zeros, matching eval_emissive.py's cond_mode="zero"
       --image PATH  -> skip rendering, run img_to_cond.img_to_cond() (data_toolkit,
                        unmodified) directly on the supplied image
       (default)     -> render the glb via data_toolkit/bpy_render.render_from_transforms
                        using the SAME transforms.json build_dataset.py's --real_cond
                        path uses, then img_to_cond.img_to_cond() on that render
  4. load ckpt: eval_emissive.load_eval_models() for the tex/shape decoders + norm
     stats + sampler (exact import, same object every eval/train_emissive quick-val call
     reuses), plus the identical state_dict "gen3dseg."-prefix-strip loading snippet used
     verbatim in eval_emissive.py main() and make_pred_glb.py main().
  5. sample --draws independent draws and average: this loop is lifted from
     eval_emissive.eval_sample()'s draw loop (same noise/sampler/decode calls) with the
     GT-scoring half removed, since there is no GT here.
  6. threshold @ --thr, then decode to mesh via inference_full.slat_to_glb — the same
     function make_pred_glb.py imports and calls, with the identical hard-threshold-to-
     white/black glue (lifted from make_pred_glb.py's main(), see comment at the call).
  7. per-face labels (`_label_mesh_faces`): map the DECODED mesh's faces (pred_mesh.glb) to
     the predicted per-voxel probability/mask by nearest voxel in the shared [-0.5,0.5]^3 /
     512-res frame (glb_to_vxz's aabb == inference_full.slat_to_glb's aabb/origin — a
     mesh-bounds check + a baked-texture-vs-face_mask agreement check are printed at
     runtime as an empirical, not assumed, verification of this: it caught a REAL axis
     mismatch — slat_to_glb's exported mesh isn't in the same axis convention as the raw
     voxel coords, see PRED_MESH_AXES/_SIGNS/_to_voxel_frame for the measured correction
     and emissive/docs/EXPERIMENTS.md for the before/after agreement numbers). Written to
     pred_mesh_labels.npz, always. With --label_input_mesh: the SAME lookup, but against
     the ORIGINAL input glb's faces (reproducing glb_to_vxz.py's own load+normalize so face
     order matches a plain trimesh.load(...).to_mesh() of the user's file — see
     `_normalized_input_mesh`), written to input_mesh_labels.npz plus a per-face-colored
     input_mesh_labeled.glb for eyeballing.

VALIDATION — GPU-tested 2026-07-16 for the base pipeline (job 232600, see
emissive/docs/EXPERIMENTS.md) and again for the per-face mesh-label feature added on top
(see EXPERIMENTS.md's "predict_emissive.py status" for both jobs' pass/fail numbers,
including the texture-vs-face_mask agreement %). `python -m py_compile` clean and `--help`
runs standalone (all heavy imports — torch/trellis2/inference_full/data_toolkit/o_voxel —
are deferred until after argparse). Smoke-test command:

    salloc -p 3dlg-hcvc-lab-debug --gres=gpu:l40s:1 --time=0:30:00 --cpus-per-task=8 --mem=64G
    source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh && conda activate trellis2
    cd <repo root, after rsync>   # emissive/slurm/*.sbatch document this convention
    export HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache
    python emissive/infer/predict_emissive.py --glb assets/example.glb \
        --out /tmp/predict_smoke --draws 4 --thr 0.5 --zero_cond --label_input_mesh
    # then, to also exercise the real-cond render path (needs bpy in the trellis2 env):
    python emissive/infer/predict_emissive.py --glb assets/example.glb \
        --out /tmp/predict_smoke_realcond --draws 4 --thr 0.5

Usage:
  python predict_emissive.py --glb mesh.glb --out out_dir/
  python predict_emissive.py --glb mesh.glb --out out_dir/ --draws 8 --thr 0.4
  python predict_emissive.py --glb mesh.glb --out out_dir/ --image render.png
  python predict_emissive.py --glb mesh.glb --out out_dir/ --zero_cond --ckpt /path/to/other.ckpt
  python predict_emissive.py --glb mesh.glb --out out_dir/ --label_input_mesh
"""
import os
import sys
import json
import time
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(ROOT, "inference_full.py")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError(f"could not locate SegviGen repo root (inference_full.py) above {__file__}")
    ROOT = parent   # walk up: this script lives nested under emissive/infer/, not repo root
SEGVIGEN = ROOT

# Recommended checkpoint per emissive/docs/EXPERIMENTS.md: no fine-tune beats the 0.219
# zero-shot oracle (which is oracle-assisted / not deployable), so this is the best
# *deployable* checkpoint — emis_1k_w5's EMA weights at its best epoch (0.117 nonzero
# IoU, full 111-shape val, 4-draw averaged, @0.5). See EXPERIMENTS.md for the full
# registry and the tie with emis_2k_bal (0.114) that this default breaks.
DEFAULT_CKPT = "/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_1k_w5/epoch_0016_ema.ckpt"
DEFAULT_TRANSFORMS = os.path.join(SEGVIGEN, "data_toolkit", "transforms.json")

# Shared by glb_to_vxz.py's o_voxel.convert.mesh_to_flexible_dual_grid (grid_size=512,
# hardcoded there) and inference_full.slat_to_glb's default `resolution=512` /
# aabb=[[-0.5]*3,[0.5]*3] — every mesh<->voxel lookup in this file assumes this one frame.
VOXEL_RES = 512
VOXEL_AABB = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))


def _common_coords_2(slat_a, slat_b):
    """2-slat analogue of data_toolkit/vxz_to_slat.py's get_common_coords(). That helper
    hardcodes 4 slats (shape/input_tex/output_tex/foreground) for the train-time case
    where a GT output slat exists to intersect against. Inference only ever has
    shape_slat + input_tex_slat (no GT), but the intersection is still required: the
    shape_encoder and tex_encoder can each downsample glb_to_vxz's input voxel grid to a
    different coords set, so shape_slat.coords and input_tex_slat.coords are not
    guaranteed to match without this step."""
    import torch
    xs = [torch.unique(x, dim=0) for x in (slat_a.coords, slat_b.coords)]
    all_coords = torch.cat(xs, dim=0)
    uniq, counts = torch.unique(all_coords, dim=0, return_counts=True)
    return uniq[counts == 2].cuda()


def _load_ckpt(gen, ckpt_path, device):
    """Identical to the state_dict loading in eval_emissive.py main() / make_pred_glb.py
    main() (the trained module is wrapped as `gen3dseg.<name>` by the standalone training
    loop's Lightning-style checkpointing; strip that prefix before load_state_dict)."""
    from collections import OrderedDict
    import torch
    sd = torch.load(ckpt_path, map_location=device)["state_dict"]
    sd = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in sd.items()])
    gen.load_state_dict(sd)
    gen.eval()


def _render_cond_image(glb_path, out_img_path, transforms_json):
    """Default (no --zero_cond, no --image) conditioning path: render the glb the same
    way build_dataset.py's --real_cond does (data_toolkit/bpy_render.render_from_transforms
    + transforms.json), for use as the DINOv3 image condition. Imports bpy lazily — only
    needed on this path, not for --zero_cond / --image."""
    sys.path.insert(0, os.path.join(SEGVIGEN, "data_toolkit"))
    from bpy_render import render_from_transforms
    render_from_transforms(glb_path, transforms_json, out_img_path)


def _as_single_mesh(glb_obj):
    """inference_full.slat_to_glb / o_voxel.postprocess.to_glb returns a single baked mesh
    in practice (one UV atlas from `to_glb`'s xatlas pass), but trimesh represents a
    just-loaded/just-built glb as either a Trimesh or a Scene depending on the exporter —
    normalize to a plain Trimesh so face-label code doesn't need to special-case both."""
    import trimesh
    if isinstance(glb_obj, trimesh.Trimesh):
        return glb_obj
    geoms = list(glb_obj.geometry.values())
    if len(geoms) != 1:
        print(f"[warn] expected 1 geometry from slat_to_glb, got {len(geoms)} — using the first", flush=True)
    return geoms[0]


def _check_frame(lo, hi, tag):
    """Empirical (not assumed) check that a mesh's vertex bounds (`lo`, `hi`) actually sit
    in VOXEL_AABB — per owner request, since the voxel-lookup in _label_mesh_faces silently
    produces garbage if this frame assumption is wrong. Prints bounds; warns (does not
    raise) if they exceed the expected box by more than a hair. Bounds-matching alone is
    NOT sufficient to confirm axis correspondence, though — see PRED_MESH_AXES/_SIGNS and
    _texture_face_agreement for the check that actually caught the real mismatch below."""
    import numpy as np
    print(f"[frame check:{tag}] bounds = {lo} .. {hi}  (expect within {VOXEL_AABB})", flush=True)
    if (np.asarray(lo) < -0.51).any() or (np.asarray(hi) > 0.51).any():
        print(f"[WARN] {tag} bounds exceed the expected {VOXEL_AABB} frame by >0.01 — "
              f"the voxel-lookup labels below may be WRONG. Investigate before trusting them.",
              flush=True)


# ANCHORED ANALYTICALLY, not just fit to one shape (2026-07-16, see EXPERIMENTS.md): found
# empirically first (bbox-extent mismatch + a 48-candidate permutation/sign search scored
# against the mesh's own baked texture — see the validation record), then located and
# confirmed in the actual code that causes it. o_voxel.postprocess.to_glb -- the function
# inference_full.slat_to_glb calls to produce pred_mesh.glb -- ends with this exact,
# unconditional (every shape, not shape-dependent) coordinate-system conversion, in the
# installed package at .../site-packages/o_voxel/postprocess.py:
#
#   o_voxel/postprocess.py:312  # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
#   o_voxel/postprocess.py:313  vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
#   o_voxel/postprocess.py:314  normals_np[:, 1], normals_np[:, 2]  = normals_np[:, 2], -normals_np[:, 1]
#   o_voxel/postprocess.py:315  uvs_np[:, 1] = 1 - uvs_np[:, 1]  # Flip UV V-coordinate
#
# i.e. FORWARD (internal/voxel frame -> exported pred_mesh.glb frame): (x,y,z) -> (x,z,-y).
# The inverse (exported -> internal, which is what we need: out_coords/glb_to_vxz's frame
# never goes through this conversion, since it voxelizes the already-Y-up INPUT glb
# directly) is, solving X=x,Y=z,Z=-y for (x,y,z): (x,y,z) = (X,-Z,Y) — EXACTLY the
# `vertices[:, (0,2,1)] * (1,-1,1)` correction below. Baked-texture-vs-face_mask agreement
# (job 232608): 0.428 uncorrected -> 0.9413 corrected — matches this derivation, not a
# coincidental fit (the same source lines' UV V-flip also independently confirms the
# `1 - v` convention used in _texture_face_agreement below). Needed ONLY for pred_mesh (the
# slat_to_glb/to_glb output) — NOT for _normalized_input_mesh, which voxelizes the same
# Y-up input directly and never goes through to_glb's conversion (measured 99.8%
# exact-voxel-hit rate with no correction).
PRED_MESH_AXES = (0, 2, 1)
PRED_MESH_SIGNS = (1.0, -1.0, 1.0)


def _to_voxel_frame(vertices):
    """Apply the pred_mesh -> out_coords axis correction documented above. Only call this
    on the DECODED mesh (pred_mesh.glb / slat_to_glb's output); the input mesh needs no
    correction."""
    import numpy as np
    return vertices[:, PRED_MESH_AXES] * np.array(PRED_MESH_SIGNS)


def _label_mesh_faces(vertices, faces, voxel_coords, voxel_prob, voxel_mask, resolution=VOXEL_RES):
    """Map each mesh face to the predicted per-voxel probability/mask. Assumes `vertices`
    live in VOXEL_AABB at `resolution` — see `_check_frame`, called by the caller before
    this.

    Voxel-lookup rule: for each face, take its centroid (mean of its 3 vertices), convert
    to a voxel index `idx = floor((centroid + 0.5) * resolution)` clipped to
    `[0, resolution-1]` (voxel i's CENTER sits at `(i+0.5)/resolution - 0.5`, the same
    convention glb_to_vxz/slat_to_glb use), then look up that EXACT voxel index in the
    predicted voxel set (`voxel_coords`). Faces whose centroid voxel has no prediction —
    expected for faces from Dual Contouring / mesh simplification, whose vertices don't
    sit exactly on the dense grid the flow was sampled on — fall back to the nearest
    predicted voxel CENTER by Euclidean distance (cKDTree, unlimited radius: every face
    gets an answer). Returns (face_prob (F,) float32, face_mask (F,) bool, n_exact,
    n_fallback) — n_fallback is a diagnostic, not an error signal by itself."""
    import numpy as np
    centroids = vertices[faces].mean(axis=1)
    idx = np.clip(np.floor((centroids + 0.5) * resolution).astype(np.int64), 0, resolution - 1)

    def _key(c):
        c = c.astype(np.int64)
        return (c[:, 0] * resolution + c[:, 1]) * resolution + c[:, 2]

    vox_keys = _key(voxel_coords)
    order = np.argsort(vox_keys)
    vox_keys_sorted = vox_keys[order]
    face_keys = _key(idx)
    pos = np.clip(np.searchsorted(vox_keys_sorted, face_keys), 0, len(vox_keys_sorted) - 1)
    hit = vox_keys_sorted[pos] == face_keys

    face_prob = np.zeros(len(faces), dtype=np.float32)
    face_mask = np.zeros(len(faces), dtype=bool)
    orig_idx = order[pos[hit]]
    face_prob[hit] = voxel_prob[orig_idx]
    face_mask[hit] = voxel_mask[orig_idx]

    n_fallback = int((~hit).sum())
    if n_fallback > 0:
        from scipy.spatial import cKDTree   # available in the trellis2 env (verified)
        voxel_centers = (voxel_coords.astype(np.float64) + 0.5) / resolution - 0.5
        tree = cKDTree(voxel_centers)
        miss = np.where(~hit)[0]
        _, nn = tree.query(centroids[miss])
        face_prob[miss] = voxel_prob[nn]
        face_mask[miss] = voxel_mask[nn]

    return face_prob, face_mask, int(hit.sum()), n_fallback


def _texture_face_agreement(mesh, face_mask):
    """Diagnostic only (not saved to disk): independently samples `mesh`'s baked
    base_color texture at each face's UV centroid and compares "is it white" against
    `face_mask`, as an empirical cross-check of _label_mesh_faces's frame assumption
    (rather than trusting it) — per owner request. Returns the agreement fraction, or
    None if the mesh has no UV/texture to check against."""
    import numpy as np
    vis = mesh.visual
    uv = getattr(vis, "uv", None)
    material = getattr(vis, "material", None)
    img = None
    if material is not None:
        img = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if uv is None or img is None:
        return None
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    face_uv = uv[mesh.faces].mean(axis=1)
    px = np.clip((face_uv[:, 0] * (w - 1)).astype(np.int64), 0, w - 1)
    py = np.clip(((1 - face_uv[:, 1]) * (h - 1)).astype(np.int64), 0, h - 1)   # glTF V is flipped vs image
    # rows -- confirmed against o_voxel/postprocess.py:315 (`uvs_np[:,1] = 1-uvs_np[:,1]`),
    # the exact source of pred_mesh.glb's UV convention (see PRED_MESH_AXES/_SIGNS above)
    sampled = arr[py, px].mean(axis=1)
    tex_white = sampled > 0.5
    return float((tex_white == face_mask).mean())


def _normalized_input_mesh(glb_path):
    """Reproduces data_toolkit/glb_to_vxz.py's glb_to_vxz() load+normalize EXACTLY (its
    `asset = trimesh.load(...); center/scale by bounding_box.bounds; to_mesh()` block) —
    lifted rather than imported because glb_to_vxz() only writes a .vxz file and doesn't
    return the intermediate mesh we need here. Returns a mesh in the same VOXEL_AABB frame
    voxel lookups use, so its face order/positions match a plain
    `trimesh.load(glb_path).to_mesh()` of the same file (asserted by the caller against a
    second, untransformed load, since --label_input_mesh writes labels onto that raw mesh
    by face index)."""
    import trimesh
    asset = trimesh.load(glb_path, force="scene")
    aabb = asset.bounding_box.bounds
    center = (aabb[0] + aabb[1]) / 2
    scale = 0.99999 / (aabb[1] - aabb[0]).max()
    asset.apply_translation(-center)
    asset.apply_scale(scale)
    return asset.to_mesh()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glb", required=True, help="input mesh, any source (not necessarily a somage asset)")
    ap.add_argument("--out", required=True, help="output dir: mask.npz, pred_mesh.glb, meta.json")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                     help=f"fine-tuned checkpoint (default: recommended in EXPERIMENTS.md, "
                          f"emis_1k_w5 best-EMA) [{DEFAULT_CKPT}]")
    ap.add_argument("--draws", type=int, default=4,
                     help="independent flow-matching samples to average (K). K=1 is noisy "
                          "(draw-std ~0.09 IoU observed in eval); K=4 matches the headline "
                          "eval protocol in EXPERIMENTS.md.")
    ap.add_argument("--thr", type=float, default=0.5, help="probability threshold for the binary mask")
    ap.add_argument("--steps", type=int, default=12, help="flow sampler steps (matches eval_emissive.py default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zero_cond", action="store_true", default=False,
                     help="skip DINOv3 conditioning entirely (cond=zeros), matching "
                          "eval_emissive.py's --cond zero. Fastest; use for a first smoke test.")
    ap.add_argument("--image", default=None,
                     help="use this image as the DINOv3 condition instead of rendering --glb "
                          "(skips the bpy render step; still runs rembg+DINOv3 via img_to_cond.py)")
    ap.add_argument("--transforms", default=DEFAULT_TRANSFORMS,
                     help="camera transforms.json for the render-from-glb cond path (default: "
                          "the same file build_dataset.py's --real_cond uses)")
    ap.add_argument("--label_input_mesh", action="store_true", default=False,
                     help="also write input_mesh_labels.npz (per-face prob/mask over the "
                          "ORIGINAL --glb's faces, same voxel-lookup as pred_mesh_labels.npz) "
                          "and input_mesh_labeled.glb (the original geometry, face-colored "
                          "white/black for eyeballing). Off by default (extra trimesh loads).")
    args = ap.parse_args()

    if args.zero_cond and args.image:
        ap.error("--zero_cond and --image are mutually exclusive")

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    # --- heavy imports deferred to here so `--help` above never needs torch/trellis2/o_voxel ---
    sys.path.insert(0, SEGVIGEN)
    sys.path.insert(0, os.path.join(SEGVIGEN, "data_toolkit"))
    sys.path.insert(0, os.path.join(ROOT, "emissive", "eval"))  # sibling dir holding eval_emissive.py
    os.environ.setdefault("HF_HOME", "/3dlg-jupiter-project/lightgen/hf_cache")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    import numpy as np
    import trellis2.modules.sparse as sp
    from trellis2 import models
    from inference_full import Gen3DSeg, Sampler, slat_to_glb   # reuse: same import make_pred_glb.py does
    from eval_emissive import load_eval_models, COND_T, COND_D  # reuse: shared decoder/norm-stat loader
    from glb_to_vxz import glb_to_vxz
    from vxz_to_slat import vxz_to_latent_slat, get_slat_by_common_coords

    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- 1. glb -> vxz -> shape_slat / input_tex_slat (steps 1-2 in the module docstring) ---
    vxz_path = os.path.join(args.out, "input.vxz")
    print(f"[glb_to_vxz] {args.glb} -> {vxz_path}", flush=True)
    glb_to_vxz(args.glb, vxz_path)

    shape_encoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16").to(device).eval()
    tex_encoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16").to(device).eval()
    shape_slat_o, tex_slat_o = vxz_to_latent_slat(shape_encoder, tex_encoder, vxz_path)
    common_coords = _common_coords_2(shape_slat_o, tex_slat_o)
    shape_slat = get_slat_by_common_coords(shape_slat_o, common_coords)
    input_tex_slat = get_slat_by_common_coords(tex_slat_o, common_coords)
    coords = shape_slat.coords
    print(f"[slat] {coords.shape[0]} common voxels", flush=True)

    # --- 2. cond (step 3) ---
    if args.zero_cond:
        cond = torch.zeros(1, COND_T, COND_D, device=device)
        print("[cond] zero", flush=True)
    else:
        from img_to_cond import img_to_cond
        from trellis2.pipelines.rembg import BiRefNet
        from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
        img_path = args.image
        if img_path is None:
            img_path = os.path.join(args.out, "cond_render.png")
            print(f"[cond] rendering {args.glb} -> {img_path}", flush=True)
            _render_cond_image(args.glb, img_path, args.transforms)
        else:
            print(f"[cond] using supplied image {img_path}", flush=True)
        rembg = BiRefNet(model_name="briaai/RMBG-2.0").to(device)
        dino = DinoV3FeatureExtractor(model_name="facebook/dinov3-vitl16-pretrain-lvd1689m").to(device)
        cond_path = os.path.join(args.out, "cond.pth")
        img_to_cond(rembg, dino, img_path, cond_path)   # reuse: identical to build_dataset.py's real_cond call
        cond = torch.load(cond_path, map_location=device)["cond"]
    cond_dict = {"cond": cond, "neg_cond": torch.zeros_like(cond)}

    # --- 3. load fine-tuned model (step 4) ---
    print(f"[ckpt] {args.ckpt}", flush=True)
    flow = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")
    gen = Gen3DSeg(flow).to(device)
    _load_ckpt(gen, args.ckpt, device)

    md = load_eval_models(device)   # reuse: eval_emissive.load_eval_models — decoders + norm stats + sampler
    tex_decoder, shape_decoder, sampler = md["tex_decoder"], md["shape_decoder"], md["sampler"]
    sm, ss, tm, ts = md["sm"], md["ss"], md["tm"], md["ts"]
    sp_params = dict(md["pipeline_args"]["tex_slat_sampler"]["params"]); sp_params["steps"] = args.steps

    shp_n = sp.SparseTensor((shape_slat.feats.to(device) - sm) / ss, coords.to(device))
    itx_n = sp.SparseTensor((input_tex_slat.feats.to(device) - tm) / ts, coords.to(device))

    # --- 4. sample K draws + decode (steps 5-6), loop lifted from eval_emissive.eval_sample() ---
    with torch.no_grad():
        shape_decoder.set_resolution(512)
        meshes, subs = shape_decoder(sp.SparseTensor(shape_slat.feats.to(device), coords.to(device)), return_subs=True)

    draw_probs = []
    for k in range(args.draws):
        with torch.no_grad():
            noise = sp.SparseTensor(torch.randn_like(itx_n.feats), coords.to(device))
            out = sampler.sample(gen, noise, itx_n, shp_n, [coords.shape[0]], cond_dict, sp_params)
            out = out * ts + tm
            pred_vox = tex_decoder(out, guide_subs=subs) * 0.5 + 0.5   # base color in [0,1]-ish
        draw_probs.append(pred_vox.feats[:, :3].mean(-1).cpu())
        print(f"[draw {k + 1}/{args.draws}] mean prob={draw_probs[-1].mean().item():.3f}", flush=True)
    prob = torch.stack(draw_probs, dim=0).mean(0)   # average across draws
    mask = prob > args.thr

    out_coords = pred_vox.coords[:, 1:].cpu().numpy().astype(np.int32)
    np.savez_compressed(
        os.path.join(args.out, "mask.npz"),
        coords=out_coords,
        prob=prob.numpy().astype(np.float32),
        mask=mask.numpy().astype(bool))
    print(f"[mask.npz] {out_coords.shape[0]} voxels, {mask.float().mean().item():.3f} frac emissive", flush=True)

    # Decode to mesh: identical hard-threshold-then-bake path as make_pred_glb.py main()
    # (lifted with attribution; slat_to_glb itself is imported, not reimplemented).
    new_feats = pred_vox.feats.clone()
    white = mask.to(new_feats.device).float()[:, None]
    new_feats[:, 0:3] = white
    pred_vox_thr = pred_vox.replace(new_feats)
    print("[glb] slat_to_glb (remesh 512 + bake 4096) ...", flush=True)
    glb = slat_to_glb(meshes, pred_vox_thr)
    glb_path = os.path.join(args.out, "pred_mesh.glb")
    glb.export(glb_path)

    # --- 5. per-face labels on the decoded mesh (step 7 in the module docstring) ---
    voxel_prob_np = prob.numpy().astype(np.float32)
    voxel_mask_np = mask.numpy()
    mesh_pred = _as_single_mesh(glb)
    _check_frame(*mesh_pred.bounds, "pred_mesh (raw slat_to_glb output)")
    # slat_to_glb's exported vertices are NOT in the same axis convention as out_coords —
    # see PRED_MESH_AXES/_SIGNS above for the empirical evidence and the fix.
    pred_vertices_voxel_frame = _to_voxel_frame(mesh_pred.vertices)
    _check_frame(pred_vertices_voxel_frame.min(0), pred_vertices_voxel_frame.max(0),
                 "pred_mesh (axis-corrected, used for the lookup below)")
    pred_face_prob, pred_face_mask, pred_n_exact, pred_n_fallback = _label_mesh_faces(
        pred_vertices_voxel_frame, mesh_pred.faces, out_coords, voxel_prob_np, voxel_mask_np)
    print(f"[pred_mesh_labels] {len(pred_face_mask)} faces, {pred_n_exact} exact / "
          f"{pred_n_fallback} nearest-fallback voxel lookups; face frac emissive="
          f"{pred_face_mask.mean():.3f} (voxel frac={voxel_mask_np.mean():.3f})", flush=True)
    agreement = _texture_face_agreement(mesh_pred, pred_face_mask)
    if agreement is not None:
        verdict = "PASS" if agreement >= 0.9 else "CHECK FRAME/LOOKUP"
        print(f"[pred_mesh_labels] texture-vs-face_mask agreement = {agreement:.4f} ({verdict})", flush=True)
    else:
        print("[pred_mesh_labels] no baked texture/UV on decoded mesh — skipped agreement check", flush=True)
    np.savez_compressed(
        os.path.join(args.out, "pred_mesh_labels.npz"),
        face_prob=pred_face_prob, face_mask=pred_face_mask,
        n_faces=np.int64(len(pred_face_mask)), thr=np.float32(args.thr),
        resolution=np.int64(VOXEL_RES), aabb=np.array(VOXEL_AABB, dtype=np.float32))

    outputs = {"mask": "mask.npz", "mesh": "pred_mesh.glb", "mesh_labels": "pred_mesh_labels.npz"}

    # --- 6. optional: labels on the ORIGINAL input mesh's faces ---
    input_mesh_diag = None
    if args.label_input_mesh:
        import trimesh
        mesh_norm = _normalized_input_mesh(args.glb)
        mesh_raw = trimesh.load(args.glb, force="scene").to_mesh()
        assert len(mesh_norm.faces) == len(mesh_raw.faces), (
            f"face count mismatch between normalized ({len(mesh_norm.faces)}) and raw "
            f"({len(mesh_raw.faces)}) loads of {args.glb} — can't attach labels to the "
            f"original mesh by face index (see _normalized_input_mesh docstring)")
        _check_frame(*mesh_norm.bounds, "input_mesh(normalized)")
        # NO axis correction here -- _normalized_input_mesh voxelizes the same Y-up input
        # glb_to_vxz uses, so it already shares out_coords' frame (measured 99.8% exact-hit
        # rate with no correction; see PRED_MESH_AXES/_SIGNS comment above for contrast).
        in_face_prob, in_face_mask, in_n_exact, in_n_fallback = _label_mesh_faces(
            mesh_norm.vertices, mesh_norm.faces, out_coords, voxel_prob_np, voxel_mask_np)
        print(f"[input_mesh_labels] {len(in_face_mask)} faces, {in_n_exact} exact / "
              f"{in_n_fallback} nearest-fallback voxel lookups; face frac emissive="
              f"{in_face_mask.mean():.3f}", flush=True)
        np.savez_compressed(
            os.path.join(args.out, "input_mesh_labels.npz"),
            face_prob=in_face_prob, face_mask=in_face_mask,
            n_faces=np.int64(len(in_face_mask)), thr=np.float32(args.thr),
            resolution=np.int64(VOXEL_RES), aabb=np.array(VOXEL_AABB, dtype=np.float32))

        face_colors = np.zeros((len(in_face_mask), 4), dtype=np.uint8)
        face_colors[in_face_mask] = (255, 255, 255, 255)
        face_colors[~in_face_mask] = (0, 0, 0, 255)
        mesh_raw.visual = trimesh.visual.ColorVisuals(mesh_raw, face_colors=face_colors)
        mesh_raw.export(os.path.join(args.out, "input_mesh_labeled.glb"))

        outputs["input_mesh_labels"] = "input_mesh_labels.npz"
        outputs["input_mesh_labeled"] = "input_mesh_labeled.glb"
        input_mesh_diag = {"n_faces": int(len(in_face_mask)), "n_exact": in_n_exact,
                            "n_fallback": in_n_fallback, "frac_emissive": float(in_face_mask.mean())}

    meta = {
        "glb_in": os.path.abspath(args.glb),
        "ckpt": args.ckpt,
        "draws": args.draws,
        "thr": args.thr,
        "steps": args.steps,
        "seed": args.seed,
        "cond_mode": "zero" if args.zero_cond else ("image" if args.image else "render"),
        "n_voxels": int(out_coords.shape[0]),
        "frac_emissive": float(mask.float().mean().item()),
        "pred_mesh_labels": {"n_faces": int(len(pred_face_mask)), "n_exact": pred_n_exact,
                             "n_fallback": pred_n_fallback, "frac_emissive": float(pred_face_mask.mean()),
                             "texture_agreement": agreement},
        "input_mesh_labels": input_mesh_diag,
        "elapsed_sec": round(time.time() - t0, 1),
        "outputs": outputs,
    }
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    print(f"[done] {args.out} ({meta['elapsed_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
