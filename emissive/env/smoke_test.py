"""Smoke test: load TRELLIS.2-4B models from HF cache, introspect the flow's cond dim
(for the zero-image-cond workaround), confirm o_voxel + SegviGen ckpt availability."""
import os, sys, glob, json
ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(ROOT, "inference_full.py")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError(f"could not locate SegviGen repo root (inference_full.py) above {__file__}")
    ROOT = parent   # walk up: this script now lives nested under emissive/env/, not repo root
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/3dlg-jupiter-project/lightgen/hf_cache")

import torch
import o_voxel
print("o_voxel OK", flush=True)
from trellis2 import models

flow = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")
print("flow loaded:", type(flow).__name__, flush=True)
for attr in ["cond_channels", "context_dim", "cross_attention_dim", "model_channels",
             "io_channels", "in_channels", "out_channels", "resolution"]:
    if hasattr(flow, attr):
        print(f"  flow.{attr} = {getattr(flow, attr)}", flush=True)
cfg = getattr(flow, "_cfg", getattr(flow, "config", None))
print("  flow config:", json.dumps(cfg) if isinstance(cfg, dict) else str(cfg)[:500], flush=True)

shp = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16")
tex = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16")
texd = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16")
print("encoders+decoder OK", flush=True)

sg = glob.glob("/3dlg-jupiter-project/lightgen/hf_cache/hub/models--fenghora--SegviGen/snapshots/*/*")
print("SegviGen files:", [os.path.basename(x) for x in sg], flush=True)
print("SMOKE_OK", flush=True)
