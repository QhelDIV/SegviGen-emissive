"""
PBR-conditioning variants for the emissive fine-tune (2026-08-09).

Two mechanisms for feeding the input PBR tex latent to the flow:

"token" — upstream SegviGen's Gen3DSeg (inference_full.py): the PBR latent is
appended as N extra sequence tokens at the same coords (sequence doubles to 2N),
per-token input [feats 32 | shape 32] = 64ch, output sliced back to the x_t half.
This is what the pretrained full_seg checkpoint and every run before 2026-08 use.

"channel" — the lightgen/TRELLIS.2 emission-DiT convention, adopted for
conditioning parity across the LightGen baselines: the sequence stays N tokens
and the PBR latent is channel-concatenated per token,
    input = [x_t 32 | shape 32 | pbr 32] = 96ch  (concat_cond = cat([shape, pbr]))
exactly mirroring LightGenSLatEmissionLatent + ElasticSLatFlowModel
(skip_pbr_cond=false) in the lightgen TRELLIS2 submodule. The pretrained
input_layer is 64ch, so on warm start it is RE-INITIALIZED (shape mismatch);
all transformer blocks are still loaded — the same regime as lightgen's
emission_dit_pbr2emission / emission_dit_albedo2emission_pbrcond configs.

Checkpoints are self-describing: flow_model.input_layer.weight is
[model_channels, 96] for channel mode and [model_channels, 64] for token mode,
so eval/inference call `detect_pbr_cond(sd)` and need no CLI flag. Training
must pass an explicit --pbr_cond (same no-silent-default ethos as --cond).

Heavy deps (inference_full -> trimesh/o_voxel, trellis2.models) are imported
lazily inside the functions that need them, so detect_pbr_cond() and the
wrapper class stay importable in a bare-torch environment.
"""
import torch
import torch.nn as nn

FLOW_PATH = "microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16"
INPUT_LAYER_KEY = "flow_model.input_layer.weight"
TOKEN_IN, CHANNEL_IN = 64, 96


class Gen3DSegChannelConcat(nn.Module):
    """Drop-in replacement for Gen3DSeg (same forward signature, so Sampler and
    every caller work unchanged): channel-concat conditioning instead of token
    doubling. concat order [shape | pbr] matches lightgen's
    LightGenSLatEmissionLatent (concat_cond = cat([shape_z, pbr_z], dim=-1))."""

    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model

    def forward(self, x_t, tex_slats, shape_slats, t, cond, coords_len_list):
        import trellis2.modules.sparse as sp
        concat_cond = sp.SparseTensor(
            torch.cat([shape_slats.feats, tex_slats.feats], dim=-1), x_t.coords
        )
        return self.flow_model(x_t, t, cond, concat_cond)


def build_flow(pbr_cond):
    """Instantiate the imgshape2tex flow for the given conditioning mode.

    token:   models.from_pretrained as before (in=64, full weight reuse).
    channel: same config with in_channels=96; pretrained tensors whose shape
             changed (input_layer.*) are dropped -> re-initialized, logged.
    """
    from trellis2 import models
    if pbr_cond == "token":
        return models.from_pretrained(FLOW_PATH)
    assert pbr_cond == "channel", f"unknown pbr_cond {pbr_cond!r}"

    # Replicate models.from_pretrained (trellis2/models/__init__.py) with an
    # in_channels override — from_pretrained(**kwargs) would raise "multiple
    # values for keyword argument" since in_channels is already in the config.
    import os, json
    from safetensors.torch import load_file
    if os.path.exists(f"{FLOW_PATH}.json") and os.path.exists(f"{FLOW_PATH}.safetensors"):
        config_file, model_file = f"{FLOW_PATH}.json", f"{FLOW_PATH}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        parts = FLOW_PATH.split("/")
        repo_id, model_name = f"{parts[0]}/{parts[1]}", "/".join(parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")
    with open(config_file) as f:
        config = json.load(f)
    args = dict(config["args"])
    args["in_channels"] = CHANNEL_IN
    flow = getattr(models, config["name"])(**args)

    pre = load_file(model_file)
    own = flow.state_dict()
    dropped = [k for k in pre if k in own and pre[k].shape != own[k].shape]
    for k in dropped:
        del pre[k]
    flow.load_state_dict(pre, strict=False)
    print(f"[pbr_cond=channel] flow in_channels={CHANNEL_IN}; re-initialized "
          f"(64->96 shape mismatch): {dropped or 'none'}", flush=True)
    return flow


def wrap_gen(flow, pbr_cond):
    if pbr_cond == "channel":
        return Gen3DSegChannelConcat(flow)
    from inference_full import Gen3DSeg
    return Gen3DSeg(flow)


def detect_pbr_cond(sd):
    """Infer the conditioning mode from a (gen3dseg.-stripped) checkpoint
    state_dict via the input projection's fan-in. No flag, can't drift."""
    w = sd[INPUT_LAYER_KEY]
    if w.shape[1] == CHANNEL_IN:
        return "channel"
    if w.shape[1] == TOKEN_IN:
        return "token"
    raise ValueError(f"{INPUT_LAYER_KEY} has unexpected fan-in {w.shape[1]} "
                     f"(expected {TOKEN_IN} or {CHANNEL_IN})")


def build_gen_from_sd(sd, device):
    """Eval/inference path: detect the mode from the checkpoint, build the
    matching flow + wrapper, load strictly. Returns (gen, pbr_cond)."""
    pbr_cond = detect_pbr_cond(sd)
    print(f"[pbr_cond] detected '{pbr_cond}' from checkpoint input_layer "
          f"fan-in {sd[INPUT_LAYER_KEY].shape[1]}", flush=True)
    gen = wrap_gen(build_flow(pbr_cond), pbr_cond).to(device)
    gen.load_state_dict(sd)
    return gen, pbr_cond


def load_warmstart(gen, sd, pbr_cond):
    """Training warm start. token: strict load (unchanged behavior). channel:
    the init ckpt (full_seg) carries a 64ch input_layer that cannot fit the
    96ch flow — drop shape-mismatched tensors (they stay at their fresh init)
    and load the rest strictly-by-name. NOTE: plain strict=False would NOT
    survive the shape mismatch (torch raises size-mismatch errors regardless
    of strict), hence the explicit filter."""
    if pbr_cond == "token":
        gen.load_state_dict(sd)
        return
    own = gen.state_dict()
    dropped = [k for k in sd if k in own and sd[k].shape != own[k].shape]
    filtered = {k: v for k, v in sd.items() if k not in dropped}
    missing, unexpected = gen.load_state_dict(filtered, strict=False)
    # every missing key must be one we deliberately dropped; anything else is a
    # real mismatch between wrapper/flow and the checkpoint
    stray = [k for k in missing if k not in dropped]
    if stray or unexpected:
        raise RuntimeError(f"warm start mismatch: missing={stray} unexpected={unexpected}")
    print(f"[pbr_cond=channel] warm start: re-initialized {dropped}; "
          f"loaded {len(filtered)} tensors from init ckpt", flush=True)
