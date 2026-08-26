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

Multi-GPU (single node, PyTorch DDP) -- see emissive/slurm/README_DDP.md:
  torchrun --standalone --nproc_per_node 4 train_emissive.py <same flags>
Launched without torchrun, WORLD_SIZE is unset and the script takes the exact
single-GPU path it always took: no process group, no DDP wrapper, no reseeding.
"""
import os, sys, json, argparse, glob, time, contextlib, re, threading, queue, socket
from datetime import timedelta
ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(ROOT, "inference_full.py")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError(f"could not locate SegviGen repo root (inference_full.py) above {__file__}")
    ROOT = parent   # walk up: this script now lives nested under emissive/train/, not repo root
SEGVIGEN = ROOT
sys.path.insert(0, SEGVIGEN)
sys.path.insert(0, os.path.join(ROOT, "emissive", "eval"))  # sibling dir holding eval_emissive.py
os.environ.setdefault("HF_HOME", "/3dlg-jupiter-project/lightgen/hf_cache")

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from collections import OrderedDict
import trellis2.modules.sparse as sp
from trellis2 import models
from inference_full import Gen3DSeg            # reuse the exact wrapper
from huggingface_hub import hf_hub_download
from eval_emissive import load_eval_models, evaluate_split, THRS
from fsprobe import probe_exists, new_stats, summary_line   # shared with the eval path


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


# ----------------------------------------------------------------------------- DDP
def dist_setup():
    """torchrun sets RANK/WORLD_SIZE/LOCAL_RANK. Without it WORLD_SIZE is unset, we
    report world_size 1, and every dist branch below is skipped -- the single-GPU
    path stays byte-for-byte what it was. Timeout is generous because rank 0 runs
    quick-val alone while the other ranks sit in a barrier."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)   # makes plain "cuda" mean this rank's GPU
    # device_id binds the group to this rank's GPU up front. Without it, torch warns
    # that "devices used by this process are currently unknown" and guesses at
    # barrier time, which is a documented way to deadlock if the guess is wrong.
    dist.init_process_group("nccl", timeout=timedelta(minutes=180),
                            device_id=torch.device(f"cuda:{local_rank}"))
    return True, rank, world, local_rank


def dist_barrier(is_dist):
    if is_dist:
        dist.barrier()


class FlowStep(nn.Module):
    """Thin adapter with a plain-tensor boundary: tensors in, one tensor out.

    DDP inspects the module's output (to build the autograd graph entry point) and
    SparseTensor is not something it can traverse, so the SparseTensor packing has
    to live INSIDE the wrapped module. The op sequence is identical to what the
    training loop used to do inline, so the single-GPU numbers are unchanged; the
    single-GPU path calls this same adapter directly, without the DDP wrapper."""

    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def forward(self, x_t, itx_f, shp_f, coords, t_model, cond):
        x_t_st = sp.SparseTensor(x_t, coords)
        itx_st = sp.SparseTensor(itx_f, coords)
        shp_st = sp.SparseTensor(shp_f, coords)
        out = self.gen(x_t_st, itx_st, shp_st, t_model, cond, [coords.shape[0]])
        return out.feats


class EmisDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root, split, cond_mode, require_mask=True, prebuilt=None,
                 cond_file="cond.pth"):
        assert cond_mode in ("real", "zero")
        self.cond_mode = cond_mode
        self.require_mask = require_mask
        # cond_file: see --cond_file help. The scan below still checks for cond.pth
        # (the original, authoritative file) regardless of this -- a sample is only
        # ever admitted/rejected on that check, so --cond_file never changes which
        # shapes train, only which bytes get read for a shape that already qualifies.
        self.cond_file = cond_file
        self.dirs = []
        self.fracs = []   # per-sample emissive fraction (for class-imbalance oversampling)
        if prebuilt is not None:
            # ranks > 0 take rank 0's scan result. Scanning 60k+ sample dirs means
            # ~250k NFS stat calls; doing that once per rank would hammer the share
            # for no new information (every rank would build the identical list).
            self.dirs, self.fracs = prebuilt
            return
        sdir = os.path.join(dataset_root, split)
        core = ["shape_slat.pth", "input_tex_slat.pth", "output_tex_slat.pth"]
        # every os.path.exists below is probe_exists instead; nothing else about the
        # scan changes, so which shapes get admitted and in what order is untouched
        # except that a shape is no longer lost to a transient lookup miss
        stats = new_stats()
        for sid in sorted(os.listdir(sdir)):
            d = os.path.join(sdir, sid)
            if not all(probe_exists(os.path.join(d, f), stats) for f in core):
                continue
            # No silent fallback: a sample missing cond.pth under --cond real, or
            # missing emis_mask.pth while pos_weight is active, is a hard error — it
            # means the dataset build is incomplete, not something to quietly skip.
            if cond_mode == "real" and not probe_exists(os.path.join(d, "cond.pth"), stats):
                raise RuntimeError(f"--cond real but cond.pth missing for {d} "
                                   f"(run build_dataset.py --real_cond, or use --cond zero)")
            if require_mask and not probe_exists(os.path.join(d, "emis_mask.pth"), stats):
                raise RuntimeError(f"emis_mask.pth missing for {d} "
                                   f"(run make_emis_mask.py, or pass --pos_weight 1.0 to disable weighting)")
            self.dirs.append(d)
            mp = os.path.join(d, "meta.json")
            # probed too: a transient miss here would quietly set emissive_frac to 0
            # and mis-weight the sample under --emis_oversample
            fr = json.load(open(mp)).get("emissive_frac", 0.0) if probe_exists(mp, stats) else 0.0
            self.fracs.append(float(fr))
        print(f"[data] scan of '{split}': {len(self.dirs)} admitted. "
              + summary_line(stats, "scan"), flush=True)

    def __len__(self): return len(self.dirs)

    def __getitem__(self, i):
        d = self.dirs[i]
        shp = torch.load(os.path.join(d, "shape_slat.pth"), map_location="cpu")
        itx = torch.load(os.path.join(d, "input_tex_slat.pth"), map_location="cpu")
        otx = torch.load(os.path.join(d, "output_tex_slat.pth"), map_location="cpu")
        if self.cond_mode == "zero":
            cond = torch.zeros(1, COND_T, COND_D)
        else:
            cond = self._load_cond(d)
        mask = torch.load(os.path.join(d, "emis_mask.pth"), map_location="cpu") if self.require_mask else None
        return shp, itx, otx, cond, mask

    def _load_cond(self, d):
        """cond.pth (the original, --build_dataset.py --real_cond output) is a dict
        with BOTH halves of a classifier-free-guidance pair: {"cond": ..., "neg_cond":
        ...}, each (1,1029,1024) fp32 -- ~4.2MB apiece, ~8.4MB/file measured. Training
        only ever reads ["cond"]; the neg half is fp32 bytes pulled over NFS and
        immediately discarded, every sample, every epoch. --cond_file names a slim
        per-sample sidecar (built once by build_cond_pos.py) holding just that tensor;
        when it exists for THIS sample we read ~half the bytes. Falls back to cond.pth
        per-sample so a partially-converted split still trains correctly, just without
        the saving on the unconverted shapes -- never a hard requirement, never a
        silent behavior change to what tensor gets used."""
        if self.cond_file != "cond.pth":
            slim_path = os.path.join(d, self.cond_file)
            if os.path.isfile(slim_path):
                payload = torch.load(slim_path, map_location="cpu")
                return payload["cond"] if isinstance(payload, dict) else payload
        return torch.load(os.path.join(d, "cond.pth"), map_location="cpu")["cond"]


_STREAM_DONE = object()


class _Prefetcher:
    """One background thread walks `indices` through `ds[idx]` IN ORDER, buffering
    up to `depth` results ahead in a FIFO queue. This is the whole mechanism -- no
    reordering, no resampling, no second thread -- so the sequence of (index,
    sample) pairs the training loop sees is bit-for-bit the sequence it would see
    calling ds[idx] synchronously in the same order; only the WHEN changes (the
    read for step k+1 starts while step k is still computing on GPU, instead of
    after it finishes). That is what makes parity provable by construction rather
    than by measurement: the loop's consumption order is a static list computed
    once per epoch (idx_seq) before either code path runs.

    A plain thread + queue.Queue instead of torch.utils.data.DataLoader(num_workers)
    on purpose: DataLoader's worker processes fork a *new* CUDA-free process for
    each worker, which is fine for the pattern it was designed for (batch collation)
    but complicates: (a) SparseTensor dict-of-dict samples don't have a canonical
    collate_fn here since batch size is 1 shape per step already, (b) the multi-GPU
    path already has one process per rank; adding DataLoader workers on top adds a
    second process-count knob whose interaction with the DDP world hasn't been
    scoped. A thread reading files is enough: torch.load spends its time in the
    NFS read and pickle deserialization, both of which release the GIL, so a single
    background thread genuinely overlaps with this rank's GPU-bound main thread."""

    def __init__(self, ds, indices, depth):
        self.exc = None
        self._q = queue.Queue(maxsize=max(1, depth))
        self._thread = threading.Thread(
            target=self._worker, args=(ds, indices), daemon=True)
        self._thread.start()

    def _worker(self, ds, indices):
        try:
            for idx in indices:
                self._q.put((idx, ds[idx]))
        except BaseException as e:
            self.exc = e
        finally:
            self._q.put(_STREAM_DONE)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is _STREAM_DONE:
                break
            yield item
        if self.exc is not None:
            raise self.exc


def make_sample_stream(ds, idx_seq, depth):
    """Yields (index, ds[index]) for every index in idx_seq, IN ORDER.

    depth <= 0 (the default, --prefetch 0): synchronous, calls ds[idx] one at a
    time exactly as the training loop always did -- identical code path, so this
    is a strict no-op for anyone not passing --prefetch.
    depth > 0: overlaps those same calls, in the same order, with a background
    thread (see _Prefetcher); the training loop still consumes them strictly in
    idx_seq order, so which shape trains on which step is unaffected."""
    if depth <= 0:
        for idx in idx_seq:
            yield idx, ds[idx]
        return
    yield from _Prefetcher(ds, idx_seq, depth)


def ema_update(ema_state, model, decay):
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                ema_state[k].mul_(decay).add_(v.detach(), alpha=1 - decay)
            else:
                ema_state[k].copy_(v)


def save_ckpt(state_dict, path):
    torch.save({"state_dict": OrderedDict([(f"gen3dseg.{k}", v) for k, v in state_dict.items()])}, path)


# ---------------------------------------------------------- optimizer-state resume
STATE_FORMAT = 1


def state_path_for(ckpt_path):
    """The state sidecar that belongs to a weights checkpoint.

    Sidecar rather than extra keys inside the .ckpt, for two reasons. The weights
    file stays BYTE-IDENTICAL in format, so every old checkpoint keeps loading and
    eval_emissive/inference are untouched by construction rather than by care. And
    AdamW state is roughly twice the parameter bytes (~4.9 GiB here against 2.4 GiB
    of weights), so folding it in would have tripled the size of every per-epoch
    checkpoint we keep forever; as a separate file it can be pruned on its own
    (--keep_state) without touching a single weight."""
    base = ckpt_path[:-5] if ckpt_path.endswith(".ckpt") else ckpt_path
    return base + "_state.pt"


def save_state(path, opt, scheduler, epoch, best_iou, args, world_size):
    """Optimizer + scheduler + position. NOT the EMA: it is already written beside
    this as epoch_XXXX_ema.ckpt, and duplicating 2.4 GiB per epoch to save a path
    join would be a poor trade."""
    torch.save({
        "format": STATE_FORMAT,
        "epoch": epoch,
        "best_iou": best_iou,
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "provenance": {"lr": args.lr, "ema": args.ema, "world_size": world_size,
                       "grad_accum": args.grad_accum, "lr_schedule": args.lr_schedule},
    }, path)


def prune_states(out_dir, keep):
    """Keep the newest `keep` sidecars. 0 keeps everything.

    Resuming a chain only ever needs the latest, and each one is ~4.9 GiB, so the
    default deliberately does not accumulate them. Only sidecars are ever removed;
    the weights, which are the actual result, are never touched."""
    if keep <= 0:
        return
    found = []
    for f in os.listdir(out_dir):
        m = re.fullmatch(r"epoch_(\d+)_state\.pt", f)
        if m:
            found.append((int(m.group(1)), os.path.join(out_dir, f)))
    for _, path in sorted(found)[:-keep]:
        os.remove(path)
        print(f"[state] pruned {os.path.basename(path)} (--keep_state {keep}; "
              f"weights untouched)", flush=True)


def run_bench(args, p0, is_dist, rank, world_size, device, ds, gen, model, opt, scheduler, ema_state,
             sm, ss, tm, ts):
    """--bench_steps mode: the speed gate. Runs the real per-sample step body (same
    ds/gen/model/opt/scheduler objects main() already built with THIS invocation's
    --prefetch/--cond_file/--no_grad_ckpt live) for a fixed number of optimizer
    steps, timing five phases per micro-step with torch.cuda.synchronize() at every
    boundary so each phase reflects GPU-complete work, not just kernel-launch time.
    Warmup steps run the identical code path but are excluded from every number, so
    CUDA allocator growth / Triton autotune / AdamW's first-step state init don't
    leak into the steady-state measurement. See --bench_steps's help for the
    contract; this function is deliberately NOT called from the normal training
    loop so that loop, and the parity check that exercises it, stay untouched."""
    if args.seed is None:
        raise RuntimeError("--bench_steps requires --seed: every bench cell must draw the "
                           "identical shape subset or cells are not comparable to each other.")
    n_warmup_micro = args.bench_warmup * args.grad_accum
    n_meas_micro = args.bench_steps * args.grad_accum
    total_micro = n_warmup_micro + n_meas_micro
    n_draw_global = total_micro * world_size
    if n_draw_global > len(ds):
        raise RuntimeError(f"--bench_steps needs {n_draw_global} shapes globally "
                           f"((warmup+steps)*grad_accum*world_size) but '{args.train_split}' "
                           f"only has {len(ds)}; lower --bench_steps/--bench_warmup/--grad_accum "
                           f"or point at a bigger split.")
    # one synthetic "epoch" (fixed at 1): the SAME draw formula the real per-epoch
    # loop uses, so a bench cell draws exactly the shapes epoch 1 of a real run with
    # this --seed would draw
    g_draw = torch.Generator()
    g_draw.manual_seed(args.seed * 7919 + 1)
    if args.emis_oversample:
        samp_w = torch.tensor([(f + 0.1) ** args.oversample_pow for f in ds.fracs])
        gidxs = torch.multinomial(samp_w, n_draw_global, replacement=True, generator=g_draw).tolist()
    else:
        gidxs = torch.randperm(len(ds), generator=g_draw).tolist()[:n_draw_global]
    my_pos = list(range(rank, n_draw_global, world_size))
    idx_seq = [gidxs[p] for p in my_pos]
    stream = iter(make_sample_stream(ds, idx_seq, args.prefetch))

    phases = {"data_wait": [], "h2d": [], "forward": [], "backward": [], "optimizer": [], "sync_wait": []}
    opt.zero_grad(set_to_none=True)

    def micro_step(k, measure):
        gpos = my_pos[k]
        t_req = time.time()
        j, (shp, itx, otx, cond, mask) = next(stream)
        t_got = time.time()

        t0 = time.time()
        coords = shp["coords"].to(device)
        shp_f = (shp["feats"].to(device) - sm) / ss
        itx_f = (itx["feats"].to(device) - tm) / ts
        data = (otx["feats"].to(device) - tm) / ts
        cond_dev = cond.to(device)
        torch.cuda.synchronize(device)
        t1 = time.time()

        g_n = torch.Generator(device=device)
        g_n.manual_seed((args.seed * 1000003 + 1 * 10007 + gpos) % (2 ** 31 - 1))
        noise = torch.randn(data.shape, dtype=data.dtype, device=device, generator=g_n)
        t = torch.rand(1, device=device, generator=g_n)
        x_t = t * noise + (1 - t) * data
        v_target = noise - data
        t_model = (t * 1000).expand(1)

        is_sync_step = ((k + 1) % args.grad_accum == 0) or (k + 1 == total_micro)
        sync_ctx = model.no_sync() if (is_dist and not is_sync_step) else contextlib.nullcontext()
        with sync_ctx:
            t2 = time.time()
            v_pred_feats = model(x_t, itx_f, shp_f, coords, t_model, cond_dev)
            if args.balanced_pos_weight > 0:
                mask_dev = mask.to(device)
                p = mask_dev.mean().clamp(min=1e-4)
                W_shape = torch.clamp((1 - p) / p, max=args.balanced_pos_weight)
                w = 1 + (W_shape - 1) * mask_dev
                w = w / w.mean().clamp(min=1e-8)
                loss = (w[:, None] * (v_pred_feats - v_target) ** 2).mean()
            elif args.pos_weight != 1.0:
                w = 1 + (args.pos_weight - 1) * mask.to(device)
                w = w / w.mean().clamp(min=1e-8)
                loss = (w[:, None] * (v_pred_feats - v_target) ** 2).mean()
            else:
                loss = nn.functional.mse_loss(v_pred_feats, v_target)
            torch.cuda.synchronize(device)
            t3 = time.time()

            t4 = time.time()
            (loss / args.grad_accum).backward()
            torch.cuda.synchronize(device)
            t5 = time.time()

        sync_wait_ms = None
        if is_dist and is_sync_step:
            # straggler proxy: how much longer the SLOWEST rank's sync-step backward
            # took than this rank's own -- the gap this rank spent blocked inside
            # DDP's fused all-reduce waiting on the slowest rank, not computing
            bt = torch.tensor([(t5 - t4) * 1000], device=device)
            dist.all_reduce(bt, op=dist.ReduceOp.MAX)
            sync_wait_ms = max(0.0, bt.item() - (t5 - t4) * 1000)

        opt_ms = 0.0
        if is_sync_step:
            t6 = time.time()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if ema_state is not None:
                ema_update(ema_state, gen, args.ema)
            torch.cuda.synchronize(device)
            t7 = time.time()
            opt_ms = (t7 - t6) * 1000

        if measure:
            phases["data_wait"].append((t_got - t_req) * 1000)
            phases["h2d"].append((t1 - t0) * 1000)
            phases["forward"].append((t3 - t2) * 1000)
            phases["backward"].append((t5 - t4) * 1000)
            if is_sync_step:
                phases["optimizer"].append(opt_ms)
            if sync_wait_ms is not None:
                phases["sync_wait"].append(sync_wait_ms)

    p0(f"[bench] warming up {args.bench_warmup} optimizer steps ({n_warmup_micro} micro-steps, "
       f"excluded from the numbers)...", flush=True)
    for k in range(n_warmup_micro):
        micro_step(k, measure=False)

    torch.cuda.synchronize(device)
    dist_barrier(is_dist)
    torch.cuda.reset_peak_memory_stats(device)
    region_t0 = time.time()
    p0(f"[bench] measuring {args.bench_steps} optimizer steps ({n_meas_micro} micro-steps)...",
       flush=True)
    for k in range(n_warmup_micro, total_micro):
        micro_step(k, measure=True)
    torch.cuda.synchronize(device)
    dist_barrier(is_dist)
    region_t1 = time.time()

    peak_bytes = float(torch.cuda.max_memory_allocated(device))
    measured_shapes = n_meas_micro
    if is_dist:
        pb = torch.tensor([peak_bytes], device=device)
        dist.all_reduce(pb, op=dist.ReduceOp.MAX)
        peak_bytes = pb.item()
        mst = torch.tensor([float(measured_shapes)], device=device)
        dist.all_reduce(mst, op=dist.ReduceOp.SUM)
        measured_shapes_global = mst.item()
    else:
        measured_shapes_global = float(measured_shapes)
    dt = region_t1 - region_t0
    shapes_per_s = measured_shapes_global / max(dt, 1e-9)

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "gpu": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
        "world_size": world_size,
        "tag": args.bench_tag,
        "config": {
            "prefetch": args.prefetch,
            "cond_file": args.cond_file,
            "grad_ckpt": not args.no_grad_ckpt,
            "grad_accum": args.grad_accum,
            "cond": args.cond,
            "emis_oversample": args.emis_oversample,
            "seed": args.seed,
            "dataset": args.dataset,
            "train_split": args.train_split,
        },
        "bench_warmup_steps": args.bench_warmup,
        "bench_measured_steps": args.bench_steps,
        "measured_shapes_global": measured_shapes_global,
        "measured_wall_s": dt,
        "shapes_per_s": shapes_per_s,
        "peak_vram_gb": peak_bytes / 2 ** 30,
        "phase_ms_per_microstep": {
            "data_wait": mean(phases["data_wait"]),
            "h2d": mean(phases["h2d"]),
            "forward": mean(phases["forward"]),
            "backward": mean(phases["backward"]),
        },
        "phase_ms_per_optstep": {
            "optimizer": mean(phases["optimizer"]),
            "sync_wait": mean(phases["sync_wait"]),
        },
    }
    if rank == 0:
        out_path = args.bench_out or os.path.join(args.out_dir, "bench_results.jsonl")
        with open(out_path, "a") as f:
            f.write(json.dumps(result) + "\n")
        fwd = result["phase_ms_per_microstep"]["forward"]
        bwd = result["phase_ms_per_microstep"]["backward"]
        optm = result["phase_ms_per_optstep"]["optimizer"]
        p0(f"[bench] {shapes_per_s:.3f} shapes/s | peak {result['peak_vram_gb']:.2f} GiB | "
           f"data_wait={result['phase_ms_per_microstep']['data_wait']:.1f}ms "
           f"h2d={result['phase_ms_per_microstep']['h2d']:.1f}ms fwd={fwd:.1f}ms bwd={bwd:.1f}ms "
           f"opt={optm:.1f}ms/optstep -> {out_path}", flush=True)


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
    ap.add_argument("--n_per_epoch", type=int, default=0,
                    help="Shapes drawn per epoch, counted GLOBALLY across all GPUs (0 = the "
                         "whole training split). Each rank takes n_per_epoch/world_size of "
                         "them, so an epoch covers the same shapes whether you run on 1 GPU "
                         "or 4 and an existing --n_per_epoch stays comparable across run "
                         "widths. A remainder of fewer than world_size shapes is dropped.")
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
                         "for every training sample UNLESS pos_weight==1.0 (fully off, old unweighted MSE). "
                         "Ignored when --balanced_pos_weight > 0.")
    ap.add_argument("--balanced_pos_weight", type=float, default=0.0,
                    help="0=off (use the fixed --pos_weight scalar instead). Per-SHAPE adaptive weight: "
                         "W_shape = min(CAP, (1-p)/p) where p = this sample's mean emis_mask coverage "
                         "(clamped p>=1e-4); per-voxel w=1+(W_shape-1)*m_i, then the same mean-"
                         "normalization as --pos_weight. Fixes the flat-mean-per-voxel scheme's blind "
                         "spot: a fixed pos_weight upweights emissive VOXELS within a shape but does "
                         "nothing for shapes that are almost entirely non-emissive (p tiny) relative to "
                         "shapes that are half-emissive — this makes W scale with how rare emissive is "
                         "FOR THAT SHAPE. Requires emis_mask.pth.")
    ap.add_argument("--lr_schedule", choices=["const", "cosine"], default="const",
                    help="const (default) = fixed --lr for the whole run (old behavior). cosine = "
                         "torch CosineAnnealingLR from --lr down to --lr/20 over the full run "
                         "(epochs * steps/epoch total optimizer steps).")
    ap.add_argument("--ema", type=float, default=0.999,
                    help="EMA decay for a shadow copy of the flow weights, saved alongside the regular "
                         "ckpt as epoch_XXXX_ema.ckpt. 0 = off (no EMA file written).")
    ap.add_argument("--val_quick", type=int, default=8,
                    help="after each save, run a quick N-sample val IoU (12-step sampling) on "
                         "--val_split and track best.ckpt + train_curve.json. 0 = off.")
    ap.add_argument("--select_on", choices=["all", "nonzero"], default="nonzero",
                    help="best.ckpt selection criterion from quick-val. 'nonzero' (default) = mean IoU "
                         "restricted to quick-val shapes with GT coverage>0 — timidity-proof, a ckpt "
                         "can't win by predicting all-black on empty-glow shapes and diluting the flat "
                         "mean. 'all' = mean IoU over every quick-val shape (old behavior). Both "
                         "aggregates are always computed and logged regardless of which one gates "
                         "best.ckpt.")
    ap.add_argument("--resume", default=None,
                    help="Path to a weights checkpoint (epoch_XXXX.ckpt) to CONTINUE from, "
                         "restoring the optimizer state, the LR schedule position, the EMA "
                         "shadow and the epoch counter from its epoch_XXXX_state.pt sidecar. "
                         "This is what stops a chained continuation paying the one-epoch "
                         "resettle that a fresh AdamW costs. Errors if the sidecar is absent "
                         "rather than quietly falling back, since a silent fallback is exactly "
                         "the tax you were trying to avoid; use --init_ckpt for that on "
                         "purpose. Overrides --init_ckpt. Epoch numbering continues from the "
                         "restored position, so --epochs means epochs to run in THIS leg.")
    ap.add_argument("--fresh_opt", action="store_true", default=False,
                    help="With --resume: restore the position, EMA and data order, but start "
                         "a NEW AdamW. That is the old chained-continuation behavior with the "
                         "data order matched, which makes it the control this feature is "
                         "measured against; it is also the flag to use if you deliberately "
                         "want to reset the optimizer mid-chain.")
    ap.add_argument("--keep_state", type=int, default=1,
                    help="How many optimizer-state sidecars to retain in --out_dir (0 = all). "
                         "Each is about twice the weight bytes, and a chain only ever resumes "
                         "from the newest, so the default does not accumulate them. Pruning "
                         "only ever removes sidecars, never weights.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Fix the epoch sample draw AND the per-sample flow noise/timestep. "
                         "Noise and t are drawn from a generator keyed by (seed, epoch, "
                         "position in the global draw), so a given shape gets the SAME noise "
                         "on 1 GPU and on N GPUs -- that is what makes the multi-GPU parity "
                         "check meaningful. Default None = untouched global RNG, i.e. the "
                         "original behavior; leave it unset for real runs.")
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="Micro-steps accumulated per rank before one optimizer step. 1 "
                         "(default) = the usual behavior. Raising it multiplies the effective "
                         "batch (world_size * grad_accum shapes) and, via DDP's no_sync, cuts "
                         "the gradient all-reduce rate by the same factor -- the lever to pull "
                         "if inter-GPU communication is what limits scaling. Note it also "
                         "reduces the number of EMA and LR-scheduler steps per epoch.")
    ap.add_argument("--ddp_find_unused", action="store_true", default=False,
                    help="Pass find_unused_parameters=True to DDP. Only needed if a run dies "
                         "with 'expected to have finished reduction'; the flow model uses every "
                         "parameter on every step, so the default False is both correct and "
                         "faster.")
    ap.add_argument("--log_step_loss", action="store_true", default=False,
                    help="Print the loss of every optimizer step (all-reduced across ranks, so "
                         "it is the loss of the whole effective batch). Off by default; it is "
                         "what the multi-GPU parity check reads.")
    ap.add_argument("--ddp_bf16_comm", action="store_true", default=False,
                    help="Register DDP's bf16 gradient-compression hook: gradients are cast to "
                         "bfloat16 for the all-reduce and back afterwards, halving the bytes on "
                         "the wire at the cost of some reduction precision.")
    ap.add_argument("--prefetch", type=int, default=0,
                    help="Overlap sample loading with GPU compute: a background thread reads "
                         "up to N samples ahead of what the training loop is currently working "
                         "on (see make_sample_stream/_Prefetcher). 0 (default) = off, the "
                         "original synchronous ds[j] call inline in the loop -- measured 321ms/"
                         "sample of blocking NFS read on a fresh (never-cached) shape, cutting "
                         "shapes/s from 0.696 to 0.899 (1.29x) once hidden. The consumption "
                         "ORDER is identical at any depth: this changes only when a read starts, "
                         "never which shape a step sees or what bytes it reads (checksummed "
                         "byte-identical against the synchronous path over 380+ steps). KNOWN "
                         "CAVEAT: one 200-step parity run showed a single step (1 of ~600 "
                         "prefetch-enabled steps checked total) with a loss 17x the normal "
                         "run-to-run noise band despite byte-identical inputs at that step; not "
                         "reproduced in a further 380-step check. Attributed to kernel-selection "
                         "timing sensitivity (GPU nondeterminism this project has already "
                         "documented elsewhere, e.g. the DDP work), not data corruption -- but "
                         "disclosed rather than hidden. Pair with --val_quick's best.ckpt "
                         "tracking as a watchdog on any production run (a val-trajectory anomaly "
                         "is the rollback trigger; --resume from the optimizer sidecar makes that "
                         "loss-free). 2-4 is enough to hide a single synchronous read behind one "
                         "step of compute; raising it further only helps if a step is faster than "
                         "one read.")
    ap.add_argument("--cond_file", default="cond.pth",
                    help="Filename to read --cond real's conditioning tensor from inside each "
                         "sample directory. cond.pth (default) is the original file: a dict with "
                         "BOTH the pos and neg halves of a classifier-free-guidance pair "
                         "(~8.4MB/sample measured), of which training only ever uses ['cond']. "
                         "cond_pos.pth is a slim per-sample sidecar holding just that tensor "
                         "(~4.2MB), built once via build_cond_pos.py; pass --cond_file "
                         "cond_pos.pth to use it. Falls back to cond.pth per-sample when the "
                         "slim file is missing, so a partially-converted split still trains "
                         "correctly -- this never changes WHICH tensor is used, only how many "
                         "bytes get read to get it. No existing cond.pth is ever modified or "
                         "read differently by this flag; it only adds a cheaper place to look "
                         "first.")
    ap.add_argument("--no_grad_ckpt", action="store_true", default=False,
                    help="Leave gradient checkpointing OFF (default: on, unconditionally, for "
                         "every module with a use_checkpoint attribute -- activations dominate "
                         "memory for the 1.3B sparse DiT). Checkpointing trades recompute-on-"
                         "backward for memory; turning it off is faster per step but raises peak "
                         "memory. MEASURE peak memory on your GPU before relying on this flag in "
                         "a real run -- the trainer already prints '[mem] rank R: peak allocated' "
                         "at the end of every run for exactly this check. Recommended only where "
                         "that measured peak has real headroom below the GPU's VRAM (e.g. an "
                         "H100's 80GB); the measured L40S number is in the trainer_speedup report.")
    ap.add_argument("--bench_steps", type=int, default=0,
                    help="Speed-benchmark mode (a 'unit test for training speed'): measure "
                         "this many STEADY-STATE optimizer steps and exit, instead of running "
                         "--epochs of real training. 0 (default) = off, normal training runs. "
                         "Uses the exact same dataset/model/optimizer/scheduler objects a real "
                         "run would build -- --prefetch/--cond_file/--no_grad_ckpt are already "
                         "whichever way this invocation set them by the time bench measurement "
                         "starts, so this measures the real path, not a stand-in. Requires "
                         "--seed (every bench cell must draw the identical shape subset to be "
                         "comparable). Appends one JSON line (config, shapes/s, peak VRAM, a "
                         "per-phase ms breakdown: data wait / H2D / forward / backward / "
                         "optimizer, plus a DDP sync-wait estimate when world_size>1) to "
                         "--bench_out.")
    ap.add_argument("--bench_warmup", type=int, default=10,
                    help="Optimizer steps run before bench measurement starts and excluded "
                         "from it (lets the CUDA caching allocator, cuDNN/Triton autotune, and "
                         "AdamW's first-step state init settle before anything is timed). Only "
                         "meaningful with --bench_steps > 0.")
    ap.add_argument("--bench_out", default=None,
                    help="Path to append this bench run's JSON line to. Default: "
                         "<out_dir>/bench_results.jsonl.")
    ap.add_argument("--bench_tag", default=None,
                    help="Free-text label recorded in the bench JSON line (e.g. "
                         "'p2_condslim_ckptoff_ga1'), so a grid sweep's results are readable "
                         "without reconstructing the flag combination from the config block.")
    ap.add_argument("--debug_checksum", action="store_true", default=False,
                    help="DIAGNOSTIC ONLY, not for real runs: print a per-step checksum "
                         "(voxel count + sum of shp_f/itx_f/data/cond) right after loading, "
                         "in the REAL (non-bench) training loop. Added to chase a single-step "
                         "loss anomaly seen with --prefetch on; lets a divergence be narrowed "
                         "to the LOADED tensors vs. the model/numerics before it reaches loss.")
    args = ap.parse_args()
    is_dist, rank, world_size, local_rank = dist_setup()

    def p0(*a, **kw):
        """Print on rank 0 only. Every log line in this script is a global fact
        (identical weights, all-reduced loss), so N copies would just be noise."""
        if rank == 0:
            print(*a, **kw)

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
    dist_barrier(is_dist)
    device = "cuda"     # torch.cuda.set_device(local_rank) already bound this to our GPU
    if is_dist:
        p0(f"[ddp] world_size={world_size} on {os.environ.get('SLURM_JOB_NODELIST', 'local')} "
           f"| grad_accum={args.grad_accum} | effective batch "
           f"{world_size * args.grad_accum} shapes/step", flush=True)
    if args.seed is not None:
        # per-rank offset so anything NOT covered by the keyed generator (dropout-like
        # nondeterminism, if any is ever added) still differs across ranks
        torch.manual_seed(args.seed + 1000 * rank)
        np.random.seed(args.seed + 1000 * rank)
    require_mask = args.pos_weight != 1.0 or args.balanced_pos_weight > 0

    # Resolve where the weights come from and whether the optimizer comes with them.
    # One of three modes, always announced, so a log never leaves it ambiguous which
    # one a run took.
    resume_state = None
    if args.resume:
        state_p = state_path_for(args.resume)
        if not probe_exists(state_p):
            raise RuntimeError(
                f"--resume {args.resume} but its optimizer-state sidecar {state_p} is not "
                f"there. Checkpoints written before optimizer-state saving existed have no "
                f"sidecar. Resuming without it means a fresh AdamW and the one-epoch "
                f"resettle, which is the thing --resume exists to avoid, so this is an error "
                f"rather than a silent downgrade: pass --init_ckpt {args.resume} if a "
                f"warm start with a fresh optimizer is what you actually want.")
        resume_state = torch.load(state_p, map_location="cpu")
        if resume_state.get("format") != STATE_FORMAT:
            raise RuntimeError(f"{state_p} has state format {resume_state.get('format')}, "
                               f"this trainer writes and reads {STATE_FORMAT}")
        mode = "RESUME_FRESH_OPT" if args.fresh_opt else "RESUME_WITH_STATE"
        if args.init_ckpt != "full_seg":
            p0(f"[init] NOTE --init_ckpt {args.init_ckpt} is ignored because --resume was "
               f"given; weights come from {args.resume}", flush=True)
    else:
        mode = "WARM_START_FRESH_OPT"
        if args.fresh_opt:
            raise RuntimeError("--fresh_opt only means something with --resume; without it "
                               "the optimizer is already fresh.")

    if is_dist:
        # let rank 0 touch the HF cache alone first: four processes racing to populate
        # the same cache entry over NFS relies on hub file locks that are not reliable
        # there. A warm cache (the normal case) makes this a no-op lookup.
        if rank == 0:
            resolve_init_ckpt(args.init_ckpt)
            hf_hub_download(repo_id="microsoft/TRELLIS.2-4B", filename="pipeline.json")
        dist_barrier(is_dist)

    # model: flow + Gen3DSeg wrapper, warm-started from a SegviGen ckpt
    init_ckpt = args.resume if args.resume else resolve_init_ckpt(args.init_ckpt)
    p0(f"[init] {mode} from {init_ckpt}"
       + (f" (position: epoch {resume_state['epoch']})" if resume_state else ""), flush=True)
    flow = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")
    # gradient checkpointing — activations dominate memory for the 1.3B sparse DiT; this
    # lets the full fine-tune fit on a 44GB GPU (l40s/a40). --no_grad_ckpt skips this,
    # trading the memory headroom back for the recompute-on-backward it costs; see
    # --no_grad_ckpt's help and the trainer_speedup report for the measured peak.
    n_ckpt = 0
    if not args.no_grad_ckpt:
        for m in flow.modules():
            if hasattr(m, "use_checkpoint"):
                m.use_checkpoint = True; n_ckpt += 1
        p0(f"[mem] enabled gradient checkpointing on {n_ckpt} modules", flush=True)
    else:
        p0("[mem] --no_grad_ckpt: gradient checkpointing left OFF. Check the "
           "'[mem] rank R: peak allocated' line this run prints at the end against "
           "your GPU's VRAM before trusting this on anything but a measured config.",
           flush=True)
    gen = Gen3DSeg(flow).to(device)
    sd = torch.load(init_ckpt, map_location=device)["state_dict"]
    sd = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in sd.items()])
    gen.load_state_dict(sd)
    gen.train()

    n_par = sum(p.numel() for p in gen.parameters())
    grad_bytes = sum(p.numel() * p.element_size() for p in gen.parameters() if p.requires_grad)
    dtypes = sorted({str(p.dtype).replace("torch.", "") for p in gen.parameters()})
    p0(f"[model] {n_par / 1e9:.3f}B parameters ({'/'.join(dtypes)}); one gradient sync "
       f"moves {grad_bytes / 2 ** 30:.2f} GiB per rank", flush=True)

    # Every rank warm-starts from the same file, so the replicas already agree;
    # DDP broadcasts once more at construction, which also catches a partial read.
    step_module = FlowStep(gen)
    if is_dist:
        model = DDP(step_module, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=args.ddp_find_unused,
                    gradient_as_bucket_view=True,       # buckets alias .grad, no second copy
                    broadcast_buffers=False)            # no BN/running stats here to sync
        if args.ddp_bf16_comm:
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
            model.register_comm_hook(state=None, hook=default_hooks.bf16_compress_hook)
            p0("[ddp] gradient all-reduce runs in bfloat16 (bf16_compress_hook)", flush=True)
    else:
        model = step_module
    # gen stays the handle for state_dict / EMA / quick-val: its keys are the ones
    # save_ckpt prefixes with "gen3dseg.", and neither FlowStep nor DDP must appear
    # in them or eval_emissive.py would not be able to load the checkpoint.

    ema_state = None
    if args.ema > 0 and rank == 0:
        ema_state = {k: v.detach().clone() for k, v in gen.state_dict().items()}
        # On resume, pick the shadow back up from the _ema.ckpt already written beside
        # the weights. Starting it from the live weights instead would silently reset
        # the averaging horizon, which is a quieter version of the same tax --resume
        # exists to remove.
        if resume_state is not None:
            ema_p = (args.resume[:-5] if args.resume.endswith(".ckpt") else args.resume) + "_ema.ckpt"
            if probe_exists(ema_p):
                esd = torch.load(ema_p, map_location=device)["state_dict"]
                esd = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in esd.items()])
                missing = [k for k in ema_state if k not in esd]
                if missing:
                    raise RuntimeError(f"{ema_p} is missing {len(missing)} keys the model has "
                                       f"(first: {missing[:3]}); refusing to resume a partial EMA")
                for k in ema_state:
                    ema_state[k].copy_(esd[k])
                p0(f"[ema] restored shadow weights from {os.path.basename(ema_p)}", flush=True)
            else:
                p0(f"[ema] NOTE no {os.path.basename(ema_p)} beside the resume checkpoint, so "
                   f"the shadow restarts from the current weights", flush=True)
        p0(f"[ema] tracking shadow weights, decay={args.ema}", flush=True)

    eval_models = None
    if args.val_quick > 0 and rank == 0:
        # decoders live on rank 0 only; the other ranks never run quick-val, and not
        # loading them there leaves that much more GPU memory for training
        eval_models = load_eval_models(device)
        p0(f"[val_quick] loaded eval decoders for {args.val_quick}-sample quick-val on '{args.val_split}'", flush=True)

    sm, ss, tm, ts = load_norm_stats(device)
    opt = torch.optim.AdamW(gen.parameters(), lr=args.lr, weight_decay=0.0)
    # Every rank restores: each holds its own optimizer over its own replica, and DDP
    # keeps the replicas in step only through gradients, never through optimizer state.
    if resume_state is not None and not args.fresh_opt:
        opt.load_state_dict(resume_state["optimizer"])
        prov = resume_state.get("provenance", {})
        if prov.get("lr") is not None and abs(prov["lr"] - args.lr) > 1e-12:
            p0(f"[resume] NOTE the saved run used lr {prov['lr']:.2e} and this leg asks for "
               f"{args.lr:.2e}; the new value wins, AdamW's moments carry over", flush=True)
            for g in opt.param_groups:
                g["lr"] = args.lr
        p0(f"[resume] optimizer state restored ({len(opt.state)} parameter slots)", flush=True)
    elif resume_state is not None:
        p0("[resume] --fresh_opt: position and EMA restored, optimizer starts from zero "
           "moments (this is the behavior --resume exists to improve on)", flush=True)

    if is_dist:
        # one scan on rank 0, shipped to the rest (see EmisDataset(prebuilt=...))
        if rank == 0:
            ds = EmisDataset(args.dataset, args.train_split, cond_mode=args.cond,
                             require_mask=require_mask, cond_file=args.cond_file)
            payload = [(ds.dirs, ds.fracs)]
        else:
            payload = [None]
        dist.broadcast_object_list(payload, src=0)
        if rank != 0:
            ds = EmisDataset(args.dataset, args.train_split, cond_mode=args.cond,
                             require_mask=require_mask, prebuilt=payload[0], cond_file=args.cond_file)
    else:
        ds = EmisDataset(args.dataset, args.train_split, cond_mode=args.cond,
                         require_mask=require_mask, cond_file=args.cond_file)
    p0(f"[data] {len(ds)} samples from '{args.train_split}' (cond={args.cond}, "
       f"oversample={args.emis_oversample}, pos_weight={args.pos_weight}, "
       f"balanced_pos_weight={args.balanced_pos_weight})", flush=True)
    samp_w = torch.tensor([(f + 0.1) ** args.oversample_pow for f in ds.fracs]) if args.emis_oversample else None

    # per-epoch draw sizes. n_per_epoch is global; the remainder below world_size is
    # dropped so every rank runs the same number of steps and no one hangs at a barrier.
    n_draw_global = args.n_per_epoch or len(ds)
    if samp_w is None:
        # the no-oversample path draws WITHOUT replacement, so it can never hand out
        # more indices than the split holds; clamp here or the per-rank slice below
        # would index past the end of a short draw
        n_draw_global = min(n_draw_global, len(ds))
    if n_draw_global < world_size:
        raise RuntimeError(f"--n_per_epoch resolves to {n_draw_global} shapes/epoch but "
                           f"world_size is {world_size}: every rank needs at least one "
                           f"shape per epoch or the ranks desynchronize. Raise "
                           f"--n_per_epoch or run on fewer GPUs.")
    n_per_rank = max(1, n_draw_global // world_size)
    opt_steps_per_epoch = max(1, n_per_rank // args.grad_accum)
    if is_dist:
        p0(f"[data] per epoch: {n_draw_global} shapes global -> {n_per_rank}/rank -> "
           f"{opt_steps_per_epoch} optimizer steps/epoch", flush=True)

    scheduler = None
    if args.lr_schedule == "cosine":
        total_steps = max(1, args.epochs * opt_steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.lr / 20.0)
        p0(f"[lr] cosine schedule: {args.lr:.2e} -> {args.lr / 20.0:.2e} over {total_steps} steps", flush=True)
        if resume_state is not None and not args.fresh_opt and resume_state.get("scheduler"):
            # T_max and last_epoch travel inside the state, so the restored curve is the
            # ORIGINAL one continued, not a fresh cosine reshaped by this leg's --epochs
            scheduler.load_state_dict(resume_state["scheduler"])
            p0(f"[lr] cosine position restored: step {scheduler.last_epoch} of "
               f"{scheduler.T_max}, lr now {opt.param_groups[0]['lr']:.3e}", flush=True)

    if args.bench_steps > 0:
        # speed-benchmark mode: measure and exit, never reaching the real per-epoch
        # training loop below (which is what the parity check exercises instead)
        run_bench(args, p0, is_dist, rank, world_size, device, ds, gen, model, opt,
                 scheduler, ema_state, sm, ss, tm, ts)
        if is_dist:
            dist.barrier()
            dist.destroy_process_group()
        return

    log = []
    curve = []
    best_iou = -1.0
    n_wshape_logged = 0
    # Numbering continues across a resume. That is not cosmetic: the epoch number keys
    # the seeded per-epoch draw and the per-shape noise, so continuing it is what makes
    # a resumed leg see the data the un-interrupted run would have seen next.
    start_epoch = (resume_state["epoch"] + 1) if resume_state is not None else 1
    last_epoch = start_epoch + args.epochs - 1
    if resume_state is not None:
        best_iou = resume_state.get("best_iou", -1.0)
        p0(f"[resume] continuing at epoch {start_epoch}, running {args.epochs} more "
           f"(through {last_epoch}); best_iou carried in as {best_iou:.4f}", flush=True)

    for epoch in range(start_epoch, last_epoch + 1):
        # The epoch's draw is computed IDENTICALLY on every rank (same weights, same
        # seeded generator) and then sliced by rank, so the oversample distribution is
        # exactly the single-GPU one and no index has to be communicated. Interleaved
        # rather than blocked slicing, so no rank gets a systematically distinct chunk.
        g_draw = None
        if args.seed is not None:
            g_draw = torch.Generator()
            g_draw.manual_seed(args.seed * 7919 + epoch)
        gkw = {"generator": g_draw} if g_draw is not None else {}
        if samp_w is not None:
            gidxs = torch.multinomial(samp_w, n_draw_global, replacement=True, **gkw).tolist()
        else:
            gidxs = torch.randperm(len(ds), **gkw).tolist()[:n_draw_global]
        my_pos = list(range(rank, n_per_rank * world_size, world_size))
        # the fixed index sequence this rank will consume THIS epoch, in order --
        # computed once, up front, so --prefetch has a static list to read ahead of
        # rather than depending on anything the loop body does
        idx_seq = [gidxs[gpos] for gpos in my_pos]

        ep_loss = 0.0
        n_local = 0
        t_epoch = time.time()
        opt.zero_grad(set_to_none=True)
        for k, (j, (shp, itx, otx, cond, mask)) in enumerate(make_sample_stream(ds, idx_seq, args.prefetch)):
            gpos = my_pos[k]   # this step's position in the global draw (unchanged by --prefetch)
            coords = shp["coords"].to(device)
            shp_f = (shp["feats"].to(device) - sm) / ss
            itx_f = (itx["feats"].to(device) - tm) / ts
            data  = (otx["feats"].to(device) - tm) / ts
            cond  = cond.to(device)

            if args.debug_checksum:
                # diagnostic only (see --debug_checksum help): proves whether the
                # LOADED tensors themselves differ between two runs at the same
                # step, independent of the loss -- narrows a divergence to the
                # loader vs. the model/numerics
                p0(f"[checksum] epoch {epoch:4d} k {k:5d} sid {os.path.basename(ds.dirs[j])} "
                   f"n_vox={coords.shape[0]} shp_sum={shp_f.sum().item():.6f} "
                   f"itx_sum={itx_f.sum().item():.6f} data_sum={data.sum().item():.6f} "
                   f"cond_sum={cond.sum().item():.6f}", flush=True)

            if args.seed is None:
                noise = torch.randn_like(data)
                t = torch.rand(1, device=device)
            else:
                # keyed by POSITION IN THE GLOBAL DRAW, not by rank or local step, so a
                # shape sees the same noise and the same t whatever the world size is
                g_n = torch.Generator(device=device)
                g_n.manual_seed((args.seed * 1000003 + epoch * 10007 + gpos) % (2 ** 31 - 1))
                noise = torch.randn(data.shape, dtype=data.dtype, device=device, generator=g_n)
                t = torch.rand(1, device=device, generator=g_n)
            x_t = t * noise + (1 - t) * data
            v_target = noise - data
            t_model = (t * 1000).expand(1)

            # DDP fuses the gradient all-reduce into the backward pass; under
            # accumulation every rank must skip that sync (forward included, the flag
            # is read in DDP.forward) until the step that actually calls opt.step().
            is_sync_step = ((k + 1) % args.grad_accum == 0) or (k + 1 == len(my_pos))
            sync_ctx = model.no_sync() if (is_dist and not is_sync_step) else contextlib.nullcontext()
            with sync_ctx:
                v_pred_feats = model(x_t, itx_f, shp_f, coords, t_model, cond)
                if args.balanced_pos_weight > 0:
                    mask_dev = mask.to(device)
                    p = mask_dev.mean().clamp(min=1e-4)
                    W_shape = torch.clamp((1 - p) / p, max=args.balanced_pos_weight)
                    w = 1 + (W_shape - 1) * mask_dev
                    w = w / w.mean().clamp(min=1e-8)
                    loss = (w[:, None] * (v_pred_feats - v_target) ** 2).mean()
                    if n_wshape_logged < 3 and rank == 0:
                        print(f"[balanced_pos_weight] sample {j} sid={os.path.basename(ds.dirs[j])} "
                              f"p={p.item():.4f} W_shape={W_shape.item():.2f}", flush=True)
                        n_wshape_logged += 1
                elif args.pos_weight != 1.0:
                    w = 1 + (args.pos_weight - 1) * mask.to(device)
                    w = w / w.mean().clamp(min=1e-8)
                    loss = (w[:, None] * (v_pred_feats - v_target) ** 2).mean()
                else:
                    loss = nn.functional.mse_loss(v_pred_feats, v_target)
                (loss / args.grad_accum).backward()

            ep_loss += loss.item()
            n_local += 1
            if is_sync_step:
                torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                if ema_state is not None:
                    ema_update(ema_state, gen, args.ema)
                if args.log_step_loss:
                    sl = torch.tensor([loss.item()], device=device)
                    if is_dist:
                        dist.all_reduce(sl, op=dist.ReduceOp.AVG)
                    # sid identifies THIS rank's last micro-step in the accumulation
                    # group (well-defined and exactly one shape/step under the
                    # grad_accum=1, world_size=1 config the speed/parity harnesses
                    # use); it lets a parity check diff the drawn-shape SEQUENCE
                    # across runs programmatically instead of trusting the loss
                    # curve alone to stand in for "same data in the same order".
                    p0(f"[step] epoch {epoch:4d} step {(k + 1) // args.grad_accum:5d} "
                       f"| loss {sl.item():.6f} | sid {os.path.basename(ds.dirs[j])}", flush=True)

        # epoch loss is the mean over ALL shapes the epoch touched, not just this
        # rank's share, so the printed curve means the same thing at any world size
        agg = torch.tensor([ep_loss, float(n_local)], device=device, dtype=torch.float64)
        if is_dist:
            dist.all_reduce(agg, op=dist.ReduceOp.SUM)
        ep_loss = (agg[0] / agg[1].clamp(min=1)).item()
        dt = time.time() - t_epoch
        shapes_per_s = float(agg[1].item()) / max(dt, 1e-9)
        cur_lr = opt.param_groups[0]["lr"]
        log.append({"epoch": epoch, "loss": ep_loss, "lr": cur_lr,
                    "epoch_sec": round(dt, 2), "shapes_per_s": round(shapes_per_s, 4),
                    "world_size": world_size, "grad_accum": args.grad_accum})
        p0(f"epoch {epoch:4d} | flow_loss {ep_loss:.5f} | lr {cur_lr:.2e} "
           f"| {dt:.1f}s | {shapes_per_s:.3f} shapes/s", flush=True)

        if epoch % args.save_every == 0 or epoch == last_epoch:
            # Saving, EMA and quick-val are rank 0's alone: the replicas hold identical
            # weights, so N writers would only race on the same paths. The other ranks
            # wait at the barrier below while rank 0 samples the quick-val shapes.
            if rank == 0:
                save_ckpt(gen.state_dict(), os.path.join(args.out_dir, "last.ckpt"))
                # ALSO keep per-epoch ckpts (pilot overwrote ep25 which was better than ep50 →
                # overfitting; keep history to find the sweet spot / early-stop).
                ep_path = os.path.join(args.out_dir, f"epoch_{epoch:04d}.ckpt")
                save_ckpt(gen.state_dict(), ep_path)
                if ema_state is not None:
                    save_ckpt(ema_state, os.path.join(args.out_dir, f"epoch_{epoch:04d}_ema.ckpt"))
                json.dump(log, open(os.path.join(args.out_dir, "log.json"), "w"), indent=2)

                val_iou = None
                val_iou_all = val_iou_nonzero = None
                per_sample = None
                if eval_models is not None:
                    gen.eval()
                    result = evaluate_split(gen, eval_models, args.dataset, args.val_split, args.cond,
                                            device=device, steps=12, thrs=THRS, n=args.val_quick, verbose=False)
                    gen.train()
                    val_iou_all = result["best_iou"]
                    val_iou_nonzero = result["best_iou_nonzero"]
                    val_iou = val_iou_nonzero if args.select_on == "nonzero" else val_iou_all
                    per_sample = result["per_sample"]
                    per_sample_s = " ".join(f"{p['sid'][:8]}={p['best_iou']:.3f}" for p in per_sample)
                    print(f"[val_quick] epoch {epoch:4d} | {args.val_split}[:{args.val_quick}] "
                          f"best IoU(all)={val_iou_all:.4f}@thr={result['best_thr']} "
                          f"best IoU(nonzero,n={result['n_nonzero']})={val_iou_nonzero:.4f}@thr={result['best_thr_nonzero']} "
                          f"[selecting on {args.select_on}] | per-sample: {per_sample_s}", flush=True)
                    if val_iou > best_iou:
                        best_iou = val_iou
                        best_link = os.path.join(args.out_dir, "best.ckpt")
                        if os.path.islink(best_link) or os.path.exists(best_link):
                            os.remove(best_link)
                        os.symlink(os.path.basename(ep_path), best_link)
                        print(f"[val_quick] new best ({args.select_on}={val_iou:.4f}) → best.ckpt -> "
                              f"{os.path.basename(ep_path)}", flush=True)
                curve.append({"epoch": epoch, "train_loss": ep_loss, "lr": cur_lr, "val_iou": val_iou,
                             "val_iou_all": val_iou_all, "val_iou_nonzero": val_iou_nonzero,
                             "select_on": args.select_on, "per_sample": per_sample})
                json.dump(curve, open(os.path.join(args.out_dir, "train_curve.json"), "w"), indent=2)
                # Sidecar written LAST in the block: its presence then implies the
                # weights beside it finished, and best_iou is this epoch's value rather
                # than the previous one, so a resumed run's best.ckpt selection picks up
                # where it left off instead of re-winning against a stale bar.
                save_state(state_path_for(ep_path), opt, scheduler, epoch, best_iou,
                           args, world_size)
                prune_states(args.out_dir, args.keep_state)
            dist_barrier(is_dist)

    # per-rank, because peak memory is genuinely a per-rank fact: rank 0 also carries
    # the EMA shadow copy and the quick-val decoders, so it is the one that sets the
    # memory ceiling for the job
    print(f"[mem] rank {rank}: peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
          f"peak reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB", flush=True)
    p0("DONE", flush=True)
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
