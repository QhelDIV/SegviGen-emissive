"""How long does one gradient sync cost on this node?

DDP hides the gradient all-reduce inside the backward pass, so when multi-GPU
scaling comes out below linear the first question is whether the interconnect or
the compute is the limit. This measures the interconnect alone: an all-reduce of
exactly the byte count the emissive fine-tune's gradients occupy, on the same
node, through the same NCCL the trainer uses.

Run it the way the trainer runs:
  torchrun --standalone --nproc_per_node 4 emissive/probes/nccl_allreduce_probe.py --gib 2.5

Reported per size:
  wall        seconds for one all-reduce (median of the timed runs)
  alg BW      bytes / wall, the rate the caller sees
  bus BW      alg BW * 2(n-1)/n, the rate the links actually carry; this is the
              number to compare against the hardware's peak
"""
import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist


def bench(numel, dtype, rank, world, warmup=3, iters=10):
    x = torch.ones(numel, dtype=dtype, device="cuda")
    for _ in range(warmup):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    del x
    torch.cuda.empty_cache()
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gib", type=float, nargs="+", default=[0.25, 1.0, 2.5],
                    help="message sizes to time, in GiB. Pass the trainer's reported "
                         "gradient size ('[model] ... moves X GiB per rank') to get the "
                         "cost of one real sync.")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="bfloat16 matches the flow model's transformer weights, which "
                         "are the bulk of the gradient traffic.")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")

    dtype = getattr(torch, args.dtype)
    esize = torch.tensor([], dtype=dtype).element_size()
    if rank == 0:
        print(f"[probe] world_size={world} dtype={args.dtype} on {os.uname().nodename}", flush=True)
        print(f"{'GiB':>8} {'wall_s':>10} {'alg_GB/s':>10} {'bus_GB/s':>10}", flush=True)
    for gib in args.gib:
        numel = int(gib * (2 ** 30) / esize)
        wall = bench(numel, dtype, rank, world)
        nbytes = numel * esize
        alg = nbytes / wall / 1e9
        bus = alg * 2 * (world - 1) / world
        if rank == 0:
            print(f"{gib:8.2f} {wall:10.4f} {alg:10.2f} {bus:10.2f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
