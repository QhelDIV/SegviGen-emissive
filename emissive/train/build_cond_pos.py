"""One-time sidecar conversion: write cond_pos.pth (just the POS conditioning
tensor) beside every sample's cond.pth, so train_emissive.py --cond_file
cond_pos.pth can read half the bytes per sample.

Why this exists. cond.pth is written by build_dataset.py --real_cond as a dict
with BOTH halves of a classifier-free-guidance pair:
    {"cond": (1,1029,1024) fp32, "neg_cond": (1,1029,1024) fp32}
measured at ~8.43MB/file (4.21MB each). train_emissive.py's --cond real path
only ever reads ["cond"]; the neg half is read over NFS and discarded, every
sample, every epoch. torch.load has no partial-read mode for a pickled dict,
so the only way to stop paying for those bytes is a file that doesn't have
them -- hence this sidecar, built once, read forever after.

This NEVER touches, renames, or deletes cond.pth. cond_pos.pth is written to a
temp name in the same directory and atomically renamed into place, so a job
killed mid-conversion leaves either nothing or a complete file, never a
truncated one that would (per train_emissive.py's os.path.isfile check) get
silently picked up half-written.

Usage:
  # convert every sample in a split
  python build_cond_pos.py --dataset /path/to/dataset_direct --split train_72k_agentic

  # convert only the sample dirs listed one per line (e.g. a bench/parity subset) --
  # for measuring the byte reduction on a fixed subset without converting a whole split
  python build_cond_pos.py --dirs_file /tmp/bench_subset_dirs.txt

  # report bytes without writing anything
  python build_cond_pos.py --dataset ... --split ... --dry_run
"""
import os
import sys
import argparse
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(ROOT, "inference_full.py")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError(f"could not locate SegviGen repo root above {__file__}")
    ROOT = parent
sys.path.insert(0, os.path.join(ROOT, "emissive", "eval"))
from fsprobe import probe_exists, new_stats, summary_line   # noqa: E402

import torch   # noqa: E402


def iter_sample_dirs(args):
    if args.dirs_file:
        with open(args.dirs_file) as f:
            for line in f:
                d = line.strip()
                if d:
                    yield d
        return
    sdir = os.path.join(args.dataset, args.split)
    for sid in sorted(os.listdir(sdir)):
        yield os.path.join(sdir, sid)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None, help="dataset root, used with --split")
    ap.add_argument("--split", default=None, help="split name under --dataset (e.g. train_72k_agentic)")
    ap.add_argument("--dirs_file", default=None,
                    help="explicit list of sample dirs, one per line, instead of --dataset/--split "
                         "(e.g. to convert exactly a bench/parity subset)")
    ap.add_argument("--out_name", default="cond_pos.pth", help="sidecar filename to write")
    ap.add_argument("--force", action="store_true", default=False,
                    help="rewrite the sidecar even if it already exists")
    ap.add_argument("--dry_run", action="store_true", default=False,
                    help="scan and report byte counts; write nothing")
    ap.add_argument("--limit", type=int, default=0, help="stop after N samples (0 = no limit)")
    args = ap.parse_args()
    if not args.dirs_file and not (args.dataset and args.split):
        raise RuntimeError("pass either --dirs_file, or both --dataset and --split")

    stats = new_stats()
    n_scanned = n_converted = n_would_convert = n_skipped_exists = n_missing_cond = 0
    bytes_before = bytes_after = 0
    t0 = time.time()
    for i, d in enumerate(iter_sample_dirs(args)):
        if args.limit and i >= args.limit:
            break
        n_scanned += 1
        cond_path = os.path.join(d, "cond.pth")
        if not probe_exists(cond_path, stats):
            n_missing_cond += 1
            continue
        out_path = os.path.join(d, args.out_name)
        if os.path.isfile(out_path) and not args.force:
            n_skipped_exists += 1
            bytes_before += os.path.getsize(cond_path)
            bytes_after += os.path.getsize(out_path)
            continue
        payload = torch.load(cond_path, map_location="cpu")
        cond = payload["cond"] if isinstance(payload, dict) else payload
        src_bytes = os.path.getsize(cond_path)
        if args.dry_run:
            # estimate the sidecar size without writing: raw tensor storage bytes
            # (this is what torch.save of a bare tensor stores, modulo a small
            # pickle header -- close enough for a byte-reduction estimate)
            out_bytes = cond.element_size() * cond.nelement()
            n_would_convert += 1
        else:
            tmp_path = out_path + f".tmp{os.getpid()}"
            torch.save(cond, tmp_path)
            os.replace(tmp_path, out_path)   # atomic within the same directory
            out_bytes = os.path.getsize(out_path)
            n_converted += 1
        bytes_before += src_bytes
        bytes_after += out_bytes
        if n_scanned % 2000 == 0:
            print(f"[build_cond_pos] {n_scanned} scanned, {n_converted} written "
                  f"({time.time() - t0:.0f}s)", flush=True)

    print(summary_line(stats, "build_cond_pos scan"), flush=True)
    print(f"[build_cond_pos] {'DRY RUN: would write' if args.dry_run else 'wrote'} "
          f"{n_would_convert if args.dry_run else n_converted} "
          f"{args.out_name} sidecars (of {n_scanned} scanned); {n_skipped_exists} already existed; "
          f"{n_missing_cond} samples had no cond.pth (left untouched)", flush=True)
    if bytes_before > 0:
        pct = 100.0 * (1 - bytes_after / bytes_before)
        print(f"[build_cond_pos] bytes measured over the converted+existing set: "
              f"{bytes_before/1e6:.1f}MB (cond.pth) -> {bytes_after/1e6:.1f}MB ({args.out_name}), "
              f"{pct:.1f}% reduction", flush=True)


if __name__ == "__main__":
    main()
