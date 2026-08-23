"""Run a trainer file with the global RNG pinned, so two versions of it can be
compared step for step.

train_emissive.py seeds nothing by default: the epoch draw comes from the global
CPU generator and the flow noise from the global CUDA one, both seeded
nondeterministically. Two runs of the SAME file therefore already disagree, which
makes "does the refactored file still behave identically" unanswerable by just
running both. Pinning the global seed before the trainer starts removes that, and
since the two versions consume the generators in the same order and the same
amounts, any remaining difference in the loss sequence is a real behavior change.

  python emissive/probes/pinned_rng_launcher.py 1234 path/to/train_emissive.py --dataset ... --out_dir ...

Everything after the trainer path is passed through as that trainer's argv.
"""
import runpy
import sys

import torch


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    seed = int(sys.argv[1])
    path = sys.argv[2]
    sys.argv = [path] + sys.argv[3:]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[pinned_rng] seed={seed} running {path}", flush=True)
    runpy.run_path(path, run_name="__main__")


if __name__ == "__main__":
    main()
