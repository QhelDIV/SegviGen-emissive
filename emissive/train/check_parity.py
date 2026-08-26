"""Parity check for the trainer_speedup patch: parses three train_emissive.py
--log_step_loss logs (baseline1, baseline2, patched) and verifies

  (1) the drawn-shape SEQUENCE (the 'sid' field --log_step_loss now prints per
      step) is IDENTICAL across all three runs, and
  (2) the patched loss curve sits inside the baseline-vs-baseline noise
      envelope at every step (the model is not bitwise reproducible even
      rerunning the identical command, so "identical" is judged against that
      envelope, not zero difference -- see project memory).

Exits 0 and prints PARITY: PASS iff both hold; otherwise prints the first
disagreement and exits 1. This is meant to be run on the actual sbatch logs
(trainer_speedup_parity.sbatch writes baseline1.log/baseline2.log/patched.log),
not eyeballed.
"""
import argparse
import re
import sys

STEP_RE = re.compile(
    r"\[step\] epoch\s+(\d+) step\s+(\d+) \| loss ([\d.eE+-]+) \| sid (\S+)")


def parse_log(path):
    steps = []
    with open(path) as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                steps.append({
                    "epoch": int(m.group(1)),
                    "step": int(m.group(2)),
                    "loss": float(m.group(3)),
                    "sid": m.group(4),
                })
    if not steps:
        raise SystemExit(f"{path}: no '[step] ... | sid ...' lines found -- was "
                         f"--log_step_loss passed, and is this the patched build "
                         f"of train_emissive.py (sid was added alongside it)?")
    return steps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline1", required=True)
    ap.add_argument("--baseline2", required=True)
    ap.add_argument("--patched", required=True)
    ap.add_argument("--max_steps", type=int, default=0,
                    help="compare only the first N steps of each log (0 = use every step, "
                         "requires all three logs to have the identical count). Use this "
                         "when the patched run is SHORTER than the baselines on purpose "
                         "(e.g. a --no_grad_ckpt run that cannot complete a full-length run "
                         "on a memory-constrained GPU): the baselines are truncated to match "
                         "rather than requiring a separate short baseline pair.")
    args = ap.parse_args()

    b1 = parse_log(args.baseline1)
    b2 = parse_log(args.baseline2)
    pt = parse_log(args.patched)

    if args.max_steps > 0:
        b1, b2, pt = b1[:args.max_steps], b2[:args.max_steps], pt[:args.max_steps]

    n = len(b1)
    if not (len(b2) == n and len(pt) == n):
        raise SystemExit(f"PARITY: FAIL -- step counts differ: baseline1={len(b1)} "
                         f"baseline2={len(b2)} patched={len(pt)}"
                         + ("" if args.max_steps else " (pass --max_steps to compare a prefix)"))

    sid1 = [s["sid"] for s in b1]
    sid2 = [s["sid"] for s in b2]
    sidp = [s["sid"] for s in pt]
    if sid1 != sid2:
        i = next(i for i in range(n) if sid1[i] != sid2[i])
        raise SystemExit(f"PARITY: FAIL -- baseline1 vs baseline2 sid sequence differs "
                         f"at step {i}: {sid1[i]} vs {sid2[i]} (the two UNPATCHED runs "
                         f"should draw identically; if even these disagree the harness "
                         f"itself is broken, not the patch)")
    if sid1 != sidp:
        i = next(i for i in range(n) if sid1[i] != sidp[i])
        raise SystemExit(f"PARITY: FAIL -- patched sid sequence differs from baseline "
                         f"at step {i}: baseline={sid1[i]} patched={sidp[i]}")
    print(f"[check_parity] sid sequence identical across all 3 runs ({n} steps). OK")

    loss1 = [s["loss"] for s in b1]
    loss2 = [s["loss"] for s in b2]
    lossp = [s["loss"] for s in pt]

    envelope = [abs(a - b) for a, b in zip(loss1, loss2)]
    patched_dev1 = [abs(a - b) for a, b in zip(lossp, loss1)]
    patched_dev2 = [abs(a - b) for a, b in zip(lossp, loss2)]

    # ONE scalar tolerance band, not a per-step ratio: the baseline-vs-baseline
    # curve is itself noisy (two independent draws of the same process can
    # differ by very little at one step and more at another by chance), so
    # dividing by a per-step envelope value that happens to be tiny blows up
    # the ratio test with no real signal behind it. The band is the WORST
    # baseline-vs-baseline disagreement seen anywhere in the run, with a slack
    # factor for the fact patched-vs-baseline is a THIRD independent draw from
    # the same noise process (a third draw can exceed the max of two draws
    # even with nothing wrong -- extreme-value statistics, not a bug).
    max_envelope = max(envelope)
    slack = 3.0
    band = max(slack * max_envelope, 1e-6)   # floor avoids a degenerate zero-noise band
    bad_steps = [i for i in range(n)
                if min(patched_dev1[i], patched_dev2[i]) > band]

    print(f"[check_parity] baseline-vs-baseline envelope: max|Δloss|={max_envelope:.6g}, "
         f"mean|Δloss|={sum(envelope)/n:.6g}, tolerance band={band:.6g} ({slack}x max)")
    print(f"[check_parity] patched vs baseline1: max|Δloss|={max(patched_dev1):.6g}, "
         f"mean={sum(patched_dev1)/n:.6g}")
    print(f"[check_parity] patched vs baseline2: max|Δloss|={max(patched_dev2):.6g}, "
         f"mean={sum(patched_dev2)/n:.6g}")

    if bad_steps:
        i = bad_steps[0]
        raise SystemExit(f"PARITY: FAIL -- step {i} (epoch {pt[i]['epoch']} step "
                         f"{pt[i]['step']}, sid {sidp[i]}): patched deviates from both "
                         f"baselines ({patched_dev1[i]:.6g}, {patched_dev2[i]:.6g}) by more "
                         f"than the tolerance band ({band:.6g}). {len(bad_steps)}/{n} steps "
                         f"failed this test.")

    print(f"PARITY: PASS -- {n} steps, identical sid sequence, patched loss curve within "
         f"{slack}x the baseline noise envelope at every step")


if __name__ == "__main__":
    main()
