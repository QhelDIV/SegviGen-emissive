"""Tests for optimizer-state save and resume in train_emissive.py.

Why this exists. Every continuation to date has been a warm start with a FRESH
AdamW, because save_ckpt only ever wrote weights. A fresh AdamW arrives with zero
first and second moments, so its first steps are effectively unscaled and the loss
visibly resettles before recovering. That has been reproduced on three arms, and
with the fir run now extended by chained +8-epoch hops the tax repeats every hop.

What is pinned here:
  * the state round-trip: optimizer moments, scheduler position, epoch and best_iou
    all come back
  * OLD checkpoints stay loadable, and a checkpoint with no sidecar is an explicit
    error under --resume rather than a silent downgrade to the very behavior
    --resume exists to avoid
  * weights files are untouched in format, so eval_emissive and inference cannot be
    affected
  * pruning removes only sidecars, never weights

The end-to-end demonstration (run A of 2N steps against run B of N steps + resume,
showing B continues A's curve without the settle) needs a GPU and the real model,
so it lives in emissive/slurm/_optstate_demo.sbatch rather than here.

Run it (real torch needed for the state round-trip; the rest runs anywhere):
  python emissive/tests/test_optimizer_resume.py
"""
import io
import json
import os
import sys
import tempfile
import types
import contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "emissive", "eval"))
sys.path.insert(0, HERE)

from test_dataset_probe import load_trainer_module, captured, check, RESULTS  # noqa: E402

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class Args:
    """Just the fields save_state reads."""
    def __init__(self, **kw):
        self.lr = 1e-5
        self.ema = 0.999
        self.grad_accum = 1
        self.lr_schedule = "const"
        self.__dict__.update(kw)


def test_state_path_naming(M):
    check("sidecar name is derived from the weights name",
          M.state_path_for("/x/epoch_0008.ckpt") == "/x/epoch_0008_state.pt",
          M.state_path_for("/x/epoch_0008.ckpt"))
    check("a path without .ckpt still gets a sidecar name",
          M.state_path_for("/x/last") == "/x/last_state.pt")


def test_weights_format_unchanged(M):
    """The whole compatibility argument rests on this: save_ckpt still writes exactly
    one key, "state_dict", with gen3dseg-prefixed names. eval_emissive reads that and
    nothing else, so it cannot be affected by anything this feature adds."""
    if not HAVE_TORCH:
        check("weights format unchanged (needs torch)", True, "skipped")
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "epoch_0001.ckpt")
        M.save_ckpt({"a.weight": torch.zeros(2), "b.bias": torch.ones(3)}, path)
        blob = torch.load(path, map_location="cpu")
        check("weights file has exactly the one original key",
              list(blob.keys()) == ["state_dict"], str(list(blob.keys())))
        check("weights keys keep the gen3dseg prefix",
              list(blob["state_dict"].keys()) == ["gen3dseg.a.weight", "gen3dseg.b.bias"],
              str(list(blob["state_dict"].keys())))
        check("no optimizer state leaked into the weights file",
              "optimizer" not in blob and "ema" not in blob)


def test_state_round_trip(M):
    """Moments, scheduler position, epoch and best_iou must all survive."""
    if not HAVE_TORCH:
        check("state round-trip (needs torch)", True, "skipped -- no torch here")
        return
    with tempfile.TemporaryDirectory() as tmp:
        w = torch.nn.Parameter(torch.ones(4))
        opt = torch.optim.AdamW([w], lr=1e-5, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=5e-7)
        for _ in range(7):                     # build up non-zero moments
            opt.zero_grad()
            (w.sum() * 3.0).backward()
            opt.step()
            sched.step()
        before_exp_avg = opt.state[w]["exp_avg"].clone()
        before_step = int(opt.state[w]["step"].item()) if torch.is_tensor(opt.state[w]["step"]) \
            else int(opt.state[w]["step"])
        path = os.path.join(tmp, "epoch_0003_state.pt")
        M.save_state(path, opt, sched, epoch=3, best_iou=0.42, args=Args(), world_size=2)

        w2 = torch.nn.Parameter(torch.ones(4))
        opt2 = torch.optim.AdamW([w2], lr=1e-5, weight_decay=0.0)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=100, eta_min=5e-7)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        opt2.load_state_dict(blob["optimizer"])
        sched2.load_state_dict(blob["scheduler"])

        check("epoch position round-trips", blob["epoch"] == 3, str(blob.get("epoch")))
        check("best_iou round-trips", abs(blob["best_iou"] - 0.42) < 1e-12)
        check("optimizer moments round-trip",
              torch.allclose(opt2.state[w2]["exp_avg"], before_exp_avg))
        check("moments restored are NOT zero (a fresh AdamW would be)",
              before_exp_avg.abs().sum().item() > 0)
        after_step = int(opt2.state[w2]["step"].item()) if torch.is_tensor(opt2.state[w2]["step"]) \
            else int(opt2.state[w2]["step"])
        check("optimizer step counter round-trips", after_step == before_step,
              f"{after_step} vs {before_step}")
        check("scheduler position round-trips", sched2.last_epoch == sched.last_epoch,
              f"{sched2.last_epoch} vs {sched.last_epoch}")
        check("scheduler horizon travels with the state, not this leg's --epochs",
              sched2.T_max == 100, str(sched2.T_max))
        check("provenance is recorded for the resume-time lr check",
              blob["provenance"]["world_size"] == 2 and blob["provenance"]["lr"] == 1e-5)


def test_fresh_optimizer_really_differs(M):
    """The premise. A fresh AdamW has zero moments; that is the resettle being paid."""
    if not HAVE_TORCH:
        check("fresh optimizer differs (needs torch)", True, "skipped")
        return
    w = torch.nn.Parameter(torch.ones(4))
    opt = torch.optim.AdamW([w], lr=1e-5, weight_decay=0.0)
    check("a fresh AdamW starts with no per-parameter state at all", len(opt.state) == 0)


def test_prune_touches_only_sidecars(M):
    with tempfile.TemporaryDirectory() as tmp:
        for e in (1, 2, 3, 4):
            open(os.path.join(tmp, f"epoch_{e:04d}.ckpt"), "w").close()
            open(os.path.join(tmp, f"epoch_{e:04d}_state.pt"), "w").close()
        open(os.path.join(tmp, "epoch_0002_ema.ckpt"), "w").close()
        open(os.path.join(tmp, "last.ckpt"), "w").close()
        with captured():
            M.prune_states(tmp, keep=1)
        left = sorted(os.listdir(tmp))
        states = [f for f in left if f.endswith("_state.pt")]
        check("prune keeps only the newest sidecar", states == ["epoch_0004_state.pt"],
              str(states))
        check("prune removes no weights",
              all(f"epoch_{e:04d}.ckpt" in left for e in (1, 2, 3, 4)), str(left))
        check("prune removes no ema file", "epoch_0002_ema.ckpt" in left)
        check("prune removes no last.ckpt", "last.ckpt" in left)


def test_prune_keep_zero_keeps_all(M):
    with tempfile.TemporaryDirectory() as tmp:
        for e in (1, 2, 3):
            open(os.path.join(tmp, f"epoch_{e:04d}_state.pt"), "w").close()
        with captured():
            M.prune_states(tmp, keep=0)
        check("keep_state 0 retains every sidecar",
              len([f for f in os.listdir(tmp) if f.endswith("_state.pt")]) == 3)


def test_prune_orders_numerically_not_lexically(M):
    """epoch_0009 before epoch_0010: zero padding makes lexical order agree here, but
    the sort is on the parsed integer so it stays right past 9999 too."""
    with tempfile.TemporaryDirectory() as tmp:
        for e in (8, 9, 10, 11):
            open(os.path.join(tmp, f"epoch_{e:04d}_state.pt"), "w").close()
        with captured():
            M.prune_states(tmp, keep=2)
        states = sorted(f for f in os.listdir(tmp) if f.endswith("_state.pt"))
        check("prune keeps the two numerically newest",
              states == ["epoch_0010_state.pt", "epoch_0011_state.pt"], str(states))


def test_missing_sidecar_is_an_error_not_a_downgrade(M):
    """An old checkpoint under --resume must fail loudly. Falling back to a fresh
    AdamW would silently reintroduce the exact cost being removed."""
    src = open(os.path.join(REPO, "emissive", "train", "train_emissive.py")).read()
    seg = src.split("if args.resume:")[1].split("else:")[0]
    check("--resume raises when the sidecar is absent", "raise RuntimeError" in seg)
    check("the error points at --init_ckpt as the deliberate alternative",
          "--init_ckpt" in seg)
    check("--resume does not silently fall back",
          "resume_state = None" not in seg.split("raise RuntimeError")[1][:400])


def test_modes_are_announced(M):
    src = open(os.path.join(REPO, "emissive", "train", "train_emissive.py")).read()
    for m in ("RESUME_WITH_STATE", "RESUME_FRESH_OPT", "WARM_START_FRESH_OPT"):
        check(f"mode {m} exists and is logged", m in src)
    check("the mode is printed on the init line", '[init] {mode} from' in src)


def test_default_path_is_untouched(M):
    """No --resume: numbering starts at 1 and no state is loaded, exactly as before."""
    src = open(os.path.join(REPO, "emissive", "train", "train_emissive.py")).read()
    check("start_epoch is 1 without a resume",
          "start_epoch = (resume_state[\"epoch\"] + 1) if resume_state is not None else 1" in src)
    check("--fresh_opt without --resume is rejected rather than ignored",
          "--fresh_opt only means something with --resume" in src)


def test_ddp_loads_on_every_rank(M):
    """Each rank owns an optimizer over its own replica; DDP syncs gradients, never
    optimizer state. So the restore must not be rank-0 only."""
    src = open(os.path.join(REPO, "emissive", "train", "train_emissive.py")).read()
    seg = src.split("opt = torch.optim.AdamW")[1].split("if is_dist:")[0]
    check("optimizer restore is not gated on rank 0",
          "load_state_dict(resume_state" in seg and "rank == 0" not in seg, seg[:200])
    save_seg = src.split("if epoch % args.save_every")[1]
    check("sidecar is written under the rank-0 guard",
          "save_state(state_path_for(ep_path)" in save_seg.split("dist_barrier")[0])


def main():
    M = load_trainer_module()
    print("optimizer-state resume tests"
          + ("" if HAVE_TORCH else "  [no torch here: round-trip checks skip]"))
    for fn in (test_state_path_naming, test_weights_format_unchanged, test_state_round_trip,
               test_fresh_optimizer_really_differs, test_prune_touches_only_sidecars,
               test_prune_keep_zero_keeps_all, test_prune_orders_numerically_not_lexically,
               test_missing_sidecar_is_an_error_not_a_downgrade, test_modes_are_announced,
               test_default_path_is_untouched, test_ddp_loads_on_every_rank):
        print(f"\n{fn.__name__}:")
        fn(M)
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
