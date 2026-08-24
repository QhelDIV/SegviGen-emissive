# Which code did this run actually execute? (convention, 2026-08-23)

## The problem

SLURM copies the **sbatch script** when you submit. It does not copy anything the
script later reads. The trainer is an ordinary `.py` opened from the shared deploy
at job **start**, so a job that waits in the queue runs whatever is committed when
it *begins*, not what was there when you submitted it.

On 2026-08-23 the agentic run had two trainer commits land under it between submit
and start, from a teammate working correctly on their own brief. Both were
improvements and neither changed the recipe, so nothing was harmed. Nothing
detected it either: the run's sbatch header still named the commit that was current
at submit, which by then was two commits stale, and only a hash check caught it.

## What we do: record the fact, do not try to control it

At job start, `capture_trainer_version.sh <out_dir> <job_id> [label]` writes
`TRAINER_AT_START.txt` into the run's output directory with the deployed file's
md5, its git blob, and the commit that blob resolves to in the canonical checkout.
It does not assume the deploy is a git checkout or that it matches canonical; it
hashes what is actually there and says plainly when the blob matches no commit,
which is the case that would matter most. Git holds the bytes, so recording which
bytes ran is enough to recover them later.

When a run must be protected from mid-flight changes, freeze trainer commits for
the window between submit and start and say so on the board. That is a
coordination cost on teammates, but it is honest and it works.

## What we tried and REVERTED: copying the code into the run directory

The obvious stronger idea is for the sbatch to copy the trainer into `$OUT_DIR`
and execute that copy. **It was implemented, it failed in production, and it is
not in use.** The failure is worth writing down, because the idea will occur to
the next person too.

`train_emissive.py` finds the repo root by walking UP from its own file until it
sees `inference_full.py`, then puts that root on `sys.path`. Copying the entry
point alone therefore pins nothing: the walk-up still lands on the deploy. Copying
enough to move the walk-up (`inference_full.py` + `trellis2/` + `emissive/`) DOES
move it, and that is exactly the trap. `sys.path` now points at the copy, so every
import must be satisfied from the copy, and `inference_full.py` imports
`data_toolkit.bpy_render` at module scope. Job 248521 died on
`ModuleNotFoundError: No module named 'data_toolkit'` **after** it had already
cancelled the running single-GPU job, costing 53 minutes of training.

Two lessons, both paid for:

- **Verifying the mechanism is not verifying the result.** The walk-up was checked
  and confirmed to resolve to the pin. The three copied surfaces were confirmed
  present. The pinned copy was never actually run, so the missing fourth surface
  went unseen until it took a job down.
- **The cost estimate was wrong by a factor of fifty.** "1.9 MB and a couple of
  seconds" counted only what had been copied. `data_toolkit` alone is 93 MB, and
  the reachable import set also includes `o_voxel` and `trellis`, whose provenance
  (repo directory versus installed package) was never established. This repo's
  import surface is not cheaply pinnable. Anyone reviving this idea should start
  by enumerating the full transitive import set and measuring it, and should prove
  the copy runs by executing it, not by reasoning about `sys.path`.

## A takeover must prove it can start before it kills the incumbent

The same incident exposed a design fault independent of pinning. The takeover
scripts cancelled the running job as soon as they had an allocation, on the
assumption that having a GPU means being able to train. A job that cannot import
its own trainer holds a GPU perfectly well.

Both takeover scripts now run an import guard **before** the cancel:

```bash
echo "IMPORT_GUARD_START $(date -Is)"
python emissive/train/train_emissive.py --help > /dev/null
echo "IMPORT_GUARD_OK $(date -Is)"
```

`--help` executes every module-level import in the job's exact interpreter, path
and working directory, then exits 0. Under `set -e` an import fault aborts the
takeover and the incumbent keeps running. It costs about 11 seconds, measured on
job 248527, which is where it first ran for real, on the exact fault that had
caused the outage four minutes earlier.

Generalise it: **any job that terminates another job should first prove it can do
the work it is taking over.** An allocation is not a capability.

## Related

- `README_LOGS.md` — where a run's logs go (the run's own output directory).
- `README_DDP.md` — multi-GPU launches, and why a continuation writes to a NEW
  output directory rather than the one it resumes from.
