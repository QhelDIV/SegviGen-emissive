# Pinning the code a long run executes, and asserting its inputs

## The problem

SLURM copies the **sbatch script** when you submit. It does not copy anything the
script later reads. The trainer is an ordinary `.py` opened from the shared deploy
at job **start**, so a job that waits in the queue runs whatever is committed when
it *begins*, not what was there when you submitted it.

On 2026-08-23 the agentic run had two trainer commits land under it between submit
and start, from teammates working correctly on their own briefs. Both were
improvements, so nothing was harmed; nothing detected it either. The run's sbatch
header still named the commit that was current at submit, two commits stale, and
only a hash check caught it.

## 1. Pin the code: copy the import surface, and execute the copy

A long run's sbatch copies the code it needs into `$OUT_DIR/code` and runs **that
copy**, so the run is immune to anything committed after it starts and the exact
bytes that produced the checkpoints sit beside them.

The reference implementation is the fir agentic run,
`~/scratch/lightgen/env_build/fir_train_agentic.sbatch`, which has trained 36+
hours clean:

```bash
PIN=$OUT_DIR/code
rm -rf "$PIN"; mkdir -p "$PIN/emissive"
cp inference_full.py "$PIN/"
cp -r trellis2 data_toolkit "$PIN/"
cp -r emissive/train emissive/eval "$PIN/emissive/"
find "$PIN" -name __pycache__ -type d -prune -exec rm -rf {} +
(cd "$PIN" && find . -name '*.py' | sort | xargs md5sum > MD5SUMS)
cd "$PIN"
torchrun --standalone --nproc_per_node 4 "$PIN/emissive/train/train_emissive.py" ...
```

Two properties make it work, and both are load-bearing.

**The surface is complete.** `train_emissive.py` finds the repo root by walking UP
from its own file to the first `inference_full.py`, then puts that root on
`sys.path` ahead of the script's own directory. So a partial copy is worse than no
copy: the walk-up lands on the pin, `sys.path` points at the pin, and every import
must now be satisfiable there. The surface is `inference_full.py`, `trellis2/`,
`data_toolkit/`, and `emissive/{train,eval}/`. Do not write that list from memory.
Derive it and check it:

```bash
grep -hoE '^(from|import) [a-z_][a-z_0-9.]*' inference_full.py \
  emissive/train/train_emissive.py emissive/eval/*.py | sort -u
```
then keep every name that resolves to a path inside the repo. Run against the
current tree this yields `data_toolkit`, `inference_full` and `trellis2`, i.e. it
does catch the surface whose omission caused the outage.

**It does not catch everything, and the gap is the same shape as the original bug.**
`eval_emissive` is imported as a bare module name that only resolves because the
entry point inserts `ROOT/emissive/eval` onto `sys.path` itself. Grepping imports
finds the name but not the directory, because the directory is supplied at runtime
rather than by the repo layout. So the derivation is a floor, not a proof: add any
path the entry point puts on `sys.path` by hand (here `emissive/train` and
`emissive/eval`), and then rely on the execution check below to catch what both
steps missed. Re-derive whenever the entry point's imports or its `sys.path`
manipulation change.

**The copy is executed before anything irreversible.** Creating a pin proves
nothing; importing from it does. Any job that will terminate another job, or
overwrite anything, must first run

```bash
python "$PIN/emissive/train/train_emissive.py" --help > /dev/null
```

which executes every module-level import in the job's exact interpreter, path and
working directory, then exits 0. Under `set -e` an incomplete pin aborts the job
here instead of later. Generalised: **an allocation is not a capability, and a job
that takes work from another job must prove it can do that work first.**

### The failure this is written against

Solar job 248521 pinned a HAND-ENUMERATED surface: `inference_full.py`, `trellis2/`
and `emissive/`, but not `data_toolkit/`, which `inference_full.py` imports at
module scope. It cancelled the running single-GPU job, then died on
`ModuleNotFoundError: No module named 'data_toolkit'`, costing 53 minutes of
training and leaving nothing running for 25.

Two lessons, both paid for:

- **Verifying the mechanism is not verifying the result.** The walk-up was checked
  and confirmed to land on the pin. The three copied surfaces were confirmed
  present. The pinned copy was never actually run, so the missing fourth surface
  went unseen until it took a job down.
- **A cost estimate that counts only what you copied is not a cost estimate.**
  "1.9 MB" omitted `data_toolkit`, which is 93 MB.

The same mistake in a different costume killed job 248528 five seconds in: a helper
whose last command was a bare `[ test ] && printf` returned 1 on an empty
directory, and under `set -e` that took the script down. Its six-scenario test
harness ran under `set -uo pipefail` **without `-e`**. Test the shell the code runs
in, not only the logic.

## 2. Record what actually ran

`capture_trainer_version.sh <out_dir> <job_id> [label]` writes
`TRAINER_AT_START.txt` with the deployed file's md5, its git blob, and the commit
that blob resolves to in the canonical checkout. It does not assume the deploy is a
git checkout or that it matches canonical; it hashes what is there and says plainly
when the blob matches no commit.

Use it always. With a pin it corroborates; without one it is the only record.

## 3. Assert the inputs, not just the code

A pinned run reading changed data is as wrong as an unpinned run, and we currently
have no defence against it. The pattern worth copying comes from a TRELLIS.2
lightning job on fir (not our lane, credited for the idea): before training, it
asserts identity on everything it consumes and dies rather than proceed.

```bash
N=$(ls "${D}/emission_latents/..." | grep -c '\.npz$')
[ "${N}" -ge 72421 ] || { echo "FATAL: latents incomplete"; exit 1; }
GOT=$(md5sum "${D}/data_splits_...json" | cut -d' ' -f1)
[ "${GOT}" = "159492b2b8d104ab63f3d13eeea394d0" ] || { echo "FATAL: split md5 ${GOT}"; exit 1; }
```

It also refuses to start on a node whose GPUs already hold >2 GB, requeueing
instead of training beside someone else's leak.

Adopt: a count check on each input directory, an md5 check on each split or
manifest file, both fatal. A silently changed split is the one error class that
survives every other control in this document.

## What none of this pins

The conda environment. `torch`, `flash_attn` and the sparse extensions come from
the deploy's miniforge, and a change there reaches a pinned run's next launch.
Pinning the environment is a different and much heavier problem.

## Related

- `README_LOGS.md` — where a run's logs go.
- `README_DDP.md` — multi-GPU launches, and why a continuation writes to a NEW
  output directory rather than the one it resumes from.
