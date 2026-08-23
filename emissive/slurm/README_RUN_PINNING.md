# Pin the code a long run executes (convention, 2026-08-23)

## The problem this exists for

SLURM copies the **sbatch script** when you submit. It does not copy anything the
script later reads. The trainer is an ordinary `.py` opened from the shared deploy
at job **start**, so a job that waits in the queue runs whatever is committed when
it *begins*, not what was there when you submitted it.

On 2026-08-23 the agentic run had two trainer commits land under it between submit
and start, from a teammate working correctly on their own brief: `b1742dc` (DDP
support) and `76eb677` (the readdir probe). Both were improvements and neither
changed the recipe, so nothing was harmed. Nothing detected it either. The run's
own sbatch header still named the commit that was current at submit, which by then
was two commits stale, and only a hash check caught it.

A shared deploy plus queued multi-day jobs plus several agents committing means
this is the normal case, not an unlucky one.

## The convention

**A long run's sbatch copies the code it needs into `$OUT_DIR/code` as its first
action and executes that copy.** The run is then immune to anything committed
after it starts, and the exact bytes that produced the checkpoints sit next to
them.

Copying the entry point alone is NOT enough, and this is the part that is easy to
get wrong. `train_emissive.py` finds the repo root by walking UP from its own file
until it sees `inference_full.py`, then puts that root on `sys.path` AHEAD of the
script's own directory. A lone copy therefore still imports `inference_full`,
`eval_emissive` and `trellis2` from the live deploy: you would pin the file you
were watching and leave everything it calls unpinned. Copy the whole import
surface under one directory so the walk-up lands on the pin instead:

```bash
OUT_DIR=/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/<run_name>
PIN=$OUT_DIR/code
test -n "$OUT_DIR"
rm -rf "$PIN"
mkdir -p "$PIN/emissive"
cp inference_full.py "$PIN/"
cp -r trellis2 "$PIN/"
cp -r emissive/train emissive/eval "$PIN/emissive/"
find "$PIN" -name __pycache__ -type d -prune -exec rm -rf {} +
(cd "$PIN" && find . -name '*.py' | sort | xargs md5sum > MD5SUMS)

python "$PIN/emissive/train/train_emissive.py" --dataset ... --out_dir "$OUT_DIR" ...
```

For a `torchrun` launch, the same pinned path goes after the launcher:

```bash
torchrun --standalone --nproc_per_node 4 "$PIN/emissive/train/train_emissive.py" ...
```

Verify the walk-up lands on the pin rather than assuming it does. From the pinned
entry point, `os.path.dirname` upward to the first `inference_full.py` must resolve
to `$PIN`. It does for anything under `outputs/`, and it would silently resolve to
the deploy root if the pin were placed outside the repo with no `inference_full.py`
above it.

Measured cost: 1.9 MB and a couple of seconds at job start, nothing in the queue.

**What this does not pin:** the conda environment. `torch`, `flash_attn` and the
rest come from `trellis2` on the deploy's miniforge, and a change there would still
reach a running job's next launch. Pinning the environment is a different and much
heavier problem; this convention covers the repo code only, which is what changes
day to day.

## When the run is already queued

You cannot fix a submitted job this way, because SLURM already took its script.
Cancelling to re-pin costs the queue slot, which on a busy cluster is hours. The
proportionate fallback is to record the **fact** of what ran rather than to control
it: `capture_trainer_version.sh <out_dir> <job_id> [label]` writes
`TRAINER_AT_START.txt` into the run's output directory with the deployed file's
md5, its git blob, and the commit that blob resolves to in the canonical checkout.
It does not assume the deploy is a git checkout or that it matches the canonical
repo; it hashes what is actually there and says plainly when the blob matches no
commit. Git holds the bytes, so pinning the fact is enough to recover them.

That capture happens shortly after the job starts rather than atomically at start,
so it is only trustworthy while trainer commits are frozen. Freezing commits is a
coordination cost on everyone else. Self-pinning has no such cost, which is why it
is the convention and the capture is the fallback.

## Related

- `README_LOGS.md` — where a run's logs go (the run's own output directory).
- `README_DDP.md` — multi-GPU launches, and why a continuation writes to a NEW
  output directory rather than the one it resumes from.
