# Running the emissive fine-tune on more than one GPU

`emissive/train/train_emissive.py` takes one shape per step and spends almost all
of that step in the 1.3B sparse transformer, so the obvious way to make a run
finish sooner is to put several shapes through several GPUs at once. That is what
the DDP support does: one process per GPU, each holding a full copy of the model,
each running its own share of the epoch's shapes, with the gradients averaged
across processes before every optimizer step.

Nothing about the recipe changes. The same flags mean the same things, and a
checkpoint written by a 4-GPU run is an ordinary checkpoint that `eval_emissive.py`
loads like any other.

## Launching

Single GPU is unchanged and needs no new flags:

```bash
python emissive/train/train_emissive.py --dataset ... --out_dir ...
```

Multi-GPU goes through `torchrun`, with the same flags after the script name:

```bash
torchrun --standalone --nproc_per_node 4 emissive/train/train_emissive.py \
  --dataset ... --out_dir ...
```

`--standalone` means one node; `torchrun` picks its own rendezvous port, so
nothing needs coordinating with SLURM. Without `torchrun`, `WORLD_SIZE` is unset,
the script skips every distributed branch, and the run behaves exactly as it did
before this support existed. Existing sbatch files therefore need no edits.

`emissive/slurm/train_ddp_template.sbatch` is a ready 4-GPU sbatch to copy.

## Switching a run that is already going to 4 GPUs

`save_ckpt` has only ever written the model's `state_dict`, so there is no
optimizer state in any checkpoint and there never was: continuing a run always
means a fresh AdamW at the stated learning rate. That is as true across a GPU
count change as it was for the single-GPU continuations already in `outputs/`.

The recipe, taking `emis_72kv2_cond_pw1b_filtered` as the example:

1. Find the run's latest per-epoch checkpoint:
   `ls -t /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72kv2_cond_pw1b_filtered/epoch_*.ckpt | head -1`
2. Copy `train_ddp_template.sbatch`, and in the copy:
   - point `--init_ckpt` at that checkpoint
   - set `--out_dir` to a NEW directory, so the continuation's curve, checkpoints
     and log do not overwrite the ones that got you here
   - carry every other recipe flag over unchanged
   - set `--epochs` to the number of ADDITIONAL epochs you want
3. Keep `--lr_schedule const`. A cosine schedule restarted mid-run would sweep the
   learning rate a second time from the top, which is not a continuation.
4. Submit.

A complete example, continuing from epoch 8 for 8 more epochs on 4 GPUs:

```bash
torchrun --standalone --nproc_per_node 4 emissive/train/train_emissive.py \
  --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct \
  --train_split train_72k_nonzero_nocopy \
  --val_split val_72k \
  --out_dir /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72kv2_cond_pw1b_filtered_ddp \
  --init_ckpt /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_72kv2_cond_pw1b_filtered/epoch_0008.ckpt \
  --cond real \
  --pos_weight 1.0 \
  --balanced_pos_weight 0.0 \
  --lr_schedule const \
  --lr 1e-5 \
  --emis_oversample \
  --ema 0.999 \
  --val_quick 8 \
  --select_on nonzero \
  --epochs 8 \
  --save_every 2
```

## What the flags mean once there is more than one GPU

**`--n_per_epoch` counts shapes GLOBALLY**, not per rank. `--n_per_epoch 4000` on
4 GPUs is 1000 shapes per GPU, the same 4000 shapes an epoch as on 1 GPU, so a
value carried over from a single-GPU run keeps its meaning and epochs stay
comparable across run widths. `--n_per_epoch 0` (the whole split) likewise stays
one pass over the split. A remainder of fewer shapes than there are GPUs is
dropped, so every rank runs the same number of steps.

**`--emis_oversample` is unchanged.** Every rank computes the same weighted draw
for the epoch from the same seeded generator, then takes every Nth entry of it.
The distribution of shapes an epoch sees is exactly the single-GPU distribution;
the only difference is which process handles which shape.

**The effective batch is `world_size * grad_accum` shapes.** This is the one real
change in training semantics. Four GPUs means four shapes averaged into every
optimizer step, so an epoch does a quarter as many optimizer steps as it did, each
on a four times less noisy gradient. Coming from a batch of one, that trade is
very likely favourable, but it is a trade, not a free speedup: wall clock per
epoch and learning progress per epoch are different quantities. It also means the
EMA at `--ema 0.999` now averages over a quarter as many updates per epoch, so its
effective horizon in shapes is four times longer.

**Learning rate.** The template keeps `--lr 1e-5`, the single-GPU value. Raising
it to match the larger batch is the textbook move and may well be right, but it is
a recipe change and should be decided and evidenced separately, not folded into a
change of GPU count.

**`--grad_accum`** accumulates that many shapes per rank before syncing. Leave it
at 1 unless the throughput evidence says the gradient all-reduce is the limit; it
multiplies the effective batch further, with all the consequences above.

**`--seed`** fixes the epoch draw and the per-shape flow noise, keyed by position
in the global draw so a shape gets the same noise at any GPU count. It exists for
comparing run widths against each other. Leave it unset for real runs.

## What runs where

Rank 0 does everything that is not a training step: it scans the split directory
once and sends the list to the others, keeps the EMA shadow copy, writes the
checkpoints, the log and the curve, and runs quick-val while the other ranks wait
at a barrier. The replicas hold identical weights, so there is nothing to gather
before saving. Rank 0 therefore has the highest memory use of the four, and it is
the one that sets the job's memory ceiling.

Only rank 0 prints. The loss on the epoch line is averaged across all ranks, so it
is the mean over every shape the epoch touched and means the same thing at any
width.

## Resource asks

Four ranks are four full processes on one node. `train_ddp_template.sbatch` asks
for 16 CPUs and 256G of host memory against the single-GPU run's 8 and 64G, and
sets `OMP_NUM_THREADS=4` so the ranks do not fight over cores while loading their
`.pth` files.
