# Emissive fine-tune: experiment & checkpoint registry

This is the registry an agent (or a person) should read before picking a checkpoint for
inference (`emissive/infer/predict_emissive.py`) or before launching a new training run.
Numbers are cross-checked against `notes/2026-07-07_autonomous_run_state.md` and
`ROADMAP.md` in the parent `lightgen` project tree; those files win on any conflict with
this one (they are updated live during a run, this file is a point-in-time snapshot).

## Quick start (condensed — see root README.md for the full version)

```
python emissive/infer/predict_emissive.py \
    --glb YOUR.glb --out OUTDIR \
    [--draws 4] [--thr 0.5] [--zero_cond | --image cond.png] [--ckpt /path/to/ckpt]
```
Default `--ckpt` = the recommendation below (`emis_1k_w5` epoch 16 EMA). Outputs
`OUTDIR/mask.npz` (`coords` int32 @512-res, `prob` float32, `mask` bool),
`pred_mesh.glb` (white=emissive), `meta.json`. Template sbatch:
`emissive/slurm/predict_smoke.sbatch`. GPU-tested 2026-07-16 (job 232600) — see
"predict_emissive.py status" below for the full validation record.

All checkpoint paths are on the `/3dlg-jupiter-project` NFS mount (jupiter cluster). Every
serious run directory holds `epoch_NNNN.ckpt` + `epoch_NNNN_ema.ckpt` (~2.62 GB each) plus
`best.ckpt` (symlink to the epoch selected by `--select_on`, see
`emissive/train/train_emissive.py`) and `last.ckpt`.

## The headline number, and why nothing here is a slam dunk

**No fine-tune in this registry beats the zero-shot SegviGen oracle (0.219 nonzero IoU).**
The oracle is not a real prediction pipeline — it labels each *ground-truth* part-segment
emissive iff >50% of its voxels are truthfully emissive, i.e. it's told the correct part
boundaries and only has to guess the label — so it is **not deployable**, but it is the bar
any real fine-tune must clear to be worth shipping. None do. The best deployable
checkpoints (emis_1k_w5 EMA at 0.117, emis_2k_bal best at 0.114) are statistically tied
with each other and sit ~0.10 below the oracle. Every model, regardless of data volume or
loss weighting, is strong on large-glow shapes (0.32–0.43 IoU) and fails on tiny-glow ones
(0.04–0.06) — tiny-emissive-region segmentation is the real bottleneck; more data and
reweighting the loss both failed to move it (see `notes/2026-07-07_autonomous_run_state.md`
for the full autopsy).

## Recommended checkpoint for inference

```
/3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_1k_w5/epoch_0016_ema.ckpt
```
(= `outputs/emis_1k_w5/best.ckpt`'s EMA counterpart — `best.ckpt` is a symlink to
`epoch_0016.ckpt`; the EMA shadow weights at the same epoch are the ones actually
recommended here, not the raw epoch_0016.ckpt.)

This is `predict_emissive.py --ckpt`'s default. **Why this one, given it's ~tied with
emis_2k_bal (0.117 vs 0.114 nonzero):**
- It's a smaller, simpler, better-understood run (1k unfiltered data, fixed pos_weight=5)
  than emis_2k_bal (2k emission-filtered data, adaptive per-shape balanced pos_weight) —
  fewer moving parts to explain if a downstream consumer asks "why does it predict this".
- The 2k arms did not improve on the 1k number despite 2x cleaner data and a fancier loss
  — i.e. the extra complexity in emis_2k_bal bought nothing measurable, so there's no
  reason to prefer it over the simpler 1k/W5 run.
- Both are well inside each other's noise band (draw-std ≈0.09 IoU at K=1; even at K=4 the
  two are not distinguishably different) — this is a coin-flip between statistically tied
  checkpoints, not a case where the "better" one is obvious. Re-run this comparison if a
  future arm clears the gap unambiguously.

If reproducing the exact eval numbers below rather than doing fresh inference, use the
non-EMA `best.ckpt` variants — the numbers in the table were measured against whichever
variant (best vs EMA) is listed in that row.

## Run registry (current / relevant)

| run | data | loss | ckpt dir | best epoch | headline nonzero IoU (K=4, @0.5) | status / notes |
|---|---|---|---|---|---|---|
| **zero-shot oracle** | n/a (SegviGen `full_seg.ckpt`, no fine-tune) | n/a | `full_seg.ckpt` (HF cache) | n/a | **0.219** | THE BAR — oracle-assisted (told correct part boundaries), **NOT deployable**. Zero-glow bucket = 1.00 (trivial). |
| emis_1k_w5 | 1k (unfiltered) | fixed pos_weight=5 | `outputs/emis_1k_w5/` | ep16 (best), ep16 EMA | best 0.102 (all) / **0.117 (nz, EMA)** | **RECOMMENDED for inference** (see above). Zero-glow bucket 0.03. |
| emis_1k_w1 | 1k (unfiltered) | fixed pos_weight=1 | `outputs/emis_1k_w1/` | ep14 (best), ep14 EMA | 0.146 (all) / 0.069 (nz, EMA) | Timid (predicts less emissive) — wins on the gamed "all" aggregate (25 zero-glow val shapes reward saying nothing), loses on nonzero. Zero-glow bucket 0.41 (near-trivial). Do not use nonzero-blind. |
| emis_2k_w5 | 2k emission-filtered (`train_2k_ef`) | fixed pos_weight=5, cosine LR, select_on=nonzero | `outputs/emis_2k_w5/` | ep6 (best) / ep6 EMA | 0.103 (best) / 0.107 (EMA) | 2x cleaner+bigger data did NOT beat 1k. Large-glow 0.319 (best) / 0.426 (EMA); tiny-glow 0.055 (best) / 0.036 (EMA). |
| emis_2k_bal | 2k emission-filtered (`train_2k_ef`) | adaptive per-shape balanced pos_weight (cap 50), cosine LR, select_on=nonzero | `outputs/emis_2k_bal/` | ep8 (best) / ep8 EMA | **0.114 (best)** / 0.112 (EMA) | Balanced weighting ≈ fixed W5, no real improvement. Large-glow 0.361 (best) / 0.419 (EMA); tiny-glow 0.049 (best) / 0.040 (EMA). |

Every arm above: strong on large-glow shapes (>0.3 GT coverage, IoU 0.32–0.43), weak on
tiny-glow shapes ((0,0.05] GT coverage, IoU 0.04–0.06). Tiny-glow shapes dominate the
111-shape val set — this is why the flat/nonzero headline numbers stay low even when
large-glow performance looks reasonable.

### Historical / pilot table (superseded — collapsed here, kept for provenance)

<details>
<summary>Pre-registry runs (232/512-sample pilots, sanity checks, gate/overfit probes)</summary>

| run | data | epoch | val IoU (protocol) | notes |
|---|---|---|---|---|
| emis_pilot | 232 | ep25 | 0.203 (16-sample val, thr 0.2) | first full fine-tune; ep50 collapsed to 0.042 |
| emis_v2 | 232 | ep30 | 0.119 (16-sample val, thr 0.2) | oversampling added; still 232 data |
| emis_v3 | 512 | ep4 | 0.176 (16-sample val, zero-cond, thr 0.2) | ep8 collapsed to 0.063 — majority-class collapse, not overfitting (train also dropped) |
| emis_real / emis_real_pbr | 512 | ep2 | 0.230 (16-sample val, real DINOv3 cond, thr 0.3) | best pre-2k result; NOT the honest K=4/full-val protocol — not comparable to the registry table above |
| emis_overfit_1 | 1 sample | ep80 | 0.968 | pipeline-verification only (single-sample convergence), not a generalization number |
| gate10_v5, gate10_w5 | 10-shape canonical gate | ep5 / ep150 | n/a | v5-trainer smoke gates, not held-out eval |
| overfit10_w5(_ext), overfit1_w5(_fixed/_zerocond/_ctrl) | 1–10 shapes | various | n/a | overfitting sanity checks (loop/decode correctness), not generalization numbers |
| train_smoke | n/a | n/a | n/a | environment smoke test, no real training |

⚠ **16-sample val is noisy** (±0.03–0.05 by WORKLOG's own estimate; the later K=4/111-shape
protocol found draw-std ≈0.09 at K=1) — none of the pilot numbers above are directly
comparable to the registry table's K=4/full-val numbers. Do not use a pilot row to argue a
pilot beats the registry's current best; re-run under the current protocol first.

</details>

## Eval protocol summary

All "headline nonzero IoU" numbers above use the protocol in
`emissive/eval/eval_emissive.py`:
- **`--draws 4`**: 4 independent flow-matching samples per shape, averaged (a single draw
  has draw-std ≈0.09 IoU — swings large enough to flip which arm looks best; K=4 is the
  minimum for an honest comparison. This caught a false +53% "improvement" during the 2k
  run — see `notes/2026-07-07_autonomous_run_state.md`).
- **`--bucket_by voxel`** (default): GT-coverage buckets for `--stratify` come from
  `emis_mask.pth` (actual surface-voxel occupancy, built in `make_emis_mask.py`), not mesh
  face-area fraction (`--bucket_by face`, kept only for comparison — biased by
  tessellation).
- **threshold = 0.5** on the decoded, averaged per-voxel base-color: GT itself is fixed
  white(1)/black(0) at exactly 0.5, so 0.5 is the natural cut; `--otsu` reports an adaptive
  per-shape alternative but is not the headline number.
- **nonzero aggregate**: mean IoU restricted to shapes with GT coverage>0 (86 of 111 val
  shapes). The flat "all" mean is gamed by the 25 zero-glow shapes (any model that predicts
  all-black scores IoU=1.0 on those by convention) — nonzero is the honest headline.
- Full CLI: `python emissive/eval/eval_emissive.py --dataset <dataset> --ckpt <ckpt> \
  --split val_96 --cond real --draws 4 --bucket_by voxel --otsu --stratify` (see
  `emissive/slurm/eval_val96.sbatch` for the canonical sbatch wrapper).

## predict_emissive.py status

**Tested on GPU 2026-07-16** (job 232600, `3dlg-hcvc-lab-debug` partition, l40s, exit
0:0, 00:01:52 elapsed). Command run (from a repo clone rsynced to
`/3dlg-jupiter-project/lightgen/segvigen_emissive/repo_reorg_smoke/`, a scratch dir — the
live `code/`/`outputs/` dirs were untouched):
```
cd /3dlg-jupiter-project/lightgen/segvigen_emissive/repo_reorg_smoke
python emissive/infer/predict_emissive.py \
  --glb /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset/overfit_10/ce28711b7d614918a7239b97c089d311/glb/ce28711b7d614918a7239b97c089d311_input.glb \
  --out /3dlg-jupiter-project/lightgen/segvigen_emissive/repo_reorg_smoke/smoke_out \
  --draws 2 --steps 12 --zero_cond
```
(uses the default ckpt, `emis_1k_w5/epoch_0016_ema.ckpt`.) Ran end-to-end in 109.9s:
glb_to_vxz → shape/tex slat encode → 2-draw sample → decode → threshold → `slat_to_glb`
remesh+bake. Verified output, not just exit code: `mask.npz` has 890,597 voxels
(coords int32, prob float32 in [0.46, 1.03], mask bool, 41.1% emissive — not degenerate),
`pred_mesh.glb` loads in trimesh (1 geometry, 80,878 vertices, 98,788 faces).

First attempt (job 232599) caught a real bug: the sibling-`eval/`-dir `sys.path` insert
used `os.path.join(ROOT, "eval")` instead of `os.path.join(ROOT, "emissive", "eval")`
(`ROOT` is the repo root after the walk-up shim runs, not `emissive/infer/`) —
`ModuleNotFoundError: No module named 'eval_emissive'`. Fixed, re-verified `py_compile`
+ `--help` locally, re-ran clean on 232600.

Not yet exercised by this smoke test: the real-cond path (`--image` / render-from-glb,
needs `bpy` in the trellis2 env) — this run used `--zero_cond`. Recommend a follow-up
smoke test dropping `--zero_cond` before relying on real-cond inference.
