# Autonomous run state (owner away until ~Wed) — live status

Status: active
TL;DR: Two 2k fine-tune arms training on the cluster; multi-draw evals + comparison
table + draft team update to follow. This file is the recovery anchor for any session.

## RESULT — HONEST FULL-VAL (111 shapes, K=4 averaged, @0.5 nonzero) — the definitive numbers
| model | @0.5 nonzero | zero-glow | tiny(0-5%] | large(>0.3) |
|---|---|---|---|---|
| **zero-shot oracle (BAR)** | **0.219** | — | — | — |
| old W5-EMA (1k) | 0.117 | 0.03 | — | — |
| new W5/2k best (ep6) | 0.103 | 0.32 | 0.055 | 0.319 |
| new W5/2k EMA | 0.107 | 0.16 | 0.036 | 0.426 |
| new balanced best (ep8) | 0.114 | 0.11 | 0.049 | 0.361 |
| new balanced EMA | 0.112 | 0.06 | 0.040 | 0.419 |

**CONCLUSION (corrects the earlier optimistic quick-val read):** NO fine-tune beats the 0.219
oracle. The 16-sample quick-val "0.179 peak" was NOISE — the same checkpoint scores 0.103 on the
full averaged eval. All four 2k checkpoints ≈ 0.10–0.11 nonzero, statistically tied with the old
1k model (0.117) — **the 2k emission-filtered data did NOT improve the honest number, and
balanced weighting ≈ W5.** Neither lever moved the ceiling. The multi-sample eval (draws=4) EARNED
ITS KEEP by catching a false +53% before it reached the team. Universal pattern: every model works
on large glow (0.32–0.43) and fails on tiny glow (0.04–0.06); tiny-glow shapes dominate the val set
→ **tiny-region segmentation is the real wall; data volume & loss weighting don't touch it.**
Next levers to consider: higher-res target/readout, tiny-region-specific objective, or accept the
gap and reframe (is voxel IoU even the right metric for tiny emitters?). Eval jobs: 231621-231624.

## LIVE ROADMAP (owner-facing real-time view) — KEEP IT FRESH
`ROADMAP.md` (repo root) → rendered to https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/roadmap/index.html
(linked at the top of the console home; auto-refreshes every 2 min). **Update protocol: on any
meaningful state change (arm finishes, evals fire, decision resolves, blocker appears), edit
ROADMAP.md and run `.venv_console/bin/python tools/build_roadmap.py`** (fast, no mkdocs). This is
the owner's quick-sync surface — a stale roadmap defeats its purpose.

## Where things stand (last updated 2026-07-07 ~03:00 UTC / 20:00 PDT)

**RUNNING — SLURM, independent of any agent session:**
- Job **231598** = arm `emis_2k_w5` — pos_weight 5, cosine LR, select_on nonzero, 20ep, train_2k_ef. ETA ~13h from 01:00 UTC → ~14:00 UTC Tue.
- Job **231599** = arm `emis_2k_bal` — balanced_pos_weight 50 (W_shape=(1-p)/p capped), cosine, nonzero, 20ep. Same ETA.
- Baseline evals **231582/231583** (old 1k W5-EMA / W1-EMA on val_96, draws=4, voxel buckets) — should finish ~03-04 UTC; results become the OLD rows of the comparison table.

**THE BAR (measured tonight, oracle_val96.json):** zero-shot SegviGen oracle on val_96 =
flat 0.395 / **nonzero 0.219** (86 glowing shapes). This is what the fine-tune must beat.
best_single_part 0.195, mean 5.5 parts. Old arms (from strata): W5-EMA ~0.12, W1-EMA ~0.08
nonzero — i.e. below oracle. Eval noise: draw-std 0.092 → single draws meaningless, always K≥4.

## BASELINE ROWS (measured, honest protocol K=4 voxel-buckets @0.5) — jobs 231582/231583 DONE
| model | @0.5 all | @0.5 nonzero | zero-glow bucket |
|---|---|---|---|
| oracle (bar) | 0.395 | **0.219** | 1.00 |
| old W5-EMA (eager) | 0.102 | **0.117** | 0.030 |
| old W1-EMA (timid) | 0.146 | **0.069** | 0.410 |
New arms must beat 0.117 (best old nonzero) and ideally approach the 0.219 oracle.
New arm ep2: W5/2k nonzero=0.107 (already ~old-W5 converged), balanced nonzero=0.037 (early).

## Dataset built tonight: train_2k_ef (task A, DONE + verified)
2,000 samples = 1,880 nonzero + 120 zero-glow (6%). Selection: 74k train partition → PBR-pass
(26,264) → has-label (−2,210) → −DC-221 −val_96 → nonzero>0 (18,210) + 120 reused negatives.
All 5 gates PASS (∩DC-221=0, ∩val_96=0, neg 6%, 2000 total, all have emis_mask). 166-sid
buffer moved to `dataset/train_2k_ef_extra/`. Full pool for scaling: `dataset/train_full_ef_pool.json`
(18,210). BUILD_INFO at `dataset/train_2k_ef/BUILD_INFO.md`.

## NEXT ACTIONS (for whoever is driving)
1. When 231598/231599 finish: for each arm find best epoch from `outputs/<arm>/train_curve.json`
   (max val_iou_nonzero), eval BOTH `best.ckpt` and `epoch_<best>_ema.ckpt`. Ready-to-paste
   (note eval_val96.sbatch is v4 = draws=1; use direct --wrap for the honest K=4 protocol):
   ```
   for A in emis_2k_w5 emis_2k_bal; do for C in best.ckpt <best_ema>.ckpt; do
     sbatch -p 3dlg-hcvc-lab-debug --gres=gpu:l40s:1 --time=3:00:00 -J emis_eval96 \
       -o /3dlg-jupiter-project/lightgen/segvigen_emissive/eval_%j.log \
       --wrap "source /3dlg-jupiter-project/lightgen/miniforge3/etc/profile.d/conda.sh && conda activate trellis2 && cd /3dlg-jupiter-project/lightgen/segvigen_emissive && python code/eval_emissive.py --dataset /3dlg-jupiter-project/lightgen/segvigen_emissive/dataset --ckpt outputs/$A/$C --split val_96 --cond real --draws 4 --bucket_by voxel --otsu --stratify"
   done; done
   ```
   Headline to read from each log: `HEADLINE IoU@0.5 nonzero` + the stratified buckets.
2. Build the comparison table: oracle(0.219 nz) vs old-W5/W1 vs new-2k-W5/balanced, all @0.5,
   nonzero-only, per-bucket. This is the Wednesday headline.
3. Update training_curves + finetune pages with new arms; refresh BRIEF; DRAFT (not send) a team
   update via the updates pipeline.

## STALLED / non-critical
- **vizworker session-limited** (resets 9:10pm PDT 2026-07-06). Stalled task: PBR-filter contrast
  gallery page (`lightgen/pbr_filter_v1`) — the "what's inside the non-PBR 56%" visual. Findings
  already recorded in battle plan; page is documentation, resume when vizworker is back or Wed.

## KEY FINDINGS BANKED TONIGHT
- **PBR filter is a tooling signature, not quality** (15× substancepainter gap; scans/baked
  hypothesis debunked). Non-PBR shapes are MORE label-emissive (86% vs 76%); filter discards
  ~25k usable train shapes to keep ~18k. Its input-quality justification was NEVER cleanly tested
  → named next experiment: PBR-filter ablation. (battle plan updated.)
- Process hygiene: ssh session ≠ remote process lifecycle (both a death and a zombie tonight) —
  memory `reference_cluster_process_hygiene.md`.

## Guardrails still in force
Cluster jobs + local/reversible only. NO git pushes to fork/team repo, NO team-facing messages
(Slack, the pending Dongchen leakage note) until owner returns. Draft, don't send.
