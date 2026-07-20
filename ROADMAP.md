# Lightgen — live roadmap

_The real-time "where are we / what's next / what needs you" view. I edit this file as
state changes and republish; the page auto-refreshes every 2 min. For the deep story see
the BRIEF / battle plan; this is the operational now._

## Pipeline
- done: 2k dataset (emission-filtered)
- done: eval v5 + oracle bar
- done: baselines measured
- done: 2 models trained (peaked ep6 / ep8)
- done: averaged eval (honest full-val)
- done: comparison table + BRIEF
- todo: update pages + BRIEF
- done: draft team update (page + PDF)
- done: paper-style mesh renders
- wait: your review (Wed) ←

## ▶ Now (active)

- 🔧 **pipelineworker** (Opus): ✅ **FIX CONFIRMED POPULATION-WIDE** — 54-shape corr-to-render:
  broken 0.25 → **fixed 0.83, PAST incumbent somage 0.72**. Fixed direct GT is now the BEST
  emissive-GT source (even recovers glow the bake missed). False-pos 0.33→0.01; teddy 0.96→0.00;
  factor-only validated (7/9 clean, 1 material-semantics footnote). Direct-GLB pipeline not just
  recovered — strictly better. Now: 4-way render panels + PR-ready patch draft (owner-gated).
- ✅ **collaborator page COMPLETE** (`_preview/emissive_gt`, master-built + master-verified):
  full 4-way comparison matrix (teddy 96%→0%, sign recovered, sword localized), all 25 renders load,
  QA-clean at every width. Awaiting owner sign-off on framing/title → promote to workspace + share.
- 🖥️ **consoleworker** (Sonnet): adding the research WORKSPACE zone (advisor-facing, lite — zone
  + switcher + zone-link guard, no versioning/annotation). Home for the collaborator page.
- ✍️ **master (me)**: hand-building the crystal-clear collaborator page — emissive-GT story: what we
  need → two extraction routes → the one-line bug (with C++) → fair 4-way visual comparison → rec.

- ⚙️ **pipelineworker** (Opus): 🚨 **verdict (a) — o_voxel's per-voxel `emissive` attr is BROKEN**
  (wrong texture lookup in the compiled voxelizer: black tex→fabricated glow 22–67% on truly
  zero-glow shapes, bright tex→missed glow; factor-scaled; confirmed by controlled factor-zeroing
  test). Explains upstream's `del attr['emissive']` — it's known-bad, not a choice. Somage GT
  corr-to-true 0.73 vs direct 0.45 → current GT is the better proxy. Pilot reframed: evidence
  batch + verdict page in progress. Constructive path: UV-sample the ORIGINAL glb's
  emissiveTexture per-face → rasterize to voxels (one hop, no somage, no broken attr).
  ⚠ The pipeline_glb_direct explainer's "supports emission, just switched off" claim needs
  correcting BEFORE team share.
- ✅ **console migrated to v13/v3-shell** (master-verified live) — left page-tree, itables pages
  database, jday daily-report genre, today's report `updates/2026-07-19`. BRIEF kept as Overview
  (deviates from somages' retire-to-LOG, deliberately — ours isn't stale). Stale v3-gating note
  in the skill corrected + pushed (dotclaude `fdd5a70`).

- ✅ **fork reorg + registry + predict_emissive.py DONE, GPU-verified** — branch
  `reorg/emissive-layout` @ `c8b7edb` on SegviGen-emissive, **awaiting your merge approval**.
  `emissive/{data_prep,train,eval,infer,slurm,env,docs}` layout, v5 code synced, EXPERIMENTS.md
  ckpt registry (recommended: emis_1k_w5 ep16-EMA). Smoke test job 232600 exit 0 (glb→mask+mesh,
  110s); master accuracy probe: pred-vs-GT voxel IoU 0.115 ≈ model's known 0.117 → pipeline faithful.
  Caveat: real-cond (`--image`) path not yet exercised (zero-cond only).
- ✅ **team explainer LIVE** (master-verified, desktop+phone): [pipeline_glb_direct](https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/pipeline_glb_direct/index.html)
  — current somage bake vs proposed direct-GLB voxelization, TRELLIS.2 pinned evidence,
  50-shape diagnostic ask. DRAFT — for you to share with the team.


- ✅ **Autonomous arc complete.** 2k dataset built + verified, both models trained + honestly
  evaluated, comparison table done, BRIEF refreshed, **draft team update ready for your review**
  (page + PDF, NOT sent). Nothing running; nothing else actionable without you.
- **Headline for Wednesday:** clean negative result — neither bigger-clean-data nor balanced
  weighting beats zero-shot SegviGen (0.219); tiny-glow regions are the wall. Full story in the
  [draft update](https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/updates/2026-07-07_2k_results/index.html).


## → Next (ordered, gated)

1. Models finish → **averaged eval** (run each shape 4× and average — single generations are noisy; voxel buckets, @0.5) on each model's best + EMA ckpt.
2. Build the **comparison table**: oracle 0.219(nz) vs old-W5 0.117 vs old-W1 0.069 vs the two new models.
3. Update `training_curves` + `finetune` pages with the new models; refresh BRIEF.
4. **Draft** (not send) the team update via the updates pipeline — for owner review Wed.

## ⏸ Waiting on you (Wednesday)

1. **Review the [draft team update](https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/updates/2026-07-07_2k_results/index.html)** (+ PDF) — the negative result & framing — before it's shared.
2. **Send the Dongchen leakage note** (drafted) — 7 train shapes in his val/test; + the 2 provenance Qs.
3. **Split-of-record decision** — adopt team's pinned split going forward? (recommendation: yes.)
4. **PBR-filter ablation** go/no-go — investigation showed the filter may discard more usable
   emissive data than it keeps; its input-quality justification was never cleanly tested.
5. **Merge `reorg/emissive-layout`** on SegviGen-emissive (@`c8b7edb`, GPU-verified) — say go.
6. **Share the [pipeline explainer](https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/pipeline_glb_direct/index.html)** with the team (draft ready, master-verified).
7. **50-shape GT diagnostic go/no-go** — the decision the explainer page asks for.

## 🎯 The thesis (why these steps)

Fine-tuned SegviGen must beat the **zero-shot oracle = 0.219 IoU (nonzero-glow shapes)** to earn
its keep. Old models peaked at 0.117. The 2k run tests two levers at once: **cleaner+bigger data**
(2k emission-filtered vs 1k unfiltered) and **balanced per-shape weighting** (fixes the "5× is
too weak for tiny glow" arithmetic). Honest metrics throughout: @0.5 headline, averaging 4 samples/shape (a single run varies ±0.09), voxel-based buckets, nonzero-only aggregate (flat mean is gamed by the
25 zero-glow val shapes).

## 🧱 Comparison (full 111-val · 4-sample avg · @0.5 nonzero)

| model | nonzero IoU | vs oracle 0.219 |
|---|---|---|
| zero-shot oracle | **0.219** | — (the bar) |
| old W5-EMA (1k) | 0.117 | below |
| new W5/2k (best/EMA) | 0.103 / 0.107 | below, ≈old |
| new balanced (best/EMA) | 0.114 / 0.112 | below, ≈old |

Every model: strong on large-glow (0.32–0.43), fails on tiny-glow (0.04–0.06). Tiny-glow is the wall.


## 📋 Recent (newest first)

- +new **per-face mesh masks merged** (fork main @ `9b71cf8`) — `predict_emissive.py` now also
  writes `pred_mesh_labels.npz` (per-face over the decoded mesh, always) and, with
  `--label_input_mesh`, per-face labels + colored GLB over the ORIGINAL input mesh. GPU-validated
  (jobs 232606/232608); the agreement gate caught a real axis bug — `o_voxel.postprocess.to_glb`
  exports Y-up (`postprocess.py:312`), voxel grid is Z-up; correction anchored to source, 0.94
  texture-agreement post-fix. Earlier same day: merged the whole `emissive/` reorg + registry +
  research-style README quick-start into fork main.

- +new **multiview render pipeline consolidated + documented** → [PR #1](https://github.com/dongchen-yang/lightgen/pull/1)
  on the lightgen repo. Stage 2 (webp packing) was untracked in omages_internal; ported it standalone
  into `data_processing/multiview_render/webp/` + wrote `PIPELINE.md` (end-to-end: GLB→PNG→animated
  webp, provenance, dataset facts). Branch `docs/multiview-webp-pipeline`, awaiting Dongchen review.

- ~15:10 **durability mirror set up** (post-reboot scare) — working tree + `.claude`
  authored bits are on LOCAL disks (die with this machine); now snapshot-mirrored to
  `/project/3dlg-hcvc/omages/lightgen/` (NFS, survives) via `tools/sync_mirror.sh`. No
  secrets/transcripts copied. Heavy data/ckpts already safe on cluster. **git fork = the
  proper next step, awaiting your go** (first outward-facing push).

- +new **interactive 3D live** on results_2k_v1 §4 — click any mesh (appearance/GT/pred) to
  orbit/zoom it (model-viewer; pred previews vertex-colored to ~2MB). Now a reusable xgpage component.

- +new **paper-style mesh view LIVE** on results_2k_v1 §4 — 8 W5 predictions decoded to smooth
  meshes (official slat_to_glb). Large-glow → clean white regions; ff6c2c51 (97.6%%-emissive GT)
  comes back near-all-black = the collapse made undeniable on a real surface.


- ~12:00 **autonomous arc done** — BRIEF updated, draft team update published (page+PDF, unsent),
  roadmap finalized. All ready for owner review Wed.

- ~11:30 **honest eval CORRECTS the quick-val read** — no fine-tune beats oracle 0.219; the 2k
  data did NOT improve on the old 1k (both ~0.10–0.12 nonzero). The 0.179 "peak" was 16-sample
  quick-val noise (full averaged eval = 0.103). Multi-sample eval caught a false +53%. Both levers
  (clean data, balanced weighting) failed to move the ceiling; tiny-glow is the unsolved wall.

- ~11:00 **both models done** (walltime timeout @ep12, but peaked earlier — best ckpts captured).
  Fired 4 averaged evals on the peaks. W5/clean-2k quick-val peaked 0.179 — LATER SHOWN to be small-sample noise (see above).

- ~00:00 **ep6 read**: W5/clean-2k climbing (nz 0.18, past old ceiling, nearing oracle 0.219); balanced-weighting flat/struggling. Early read: **data quality > loss-engineering** so far.

- 22:30 **live roadmap shipped** (this page) + linked from console home; PBR-contrast page
  verified (48 thumbs load, thesis visually confirmed) → `lightgen/pbr_filter_v1`.
- 20:00 models confirmed healthy post-refresh (ep3, no preemption); heartbeat + roadmap re-armed.
- 19:57 **both 2k models launched** (231598 W5, 231599 balanced).
- 19:5x train_2k_ef verified — 2000 samples, all 5 gates pass, 166-sid buffer set aside.
- ~19:4x baseline evals done → oracle 0.219(nz), old-W5 0.117, old-W1 0.069 — the table's old rows.
- earlier: PBR-filter investigation (tooling-signature finding), eval v5 + trainer v5 shipped,
  train_2k_ef built (13 chunks), typography + filepath() + code-links across all pages.

<!-- Guardrails in force: cluster+local/reversible only; NO git pushes / NO team-facing sends until owner back. -->
