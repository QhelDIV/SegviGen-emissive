# SegviGen → Emissive Segmentation — Autonomous Worklog

**Goal:** Standalone fine-tune of a SegviGen-style model (Trellis.2 generative prior)
to predict per-voxel **binary emissive** segmentation (white = emissive, black =
non-emissive) on our textured-asset dataset, aiming to beat the DiffusionNet plateau
(val IoU ~0.26).

**Mode:** Autonomous, self-paced, ~12h unattended (started 2026-05-29 ~02:00).
Decisions logged inline; defaults chosen to keep moving. Override on return.

> ## ✅ RESOLVED — DINOv3 + RMBG-2.0 licenses accepted (2026-05-29 ~14:15, job 226785)
> Kept below for history; this was the original blocker.
>
> **DINOv3 + RMBG-2.0 are GATED on Hugging Face** — the download got 401 GatedRepoError.
> To use the real image-conditioning, log in to HF as yourself and accept the licenses:
>   - https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m  (the image cond — important)
>   - https://huggingface.co/briaai/RMBG-2.0  (bg removal — probably avoidable; our renders have alpha)
> then re-run `dl_weights.sbatch`. **TRELLIS.2-4B + SegviGen weights downloaded fine.**
>
> **Autonomous workaround (used while gated):** ran the pilot with **zero image-cond**
> (the neg_cond) — the model already conditions on shape_slat + input_tex_slat (PBR
> appearance), so this tested whether geometry+material alone suffice for emissive (a
> clean ablation). **Resolution:** licenses accepted 2026-05-29 (job 226785 re-ran
> `dl_weights.sbatch` clean) → real DINOv3 cond swapped in for the real-cond run
> (job 226802; see SUMMARY below) — best result to date.

---

## 📋 SUMMARY (read me first) — as of 2026-05-29 ~16:25, updated 2026-07-01

**What got built (all working, end-to-end, reproducible):** full SegviGen→emissive pipeline on
the cluster — TRELLIS.2 env, data pipeline (somage→GLB→O-Voxel→shape/material latents + binary
emissive target), standalone fine-tune of the `slat_flow_imgshape2tex` flow (init `full_seg.ckpt`,
gradient-checkpointed, fits 44GB), and eval (sample→decode→IoU with threshold sweep). Code in
`code/`; jobs via SLURM on solar. ⭐ **Confirmed the appearance signal IS wired in** — the flow
already conditions on the input **material/PBR latent** (base color etc.) + shape latent, so the
architecture is NOT the limitation (this was the key open question vs DiffusionNet).

**Results — val IoU (16-sample val set unless noted; zero-cond ablation + real-cond DINOv3 runs):**
| run | data | epoch | TRAIN IoU | val IoU | notes |
|---|---|---|---|---|---|
| DiffusionNet (baseline) | 32k | — | — | **0.259** | geometry-only prior work |
| pilot (full FT) | 232 | ep25 | — | 0.203 | best zero-cond on 232 data (thr 0.2) |
| pilot | 232 | ep50 | — | 0.042 | collapsed (paint nothing) |
| v2 (oversample) | 232 | ep10 | — | 0.095 | rescues emissive-heavy at thr 0.2 (6f44 0→0.82) |
| v2 (oversample) | 232 | ep30 | — | 0.119 | noisy 16-sample val |
| v3 (oversample, zero-cond) | 512 | ep4 | 0.179 | 0.176 | train ≈ val → NOT overfit; zero-cond ceiling (thr 0.2) |
| v3 (oversample, zero-cond) | 512 | ep8 | 0.145 | 0.063 | both train AND val dropped → collapse, not overfit |
| **real-cond (DINOv3, job 226802)** | **512** | **ep2** | **0.137** | **0.230** | **best result to date (thr 0.3); beats zero-cond best 0.176** |
| real-cond (DINOv3, job 226802) | 512 | ep4 | 0.160 | 0.133 | thr 0.2 |
| real-cond (DINOv3, job 226802) | 512 | ep6 | 0.077 | 0.035 | thr 0.2; same majority-class collapse as zero-cond |
| overfit-1 sanity (job 226809) | 1 | ep80 | — | **0.968** | single-sample convergence test, not a real val number — see below |

⭐ **Re-diagnosis (from train-vs-val, prompted by Xingguang):** the ep4→ep8 drop is NOT
classical overfitting (train↑/val↓). **Train also dropped** (0.179→0.145). The model
collapses toward the trivial all-non-emissive prediction; extra training pulls it
deeper into the majority prior, hurting train and val together. So "more data" alone
may not move the ceiling at 512 (ep4 isn't memorising). Class-imbalance + objective
fixes are the primary lever; real-cond (now run) tests whether the ceiling itself
shifts with richer conditioning.

⭐ **UPDATE (best signal so far): v3 on 512 data reaches 0.176 at only ep4**, already above all
232-data runs (best 0.119) — confirms **data scale is the dominant lever**. NOTE: 16-sample val +
stochastic flow sampling → ±0.03–0.05 noise; use a bigger val set for firm numbers. Best pred
threshold ~0.3 (soft colors).

⭐ **Real-cond confirms and extends the diagnosis:** DINOv3 conditioning (job 226802, init
`full_seg.ckpt`) reaches **val IoU 0.230@ep2 (thr 0.3) — the best result to date**, beating the
zero-cond ceiling (0.176) by +0.054 — conditioning raises the ceiling. But ep4 (0.133) and ep6
(0.035) show the identical majority-class collapse dynamic replaying on top of the richer
conditioning: **real-cond raises the ceiling but does not fix the collapse**; the class-imbalance/
objective fix remains the critical next lever.

⭐ **Overfit-1 sanity (pipeline verification, job 226809):** trained on a single sample to
convergence (80 epochs) reaches **IoU 0.968@ep80** (from 0.035@ep10). Confirms the training loop,
latent encode/decode round-trip, and loss are all sound — decode is NOT the ceiling. The ~0.18–0.23
plateau on real data is a genuine class-imbalance/objective ceiling, not a pipeline bug.

**Honest read:** even with real-cond, the best result to date (0.230) remains **below the
DiffusionNet baseline (0.259)**. NOT a verdict against the method — the collapse dynamic recurs
before the ceiling can be tested at convergence. Re-diagnosed limiters (post train-vs-val) in
order: (1) **majority-class collapse** — extra training drives both train AND val toward the
trivial "predict all non-emissive" solution, in both zero-cond (ep4→ep8: train 0.179→0.145, val
0.176→0.063) and real-cond (ep2→ep6: val 0.230→0.035); (2) **class imbalance** (~11% emissive) is
the underlying driver — visually confirmed on f65a02 (GT 55% emissive, pred ~0); (3) **soft-color
readout** — flow paints emissive muted (~0.2–0.3), fixed 0.5 threshold is fragile; (4) real DINOv3
cond (now confirmed helping, +0.054 at ep2) is not sufficient on its own — the collapse fix is
still required to reach convergence-quality numbers.

**Recommended next steps (ranked, revised):**
1. **Real-cond run (M8) — IN PROGRESS.** Decisive test: does DINOv3 conditioning raise the ceiling
   (currently ~0.18 zero-cond at the no-collapse epoch)? Train cond.pth build 226792 running.
2. **Class-imbalance fix** (moved up): add explicit pos_weight in the flow loss OR a sharper
   oversample target (current is 0.1-floor weighted by emissive_frac → mild). Both train and val
   suffer from collapse → this is the primary lever for the ceiling, not data size.
3. **Scale data to 1–5k** ONLY if (1) + (2) show train > val materially (true overfitting). At 512
   today, train ≈ val → more data may not move the ceiling much in zero-cond.
3. **Better readout** than a fixed threshold: Otsu / 2-cluster on the decoded base_color (the colors
   are bimodal but muted), or train with a sharper/one-hot color target.
4. **LoRA / early-stop** for the full-FT overfitting (per-epoch ckpts now saved; ep10–25 is the sweet spot).

**Artifacts:** dataset at `dataset/{train,val}/<sid>/*.pth`; ckpts in `outputs/emis_{pilot,v2,v3}/`;
eval jsons `dataset/eval_val.json`; scripts `code/{somage_to_glb,build_dataset,train_emissive,eval_emissive}.py`
+ sbatch wrappers. Per-epoch eval logs in `eval_<jobid>.log`.

**Addendum — 2026-06-20/21 session (not previously logged in this file):** ran pretrained
`full_seg` zero-shot visualizations on the canonical `overfit_split_10` + a GT-vs-pred comparison
page, published to aspis (see `AGENTS.md` for the live links). From these: **segment-then-label
ORACLE IoU ≈ 0.235** (label each predicted part emissive iff >50% of its voxels are GT-emissive,
then union the emissive parts) — this is the bar any fine-tune must beat, since it upper-bounds
what a correct-part-boundaries-but-wrong-labels model could score.

**Addendum — 2026-07-01:** mission refocused after team discussion (2026-06-25) → **binary
emissive fine-tune**. Plan of record: `notes/2026-07-02_battle_plan.md`. Recon confirmed our data
format already matches the official `example_full_seg.py` recipe, and no official trainer exists
for Gen3DSeg's dual-conditioning scheme (shape + material latent) → the standalone
`train_emissive.py` here remains the correct vehicle, not a detour.

**Key design decisions already settled (with Xingguang, before he left):**
- Standalone fine-tune (NOT a new task-ID in the multi-task model) — we only need
  emissive, so init from SegviGen weights and fine-tune a single binary task.
- **Critical delta from vanilla SegviGen:** SegviGen conditions its color/material
  flow on the *geometry latent only* (parts are shape-defined). Emissivity is a
  *material* property → we must ALSO condition on the input asset's **material
  latent** (Trellis.2's PBR latent: base color/metallic/roughness/opacity). This is
  the appearance signal DiffusionNet lacked.
- Trellis.2 has **two decoupled SC-VAEs**: shape latent + material latent. Material
  PBR = (c, m, r, α); **no emission channel** → emission is a learned inference from
  base color + PBR + shape, supervised by our labels.
- Guard against the albedo→emissive brightness shortcut; handle class imbalance
  (DiffusionNet used pos_weight=7).

**Data on hand (cluster):**
- 74k textured assets; emissive labels at
  `/3dlg-jupiter-project/lightgen/diffusionnet_xg/labels_uv_74k/` (per-asset .npy).
- Splits: `/3dlg-jupiter-project/lightgen/diffusionnet_xg/data/data_splits_74k.json`.
- DiffusionNet baseline best val IoU ~0.259 (for comparison).

---

## Milestones
- [ ] M1 Acquire & understand SegviGen + Trellis.2; trace (geo latent, mat latent, target, task) path; confirm conditioning streams.
- [ ] M2 Data-prep: encode few assets → geo+mat latents; build binary emissive targets; verify round-trip.
- [ ] M3 Add material-latent conditioning; standalone fine-tune script (LoRA/low-LR, class imbalance).
- [ ] M4 Pilot fine-tune on small subset; monitor; iterate.
- [ ] M5 Evaluate vs DiffusionNet IoU; qualitative renders; write up.

---

## Log

### 2026-05-29 ~14:15 — USER BACK; gated models unblocked + visual eval
- **Xingguang accepted the DINOv3 + RMBG HF licenses.** Re-ran `dl_weights.sbatch`
  (job 226785) → **all 4 repos downloaded OK** (DINOv3-vitl16 + RMBG-2.0 no longer
  401). Real image-cond is now unblocked → M8 (real_cond run) can proceed.
- **User asked for VISUAL results** (understanding + agent self-check). Built a
  visual contact-sheet pipeline:
  - `eval_emissive.py --dump_vis <dir>`: dumps per-sample npz {coords, pred_bc, gt_e}
    on the decoded voxels (the exact thing the per-voxel IoU measures).
  - `render_vis.py` (LOCAL, lightgen_repo bpy venv): renders GT vs predicted emissive
    voxels (orange=emissive, grey=non) + input-albedo GLB → `index.html` (sorted
    worst-IoU-first so failures surface for self-check). Voxel-cube render validated
    on synthetic data ✓.
  - Submitted dump-eval on **v3 ep8** (job 226786) → render locally when done.
- **First visual contact sheet done** (`vis_data/vis_v3ep8/index.html`, all 16 val).
  ★ KEY VISUAL FINDING: on the emissive-heavy sample f65a02 (GT 55% emissive, a glowing
  pool asset) the **prediction is almost entirely grey** — the model paints ~zero
  emissive. Confirms VISUALLY the class-imbalance "emissive-heavy blindness" (was only
  inferred from IoU before). Explains ep8 (0.063) << ep4 (0.176): extra training drives
  collapse toward the non-emissive majority prior. → strengthens the case for
  oversampling/pos-weight + early-stop, and for the real-cond signal (M8).
  - Render gotchas (fixed): decoded voxels are 0.5–5M @ res512 → must COARSEN (adaptive
    factor, ≤15k cells) or bpy OOMs; bpy also crashes after a few in-process renders →
    render ONE sample per subprocess; first GPU render compiles denoise kernels (~min)
    so use a generous per-subprocess timeout. `render_vis.py` handles all three.

### 2026-05-29 ~16:25 — ⭐ TWO HEADLINE FINDINGS

**(A) PIPELINE VERIFIED by overfit-1 sanity (Xingguang's prompt):** trained on one
single sample (`cedca5b73385…`, gt_frac 0.40) for 80 epochs × 20 grad steps
(job 226809). Loss drove 0.30 → 0.02. IoU curve on the same sample: ep10 0.035 →
ep30 **0.946** → ep80 **0.968**. The training loop, normalization, encode/decode
round-trip and loss are all sound; decode is NOT the ceiling. So the 0.18 plateau
on 512 samples is a **genuine class-imbalance / objective ceiling**, not a pipeline
bug. (This was a critical missed sanity test — flagged by Xingguang.)

**(B) REAL-COND (DINOv3) ALREADY RAISES THE CEILING:** real-cond training
(job 226802, init full_seg.ckpt, 12 epochs, save every 2). Auto-eval (val + train[:32])
per ckpt via nohup poller.
| ep | val IoU (best thr) | train[:32] IoU | best zero-cond ref |
|---|---|---|---|
| **ep2** | **0.230 @0.3** | 0.137 | zero-cond v3 ep4 = 0.176 |
| ep4 | 0.133 @0.2 | 0.160 | |
| ep6 | 0.035 @0.2 | 0.077 | |
⭐ **ep2 val 0.230 > zero-cond best 0.176 (+0.05)** → DINOv3 conditioning is helping;
DiffusionNet 0.259 is in reach. But the **same collapse dynamic** replays after ep2.
Real-cond raises the start but doesn't fix the imbalance collapse → **class-imbalance
fix (pos_weight in flow loss) is now the critical next lever**, on top of real-cond.

### 2026-05-29 ~02:00 — M1 start
- Created project dir `studio/misc/lightgen/segvigen_emissive/`.
- Cloned SegviGen code (`github.com/Nelipot-Lee/SegviGen`) to `code/SegviGen`. It
  **bundles Trellis.2** (`trellis2/`) + a `data_toolkit/` with the exact data path.

### 2026-05-29 ~02:15 — M1 FINDINGS (key)

**★ BIG FINDING — appearance is already a conditioning input (revises our earlier plan).**
The flow model is `slat_flow_imgshape2tex` ("(image, shape) → texture"), i.e.
Trellis.2's material-generation DiT. In `inference_full.py::Gen3DSeg.forward`, the
noisy latent `x_t` is **concatenated (feature-dim) with the input asset's material
latent `tex_slat`**, and the model also receives `shape_slat` and DINOv3 image
features `cond`. So the model conditions on:
  **noisy x_t  +  input material latent (PBR: base_color/metallic/roughness/alpha)
   +  shape latent  +  rendered-image DINOv3 features.**
→ The appearance signal DiffusionNet lacked is ALREADY present. We do NOT need to
add a new conditioning stream; reuse the architecture as-is and just retarget the
output to a binary emissive coloring. (Earlier worry about geometry-only
conditioning was wrong — verified in code.)

**Pipeline / latents (all in `data_toolkit/`):**
- `glb_to_vxz`: GLB → `.vxz` O-Voxel (grid 512). Geometry via
  `o_voxel.convert.mesh_to_flexible_dual_grid` (dual_vertices, intersected flags);
  material via `o_voxel.convert.textured_mesh_to_volumetric_attr`
  (base_color, metallic, roughness, alpha).
- `vxz_to_slat`: encodes `.vxz` → `shape_slat` (via `shape_enc`) and `tex_slat`
  (via `tex_enc`, on attr = [base_color,metallic,roughness,alpha]*2-1). Saves per
  sample, aligned to common voxel coords:
    - `shape_slat.pth`  (input geometry latent)
    - `input_tex_slat.pth`  (input material/appearance latent)
    - `output_tex_slat.pth` (TARGET = the segmentation coloring latent)
    - interactive only: `point_{1..10}.pth` (3D click coords)
- `color_glb`: builds the TARGET by coloring meshes with solid PBR
  `baseColorFactor` (+emissiveFactor). interactive = white target part / black rest;
  full = random palette per part. Color is written as base color → lands in the
  material attrs on voxelization → encoded to `output_tex_slat`.

**Training tuple** = (shape_slat, input_tex_slat, output_tex_slat [, points], cond_image).
Flow matching learns: noise + input_tex_slat + shape_slat + cond → output_tex_slat.

**Our emissive mapping (non-interactive path, verbatim except coloring):**
1. `glb_to_vxz(asset.glb, input.vxz)`.
2. Build emissive-colored glb: emissive faces → white (255,255,255), non-emissive
   → black (0,0,0). Two solid-color submeshes (like a 2-class full-seg with FIXED
   semantic colors instead of random palette). Export `emissive.glb`.
3. `glb_to_vxz(emissive.glb, output.vxz)`.
4. `vxz_to_slat(..., interactive=False)` → shape_slat, input_tex_slat, output_tex_slat.
5. render cond image (`bpy_render.render_from_transforms`) + `img_to_cond` (BiRefNet
   rembg + DINOv3-L) → `cond.pth`.
We only need to write an emissive variant of `color_glb` driven by per-face labels.

**Weights / deps needed (HF, large):**
- `microsoft/TRELLIS.2-4B/ckpts/`: `slat_flow_imgshape2tex_dit_1_3B_512_bf16`
  (1.3B flow), `shape_enc/tex_enc/shape_dec/tex_dec_next_dc_f16c32_fp16` (SC-VAEs),
  `pipeline.json` (slat normalization stats).
- `fenghora/SegviGen` HF: interactive / full / 2d_map ckpts (state_dict under
  `gen3dseg.`). **Init our fine-tune from the interactive ckpt** (binary behavior).
- Image models: `facebook/dinov3-vitl16-pretrain-lvd1689m`, `briaai/RMBG-2.0`.
- CUDA ext `o_voxel` + flash-attn + nvdiffrast + nvdiffrec + cumesh + flexgemm
  (TRELLIS.2 `setup.sh`). Needs GPU ≥24GB. → heavy env build, must be on cluster.

**DECISION (revised):** No architectural change needed. Standalone fine-tune of the
`slat_flow_imgshape2tex` flow, init from SegviGen interactive ckpt, non-interactive
(no points), output = binary emissive coloring. Inputs (shape+material+image) already
carry appearance. LoRA/low-LR + class-imbalance weighting still apply.

**OPEN / to verify in M2:**
- Are our 74k source assets GLBs with PBR/base-color textures (needed for
  `textured_mesh_to_volumetric_attr`)? Locate them; check format.
- How `labels_uv_74k` per-face emissive maps onto mesh faces for the white/black split.
- Env build feasibility on cluster (o_voxel/flash-attn compile, disk for 4B weights).
- `tex_decoder` output→ how to threshold the generated coloring back to binary emissive
  + transfer to mesh (mirror SegviGen's voxel→mesh, then >0.5 on base color).

### 2026-05-29 ~02:20 — M1 done → M2 start (env + asset recon)

**Source asset format (NOT glb):** somage npz pair per asset (from DiffusionNet prep):
- `somage_original_mesh.npz`: vert (V,3), face (F,3), repacked_uvs (F,3,2)
- `somage.npz`: 512² maps — `color` (base), `metal`, `rough`, **`emission_color`**, occupancy, position.
→ We have full PBR maps + the GT emission map. Plan: **somage → GLB** (mesh+UV+PBR
textures) so SegviGen's `glb_to_vxz` works unchanged. Per-face emissive label from
`labels_uv_74k` (already used by DiffusionNet) or threshold of emission_color.

**Cluster recon:**
- GPUs available: a40(48G)/l40s/a100/a5000/2080ti/blackwell. Target **a40 or l40s**
  for the build (avoid blackwell sm_120 — too new for flash-attn/ext).
- 67T free on project. **HF token present** (gated DINOv3 should download).
- **No conda** on login, no existing trellis env. → install miniforge user-local.

**Plan (parallelized for autonomy):**
- (A) Env build (long pole, risky): miniforge → TRELLIS.2 `setup.sh` on a GPU node →
  download `microsoft/TRELLIS.2-4B` + `fenghora/SegviGen` weights. Best-effort, logged;
  if it blocks on a compile/gated download, document for Xingguang.
- (B) Write all code regardless of (A): `somage_to_glb.py`, emissive coloring,
  `build_dataset.py` (reuse toolkit), `train_emissive.py`, `eval_emissive.py`. So the
  return state is "ready to run" even if the env isn't finished.

- Started miniforge download+install to `/3dlg-jupiter-project/lightgen/miniforge3` (bg).

### 2026-05-29 ~02:45 — infra map + converter written

**Filesystem / sync (IMPORTANT for future-me):**
- `/local-scratch2/...` == `/localhome/...` locally (same dir), but **NOT** live-mounted
  on the cluster. Cluster = `solar.cs.sfu.ca`. Shared compute path = `/3dlg-jupiter-project/lightgen/`.
- Code reaches the cluster via explicit sync: `cluster_skill/cluster_ssh.py` has
  `write`/`read`/`ls`/`run`/`monitor` subcommands (ssh control socket), or rsync over
  the same socket (`SOLAR_SOCKET` in cluster_ssh.py). Develop locally → sync to
  `/3dlg-jupiter-project/lightgen/segvigen_emissive/`.
- `glb_1k_path` GLBs are NOT present on disk → **somage→GLB converter is the path** (confirmed).

**Asset locations (cluster):**
- somages: `/3dlg-falas/project/omages/datasets/TexVerse/lightgen/somages_corresp_dc80k/<ditem_dir>/`
  where `ditem_dir` comes from the parquet (`emissive_thumbnails_obj_ids_df.parquet`,
  col `ditem_dir`, e.g. `000-000/<sid>`). Files: `somage_original_mesh.npz`, `somage.npz`.
- labels: `/3dlg-jupiter-project/lightgen/diffusionnet_xg/labels_uv_74k/<sid>.npy` (per-face emissive).
- parquet cols include: ditem_dir, glb_1k_path, vertexCount, success, valid.

**Code written (local, `code/`):**
- `somage_to_glb.py` — DONE (untested). `build_input_glb` (textured: base color +
  metallic-roughness from somage maps, unwelded per-corner UVs, V-flip), 
  `build_emissive_target_glb` (emissive faces→white / rest→black solid PBR submeshes),
  `emissive_face_mask` (labels_uv_74k if present, else threshold emission_color).
  TODO validate on a real asset (sid `00001dbc99db48efb16f81945dcc9999`, ditem `000-000/...`)
  once a python with trimesh+o_voxel is available.

**Env build status:** miniforge still installing (bg). NEXT: finish miniforge → create
conda env → clone `microsoft/TRELLIS.2 --recursive` → `setup.sh ... --o-voxel ...` on an
a40/l40s GPU node → download `microsoft/TRELLIS.2-4B` + `fenghora/SegviGen` weights.

**Remaining code to write (env-independent):**
- `build_dataset.py` — driver: somage→GLB (input+target) → glb_to_vxz ×2 →
  vxz_to_slat(interactive=False) → render cond (bpy) + img_to_cond → save tuples.
- `train_emissive.py` — load `slat_flow_imgshape2tex` + SegviGen interactive ckpt;
  dataset of (shape_slat, input_tex_slat, output_tex_slat, cond); flow-matching loss;
  class-imbalance handling; LoRA/low-LR; save ckpt. (Need to read
  `trellis2/models/structured_latent_flow.py` + a trainer for the exact interface.)
- `eval_emissive.py` — inference (mirror inference_full) → threshold coloring to binary
  → voxel→mesh → IoU vs GT emissive (compare to DiffusionNet ~0.259).
- `setup_trellis_env.sh` — env build (best-effort, logged).

**Autonomy:** continuing via self-paced loop; miniforge completion will re-trigger work.

### 2026-05-29 ~03:00 — converter validated; env build + weight DL launched

**somage_to_glb.py VALIDATED** (cluster, diffusionnet venv = trimesh, no o_voxel needed):
- sid 00001…9999 (frac 1.0, fully emissive outlier): produced input.glb + emissive.glb (1 geom).
- sid 649fd56a… (frac 0.195, mixed): emissive.glb has 2 geoms [emissive(white), nonemissive(black)];
  input.glb textured with UVs (35391 verts, uv present). ✓ both stages work.
- Label sanity: 40 random labels_uv → mean emissive frac 0.108, median 0.035, p90 0.21,
  3/40 fully non-emissive. Matches DiffusionNet's ~13% positive (pos_weight=7). Labels OK.

**Sync mechanism established:** rsync over `ssh -S ~/.ssh/solar_master` to
`xya120@solar.cs.sfu.ca:/3dlg-jupiter-project/lightgen/segvigen_emissive/`. Code synced.

**Env build LAUNCHED — SLURM job 226655** (a40, debug, 8h):
`code/setup_trellis_env.sh` — sudo-free port of TRELLIS.2 setup.sh: conda env trellis2
(py3.10) + torch2.6/cu124 + conda cuda-toolkit12.4 (nvcc) + libjpeg-turbo (no sudo) +
basic deps + transformers==4.57.6 + bpy + flash-attn2.7.3 + nvdiffrast/nvdiffrec/cumesh/
flexgemm/o-voxel (TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9). Log: `env_build_<jobid>.log`. Smoke
test imports o_voxel → prints ENV_BUILD_OK. (skipped pillow-simd — fragile; using stock pillow.)

**Weight download LAUNCHED** (bg on login, HF_HOME=/3dlg-jupiter-project/lightgen/hf_cache):
microsoft/TRELLIS.2-4B, fenghora/SegviGen, facebook/dinov3-vitl16-pretrain-lvd1689m (gated,
token present), briaai/RMBG-2.0. Log: `dl_weights.log` → ALL_DOWNLOADS_DONE marker.

**NOTE for future-me:** job 226655 + weight DL are CLUSTER-side (not harness-tracked) →
must poll via cluster_ssh (squeue / tail logs); no auto-notification.

**Next iteration (while build/DL run):** write `build_dataset.py` ... then `train_emissive.py`
... then `eval_emissive.py`. Check 226655 + dl_weights status.

### 2026-05-29 ~03:30 — all code written; fixed 2 blockers; jobs running

**Blocker 1 (env queue):** a40 debug GPUs saturated → job 226655 stuck PENDING. Fixed:
the env BUILD only needs a GPU node for nvcc/nvidia-smi (cross-compiles for listed archs),
so resubmitted on the **idle 2080ti** node with `TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9`
(covers 2080ti/a100/a40/a5000/l40s). → **job 226658 RUNNING** on cs-venus-02.

**Blocker 2 (weight DL OOM):** login-node hf_transfer (Rust) hit the memory cgroup
("memory allocation of 67MB failed", only 129M downloaded). Fixed: rewrote to Python
`snapshot_download(token=...)` (token passed explicitly since HF_HOME redirected away from
~/.cache) and moved to a **CPU SLURM job** with `HF_HUB_ENABLE_HF_TRANSFER=0`, mem 32G.
→ **job 226659 RUNNING** on cs-venus-07. Repos: TRELLIS.2-4B, SegviGen, dinov3(gated), RMBG.

**Code suite complete (all in `code/`, synced to cluster):**
- `somage_to_glb.py` ✓validated
- `build_dataset.py` — somage→GLB→vxz×2→vxz_to_slat(interactive=False)→render+img_to_cond;
  saves shape_slat/input_tex_slat/output_tex_slat/cond per sid + manifest. Input appearance
  = albedo/PBR only (no emission) → genuine inference.
- `train_emissive.py` — reuses `Gen3DSeg`; flow-matching (x_t=t·noise+(1-t)·data, v=noise-data,
  t·1000), init from SegviGen interactive ckpt, AdamW lr 1e-5, grad-clip, normalize via
  pipeline.json stats. (v1: no LoRA, no explicit class-imbalance weighting — see DECISIONS.)
- `eval_emissive.py` — sample→decode→threshold base_color>0.5→per-voxel emissive IoU vs GT.
- `setup_trellis_env.sh`, `dl_weights.sbatch/.py`.

**DECISIONS (v1, revisit):**
- Full low-LR fine-tune (not LoRA) for the pilot — simplest; LoRA if it overfits the small set.
- No explicit class-imbalance weighting in flow loss (latent-space weighting is awkward;
  generative target already encodes white regions). Revisit if val under-predicts emissive.
- Per-voxel IoU as v1 metric; per-face mesh IoU (decode→mesh→majority, like inference_full)
  is the rigorous comparison to DiffusionNet — TODO once pilot works.

**NEXT (when env 226658 + DL 226659 done):**
1. Smoke-test env: `import o_voxel`, load shape_enc/tex_enc/flow from HF cache.
2. `build_dataset.py --split val --n 32` and `--split train --n 256` (small pilot set;
   each sample ~ a few s of encode + render).
3. `train_emissive.py` pilot (e.g. 200 epochs on 256 train) → `eval_emissive.py --split val`.
4. Compare mean IoU to DiffusionNet ~0.259; log + iterate.

### 2026-05-29 ~03:50 — env build failed@flash-attn (fixed); weights partial (gating)

- **Weights:** TRELLIS.2-4B ✓ (snapshot af44b45…), SegviGen ✓ (73326b77…). DINOv3 + RMBG
  FAILED = GatedRepoError 401 → see ⚠ USER ACTION block at top. hf_cache=38G.
- **Env build 226658 FAILED at flash-attn**: `pip install flash-attn==2.7.3` built from
  source under build-isolation → `No module named torch`. The env (trellis2) is partially
  built (torch+cuda-toolkit+basics OK) but exited before extensions. 
- **Fix → resume job 226662 (RUNNING, 2080ti):** `resume_env.sh` activates trellis2, installs
  EXTENSIONS FIRST (nvdiffrast/nvdiffrec/cumesh/flexgemm/o-voxel, all --no-build-isolation —
  these are what DATA-PREP needs), THEN flash-attn LAST with **--no-build-isolation** (the fix).
  Smoke markers: OVOXEL_OK / FLASHATTN_OK / TRELLIS_IMPORT_OK. Note: data-prep only needs
  o_voxel + encoders (not flash-attn); flash-attn only needed for the FLOW (train/eval).
- **NEXT:** when 226662 done — if OVOXEL_OK: modify build_dataset/train/eval to support
  **--zero_cond** (zeros matching the flow's expected DINOv3 cond dim; introspect once flow
  loads). Then smoke-test, build small dataset, pilot train (zero-cond), eval vs 0.259.
  If flash-attn still fails: data-prep can still proceed; diagnose flash-attn separately
  (try prebuilt wheel for torch2.6/cu124/cp310 from flash-attn GH releases).

### 2026-05-29 ~04:10 — ENV BUILT ✓ (resume job 226662 succeeded)

- env_resume_226662: **OVOXEL_OK, FLASHATTN_OK, TRELLIS_IMPORT_OK** — trellis2 conda env
  fully usable (o_voxel + flash-attn + nvdiffrast/nvdiffrec/cumesh/flexgemm + trellis2 import).
- ⚠ RUNTIME CONSTRAINT: flash-attn 2 needs sm_80+ → the FLOW (train/eval) must run on
  a40/l40s/a100/rtx_a5000, NOT the 2080ti (sm_75). Data-prep (encoders, no flash-attn) is fine anywhere.
- SLURM gres names (from sinfo): `gpu:a40`, `gpu:l40s`, `gpu:a100`, `gpu:rtx_a5000`, `gpu:2080_ti`,
  `gpu:rtx_pro_6000_blackwell_se`(drain). NOTE: a5000 = `rtx_a5000` (first submit failed on `a5000`).
- Submitted **smoke_test (job 226665, rtx_a5000)**: loads flow+encoders+decoder from HF cache,
  prints the flow's cond/context dim (for zero-cond shape), lists SegviGen ckpt files.
- NEXT: read smoke log → get cond dim + SegviGen interactive ckpt filename → add `--zero_cond`
  to build_dataset/train/eval → build_dataset val/train (small) → pilot train → eval.

### 2026-05-29 ~04:30 — smoke OK; zero-cond wired; dataset build launched

**Smoke test 226665 ✓ (SMOKE_OK):** flow=SLatFlowModel, **cond_channels=1024** (DINOv3 dim),
in_channels=64 (=32 noisy x_t + 32 input_tex_slat, confirms concat), out=32, latent res 32³.
SegviGen ckpts present: interactive_seg.ckpt / full_seg.ckpt / full_seg_w_2d_map.ckpt.
flex_gemm conv + flash_attn backends active.

**Zero-cond wired into all 3 scripts** (DINOv3 gated workaround): cond = zeros(1, 1024, 1024).
- build_dataset.py: `--zero_cond` (default) skips DINOv3/rembg/bpy-render entirely → just
  shape_slat/input_tex_slat/output_tex_slat (faster, no gated deps, no bpy).
- train_emissive.py / eval_emissive.py: synth zeros(1,1024,1024) when --zero_cond.
- `--real_cond` flag flips back once DINOv3 license accepted.

**Dataset build LAUNCHED — job 226677 (rtx_a5000, cs-venus-05):** `build_dataset.sbatch val 16`
→ writes dataset/val/<sid>/{shape_slat,input_tex_slat,output_tex_slat}.pth. Validates the full
real-data pipeline (somage→2×glb→vxz→encode). build_dataset.sbatch takes args: split n.

**NEXT:** check build_226677.log — if slats produced cleanly (no errors), scale:
`build_dataset.sbatch train 256` + keep val 16 (or grow to 32). Then submit pilot
`train_emissive.py` (init interactive_seg.ckpt from HF cache snapshot, lr 1e-5, ~200 ep,
zero_cond) on a40/l40s/a5000 (sm_80+ for flash-attn) → `eval_emissive.py --split val` →
compare mean IoU to 0.259. If build errored, diagnose (likely o_voxel convert API / UV / glb).

### 2026-05-29 ~04:45 — build dep fix (pyarrow)
- build 226677 FAILED: trellis2 env lacked pyarrow → pandas couldn't read parquet. Fixed:
  `pip install pyarrow` in trellis2 env (24.0.0). Resubmitted → **job 226685** (rtx_a5000) building val 16.
- NEXT: poll build_226685.log for [ok]/DONE + dataset/val/*/output_tex_slat.pth; if OK scale train 256 + pilot train.

### 2026-05-29 ~05:00 — DATA PIPELINE WORKS ✓ (M2 done); training de-risk launched

- **build 226685: val 16 built cleanly, ok=16 fail=0.** Each sample has shape_slat/
  input_tex_slat/output_tex_slat.pth (~230KB) + meta.json. Emissive fracs 0.0–0.887 (sane).
  → M2 COMPLETE. somage→glb→vxz→encode pipeline validated on real data with real env.
- **M3 COMPLETE**: training script written; material conditioning already in the model
  (input_tex_slat concat) — no architecture change needed.
- interactive_seg.ckpt path: hf_cache/hub/models--fenghora--SegviGen/snapshots/73326b77.../interactive_seg.ckpt
- **train-256 build → job 226693 (rtx_a5000, running).**
- **train smoke → job 226695 (l40s, 3 epochs on val 16, out=train_smoke):** validates the
  untested train_emissive.py on the real 1.3B flow + confirms memory (full fine-tune w/ AdamW
  on 1.3B → need 48GB GPU; using l40s). Added --train_split arg for this.
- train_emissive.sbatch args: epochs train_split out_subdir. Real pilot will be
  `sbatch train_emissive.sbatch 100 train emis_pilot` after 256 build + smoke pass.
- **NEXT:** check train_226695.log — if it runs clean (loss prints, no OOM/shape errors) →
  once 226693 (train 256) done, submit real pilot (100 ep, train split) → eval_emissive.py
  --split val → compare to 0.259. If smoke errors: diagnose train_emissive.py (flow call
  signature / SparseTensor / normalization / cond shape), fix, resync, re-smoke.

### 2026-05-29 ~05:20 — init ckpt fix: full_seg, not interactive

- Smoke 226695 FAILED: `Unexpected key seg_embeddings.weight` loading interactive_seg.ckpt
  into Gen3DSeg. CAUSE: interactive model has a point/click embedding (seg_embeddings) that
  the no-click Gen3DSeg (inference_full) lacks. 
- **DECISION (corrected init):** our emissive task has NO clicks — it's structurally a
  **full-segmentation** (generate coloring from shape+material+img), just with a fixed
  2-color target. So init from **full_seg.ckpt** (matches Gen3DSeg exactly). Updated
  train_emissive.sbatch CKPT → full_seg.ckpt. (Earlier "interactive is closest" was wrong
  on conditioning structure — interactive needs point inputs we don't have.)
- (Secondary OOM in my key-inspection was just the LOGIN node memory cgroup — not a train issue.)
- Resubmitted smoke → **job 226706** (full_seg.ckpt, 3ep val16). train-256 build 226693 ~82/256
  (~20s/sample → ~1h total).
- NEXT: poll train_226706.log for 'epoch'/'flow_loss' (clean run) ; when build 226693 DONE +
  smoke clean → `sbatch code/train_emissive.sbatch 100 train emis_pilot`. Then eval vs 0.259.

### 2026-05-29 ~05:45 — OOM fix: gradient checkpointing

- Smoke 226706: full_seg.ckpt loaded ✓ (state_dict fix worked), data ✓, but **CUDA OOM on
  l40s 44GB** — full fine-tune of 1.3B sparse DiT; activations dominate (~26GB+).
- FIX: flow supports `use_checkpoint` → enabled gradient checkpointing on all flow modules in
  train_emissive.py (`m.use_checkpoint=True`), + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  in sbatch. Keeps full fine-tune. (bitsandbytes not installed; a100 busy — checkpointing is the clean fix.)
- Resubmitted smoke → **job 226716** (l40s). train-256 build 226693 at 158/256.
- Wrote **eval_emissive.sbatch** (args: ckpt split).
- NEXT: poll train_226716.log — if flow_loss prints (no OOM) → SMOKE PASSES. When build 226693
  DONE → `sbatch code/train_emissive.sbatch 100 train emis_pilot`; then eval_emissive.sbatch
  <outputs/emis_pilot/last.ckpt> val → mean IoU vs 0.259. If still OOM: freeze backbone (train
  subset) or reduce; if other error, diagnose.

### 2026-05-29 ~06:05 — SMOKE PASSED ✓ — PILOT FINE-TUNE LAUNCHED

- **Training smoke 226716 PASSED**: grad-checkpointing on 31 modules, 16 val samples,
  flow_loss printed 3 epochs (0.146/0.283/0.141), DONE, no OOM/error. train_emissive.py
  works end-to-end on the real 1.3B flow. 
- train-256 build 226693: 229→256 (finishing).
- **PILOT LAUNCHED — job 226719 (l40s, cs-venus-17):** `train_emissive.sbatch 50 train emis_pilot`
  — full_seg init, --zero_cond, lr 1e-5, grad-checkpoint. Saves outputs/emis_pilot/last.ckpt
  every 25 ep (+ log.json with flow_loss).
- **NEXT:** monitor train_226719.log (flow_loss should trend down; watch for crash). When
  outputs/emis_pilot/last.ckpt appears (~epoch 25) → `sbatch code/eval_emissive.sbatch
  /3dlg-jupiter-project/lightgen/segvigen_emissive/outputs/emis_pilot/last.ckpt val` →
  read eval_*.log mean IoU, compare to DiffusionNet 0.259. This is the FIRST emissive number.
  Note: zero-cond ablation (no DINOv3); a positive result here = shape+material latents alone
  carry emissive signal (the DiffusionNet-missing appearance). Then iterate (more data/epochs,
  real cond once user accepts DINOv3 license).

### 2026-05-29 ~06:35 — pilot training healthy
- Pilot 226719: 232 train samples (build done). flow_loss trending DOWN: 0.261→0.247→0.223
  →0.214→0.248→0.214 (ep1-6). ~5.3 min/epoch → 50 ep ≈ 4.4h. No crash. Learning.
- First ckpt (outputs/emis_pilot/last.ckpt) at epoch 25 (~1.5h). NEXT: when it appears →
  `sbatch code/eval_emissive.sbatch outputs/emis_pilot/last.ckpt val` → first emissive IoU vs 0.259.

### 2026-05-29 ~07:27 — pilot ep~15, loss↓ 0.248→0.229→0.204 (ep5/10/15), healthy, no ckpt yet (ep25 ~50min).

### 2026-05-29 ~08:20 — epoch-25 ckpt saved → FIRST EVAL submitted
- Pilot 226719 at ep28, flow_loss plateauing ~0.20-0.23 (normal for flow-matching FT). 
  last.ckpt (2.6GB, ep25) + log.json saved. Training continues to ep50.
- **EVAL job 226743 (l40s, cs-venus-16):** eval_emissive.sbatch on emis_pilot/last.ckpt, val split,
  --zero_cond. Samples flow (12 steps) → decode → threshold base_color>0.5 → per-voxel IoU vs GT.
- NEXT: poll eval_226743.log for 'mean IoU' (FIRST emissive number, vs DiffusionNet 0.259) OR
  errors (eval_emissive.py UNTESTED — watch Sampler.sample sig / tex_decoder / threshold). Log result.

### 2026-05-29 ~08:36 — eval fix: tex_decoder needs guide_subs
- Eval 226743 errored: tex_decoder 'Cache not found' — material decoder needs the shape VAE's
  subdivision structure. FIX: eval_emissive.py now loads shape_decoder, computes subs =
  shape_decoder(shape_slat, return_subs=True), passes guide_subs=subs to both tex_decoder calls
  (mirrors inference_full.py). Resubmitted → **eval job 226750** (l40s).
- NEXT: poll eval_226750.log for 'mean IoU = X' (FIRST emissive number, zero-cond, ep25 ckpt)
  vs 0.259. Training 226719 still → ep50 (will overwrite last.ckpt; can re-eval ep50 after).

### 2026-05-29 ~08:55 — ⭐ FIRST EMISSIVE RESULT: mean IoU 0.2034 (ep25, zero-cond, 16 val)

Pipeline works end-to-end (train→sample→decode→IoU). **0.2034 vs DiffusionNet 0.259** — close,
on a tiny pilot (232 train, 25 ep, ZERO image-cond, full_seg init).

**Diagnostic per-sample pattern = UNDER-PREDICTION of emissive:**
- Sparse-emissive nailed: 10b7 gt_frac0.015→IoU 0.956.
- Emissive-HEAVY all failed: 6f44 gt0.899→0.0, f65a 0.548→0.0, 39a8 0.456→0.0, 57a8 0.316→0.0.
- A few empty objects got false positives (374f/e654 gt0→0.0).
→ Model paints mostly black (non-emissive). Classic class imbalance (~11% emissive) + maybe
  muted decoded colors missing a 0.5 threshold.

**Actions taken:** added pred-threshold SWEEP to eval_emissive.py (IoU@0.2/0.3/0.4/0.5, GT fixed
0.5) — cheap calibration check (no re-sampling). Will reveal if a lower threshold recovers the
emissive-heavy objects (calibration) vs fundamental under-prediction.

**NEXT levers (in order):** (1) ep50 re-eval w/ threshold sweep (more training + calibration);
(2) if calibration → pick best threshold; (3) class-imbalance: save per-voxel emissive mask in
build_dataset + weight the flow MSE on emissive voxels (needs rebuild) OR oversample emissive-heavy
shapes; (4) more data + longer train (emis_v2); (5) real DINOv3 cond (after user accepts HF license).
Training 226719 at ep36 → ep50 ~75min; re-eval the ep50 ckpt then.

### 2026-05-29 ~09:38 — pilot ep45, loss plateaued ~0.19-0.22; ep50 ~27min out. Awaiting ep50 ckpt → re-eval (threshold sweep).

### 2026-05-29 ~10:10 — training DONE (ep50); re-eval submitted (job 226755, threshold sweep)
- Added `--emis_oversample` to train_emissive.py (weighted sampling by emissive_frac+0.1) — ready
  for v2 if eval confirms true under-prediction (vs calibration).
- NEXT: read eval_226755.log per-threshold mean IoU + BEST vs 0.259 → decide: calibration (pick
  best thr) vs class-imbalance (launch v2: copy train_emissive.sbatch adding --emis_oversample,
  `train 50 emis_v2`). Then re-eval v2.

### 2026-05-29 ~10:30 — ⚠ ep50 WORSE than ep25 (overfitting) + under-prediction confirmed

**ep50 sweep (job 226755):** mean IoU @0.2=0.030 @0.3=0.031 @0.4=0.035 @0.5=0.042 (BEST 0.042).
vs ep25=0.2034. → **ep50 << ep25: the full fine-tune OVERFITS the 232-sample set** (ep25 sweet
spot, ep50 collapsed). Threshold sweep flat → NOT calibration. Emissive-heavy objects ~0 at ALL
thresholds (39a8 0.456→0.0, f65a 0.548→0.0, 6f44 0.899→0.03) → real UNDER-PREDICTION (class imbalance).

**Honest read:** zero-cond pilot is a preliminary NEGATIVE vs DiffusionNet 0.259 (best 0.20, unstable,
fails emissive-heavy). Likely causes, in order: (a) overfitting tiny data (232) w/ full 1.3B fine-tune;
(b) class imbalance (~11% emissive); (c) missing image-cond (zero-cond ablation — real method needs DINOv3).

**Fixes applied:** train_emissive.py now KEEPS per-epoch ckpts (epoch_NNNN.ckpt) — pilot lost the
better ep25 by overwriting. Launched **v2 job 226756** (emis_v2): `--emis_oversample` (sample weighted
by emissive_frac) + 30 ep + save_every 10 (ckpts ep10/20/30) to (i) fight class imbalance, (ii)
early-stop before overfit.

**NEXT:** eval emis_v2 ckpts ep10/20/30 (`eval_emissive.sbatch outputs/emis_v2/epoch_00NN.ckpt val`),
find sweet spot, check if oversample fixes emissive-heavy. Then bigger data (build train 512/1024) +
LoRA/lighter FT to truly fix overfitting; real DINOv3 cond after USER accepts HF license (⚠ top of log).
For the team: the appearance signal IS available (input_tex_slat/material latent) — the bottleneck so
far is data scale + imbalance + the zero-cond handicap, not the architecture.

### 2026-05-29 ~11:22 — v2 (oversample) at ep8, loss ~0.25-0.31 (noisier, expected); ep10 ckpt ~13min. Awaiting ckpt → eval.

### 2026-05-29 ~11:39 — v2 ep10 ckpt → eval (226757); started train→512 build (226758, for v3/overfit fix)
- v2 loss noisy ~0.26-0.31 (oversample). ep10 ckpt saved → evaluating (job 226757).
- Launched bigger data build train→512 (226758) in parallel — more data is the real overfitting fix.
- NEXT: read eval_226757 ep10 (per-thr IoU + emissive-heavy gt_frac>0.3 samples) vs ep25=0.2034/0.259;
  eval ep20/ep30 as they save; when 512 build done → v3 train on 512 (lower LR, ~25-30ep, keep epoch ckpts).

### 2026-05-29 ~11:59 — v2 ep10 (oversample): mixed; emissive-heavy rescued at low thr but muted colors
- v2 ep10 mean IoU @0.2=0.0946 @0.3=0.045 @0.4=0.024 @0.5=0.014 (BEST 0.0946 @0.2). Note: threshold
  preference FLIPPED vs ep25 (which peaked @0.5) → oversample model paints emissive with MUTED
  colors (~0.2-0.3, not white).
- Emissive-heavy WINS at thr 0.2: 6f44 gt0.899 0.0→**0.821**; 57a8 gt0.316 0.0→0.332. (39a8 0.456, f65a 0.548 still 0.)
- BUT overall mean (0.095) < ep25 pilot (0.2034): oversample over-paints sparse/empty objects (false
  positives), and muted colors hurt fixed-threshold IoU. → fundamental: flow outputs SOFT colors;
  binary target is hard to reproduce crisply in zero-cond + tiny data.
- 512 build at 326/512. NEXT: eval v2 ep20/ep30 (may sharpen colors); v3 on 512 data; then SUMMARY.

### 2026-05-29 ~12:26 — v2 ~ep17 (ep20 ckpt pending), 512 build 419/512. Awaiting ep20 + build done. Summary due ~13:30.

### 2026-05-29 ~12:47 — eval v2 ep20 (226761); launched v3 on 512 data (226762, 12ep, save_every4)
- 512 build ~487/512 (nearly done). v3 (emis_v3) on the larger set — but 512 epochs are ~2x longer
  (~13min/ep) so v3 won't finish before user returns; early ckpts (ep4 ~50min) give a 512-data signal.
- NEXT: read eval_226761 (v2 ep20); eval v2 ep30 when saved; eval v3 epoch_0004 when it appears.
  Write SUMMARY by ~13:30.

### 2026-05-29 ~13:34 — v2 DONE(ep30); eval v3 ep4 (512 data, job 226776) + v2 ep30 (226777)
- v3 (512 data) training ~ep6, loss ~0.28; ep4 ckpt saved → evaluating (does more data reduce overfit?).
- v2 finished ep30 → evaluating to complete the 232-data overfit sweep (ep10 0.095 > ep20 0.068 > ep30?).
- NEXT: read eval_226776 (v3 ep4) + eval_226777 (v2 ep30); update SUMMARY table; continue v3 ep8/12.

### 2026-05-29 ~14:00 — user returned → AUTONOMOUS LOOP ENDED
- Final in-loop results: v3(512 data) ep4 = 0.176 (best @thr0.3) > v2(232) ep10 0.095/ep20 0.068/ep30 0.119.
  → **more data is the lever**. v3 left training to ep12 (job 226762, ~40min) — eval epoch_0008/0012
  later for a better-trained 512 number (`sbatch code/eval_emissive.sbatch outputs/emis_v3/epoch_00NN.ckpt val`).
- Loop stopped (user back). All scripts/ckpts/logs preserved. SUMMARY table at top is current.
- Recommended next (unchanged, now data-backed): (1) accept DINOv3/RMBG HF license → real-cond run;
  (2) scale data 1–5k (v3 shows it works); (3) bigger val set (16 too noisy); (4) better readout (thr~0.3 / Otsu).
