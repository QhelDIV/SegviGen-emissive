# Fact sheet: train/val/test split alignment across all four models

Assembled 2026-08-07 by a split-audit session. Every number below was computed this
session from split JSONs, resolved sha lists, and directory listings, and is reproducible
from `/project/3dlg-hcvc/omages/yanxg_scratch/split_audit/split_audit.py`
(log: `split_audit_output.log` in the same directory; outputs: `usable_381_sids.json`,
`clean_intersection_sids.json`, `emissive_frac_profile.json`, `audit_summary.json`).
Numbers carried over from prior work (marked **[PRIOR]**) were spot-checked, not re-derived
from scratch, except where noted as independently reproduced.

## 0. The headline, stated once so it is not missed

**Only 1 of the 381 "usable" evaluation shapes is held out from every model's training set.**
TEXGen v1 memorized 378/381. Our own canonical pipeline (SegviGen direct-ovoxel) has
311/381 in its training set — not by any deliberate choice, but because its split file was
never coordinated with the eval-set curation at all (see section 4). The only model family
that is genuinely clean against `usable` is the current TRELLIS.2/TEXGen `_v2` generation,
because `usable` was deliberately carved from their shared split's held-out portion. Our
model was not built against that same split, so it does not inherit that cleanliness.

## 1. Inventory: every split file and who trained on it

| # | Split file | Location | train / val / test | Used by |
|---|---|---|---|---|
| 1 | `data_splits_74k.json` | `/cs/3dlg-jupiter-project/lightgen/diffusionnet_xg/data/` | 59,602 / 7,450 / 7,451 (total 74,503, seed=42) | **Ours**: SegviGen direct-ovoxel, via `SPLIT_JSON` at `build_dataset_direct.py:91`. Indices are positional into `success==True`-filtered `emissive_thumbnails_obj_ids_df.parquet` (`/cs/3dlg-falas/.../TexVerse/lightgen/emissive_thumbnails_obj_ids_df.parquet`) — a **different parquet** than the one the other three splits below resolve against. |
| 2 | `data_splits_emissive_74k_pinned.json` | `data_processing/annotation/` (in-repo) | 73,251 / 112 / 109 (of 824,858 success rows) | TEXGen `output_emissive_74k_vanilla` (**v1**, superseded — see §2026-08-03 in EXECUTION_LOG) |
| 3 | `data_splits_emissive_74k_stratified_voxonly.json` | `/localhome/dya78/lightgen_74k_staging/` (**permission denied**, not independently readable this session) | reported 0/381 train-overlap with `usable`; full train/val/test counts not independently re-derivable | TRELLIS.2 `emission_dit_74k` (albedo→emission, **v1**) and `emission_dit_twostream_74k` (pbr→emission, **v1**), both superseded |
| 4 | `data_splits_emissive_74k_stratified_newbake_vae.json` | `/cs/3dlg-jupiter-project/lightgen/trellis2_bw/lightgen_74k_newbake/` | 71,646 / 387 / 388 (of 824,858 success rows, same parquet as #2) | **Current** generation, ONE shared split for all three: TEXGen vanilla `_v2`, albedo→emission `_v2`, pbr→emission `_v2` (`docs/status/checkpoints.md` §2026-08-03) |
| 5 | (dir listing, no separate JSON) `dataset_direct/train_1k`, `train_2k_ef` | `/cs/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset/` | 1,123 / — / — (`train_2k_ef` 2,000) | Path A legacy pilot runs `emis_1k_w1`/`emis_1k_w5`/`emis_2k_*` — explicitly labeled `"OOD: Path A data"` in `three_ckpt_table.py:24-25`, i.e. the superseded somage/GLB round-trip pipeline, not the current canonical model |

Split #1's generation script could not be located in the `diffusionnet_xg` tree available to
this session — its provenance is established only by consumer (`build_dataset_direct.py`) and
physical location (under `diffusionnet_xg/data/`, i.e. it is **DiffusionNet's split**, reused
by SegviGen because both draw from the same 74,503-shape pool, not because it was built with
SegviGen's evaluation needs in mind). Split #3's raw JSON is on a path owned by user `dya78`
with mode that denies this session read access; every number about it in this fact sheet is
either **[PRIOR]** (taken from `lightgen_repo/evaluation/newdata_eval/EXECUTION_LOG.md`, itself
measured by a prior session with access) or derived from the repo's own
`eval_sets/train_overlap_trellis_stratified_voxonly.txt` artifact (0 lines, confirmed empty
this session).

The "our own splits materialize as directories" claim was verified directly: `dataset_direct/
{train_72k,val_72k,test_72k}` under `/cs/3dlg-jupiter-project/lightgen/segvigen_emissive/`
have exactly 57,968 / 7,290 / 7,288 directory entries (counted this session, matches the brief),
pairwise disjoint (checked).

## 2. The eval set that makes v2 legitimate, and why "ours" doesn't inherit it

`usable.txt` (381 shas, `evaluation/newdata_eval/eval_sets/usable.txt`) was built by
`build_eval_sets.py` from a hand-curated `eval_shas.txt` filtered to shapes with on-disk GT
artifacts. Critically, `eval_shas.txt` was **itself carved from split #3's held-out portion**
(`pin_remaining_cohort.py:29`, per EXECUTION_LOG) — so TRELLIS.2 v1 is clean against `usable`
by construction, not by luck. `pin_v2_split_shas.py` (repo, resolved+gated this session)
confirms the same relationship holds for split #4: `usable` is 381/381 a subset of the
resolved v2 `val` (387), zero overlap with v2 `train`/`test`. **No such relationship was ever
built for split #1** — nothing in `build_dataset_direct.py` or `data_splits_74k.json`
references `usable.txt`, `eval_shas.txt`, or either TRELLIS.2 split. Split #1 is a plain
`seed=42` random partition of the 74,503-shape pool with no coordination to any other split.

## 3. The contamination matrix

Overlap = number of shapes of the eval set (column) present in the model's TRAIN set (row).
`usable`, TEXGen v1, and v2's own val/test counts were resolved and cross-checked against the
shipped resolution algorithm this session (824,858 `success==True` rows of the 854,287-row
`df_SomgProc_final.parquet` at
`/cs/3dlg-falas/project/omages/datasets/TexVerse/somages/v1201_homages_512charts/somages/`,
`success==True` filter, positional indexing — same algorithm `pin_v2_split_shas.py` documents
and gates). The TEXGen-v1/`usable` cell (378) reproduces the file already in the repo exactly.

| Model (train set) | `usable` (381) | TEXGen v1 val (112) | TEXGen v1 test (109) | v2 shared val (387) | v2 shared test (388) | our val_72k (7,290) | our test_72k (7,288) |
|---|---|---|---|---|---|---|---|
| **TEXGen v1 (pinned)**, 73,251 shapes | **378 (99.2%)** | 0 | 0 | 384 (99.2%) | 388 (100%) | 7,275 (99.8%) | 7,259 (99.6%) |
| **TRELLIS.2 v1** albedo/pbr→emission (stratified_voxonly) | **0 [PRIOR]** | not resolvable (file inaccessible) | not resolvable | not resolvable | not resolvable | not resolvable | not resolvable |
| **TEXGen v2 / albedo→emission v2 / pbr→emission v2** (one shared split), 71,646 shapes | **0** | 110 (98.2%) | 108 (99.1%) | 0 | 0 | 7,191 (98.6%) | 7,167 (98.3%) |
| **Ours: SegviGen direct-ovoxel** (`train_72k`), 57,968 shapes | **311 (81.6%)** | 87 (77.7%) | 90 (82.6%) | 315 (81.4%) | 314 (80.9%) | 0 | 0 |
| Path A legacy `emis_1k` (train_1k), 1,123 shapes | 2 (0.5%) [PRIOR, reproduced] | 3 | 4 | 2 | 9 | 0 | 0 |
| Path A legacy `emis_2k` (train_2k_ef), 2,000 shapes | 8 (2.1%) [PRIOR, reproduced] | 0 | 0 | 8 | 17 | 0 | 0 |

Reading this table:

- **Every diagonal-looking "0" in the top-right block is by construction, not by accident**:
  a model's train set is disjoint from its own val/test because the split JSON guarantees it,
  and `usable`/v2-val/v2-test are disjoint from v2-train because `pin_v2_split_shas.py`'s gate
  checked it. Those cells confirm the machinery is sound, they are not new findings.
- **The `usable` column is the paper's actual comparison ground**, and on it: TEXGen v1 is
  99.2% memorized, TRELLIS.2 v1 and the whole v2 generation are 0% memorized, and **Ours is
  81.6% memorized** — the largest contamination figure after TEXGen v1's, and far larger than
  either Path A legacy run.
- **The last two columns are the sharpest finding.** They ask: if you used *our own* held-out
  set (`val_72k`/`test_72k`, 7,290/7,288 shapes) as the comparison ground instead of `usable`,
  how contaminated would the *baselines* be? Answer: TEXGen v1 98-100%, the entire v2
  generation 98.3-98.6%. There is effectively **no eval set anyone can name today on which both
  "ours" and the baselines are simultaneously clean** — using `usable` contaminates us, using
  our own held-out set contaminates every baseline.
- 81.6% is not a targeted leak, it is close to the **base rate**: split #1's train fraction is
  59,602/74,503 = 80.0%, and 81.6%/81.4%/80.9% track that almost exactly. This is what an
  uncoordinated `seed=42` random split looks like when laid against an eval set drawn from the
  same 74k population — not evidence of anything adversarial, but exactly as damaging to the
  comparison as if it were.
- The breakdown of all 381 `usable` shapes against our own three buckets: 311 in `train_72k`,
  31 in `val_72k`, 38 in `test_72k`, 1 absent from `dataset_direct` entirely (311+31+38+1=381,
  checked). So even the 70 `usable` shapes we did NOT train on are scattered across our own
  val/test, not curated as a shared holdout with anyone else.

## 4. Our split's provenance and the disjointness question, answered directly

**Is `train_72k` disjoint from TRELLIS.2's val/test? No — for either generation.**

- Against TRELLIS.2 v1 (stratified_voxonly): could not be checked directly (file inaccessible),
  but since `usable` was carved from stratified_voxonly's held-out portion and 311/381 (81.6%)
  of `usable` is in `train_72k`, at minimum those 311 shapes are shared between our train set
  and TRELLIS.2 v1's declared-clean holdout.
- Against TRELLIS.2/TEXGen v2 (the split of record for the paper's current baseline numbers):
  measured directly — `train_72k` ∩ v2 val = 315/387 (81.4%), `train_72k` ∩ v2 test =
  314/388 (80.9%).

So yes: our training set contains most of TRELLIS.2's evaluation shapes, in both generations
checked. Per the brief's framing, this means **any comparison that scores "ours" against
TRELLIS.2 using TRELLIS.2's eval shapes as ground truth is inflated for us** on that subset —
and, symmetrically, section 3's last two columns show a comparison run the other way (baselines
scored on our held-out set) would be inflated for the baselines. Neither direction is a valid
head-to-head today.

`train_72k`'s pool overlaps v2's training pool almost completely (57,128/57,968 = 98.5% of our
train shapes are also in v2's 71,646-shape train set) — the two splits draw from nearly the
same population, they are simply partitioned with no shared logic.

## 5. The clean intersection

Held out from every model with a known train set (TEXGen v1, the v2 shared split, and Ours;
TRELLIS.2 v1's train is inaccessible so it contributes only the already-published 0-overlap
fact, which does not shrink this set further since it's already a subset check against
`usable`):

**Size: 1 shape**, sid `2fd0a8ecb9564375b4e3cb4622a73d08`. Saved at
`/project/3dlg-hcvc/omages/yanxg_scratch/split_audit/clean_intersection_sids.json`.

This is, plainly, the finding: **there is no usable clean evaluation set for a four-way
comparison under the splits in use today.** One shape cannot support an aggregate metric,
let alone a stratified one. Its own `emissive_frac` (from `dataset_direct` meta.json) is
0.112 — unremarkable, not degenerate, but a sample size of 1 says nothing about whether a
real clean set would skew toward degenerate shapes.

For context, the profile of the full 381-shape `usable` set (380/381 found in `dataset_direct`
meta.json, 1 absent): mean `emissive_frac` 0.218, median 0.018, 87.1% nonzero, 33.4% above
0.1, 20.5% above 0.5 — bimodal, consistent with the dataset-wide profile in
`FACTSHEET_experiment_overview.md` §6. There is no meaningful comparison to be drawn between
a distribution over 380 shapes and a single point.

## 6. What it would cost to fix — recommendation (SUPERSEDED, see §7)

**This section's original recommendation was to adopt the v2 shared split
(`data_splits_emissive_74k_stratified_newbake_vae.json`).** The owner has since decided
differently: adopt `data_splits_emissive_74k_stratified_voxonly.json` (TRELLIS.2's own split)
instead, because it is the one baseline whose numbers are not memorization. That decision
stands; this section is kept for the reasoning trail and §7 below has the costing for the
path actually chosen.

Original reasoning, still correct as an explanation of *why v2 was tempting* (cheap, and
already shared by 3 of 4 model families): the two pools are almost the same 72k shapes, just
cut differently.

Measured this session:
- v2's full split (train+val+test) totals 72,421 shapes. Of those, **72,260 (99.8%) are
  already built somewhere** in our `dataset_direct` (train/val/test_72k combined, 72,546
  total). Only **161 shapes** would need building from scratch (160 destined for v2's train,
  1 for v2's val) — at the measured ~1.25 s/shape build cost, this is minutes of compute, not
  hours.
- The remaining work is **re-partitioning, not rebuilding**: 14,358 shapes currently sitting in
  our `val_72k`/`test_72k` would need to move into the v2-defined train bucket (a directory
  move/symlink operation, not a recompute) to match v2's 71,646-shape train.
- After re-partitioning, the real cost is **retraining the SegviGen DiT fine-tune** on the new
  71,646-shape train set — same order of cost as the run already being planned per
  `FACTSHEET_experiment_overview.md` §9 (~20 h/epoch at 57,968 shapes; a ~71,646-shape epoch
  scales proportionally to roughly 25 h), times however many epochs the real run budget decides
  on (still an open decision per that fact sheet's §12).

This single change would make all four models share one split of record, make `usable` (and
v2's val/test) simultaneously clean for every model, and retire the two files (`data_splits_74k
.json` for us, `stratified_voxonly` for TRELLIS.2 v1) that are the source of every
contamination number in section 3 above. The alternative — evaluating everything on the current
1-shape clean intersection, or keeping per-model contamination annotations the way TEXGen v1's
UV-space numbers already are — either produces a statistically meaningless comparison or
permanently caveats the paper's central table. Retraining once on a shared split is bounded,
mostly reuses work already done, and removes the caveat rather than managing it.

## 7. Owner decision: adopt TRELLIS.2's split — precise cost

**DECISION UPDATED 2026-08-08 (supersedes the paragraph below): the owner confirmed the
v2 shared split, `data_splits_emissive_74k_stratified_newbake_vae.json`, as canonical.**
The intent was always "the split Dongchen's TRELLIS.2 emission models trained on"; the
costing in this section established that the v1 file named below is unreadable (7.0) and
that the current baseline generation trained on v2, so v2 IS that split for every number
we compare against today. The launch sequence executes against v2. Section 6's analysis
is therefore operative again; this section's v1 costing is kept as the record of why.

Original decision text (superseded): adopt `data_splits_emissive_74k_stratified
_voxonly.json` as the canonical split for our models, not v2's shared split. Rationale
given: it is the clean baseline, so aligning to it makes our numbers comparable to the one
baseline whose numbers are not memorization. This section costs that exact path.

### 7.0 A blocker that limits precision here, stated plainly

**The raw split file could not be read this session and still cannot be.** It lives at
`/localhome/dya78/lightgen_74k_staging/data_splits_emissive_74k_stratified_voxonly.json`.
`/localhome/dya78/` itself is `drwx------` (mode 700, owner `dya78`, no ACL entries for anyone
else) — not a permissions gap on the one file, the whole home directory denies traversal to
every other account, confirmed via `stat`/`getfacl` (both permission-denied). The only other
known copy is on the fir cluster (`/scratch/dya78/lightgen/lightgen_74k/data_splits_emissive
_74k_stratified_voxonly.json`, per `TRELLIS2/CLAUDE.md` and several superpowers docs), which
requires an interactive MFA login this session cannot complete unattended. No generation
script for this specific file was found in the `TRELLIS2/data_toolkit` tree available locally,
so it cannot be reproduced from code either. **Two ways to unblock, either is enough:**
(a) ask Dongchen to `chmod o+rx /localhome/dya78` (or narrower, just enough to traverse to the
one file) or copy the 570 KB file to any `/cs` or `/project` path, or (b) have a session with
fir access run the tool below and hand back its output files. Everything below this point is
built so that the moment either happens, the remaining work is one command, not a
re-investigation.

**The tool, written and validated this session:**
`/project/3dlg-hcvc/omages/yanxg_scratch/split_audit/resolve_and_repartition.py`. It
generalizes the repo's own `pin_v2_split_shas.py` (same resolution algorithm: positional
indexing into the `success==True`-filtered `df_SomgProc_final.parquet`) so it isn't hardcoded
to v2's expected counts, resolves any split JSON into train/val/test sha lists, runs the same
`usable.txt ⊂ val` safety gate, and computes a repartition plan against the live
`dataset_direct` tree. **Validated by dry-running it against the already-known v2 split** — it
reproduced the exact counts (71,646/387/388, gate PASS, 161 shapes needing fresh build) that
this session derived independently in §5/§3, confirming the script's logic before betting the
real run on it. Usage once the file is readable:

```
python3 resolve_and_repartition.py \
  --split-json /path/to/data_splits_emissive_74k_stratified_voxonly.json \
  --label v1_stratified_voxonly \
  --out-dir /project/3dlg-hcvc/omages/yanxg_scratch/split_audit
```

### 7.1 How different is it from what we have — measured directly where possible

Exact numbers against TRELLIS.2 v1's actual val/test require the blocked file. What could be
measured directly this session, without it:

- `usable.txt` (381 shas) is documented (EXECUTION_LOG, `pin_remaining_cohort.py:29`) as carved
  from stratified_voxonly's **held-out** portion — i.e. `usable ⊆ stratified_voxonly.val`, a
  fact independent of reading the file. Against that known-clean subset: **`train_72k` ∩
  usable = 311/381 (81.6%)**. This is a real, exact, measured lower bound on how much of
  TRELLIS.2 v1's actual held-out set our current train set covers — not a proxy number.
  `train_1k` (emis_1k) ∩ usable = 2/381, `train_2k_ef` (emis_2k) ∩ usable = 8/381 (both
  reproduced this session, matching the manager's prior figures exactly).
- As a magnitude proxy (NOT the same split, same "stratified" family run on a different bake
  generation — do not read as exact): against the v2 shared split's val+test (775 shapes),
  `train_72k` ∩ (v2 val ∪ test) = **629/775 (81.2%)**, `train_1k` ∩ (v2 val ∪ test) = 11/775,
  `train_2k_ef` ∩ (v2 val ∪ test) = 25/775. These track the same ~80-82% base-rate pattern as
  every other number in this fact sheet, for the same reason (§3): `data_splits_74k.json` is an
  uncoordinated `seed=42` random split at 80.0% train fraction, so it lands on roughly that
  fraction of *any* eval set drawn from the same population, regardless of which "stratified"
  generation that eval set is.
- **Best estimate for the real number**: given the pattern above holds across every
  measurement made against every split family this session (v1's usable-derived lower bound,
  v2's val+test), the true `train_72k` ∩ stratified_voxonly-(val ∪ test) figure is expected to
  land in the same ~80% range — likely several hundred shapes out of whatever stratified_voxonly's
  val+test total is (documented elsewhere in the repo as val/test on the order of ~400/~396 for
  the "stratified" split family, `docs/superpowers/plans/2026-06-22-trellis2-emission-voxel-uv-bridge-eval.md:21`,
  though that count was not independently re-verified this session for `_voxonly` specifically).
  This is an estimate, not a measurement — run the tool above for the real number.

### 7.2 What has to be redone, and what does not

**All five named checkpoints are invalid under the new split.** This is provable now, without
the blocked file, because each one's training set has a nonzero *measured* overlap with a
confirmed subset of stratified_voxonly's held-out portion (`usable.txt`, §7.1):

| checkpoint | trained on | usable overlap (proven ⊆ stratified_voxonly val) | verdict |
|---|---|---|---|
| `emis_1k_w1` | `dataset/train_1k` (1,123 shapes) | 2/381 | **invalid** |
| `emis_1k_w5` | `dataset/train_1k` (same 1,123) | 2/381 | **invalid** |
| `emis_2k_bal` | `dataset/train_2k_ef` (2,000 shapes) | 8/381 | **invalid** |
| `emis_2k_w5` | `dataset/train_2k_ef` (same 2,000) | 8/381 | **invalid** |
| `emis_72k_unfilt` | `dataset_direct/train_72k` (57,968 shapes) | 311/381 | **invalid** |

None of these overlap counts are proxies — `usable.txt` is a genuine, documented subset of
stratified_voxonly's own val split, so any nonzero overlap is a real leak against the split
being adopted, not an estimate. All five saw evaluation shapes; all five need retraining on
the re-partitioned data before their numbers can be used in the paper's comparison.

**The built dataset needs reorganizing, not rebuilding, for shapes already built.** Confirmed
by reading `build_dataset_direct.py`: a shape's on-disk content (`shape_slat.pth`,
`input_tex_slat.pth`, `output_tex_slat.pth`, `emis_mask.pth`, `input.vxz`, `output.vxz`,
`meta.json`) is a pure function of the shape itself (via `OVOX_ROOT`/`GLB_ROOT` lookups keyed
by sid) — nothing in the per-shape build depends on which named split directory
(`train_72k`/`val_72k`/`test_72k`) it ends up written under; `--split`/`--out_split_name`
only choose the destination folder. `EmisDataset.__init__` in `train_emissive.py` reads splits
by **directory membership alone** (`for sid in sorted(os.listdir(sdir))`, no separate manifest
or split JSON consulted at train time). So adopting a new split is mechanically: move each
shape's already-built directory into whichever of `train_72k`/`val_72k`/`test_72k` the new
split assigns it to, and only run `build_dataset_direct.py` for shapes the new split wants that
are not present in `dataset_direct` under any current split at all.

Quantified with the v2 split as the closest available worked example (§5, validated with the
tool above): of 72,421 total shapes in that split, 72,260 (99.8%) were already built somewhere
in our tree, only 161 needed fresh building, and 14,358 needed moving between our buckets — a
directory-rename operation, not a recompute, and fast (metadata-only, same filesystem). The
real stratified_voxonly numbers should be pulled the same way once the file is readable, but
given `dataset_direct` already covers 72,546/74,503 (97.4%) of the full population any
"74k stratified" split draws from, a similarly small fresh-build count (very likely under a
few hundred shapes, almost certainly not thousands) is the reasonable expectation — stated as
an estimate, not asserted as measured.

**The 57,368 `cond.pth` files already built (train_72k) survive a re-split unchanged.**
Confirmed by reading `build_cond_thumbnail.py`: `cond.pth` is computed from `sid →
TexVerse-thumbnail` lookup (`thumb_index.json`) alone — the DINOv3 embedding of a shape's own
thumbnail image, background-removed and cropped. Nothing about the computation references
split membership; `--split` only selects which directory tree to iterate for writing outputs.
A directory move after the fact does not invalidate or require regenerating an existing
`cond.pth` — it is exactly as valid in its new location as its old one. Current counts,
recounted this session: `train_72k` 57,368/57,968 have `cond.pth`, `val_72k` 7,211/7,290,
`test_72k` 0/7,288 (not yet backfilled). After repartitioning, run `build_cond_thumbnail.py
--split <bucket>` again per bucket — it is idempotent/resumable (skips any sid that already has
`cond.pth`) and will only compute the ~600 train-side stragglers plus whatever the new split
adds that wasn't in any old bucket.

### 7.3 What the next (image-conditioned) training run should consume

**Exact split definition: pending §7.0's blocker.** The run must consume the `dataset_direct`
tree re-partitioned so that `train_72k`/`val_72k`/`test_72k` directory membership exactly
matches the resolved train/val/test of `data_splits_emissive_74k_stratified_voxonly.json` — no
new manifest format is needed, because `EmisDataset` already treats directory membership as
the split (§7.2). The existing three directory names should be **kept as-is** (`train_72k`,
`val_72k`, `test_72k`) so no training-code changes are required; only their *contents* change.
Build order, precisely enough to execute without re-deriving any of this:

1. Get the split file readable (§7.0).
2. Run `resolve_and_repartition.py` (above) → `v1_stratified_voxonly_{train,val,test}_shas.txt`
   and `v1_stratified_voxonly_repartition_plan.json`.
3. For every sid the plan reports as "already correctly placed": no action.
4. For every sid reported as "needs move between our buckets": `mv` its directory from its
   current `dataset_direct/<old_split>/<sid>` to `dataset_direct/<new_split>/<sid>` (same
   filesystem — a rename, not a copy; do this before step 5 so cond-backfill iterates the
   final membership).
5. For every sid reported as "needs fresh build": run `build_dataset_direct.py --sid_file
   <list> --out_split_name <target_bucket>` (per `build_dataset_direct.py`'s own usage docstring).
6. Run `build_cond_thumbnail.py --split train_72k`, then `--split val_72k`, then `--split
   test_72k` (idempotent — only computes what's missing after steps 3-5).
7. **Verification before launch, not after**: re-run `resolve_and_repartition.py`'s repartition
   check (or equivalent) and confirm zero shapes remain misplaced, and separately confirm
   `train_72k ∩ (resolved val ∪ test) == 0` and `(val_72k ∪ test_72k) ∩ resolved train == 0` —
   i.e. re-derive this fact sheet's §3 contamination numbers against the NEW partition and
   confirm they are all zero before spending GPU time. Do not launch on the assumption that
   steps 1-6 were sufficient; the whole reason this fact sheet exists is that assumed-clean
   splits were not.

## 8. What must NOT be claimed

- Do not report a "TEXGen vs. TRELLIS.2 vs. Ours" table on `usable` (or any subset of it) as a
  head-to-head. Ours is 81.6% memorized on that set; TEXGen v1 is 99.2%. Only the v2 TRELLIS.2/
  TEXGen numbers are clean there, and Ours is not a like-for-like fourth column next to them.
- Do not report anything on our own `val_72k`/`test_72k` as a fair baseline comparison either —
  section 3's right two columns show 98%+ of those shapes are in the v2 baselines' training
  set. That direction is contaminated the other way.
- Do not treat "0 overlap" for TRELLIS.2 v1 (stratified_voxonly) as independently re-verified
  this session — it is carried over from the repo's own artifact and EXECUTION_LOG, because the
  source JSON was not readable. If that file becomes readable, the full val/test lists should
  be re-resolved and this fact sheet's TRELLIS.2 v1 row completed.
- Do not treat the 1-shape clean intersection as evidence about anything except that no clean
  comparison currently exists — n=1 cannot characterize an emissive-fraction distribution, a
  model's performance, or a bias in what "clean" shapes look like.
- Do not read the 81.6%/81.4%/80.9% overlap figures as a deliberate leak or a bug in
  `build_dataset_direct.py` — they are explained fully by `data_splits_74k.json`'s train
  fraction (80.0%) under an uncoordinated seed=42 split. The mechanism is mundane; the
  consequence for the comparison is not.
- The Path A (`emis_1k`/`emis_2k`) contamination numbers in section 3 describe superseded
  pilot runs, not the current canonical SegviGen model. They are included for the requested
  sanity check and because they are real numbers, not because they bear on the paper's current
  comparison.
- Do not treat §7.1's "several hundred shapes, likely ~80%" figures against
  stratified_voxonly's actual val+test as measured. They are an estimate reasoned from the
  pattern every other split-pair in this fact sheet shows, plus one genuine exact measurement
  (the 311/381 `usable`-derived lower bound). Run `resolve_and_repartition.py` against the real
  file before quoting an exact overlap count or sizing the retrain.
- Do not start the image-conditioned run on the assumption that §7.3's steps 1-6 are sufficient
  on their own. Step 7 (re-verify zero contamination against the actual resolved split) is not
  optional — it is the same kind of check that was skipped when `data_splits_74k.json` was
  first adopted, which is the entire reason this fact sheet exists.
