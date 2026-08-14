# Data subsets registry — which datasets exist and where they came from

Status: active
TL;DR: One registry for every data subset in the project — full 80k corpus, the 74k split file, the canonical overfit-10, our new pbr-filtered train_1k/val_96 — plus the unresolved historical "1000" subset.

The owner remembers "three subsets: 10 for overfitting, 1000 simple, and 84k full set."
Here is what actually exists (verified on the cluster 2026-07-03):

## Upstream of the corpus — TexVerse 858K → 80.7K (traced 2026-07-06)

- **TexVerse** (arXiv 2508.10868): 858K unique Sketchfab models (≥1024px textures,
  CC-licensed, NoAI-excluded); 158K of them carry PBR materials.
- **The emissive flag pass (Dongchen, from the team Slack thread)**: iterate all ~850K
  GLBs, load each, check whether ANY material has non-zero emissive strength → **~80.7K
  flagged "emissive" objects**; thumbnails grabbed for manual inspection →
  `emissive_thumbnails_obj_ids_df.parquet` (80,735 rows). THIS is the sole criterion that
  made our corpus: *declared* emissive strength > 0 on at least one material.
- ⚠ **Reconciliation — why 24.5% of train_1k is still "zero-glow" despite an all-emissive
  corpus**: "declared emissive" ≠ "effectively emissive." glTF materials can carry
  emissiveFactor > 0 with an all-black emissive texture, or negligible emission that the
  per-face labeling (labels_uv_74k thresholds) rejects → emissive_frac = 0 at label level.
  Same reason Dongchen's 1099-shape split notes "55 zero-emission samples removed."
- Historical footnote: the early "~45k simple shapes (≤8 UV patches)" parquet from the same
  Slack thread is where the "1000 simple" subset idea originated.

## The full corpus — "80k"
- `somages_corresp_dc80k/` (TexVerse somages, `/3dlg-falas/project/omages/datasets/TexVerse/lightgen/`),
  **80,735 assets** (parquet rows, verified 2026-07-05). Emissive metadata:
  `emissive_thumbnails_obj_ids_df.parquet`. PBR-workflow tags over the full corpus:
  metalness 33,091 / specular 1,815 (both pass the pbr filter) / `<NA>` 45,829 (dropped —
  no PBR workflow tagged; typically fully-lit/baked/shadeless). That tag is the pbr filter's
  ONLY criterion.

## ⚠ Leakage + split-version findings (2026-07-06 investigation)

- **train_1k leaks 7 shapes into Dongchen's 1099-set held-outs**: 3 in his val
  (`acdb3e40…`, `b970cfee…`, `d96d5471…`), 4 in his test (`24d4dc58…`, `57f435fe…`,
  `83442a70…`, `bb4da801…`). Our own val_96 is clean (0 overlap with DC val/test).
  Our published numbers are unaffected (they're on val_96), but **any run compared on
  DC's benchmark must use a cleaned manifest excluding these 7**.
- **Three different "74k"s exist**: the live emissive-complete pool = **74,353**
  (`data_splits_emissive_complete.json`; criterion inferred: emissive-flagged +
  complete somage data; generating script for `emissive_complete_obj_ids.txt` not
  found anywhere — unresolved); OUR split = **74,503** (seed-42, 80/10/10 —
  150-shape discrepancy vs the pool, unexplained); the team's **pinned** split =
  **73,472** (73,251/112/109, val/test = DC's 1k baseline held out —
  `data_splits_emissive_74k_pinned.json`, in the team repo). We used the oldest,
  unpinned variant — which is how the leak happened. Dongchen's own TEXGen training
  uses the pinned split.
- **Positional-index landmine**: `df_SomgProc_final.parquet` (854,287 rows;
  success==True: 824,858) was REBUILT after the 74k-era splits were made —
  positional indices from those split JSONs resolve at only ~9% against the current
  parquet. Never resolve 74k-era indices positionally; use literal sid lists. Our
  own manifests (train_1k/val_96) are literal sid lists and remain ground truth.

## The 74k split file
- `data_splits_74k.json` (`/3dlg-jupiter-project/lightgen/diffusionnet_xg/data/`):
  **74,503 total, seed 42, train 59,602 / val 7,450 / test 7,451** (indices into the dc80k
  ordering). Made for the DiffusionNet project (abandoned); still our sid pool for everything
  SegviGen. ⚠ The 80,735 → 74,503 drop criterion is **not recorded** anywhere (no note in the
  file; doesn't match the 65,913 labels_uv count either) — a DiffusionNet-era processing
  filter, criterion undocumented.

## overfit_split_10 — "the 10"
- The canonical 10 shapes = **rows 0–9 of `df_SomgProc_emission_filtered.parquet`**. Used by the
  TEXGen overfit10 experiments and by our SegviGen overfit gates
  (`dataset/canon_overfit10/`). NOT the same as any ad-hoc 10-shape set.

## Dongchen's curated subsets — RESOLVED (Dongchen's Slack message, recorded 2026-07-03)
All under `/cs/3dlg-jupiter-project/lightgen/data/baked_uv_local_subset/` (owner dya78);
counts verified on the cluster 2026-07-03:
- **"1k" parquet**: `df_SomgProc_emission_filtered.parquet` — **1099 rows** (emission-filtered;
  a parent split had 55 zero-emission samples removed, so ALL 1099 have some emission).
  Owner's note: the shapes are **mostly flashlights** — category-biased, which is why we are
  NOT necessarily bound to this list for fine-tuning.
- **1k split**: `data_splits_emission_filtered.json` — seed 42, **train 878 / val 112 /
  test 109** (indices into the 1099-row parquet).
- **overfit-1 split**: `overfit_split_single.json` — index 0 of the parquet.
- **overfit-10 split**: `overfit_split_10.json` — indices 0–9 ✓ this IS our
  `canon_overfit10` (confirms the canonical 10 we've been using match Dongchen's).
- **Contrast with our `train_1k`** (below): DC's 1099 are all-emissive, category-biased
  (flashlights), 878-shape train; ours is a category-diverse seeded-random PBR-filtered draw
  from the 74k split, includes 24.5% zero-emission negatives, 1123-shape train. Different
  design goals — DC's is a clean all-positive set; ours stresses the real distribution.
  Decision of record (owner, 2026-07-03): we are not required to use DC's 1k list.

## Our SegviGen fine-tune builds (all under `/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset/`)
| set | n | built | selection | notes |
|---|---|---|---|---|
| pilot train | 232 | 2026-05-29 | random from 74k-train, UNfiltered | zero-cond era |
| v3 train | 512 | 2026-05-29 | random from 74k-train, UNfiltered | `--pbr_only` silently broken then |
| `canon_overfit10` | 10 | 2026-06 | the canonical 10 | real cond.pth present |
| `train_1k` | **1123** | 2026-07-02 | deterministic **prefix of the pbr-passing pool in split order** (pool 26,264/59,602 = 44%; fresh = `train_pbr_sids_all[224:1124]`; randomness inherited from the split's seed-42 shuffle — no fresh random draw) | 224 reused + 899 fresh (1 dropped, no cond.pth); binary targets, real cond, emis_mask |
| `val_96` | **111** | 2026-07-02 | same rule, 74k-val pbr pool | the eval split for Phase 4 |
| `train_1k_gate10` | 10 | 2026-07-02 | 8 mid-frac + 2 low-frac canaries from train_1k | overfit gate |

Stats + a 48-example gallery of train_1k/val_96: see the dataset gallery page (in the live
visuals index) once published.
