# Fact sheet: image conditioning failures on the direct-ovoxel dataset (cond.pth backfill)

Assembled 2026-08-07. Every number below was measured this session, either by reading the
completed backfill's own reports/logs (SLURM job 240320 for val_72k, 240351 for train_72k,
under `/cs/3dlg-jupiter-project/lightgen/segvigen_emissive/cond_thumb/`), or by a new
header-only census over the full 72,546-shape dataset (solar job 240547) plus a small
GPU sample (jobs 240588, 240589). Read directly by the master session; do not invent,
re-round, or extrapolate beyond what section 6's estimate already states as an estimate.

Sid lists (placeholder / tiny-object-confirmed / ok-or-unverified, per split) are saved at
`/project/3dlg-hcvc/omages/yanxg_scratch/thumb_audit/classified_sids.json`. Raw census output:
`census_full.json` in the same directory. Repair-sample embeddings: `repair_sample_embeddings.npz`.

## 0. What was asked, and the one-line answer

Whether image conditioning can be trusted for tonight's 24-hour run. **No, not everywhere:**
591 shapes (0.81% of 72,546) have a thumbnail that is TexVerse's generic placeholder icon, not
a photo of the object; another 147 (measured; ~163 estimated dataset-wide) have a real photo
whose foreground is too small for BiRefNet to find. Both groups fail Path A's verbatim
preprocessing (`build_cond_thumbnail.py`, no `--repair`) with an exception, so no `cond.pth`
was ever written for them. Separately, and more urgently for a tonight decision: **test_72k
has ZERO cond.pth today (0/7,288) because the backfill was never run on it at all.** That is
a bigger and more immediate problem than the 679 known failures, because a run cannot be
evaluated against an image-conditioned model without any conditioning on its evaluation split.

## 1. Full census, all 72,546 shapes, by header only (no pixel decode, no model)

The placeholder is TexVerse's generic cube icon, always JPEG, always single-component (PIL
mode `'L'`), almost always exactly 1024x640. That signature alone (`mode == 'L'`) is what
`build_cond_thumbnail.py`'s `preprocess_image` cannot survive: `input.split()[3]` raises
`IndexError` on ANY 1-band image regardless of size, so mode alone is both necessary and
sufficient. This was checked against ground truth: for train_72k and val_72k, where the real
backfill already attempted every shape, the header-only placeholder list is an EXACT match to
the real `IndexError` failures (0 mismatches, both directions).

| split | n | placeholder (header) | % | RGB (header) |
|---|---|---|---|---|
| train_72k | 57,968 | 466 | 0.80% | 57,502 |
| val_72k | 7,290 | 66 | 0.91% | 7,224 |
| test_72k | 7,288 | 59 | 0.81% | 7,229 |
| **total** | **72,546** | **591** | **0.81%** | **71,955** |

**This is larger than the previously known 532.** The known 532 only covered train_72k+val_72k,
the two splits the backfill actually touched. test_72k's 59 placeholders were invisible until
this census because nothing had ever tried to condition on that split.

Signature check on all 532 known train/val placeholders: 530/532 are exactly 1024x640, and of
those, 530/532 are byte-identical (md5) to each other -- literally the same JPEG file. The 2
size outliers are not the cube icon either: one is a perfectly flat solid grey image (std=0.0,
constant pixel value 51), the other a dark near-black image (mean 27, std 15) -- both still
non-content, confirming the `mode=='L'` signature catches "broken/non-photo thumbnail" as a
class, not one specific file.

Tiny-object failures (RGB thumbnail, empty BiRefNet foreground) can only be measured where the
backfill actually ran preprocessing on every shape, i.e. train_72k+val_72k:

| split | attempted | tiny-object (measured) | rate among RGB |
|---|---|---|---|
| train_72k | 57,502 RGB | 134 | -- |
| val_72k | 7,224 RGB | 13 | -- |
| **train+val** | **64,726 RGB** | **147** | **0.227%** |
| test_72k | never attempted | unknown | -- |

Applying the measured 0.227% rate to test_72k's 7,229 RGB thumbnails gives an ESTIMATE of ~16
more tiny-object failures there -- not measured, no sids identified, stated as an estimate only.

**Dataset-wide total, measured + estimate:** 591 placeholder (measured, exact) + 147
tiny-object (measured, train+val) + ~16 tiny-object (estimated, test_72k) = **~754 shapes,
~1.04% of 72,546.** Bigger than the 679 the manager's brief opened with, because that number
only reflected the two splits actually attempted.

## 2. Correlation with emissive_frac and material class

No dataset field for object category exists (`df_metadata.parquet` for TexVerse carries
`pbrType`, `animation`, `isRigged`, `max_texture`, `vertexCount`, no category/taxonomy column).
`pbrType` (glTF material class, not object type) is the nearest available categorical field;
treat it as that, not as "category."

**emissive_frac** (from each shape's own `meta.json`), placeholder/tiny groups vs a 5,000-shape
random sample of successfully-conditioned shapes:

| group | n | mean | median | p90 | frac exactly 0 |
|---|---|---|---|---|---|
| placeholder | 530 | 0.220 | 0.0058 | 0.987 | 9.06% |
| tiny-object | 147 | 0.179 | 0.0141 | 0.901 | 8.84% |
| ok (random 5,000) | 5,000 | 0.229 | 0.0243 | 0.964 | 8.84% |

Means are close across all three groups and the zero-emission rate is nearly identical
(~8.8-9.1% in every group) -- **no bias toward the zero-emission or fullbright extremes.**
Medians differ more: both failure groups skew toward the LOW end of the nonzero range
(placeholder median 4x lower than ok, tiny-object 1.7x lower). Excluding these shapes would
mildly under-represent sparsely-emissive objects, not change the fullbright/dark composition.

**pbrType:**

| group | n | NA % | metalness % | specular % |
|---|---|---|---|---|
| overall | 80,735 | 56.76 | 40.99 | 2.25 |
| placeholder | 532 | 68.80 | 30.64 | 0.56 |
| tiny-object | 147 | 63.95 | 34.01 | 2.04 |

Both failure groups over-represent the `NA` pbrType class by ~7-12 percentage points and
under-represent `metalness`. Not dramatic, but real and consistent in direction across both
groups -- worth naming, not worth calling a strong confound at n=532/147 against a 41%/57%
population split.

## 3. Do the baselines eat the same file? Yes, code-verified, and they do NOT crash

**TEXGen**: `evaluation/pointsampled/texgen_infer.py:14` --
`THUMB = "/cs/3dlg-falas/datasets/TexVerse/thumbnails/thumbnails_batch/*/{}.png"` -- the
identical absolute path SegviGen's `build_cond_thumbnail.py` reads (`THUMB_ROOT` there is the
same directory, `/cs/3dlg-falas/datasets/TexVerse/thumbnails/thumbnails_batch`).

**TRELLIS.2**: `TRELLIS2/trellis2/datasets/lightgen_slat.py:551`, `thumbnail_dir` param. The
74k-scale configs (`configs/gen/emission_dit_74k.yaml`, `emission_dit_twostream_74k.yaml`)
point at `data/lightgen_74k/thumbnails`, a local materialized copy of the same TexVerse
thumbnail set (`script/fir/unpack_74k_dit_data.sh` unpacks it from the same source tree).

**Neither baseline's loader crashes on the placeholder or tiny-object images** -- verified by
running both code paths directly against a real placeholder thumbnail:
- TEXGen's `load_thumb`: `Image.open(p).convert('RGB').resize((224,224))` -- a 1-band image
  converts to RGB with no error (confirmed: produced a normal `(224,224)` RGB tensor).
- TRELLIS.2's loader: `Image.open(p).convert("RGBA")` then reads the alpha channel -- PIL fills
  a missing alpha channel as fully opaque (confirmed: `alpha` array was uniformly 255), so
  `cond = rgb * alpha` becomes just the raw placeholder pixels, uncropped, no foreground
  extraction at all (neither baseline runs BiRefNet on thumbnails -- only SegviGen does).

**So all three models are blind on these shapes, but by different mechanisms.** SegviGen's
preprocessing crashes and (without `--repair`) writes nothing, so the shape gets `--cond zero`
at train time -- an explicit, code-visible "no information" signal. TEXGen and TRELLIS.2 do not
crash: they silently embed the placeholder icon itself (for the 591) or an uncropped frame with
a speck of object in it (for the ~163) as if it were a normal, informative conditioning image.
Both are uninformative in different ways -- SegviGen's failure is at least legible as "no
signal here"; the baselines' is not.

## 4. Is --repair's output usable? Measured on 20+20 shapes, GPU job 240588

Ran the real `preprocess_image_repair` + DINOv3 embedding (verbatim `build_cond_thumbnail.py`
code, outputs to scratch only, dataset untouched) on the first 20 placeholder sids, the first 20
tiny-object sids, and read 8 real `cond.pth` from already-successful shapes as a normal-shape
reference. Pairwise cosine similarity, full flattened (1, 1029, 1024) token tensor (what the
DiT's cross-attention actually sees):

| within-group | n pairs | mean cosine | min | max |
|---|---|---|---|---|
| placeholder (repaired) | 190 | **1.000000** | 1.000000 | 1.000000 |
| tiny-object (repaired) | 190 | 0.989565 | 0.895652 | 1.000000 |
| normal (real cond.pth, baseline diversity) | 28 | 0.869827 | 0.847263 | 0.925785 |

**The placeholder group's repaired embeddings are not merely similar, they are numerically
identical.** 19/20 sampled placeholder outputs matched the first sample to max abs diff
0.0000 (float32), because the source images are themselves byte-identical (section 1) and the
pipeline is deterministic. Training on `--repair` output for these 591 shapes means feeding the
DiT the exact same conditioning vector for every one of them -- worse than `--cond zero` in
that it is a specific, nonzero, information-free vector the model cannot distinguish from a
real photo it should attend to, and better than zero only in that it is at least consistent.

The tiny-object group is NOT collapsed (min cosine 0.896, real spread) but is still tighter
than normal-shape diversity (0.870 baseline mean) -- consistent with the repair log: **all 10
sampled tiny-object shapes fell back to `used=full_frame`**, meaning BiRefNet found no
foreground even at the relaxed 0.1 threshold, so the "crop" is the entire uncropped frame in
every case. The embedding differs shape to shape (real signal, from whatever generic
background varies across frames) but does not zoom into the actual tiny object.

## 5. The render alternative: cost measured, correctness assessed

Timed Path A's own render path (`bpy_render.py`, CYCLES, 512x512, 128 samples, GPU attempted)
on 4 real failed shapes, GPU job 240589, idle L40S/A40-class solar node:

  cold (incl. bpy/Cycles init + first glb load): 5.47s
  warm: 12.99s, 1.55s, 2.73s (mean 5.76s)

At ~6s/shape in a single persistent process, all ~754 shapes (placeholder + tiny-object,
measured + estimated) render in **roughly 75 minutes on one GPU**, and proportionally less if
sharded the way the original backfill was (24-way). Cheap next to a 24-hour training run.

**Correctness is the real question, and the docstring already answers it:** Path A's render is
albedo-only, NO emission -- the model must infer emission from geometry+PBR alone for those
shapes. The TexVerse thumbnail every other shape conditions on is a fully-lit render where
genuine emission is typically blown out to white, i.e. the conditioning SHOWS where the
emission is for ~71,955 shapes and would HIDE it for the ~754 replaced ones. Rendering
replacements does not just add noise, it changes what the conditioning image means for exactly
the subset being patched. At 1.04% of the dataset this is a small fraction, but it is not
random -- section 2 shows the affected shapes skew toward lower emissive_frac and slightly
toward the NA-pbrType class, so the two conditioning regimes would not be evenly mixed across
the target distribution either.

## 6. What must NOT be claimed

  - Do not say "679 shapes are affected" without qualifying it as train_72k+val_72k only; the
    real dataset-wide count is ~754 (measured 738 + ~16 estimated), and test_72k's true
    tiny-object count is genuinely unknown, not zero.
  - Do not call the tiny-object repaired embeddings "informative" -- they are measurably less
    diverse than normal shapes and, per the repair log, are built from a full uncropped frame
    in every sampled case, not the real object crop.
  - Do not call the placeholder repaired embeddings "near-constant" as a hedge -- measured,
    they are numerically identical (cosine 1.000000, max abs diff 0.0000) across the sample.
  - Do not claim the render alternative is a strictly better fix without noting it introduces a
    second, materially different conditioning distribution into ~1% of the training set, per
    Path A's own documented warning in `build_cond_thumbnail.py`.
  - Do not treat pbrType as an object-category field; TexVerse carries no category/taxonomy
    column in `df_metadata.parquet`. The pbrType skew in section 2 is real but is a material
    class skew, not a shape-category skew.

## 7. Recommendation

**Before anything else: test_72k needs its own conditioning backfill run tonight regardless of
how the 754 are handled.** It is at 0/7,288 cond.pth. Run `build_cond_thumbnail.py --split
test_72k` (no `--repair`) first -- ~90 minutes at the measured per-shape rate from the
train_72k backfill logs (0.48-0.50s/shape, 24-way sharded), which also produces test_72k's own
real placeholder/tiny-object failure lists instead of the estimate in section 1.

For the 591 placeholder shapes (all splits): **do not use `--repair`.** Its output is proven
numerically identical across shapes and is worse than the codebase's own explicit `--cond zero`
path, because it looks like real conditioning to the training loop while carrying no
shape-specific signal. Two defensible options, in order of preference given the 24h timeline:
  1. **Leave them at `--cond zero`** (the natural state today, since `--repair` was never run)
     -- explicit, honest, zero engineering risk, costs nothing tonight. The dataset already
     supports mixed real/zero conditioning per shape.
  2. **Render replacements** (section 5, ~75 min for all ~754) if the team wants every shape to
     carry SOME image signal -- real cost is low, but accept the documented distribution shift
     for that 1.04% and consider excluding those shapes from the held-out eval set so a
     conditioning-distribution artifact cannot masquerade as a model capability difference.

For the 147 measured (+~16 estimated) tiny-object shapes: **`--repair` is fine to use as
designed.** The embeddings are real, per-shape, and not fabricated -- weak conditioning
(section 4), but weak and genuine is a materially different risk than the placeholder group's
proven-constant output. This is exactly the case `--repair` was built for.

**Net effect on tonight's run:** patching test_72k is the blocking item (~90 min). The
placeholder/tiny split, whichever option is chosen, touches ≤1.04% of shapes and does not by
itself justify delaying a 24-hour run -- but starting with placeholder shapes silently
`--repair`-ed would be the one outcome to avoid, since it would look like real conditioning
coverage went up when the information content did not.

## 8. What actually shipped: placeholder shapes carry an ALL-ZERO cond.pth

Added 2026-08-12, verified against the live dataset on that date. Sections 0-7 record the
analysis and the recommendation; this section records the state the dataset is in, because the
two differ in one way that matters at load time.

**Placeholder shapes have a `cond.pth` file, and its `cond` tensor is all zeros.** Section 7
recommended leaving them at `--cond zero`, described there as "the natural state today, since
`--repair` was never run" -- i.e. no `cond.pth` file at all. What was implemented instead, in
the zero-write round of task 21, was to WRITE a shape-correct `(1, 1029, 1024)` float32 tensor
of zeros. Measured on the live dataset: **40 of 40 sampled placeholder sids are all-zero
(`absmax` exactly 0.0), and 40 of 40 sampled tiny-object sids are nonzero**, so the two groups
were handled differently and deliberately -- placeholders zero-written, tiny-object shapes
`--repair`-ed as section 7 recommended. Scope is the full placeholder class, 591 shapes
(0.81% of 72,546), not a handful.

**Why the distinction is worth stating rather than leaving implicit.** An absent `cond.pth`
makes `train_emissive.py --cond real` raise on that shape at dataset construction: loud,
immediate, impossible to miss. A present all-zero `cond.pth` loads silently and is
bit-for-bit what the `--cond zero` path constructs, so those shapes are **unconditioned inside
a run whose configuration says `--cond real`**, and nothing at train or eval time distinguishes
them. That is the intended meaning -- an explicit "no conditioning available" marker -- and it
is defensible; it is only dangerous while it is undocumented. Note also that the zero tensor
here is 1029 tokens, matching real cond, whereas `train_emissive.py`'s `--cond zero` path
builds 1024 (`COND_T`); the two zero paths therefore differ in sequence length.

**Three of these sit in the frozen 300-shape evaluation list**
(`outputs/three_ckpt_eval/eval300_sids_frozen.json`):

    013bdc7019584f0b8d8b5264d5da4dcc
    cfd4e277f0054c6783110a5db69e2df1
    f14d122e015445c28046474f32144af9

Any conditioned-vs-unconditioned comparison scored on that list is comparing, for these three,
zeros against zeros. They should be named wherever such a comparison is reported, not silently
averaged in. (Two further frozen-300 shapes, `1b98038d95c845068926db741c29b9d8` and
`30567c38761642f2988555df33e04bba`, are tiny-object cases: they carry real nonzero embeddings,
built per section 4 from an uncropped full frame, so they are weakly conditioned rather than
unconditioned. Different failure, different handling, worth keeping separate.)

Not relitigated here; recorded so it is stated rather than silent.
