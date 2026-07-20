# Clarifications Needed Before Implementation

## 1. Data Source: Which directory to use?

Two versions of the dataset exist:
- `baked_uv_local_subset/` — UV maps only (`somage.npz`), **no mesh geometry**
- `baked_uv_local_subset_uv/` — UV maps **plus** `somage_original_mesh.npz` (verts, faces, UVs) and `somage_repacked_trimesh.npz`

DiffusionNet requires 3D mesh geometry. But the parquet file and all split JSONs point to `baked_uv_local_subset`.

- Should we use `baked_uv_local_subset_uv` instead, or generate new split files for it?
- Are the sample IDs in both directories the same set (some shards are in `_uv` but not the other)?

**Answer:**
we should use baked_uv_local_subset_uv
I think they have the same set of IDs
---

## 2. Which mesh representation to use?

`somage_original_mesh.npz` contains:
- `vert`: (V×3), `face`: (F×3) — the actual geometry DiffusionNet wants
- `uv`: (F×3×2) — per-face UV coordinates for mapping labels back

`somage_repacked_trimesh.npz` is a different re-parameterization.

- Should we use `original_mesh` or `repacked_trimesh`? What's the difference in practice?
- Some objects have multiple sub-meshes (`patch2mesh` maps patches to mesh IDs, e.g. 47 meshes in one sample). Should DiffusionNet process the **full scene as one concatenated mesh**, or **each sub-mesh independently**?

**Answer:**
we can use original meshj
use full scene as one 
---

## 3. Label derivation: UV → per-vertex/face

The emission ground truth is in UV space (512×512 image). To get per-vertex/face labels for DiffusionNet we need to project back from UV to 3D:

- UV coordinates are per-face-corner `(F, 3, 2)` — a vertex can appear in multiple faces with different UV positions. When UV readings disagree (a vertex sits on a UV seam), how to resolve the conflict? (any-emissive-wins? majority vote?)
- Is the emission threshold the same as TEXGen's `0.001` on [0,1] scale?
- Should labels be **per-vertex** or **per-face**?

**Answer:**
any-emissive-wins
yes
per-face, also is it possible to  sample points so we can have denser labels? well we can go with per-face first since it is simpler
---

## 4. Problem formulation

- Is this **binary** (emissive / not-emissive) or should we also predict emission intensity/color?
- What happens at inference time on objects with zero emissive area? The training set is filtered to emission-only.
- Class imbalance: emissive pixels are ~13% of occupied area. Should we use weighted BCE or focal loss?

**Answer:**
let's go binary
you decide on how to deal with zero emission thing in inference time
you decide on how to deal with class imbalance
---

## 5. Input features

DiffusionNet supports arbitrary per-vertex features. Options:

- **XYZ only** (3 channels — purest geometry baseline)
- **XYZ + normals** (6 channels)
- **XYZ + normals + albedo + metal + rough** (11 channels — same as TEXGen's input)
- **XYZ + HKS** (heat kernel signatures — DiffusionNet's canonical isometry-invariant choice)

Which feature set should the baseline use? Geometry-only is the most principled; using albedo/material would be easier but closer to TEXGen.

**Answer:**
let's do the same as TEXGen's input
---

## 6. Evaluation metric

- **IoU** of the emissive mask?
- **F1 score** (better for imbalanced classes)?
- **Per-vertex accuracy**?
- Evaluated on 3D vertices directly, or rasterized back to 512×512 UV space (to match TEXGen's output space)?

**Answer:**
let's use both IoU and F1
---

## 7. Scale and compute

Meshes are ~1,500–2,500 vertices. DiffusionNet needs to precompute Laplacian eigenvectors per mesh (one-time, cacheable).

- Where should the operator cache be stored (`/local-scratch2`? `/cs/3dlg-jupiter-project/`)?
- Is there a GPU job queue? Expected training time budget?

**Answer:**
- We usually do everything on solar cluster
(check this /localhome/xya120/studio/misc/lightgen/cluster_skill.tar.gz) and storage should be in /project/... (which is in jupiter storage server)
Oh, let's put everything (output, intermittent results or anything) into /project/3dlg-hcvc/lightgen/diffusionnet
- See the solar cluster info (also read the files in /project/3dlg-hcvc/omages/xgutils/src/xgutils/misc/solar*)
---

## 8. Relationship to the paper

- Is DiffusionNet the **geometry-only** baseline (no image conditioning), in contrast to TEXGen which uses CLIP?
- What is the fair comparison claim: same 3D geometry features, same train/test split, same metric?
- Should we start with the 10-sample overfit split to validate the pipeline before running on the full 1k?

**Answer:**
- Yes, it is geometry-only (online geometry, and the textures are condition)
- all same
- Yes, start with 10-sample overfit
---

## Summary of the Two Biggest Blockers

1. **Data path mismatch**: Split files point to `baked_uv_local_subset` (no mesh), but meshes live in `baked_uv_local_subset_uv`. Need to confirm IDs match and rebuild splits for the UV directory.

2. **UV-to-vertex label mapping**: No existing code maps the UV-space emission mask onto per-vertex labels on the original mesh. This projection step (handling UV seams) must be built before DiffusionNet training can start.
