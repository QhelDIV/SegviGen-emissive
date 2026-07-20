# direct_pilot — verdict scripts for the direct-GLB emissive pilot (2026-07-19)

Evidence code behind the go/no-go verdict that o_voxel's per-voxel `emissive`
attribute is BROKEN on TexVerse materials (fabricates glow on non-emissive shapes,
misses genuine emission). Verdict page:
https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/glb_direct_pilot_v1/index.html

Run on solar (env trellis2); working dir was
`/3dlg-jupiter-project/lightgen/segvigen_emissive/direct_pilot`.

- `verify_emissive.py` — voxelize a GLB, read back vxz, dump attrs (attr-exists gate).
- `diag_emissive.py`   — per shape: true glb emission (texture×factor) vs direct-vxz
                          emissive frac vs somage GT frac.
- `controlled_test.py` — with/without preprocess and with emissiveFactor zeroed;
                          shows the attr is factor-scaled with a wrong texture lookup.
- `evidence_run.py`    — 54-shape batch: build_dataset_direct + true + somage per shape.
- `ab_compare.py`      — coord overlap / agreement / IoU, direct vs somage (CPU stub io).
- `merge_stats.py`     — merge + false-positive rate + correlations + example picks.
- `render_voxels.py`   — bpy voxel-mask render (emissive=orange, surface=grey).

Ground truth for emission is the emit_only multiview render, NOT the voxel attr.
Fix path (see build_dataset_direct.py stage-2 notes): UV-sample the original GLB's
`emissiveTexture × emissiveFactor` per-face; handle factor-only + emissive_strength.

## UPDATE (2026-07-19): root cause found + fixed — it was a one-line bug

`ovoxel_emissive_mipmap_fix.patch` — the fix for the broken emissive attr. In
microsoft/TRELLIS.2 (commit 75fbf01) `o-voxel/src/convert/volumetic_attr.cpp:559`,
the emissive texture sampler was passed `roughnessMipmaps[mid]` (copy-paste from the
roughness block) instead of `emissiveMipmaps[mid]` — the emissive mipmaps are built at
L364-368 but never used. So the emissive sampler indexed the ROUGHNESS mip pyramid,
returning achromatic roughness-like garbage (the grey R=G=B emissive we measured),
factor-scaled.

Apply + rebuild (isolated, does NOT touch the installed env):
  git clone --filter=blob:none TRELLIS.2, sparse-checkout o-voxel @ 75fbf01,
  fetch the eigen submodule (commit 21e4582), `git apply` this patch,
  `python setup.py build_ext --inplace`, run with PYTHONPATH=<o-voxel dir>.

Verified (job 233531): teddy 0.96→0.00, a535 0.48→0.00 (fabrication gone),
7d46 genuinely-glowing 0.86→0.95 (real emission preserved, now properly structured).
Factor-only emissive path unchanged (default {1,1,1}×factor when no texture — correct).
Residual: KHR_materials_emissive_strength (strength>1) is dropped by trimesh before the
C code sees it, so it is NOT fixed here — a known residual for the UV-sample path too.
