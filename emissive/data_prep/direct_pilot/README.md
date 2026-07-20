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
