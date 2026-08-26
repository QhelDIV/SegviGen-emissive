title: Processed GT emission masks of the agentic split, next to the authored GT
executor: gt-mask-scout
track: research
status: done
started: 2026-08-23 16:00
updated: 2026-08-23 22:26
slurm: 
link: 
page: agentic_gt_masks
upstreams: agentic_train
motivation: Owner wants to see samples of the processed GT mask for the filtered 37k dataset: the binary emission target the agentic run actually trains on, rendered from dataset_direct, shown beside the authored GT emission so processing errors would be visible.
log:
- 2026-08-23 16:00 [master] Job started.
- 2026-08-23 16:08 [gt-mask-scout] Read the derivation code (build_dataset_direct.py, make_emis_mask.py, train_emissive.py, eval_emissive.py) and found the prior render helpers (render_target_check.py, render_pred_mask.py, render_voxels.py) to reuse for the voxel mask panels.
- 2026-08-23 17:02 [gt-mask-scout] Sampled 2000 shapes for stats and 16 for the gallery, dumped exact masks from output.vxz, rendered them as voxel cubes with a self-lit material so tiny masks stay visible, built the page, and verified it live at 1600 and 390 pixels wide.
- 2026-08-23 17:02 [gt-mask-scout] The processed emission mask matches the authored ground truth. Checked 16 shapes across the emissive_frac range: 15 show the mask landing on exactly the same surface as the authored emit-only render, including fine internal patterns (a cross-hatched panel, parallel body-seam lights), with no dilation or dropped region. One wheel-rim shape is angle-ambiguous between the two render viewpoints rather than a clear mismatch. Across a 2000-shape sample, most training masks are small: 9 of 2000 carry no emission after processing, only 205 exceed a fifth of the shape's voxels. Page: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/_preview/agentic_gt_masks/index.html
- 2026-08-23 17:05 [gt-mask-scout] Removed the em dashes the team lead flagged from the gallery intro and column headers, rebuilt, republished, and confirmed live that none remain.
- 2026-08-23 21:46 [master] Reopened, owner follow-up: scale the gallery to 120+ examples (input, authored GT, processed mask) so GT sanity is judged on a real sample, not 16 shapes. Dispatched to gt-mask-scout.
- 2026-08-23 21:48 [gt-mask-scout] Starting the 120-shape gallery expansion: stratified sample from the full 36,226-shape population, exact mask dump, single centroid-aimed render per shape, and a full 136-shape honest-reading tally.
- 2026-08-23 22:26 [gt-mask-scout] Expanded the gallery to 136 shapes total: sampled 120 more from the full 36226-shape population, dumped exact masks, rendered them in parallel on solar, ran an automated presence check on all 136, and eyeballed every shape (16 individually, 120 via compact review grids). Verdict: 132 match cleanly, 3 angle-ambiguous (ring or disc emitters seen edge-on), 1 inconclusive (both camera angles view it edge-on), 0 confirmed mismatches. Republished and verified live at 1600 and 390 pixels wide, all 424 images loading.
- 2026-08-23 22:26 [gt-mask-scout] Expanded to 136 shapes (16 individually inspected + 120 more stratified across the full 36226-shape population, seed 20260824). Verdict: 132/136 match cleanly, 3 angle-ambiguous (a ring or disc emitter seen edge-on by the mask's zoom camera: cdfcb4a8, 929ed98d1a, 510ad761b9), 1 inconclusive (6e970c9740, both camera angles view it edge-on), 0 confirmed mismatches. Backed by two automated 100%-coverage checks (render shows the glow color: 135/136; authored emit-only shows some bright pixel: 136/136) plus a real eyeball pass on every shape. Page: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/_preview/agentic_gt_masks/index.html
outcome: Expanded to 136 shapes (16 individually inspected + 120 more stratified across the full 36226-shape population, seed 20260824). Verdict: 132/136 match cleanly, 3 angle-ambiguous (a ring or disc emitter seen edge-on by the mask's zoom camera: cdfcb4a8, 929ed98d1a, 510ad761b9), 1 inconclusive (6e970c9740, both camera angles view it edge-on), 0 confirmed mismatches. Backed by two automated 100%-coverage checks (render shows the glow color: 135/136; authored emit-only shows some bright pixel: 136/136) plus a real eyeball pass on every shape. Page: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/_preview/agentic_gt_masks/index.html
