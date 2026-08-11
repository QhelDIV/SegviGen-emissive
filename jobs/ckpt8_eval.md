title: Epoch 8 checkpoint: averaged eval and rendered examples
executor: overfit-test
track: research
status: ongoing
started: 2026-08-10 16:38
updated: 2026-08-10 18:14
slurm: 
link: 
page: none (page pending; will mirror the epoch 4 eval page)
motivation: The run's final checkpoint improved sixfold on the trainer's tiny probe; the project needs the same averaged held-out evaluation as epoch 4 to know whether the jump is real and whether continuing training is worth the compute.
log:
- 2026-08-10 16:38 [overfit-test] Job started.
- 2026-08-10 16:40 [overfit-test] Started the identical protocol used for epoch 4: same held-out subset, same historical and familiar-shape populations, five draws, both weight sets, both conditioning modes, plus the render figure. Reused everything from the epoch-4 run except the checkpoint. Keeping the same node exclusions as before, including the one node that showed the access problem.
- 2026-08-10 17:36 [master] Owner extended this job: three example galleries (the paper's examples picked by Dongchen, our picked emissive set with the pumpkin, and the familiar 8) with parallel compute ahead of the validation matrix. An earlier attempt to record this crashed in a tooling migration window and was silently lost; the master is now submitting the gallery jobs directly.
- 2026-08-10 17:48 [overfit-test] gallery-runner: fig7 set (11 sids) submitted. survey build job 242799 (CPU, material slot order for mask transfer), inference dump job 242800 (GPU a40, raw+ema+zero cond, 5 draws, split fig7_11). set b sid list has a discrepancy vs the brief, flagged to team-lead, not yet started.
- 2026-08-10 17:51 [overfit-test] gallery-runner: all 11 fig7 sids sit in dataset_direct/val_72k (clean held-out split, not train_72k) -- their epoch-8 numbers are a genuine held-out read, not contaminated.
- 2026-08-10 17:54 [overfit-test] gallery-runner: set b (paper_v3 12 sids: pumpkin, sci-fi weapon, vending machine, candles x3, headphone stand, plus 6 others) confirmed present in the TexVerse catalog and in Dongchen's ovoxel data -- not a from-scratch build. Preprocessing job 242802 submitted (build_dataset_direct + build_cond_thumbnail chained, GPU a40).
- 2026-08-10 18:04 [overfit-test] gallery-runner: IMPORTANT finding while writing example_sets.json -- the familiar fbv1_8 showcase set is NOT held out. dataset_direct/fbv1_8/<sid> is a symlink to dataset_direct/train_72k/<sid> for all 8 shapes (verified via readlink + content-identical meta.json). Every prior gallery using this set (including the first-72k-checkpoint page) reported train-split numbers under a held-out framing. fig7_11 and the new showcase_12 are genuinely held out (val / never-in-catalog respectively); fbv1_8 is not. Canonical sid/caption/split registry for all three sets now at lightgen/example_sets.json.
- 2026-08-10 18:14 [overfit-test] gallery-runner: fig7 mask transfer (242806) COMPLETED clean, 33/33. render job 242877 submitted (64 cores, gres=gpu:0, GT/raw_real/ema_real box variants). showcase_12 mask transfer (242807) FAILED all 36 tasks -- my error: I submitted mask-transfer for showcase_12 before ever submitting its inference dump, so pred_voxels was empty. Fixed: dump job 242878 submitted now, mask-transfer will be resubmitted once it lands.
outcome: 
