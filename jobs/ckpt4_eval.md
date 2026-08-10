title: First 72k conditioned checkpoint: averaged eval and rendered examples
executor: overfit-test
track: research
status: ongoing
started: 2026-08-10 01:38
updated: 2026-08-10 03:05
slurm: 
link: 
page: none (page pending; will hold the eval results and renders)
motivation: The capped 72k run finally produced a checkpoint; the project needs to know whether conditioned training on the full dataset beats the zero-shot oracle and the old runs, measured with the averaged protocol so single-draw luck cannot mislead again.
log:
- 2026-08-10 01:38 [overfit-test] Job started.
- 2026-08-10 01:46 [overfit-test] Found that the historical val_96 set is mostly this checkpoint's own training data: 105 of its 111 shapes are inside the current train_72k split, only 2 are held out. Using val_72k instead as the true held-out standard set, and still reporting val_96 alongside labeled contaminated, for continuity with the old oracle and honest-old reference numbers. Also checked the 8 familiar render shapes: all 8 are in train_72k for this checkpoint too, so every render panel will be labeled seen-in-training rather than held-out.
- 2026-08-10 03:04 [overfit-test] Mask transfer for the render figure is done, including a retry for two dense many-material shapes that ran past the first time budget. Averaged eval draws are running now across the held-out set, the historical val_96 set, and the 8 familiar shapes, at both checkpoint weights and both conditioning modes; a few have already finished. Renders come next once the mask transfer is confirmed complete.
- 2026-08-10 03:05 [overfit-test] One of the held-out eval runs crashed partway through: a shape in val_72k is missing its conditioning file, which the real-conditioning mode needs. Built a clean 96-shape subset of val_72k where every shape has its conditioning file present, and switched all four held-out configurations to that subset so the comparison is on identical shapes throughout. This is the same conditioning backfill gap already tracked as a separate open item; nothing else needed on my end. Renders have started.
outcome: 
