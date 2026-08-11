title: Mask bake debug: why a saturated prediction renders streaked, not uniform
executor: gallery-runner
track: research
status: ongoing
started: 2026-08-11 14:07
updated: 2026-08-11 14:07
slurm: 
link: 
page: none (page pending; the debug page is the deliverable)
motivation: The owner proved by argument that a saturated prediction with correct UV application must render uniformly lit, yet the showcase pumpkin streaks while the fig7 lantern renders uniform under the same degenerate prediction; the streaks therefore indicate a real coverage or UV-application defect in the mask bake, and the earlier exoneration's ground-truth control was blind to misalignment.
log:
- 2026-08-11 14:07 [gallery-runner] Job started.
- 2026-08-11 14:07 [owner] We need a decicated page to debug this! illustrating / visualizing the data format of the predicted mask, and how it is converted to ... so illustrating / visualizing / debugging everything. In thoery, we should not have the bad 'pattern'
outcome: 
