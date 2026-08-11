title: Epoch 8 checkpoint: averaged eval and rendered examples
executor: overfit-test
track: research
status: ongoing
started: 2026-08-10 16:38
updated: 2026-08-10 16:40
slurm: 
link: 
page: none (page pending; will mirror the epoch 4 eval page)
motivation: The run's final checkpoint improved sixfold on the trainer's tiny probe; the project needs the same averaged held-out evaluation as epoch 4 to know whether the jump is real and whether continuing training is worth the compute.
log:
- 2026-08-10 16:38 [overfit-test] Job started.
- 2026-08-10 16:40 [overfit-test] Started the identical protocol used for epoch 4: same held-out subset, same historical and familiar-shape populations, five draws, both weight sets, both conditioning modes, plus the render figure. Reused everything from the epoch-4 run except the checkpoint. Keeping the same node exclusions as before, including the one node that showed the access problem.
outcome: 
