title: 24h conditioned training, pos_weight 1, capped epochs
executor: team-lead
track: research
status: ongoing
started: 2026-08-09 13:40
updated: 2026-08-09 21:31
slurm: 242125
link: 
page: none (training run; results land on the overfit/eval pages)
motivation: First image-conditioned training on the full 72k dataset that can actually save checkpoints, after the uncapped runs timed out with nothing saved.
log:
- 2026-08-09 15:07 first checkpoint due ~4-5h in; the run that tests whether pw1 learns at scale
- 2026-08-09 18:29 [master] Health check at five hours in: the run is alive, the first epoch finished with the loss curve starting normally, and the first checkpoint is expected after epoch four. A watcher is now on the checkpoint file.
- 2026-08-09 21:31 [master] Eight hours in: epoch 2 finished, loss edging down (0.327 to 0.322). Epochs run about 2.5 hours each, so the first checkpoint (after epoch 4) is expected in roughly two more hours. The run is healthy, just slow from data reading.
outcome: 
