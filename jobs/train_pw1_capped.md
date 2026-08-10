title: 24h conditioned training, pos_weight 1, capped epochs
executor: team-lead
track: research
status: ongoing
started: 2026-08-09 13:40
updated: 2026-08-10 09:35
slurm: 242125
link: 
page: none (training run; results land on the overfit/eval pages)
motivation: First image-conditioned training on the full 72k dataset that can actually save checkpoints, after the uncapped runs timed out with nothing saved.
log:
- 2026-08-09 15:07 first checkpoint due ~4-5h in; the run that tests whether pw1 learns at scale
- 2026-08-09 18:29 [master] Health check at five hours in: the run is alive, the first epoch finished with the loss curve starting normally, and the first checkpoint is expected after epoch four. A watcher is now on the checkpoint file.
- 2026-08-09 21:31 [master] Eight hours in: epoch 2 finished, loss edging down (0.327 to 0.322). Epochs run about 2.5 hours each, so the first checkpoint (after epoch 4) is expected in roughly two more hours. The run is healthy, just slow from data reading.
- 2026-08-10 00:33 [master] Eleven hours in: epoch 3 finished, loss still edging down (0.318). The first checkpoint saves after epoch 4, now expected two to three hours from now. Healthy, just slow.
- 2026-08-10 03:33 [master] Fourteen hours in: the epoch 4 checkpoint is saved and its quick validation scored 0.03 on the nonzero-only metric, a weak early signal on a small noisy subset. The proper averaged evaluation of this checkpoint is running as its own job; the run continues toward the epoch 8 checkpoint.
- 2026-08-10 06:34 [master] Seventeen hours in: epoch 5 done, loss 0.312 and still easing down. The epoch 8 checkpoint should land near the end of the 24 hour window; its averaged evaluation will reuse the pipeline being built for epoch 4 right now.
- 2026-08-10 09:35 [master] Twenty hours in: epoch 6 done, loss flat around 0.31 to 0.32. About four hours remain in the window; whether the epoch 8 checkpoint saves before the time limit is now close, and the run may end at epoch 7 with only the epoch 4 checkpoint plus the last-state file.
outcome: 
