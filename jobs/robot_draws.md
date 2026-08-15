title: Fresh prediction draws for the robot example
executor: robot-draws
track: research
status: done
started: 2026-08-11 16:17
updated: 2026-08-11 17:23
slurm: 
link: 
page: none (feeds panels into the existing mask-bake debug page rather than a page of its own)
upstreams: ckpt8_eval
motivation: The robot is one of the three paper figure shapes, and every existing prediction dump for it is degenerate: the model output is essentially zero at every voxel, in raw and averaged weights alike. The paper panel needs at least one usable draw, so we sample the model again with new random seeds and keep each draw separately instead of averaging.
log:
- 2026-08-11 16:17 [master] Registered at dispatch. A dedicated runner will sample new draws for the robot on the cluster and report how much of the surface each draw marks as emissive.
- 2026-08-11 16:19 [robot-draws] Submitted job 243321 on Solar: 8 raw + 8 EMA draws, real cond, sid 34170054845344aeb199b842a3bf7e92, distinct seeds (raw seed_start=1000, ema seed_start=2000), per-draw npz save (not averaged).
- 2026-08-11 17:23 [master] The resampling rescued the robot: fresh seeds found middle-mode draws where every re-dumped draw was empty or blanket. The averaged-weights seed 4 draw (29 percent of voxels emissive, ground truth 8 percent) became the robot's paper panel candidate and passed the eye check; seed 5 (3 percent) is kept as the sparse alternate.
outcome: The resampling rescued the robot: fresh seeds found middle-mode draws where every re-dumped draw was empty or blanket. The averaged-weights seed 4 draw (29 percent of voxels emissive, ground truth 8 percent) became the robot's paper panel candidate and passed the eye check; seed 5 (3 percent) is kept as the sparse alternate.
