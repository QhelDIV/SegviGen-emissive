title: Fresh prediction draws for the warhammer example
executor: robot-draws
track: research
status: done
started: 2026-08-11 16:23
updated: 2026-08-11 17:23
slurm: 
link: 
page: none (feeds panels into the existing mask-bake debug page rather than a page of its own)
motivation: Every saved draw for the warhammer is empty or nearly so, with at most 0.6 percent of voxels marked emissive against a ground truth of 8.5 percent. Fresh random seeds rescued the robot, whose first draws looked just as hopeless, so the same resampling is worth trying for the hammer before we accept a near-black paper panel.
log:
- 2026-08-11 16:23 [master] Registered at dispatch. The same runner that resampled the robot will sample fresh draws for the warhammer with the identical protocol.
- 2026-08-11 16:24 [robot-draws] Submitted job 243329 on Solar: 8 raw (seed_start=3000) + 8 EMA (seed_start=4000) draws, real cond, sid 4e383188516c46a58e96b1b7fc2f16a7 (warhammer), same per-draw npz protocol as robot_draws, fresh seed ranges to avoid collision.
- 2026-08-11 16:26 [master] The resampling worked: one raw draw marks 10.7 percent of the voxels emissive, right next to the ground truth's 8.5 percent, while every averaged-weights draw stayed near empty. That draw becomes the hammer's paper panel candidate and is being rendered now.
- 2026-08-11 17:23 [master] The resampling rescued the hammer as well: one raw draw marks 10.7 percent of voxels emissive against the ground truth's 8.5 percent, while all other draws stayed near empty. That draw's render matches the ground truth closely and became the hammer's paper panel.
outcome: The resampling rescued the hammer as well: one raw draw marks 10.7 percent of voxels emissive against the ground truth's 8.5 percent, while all other draws stayed near empty. That draw's render matches the ground truth closely and became the hammer's paper panel.
