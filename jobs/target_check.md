title: Decode the training targets: is the model being taught albedo instead of emission
executor: overfit-test
track: research
status: ongoing
started: 2026-08-10 16:50
updated: 2026-08-10 16:50
slurm: 
link: 
page: none (small diagnostic; verdict goes in the log and the overfit page if confirmed)
motivation: The owner noticed overfit predictions glow in each shape's own albedo colors while matching nothing in the emission ground truth, which points at the dataset builder feeding the wrong target; if true, the entire multi-shape training failure is a data bug, not a capacity limit.
log:
- 2026-08-10 16:50 [overfit-test] Job started.
outcome: 
