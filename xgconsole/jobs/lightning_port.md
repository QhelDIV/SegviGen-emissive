title: SegviGen training on the team's lightning framework
executor: master
track: research
status: ongoing
started: 2026-08-26 10:31
updated: 2026-08-26 10:40
slurm: 
link: 
page: none (code port; the results page comes with the first training run)
upstreams: lightning_feasibility
motivation: Owner: move SegviGen-emissive training onto TRELLIS.2-lightning to get real batching, a proper data loader, and the team's shared training infrastructure; target the 6-GPU Blackwell node once its maintenance drain lifts. Feasibility was established earlier (lightning_feasibility); this job is the port itself, executed by the master.
log:
- 2026-08-26 10:31 [master] Job started.
- 2026-08-26 10:40 [master] Port code is written: the SegviGen conditioning wrapper, a dataset class reading our own training files, a trainer preserving our weighted-loss recipe, and the config. The pretrained checkpoint was rewritten into the framework's format with all 640 weight tensors verified. First 30-step smoke on our own data is now running on solar.
outcome: 
