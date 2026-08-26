title: Match the baselines' prefiltered training subset and continue-train epoch 8
executor: filter-scout
track: research
status: ongoing
started: 2026-08-19 16:19
updated: 2026-08-19 16:59
slurm: 
link: 
page: none (investigation first; the training run and its page come after the rules are pinned down)
upstreams: ckpt8_eval
motivation: Dongchen retrained the other baselines on a prefiltered subset of the 74k cohort; our epoch-8 checkpoint must continue training on the same filtered data or the head-to-head confounds data curation with method. First step is pinning down the exact filter rules, the subset lists, and which baseline checkpoints used them.
log:
- 2026-08-19 16:19 [master] Job started.
- 2026-08-19 16:19 [owner] Dongchen retrains other baselines on some prefiltering rules, have an agent investigate this, and we shall also continue to train our ckpt 8 based on this filtered subset
- 2026-08-19 16:23 [filter-scout] Found the prefilter rules on an unmerged branch, agentic-qc-study1000: drop zero-emission (crosstab new_max==0) and albedo-copy shapes (frac_exact>=0.99 and emit_area>=0.5) from train only, val and test unchanged.
- 2026-08-19 16:24 [filter-scout] Filtered split file and the sha list live on jupiter at trellis2_bw/lightgen_74k_newbake/: train drops from 71,646 to 63,194 shapes, val 387 and test 388 unchanged. Our train_72k set has the exact same 71,646 shapes as Dongchen's split, so all 63,194 kept shapes are already in our data and no id mapping is needed.
- 2026-08-19 16:24 [filter-scout] The three retrainable baselines (TRELLIS.2 pbr to emission, TEXGen alpha, Hunyuan3D-Paint alpha) were all submitted on this filtered subset on 2026-08-19; none had finished as of the branch tip I read. The 381-shape evaluation set is untouched by this filter.
- 2026-08-19 16:24 [filter-scout] This filter (drop zero emission and albedo-copy shapes, keeping 88 percent of train) is a separate, much gentler rule than our own earlier funnel (72,427 to 26,989 trainable); they should not be conflated. Wrote the surviving 63,194-shape id list to my scratchpad for the continue-training step.
- 2026-08-19 16:31 [filter-scout] Pre-flight check passed: recomputed the albedo-copy rule myself on 200 random shapes from Dongchen's drop list, reading our own atlas source directly. 200 of 200 agreed with his numbers, so the filter reproduces cleanly on our side. Proceeding to set up the continuation run.
- 2026-08-19 16:36 [filter-scout] Built the filtered training view: dataset_direct/train_72k_nonzero_nocopy, 63,129 of the 63,194 kept shapes symlinked (65 short because our own data is missing slat files for them, same small gap the original run had). No shape is missing its conditioning file, so the run will not crash partway through.
- 2026-08-19 16:59 [filter-scout] Smoke test (job 246921) passed cleanly: the epoch-8 checkpoint resumed with no errors, the filtered sampler loaded exactly the 63,129 shapes we built, and one training step plus a checkpoint save worked. The 65 shapes short of the full 63,194 keep list all have a directory but no encoded training data at all (never finished the encoding step in our dataset build); 52 of 65 have a recorded reason (missing source file or a GPU running out of memory during encoding), 13 have no record and are being left as an unexplained small gap, listed on request.
- 2026-08-19 16:59 [filter-scout] Main continuation run submitted: job 246931 on solar, resuming the epoch-8 checkpoint for 8 more epochs on the filtered 63,129-shape set, checkpoint and quick eval every 2 epochs, full averaged evaluation to follow on each checkpoint. Output: outputs/emis_72kv2_cond_pw1b_filtered/. Expected to take roughly 18 to 21 hours based on the original run's per-epoch time, well inside the 48 hour budget.
outcome: 
