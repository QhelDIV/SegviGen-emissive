title: Can SegviGen train on TRELLIS.2-lightning? Assessment and smoke test
executor: lightning-scout
track: research
status: done
started: 2026-08-23 19:45
updated: 2026-08-23 21:02
slurm: 
link: 
page: lightning_feasibility
upstreams: train_rungraph,agentic_train
motivation: Owner: Dongchen trained the TRELLIS.2 emissive baselines on the team's TRELLIS.2-lightning repo; maybe SegviGen can train under the same setup. The framework already carries configs for the exact 1.3B slat flow SegviGen wraps, so the port is plausibly one model class plus one config. Assess it concretely and prove the framework runs in our hands with a smoke of an existing config.
log:
- 2026-08-23 19:45 [master] Job started.
- 2026-08-23 19:48 [lightning-scout] Cloned the team's TRELLIS.2-lightning framework into a scratch working copy at the exact commit Dongchen used, and started reading how its training loop is put together.
- 2026-08-23 20:02 [lightning-scout] The framework now runs in our hands: a short training run of one of Dongchen's own configs executes real steps on a lab GPU, logs its loss, and writes checkpoints.
- 2026-08-23 20:18 [lightning-scout] Confirmed the checkpoint the run wrote can be picked up again: a second run resumed from it and carried on from the exact step it stopped at.
- 2026-08-23 20:32 [lightning-scout] Wrote up the assessment as a page: the framework can train our model, the change is a few small pieces of code rather than a rewrite, and the one thing that would go wrong quietly is our saved weights failing to load while the run carries on training from scratch.
- 2026-08-23 20:32 [lightning-scout] Yes, our model can be trained under the team's framework, and the framework now demonstrably runs on our cluster. The flow model the two sides share is the same file, so the work is a small wrapper plus a data reader plus a config, about three to four days to a checked first run. It does not put our numbers and Dongchen's on the same scale, because the two predict different things. Page: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/_preview/lightning_feasibility/index.html
- 2026-08-23 20:59 [lightning-scout] Owner moved the working directory onto our own project storage, so the whole check was redone there.
- 2026-08-23 20:59 [lightning-scout] Moved the scratch working copy onto our project storage as the owner asked, re-ran both short training jobs from the new place to confirm the cluster can read and write it, deleted the old copy, and updated the page so every path and number on it points at where the work now lives.
- 2026-08-23 21:00 [lightning-scout] Yes, our model can be trained under the team's framework, and the framework now demonstrably runs on our cluster from our own project storage. The flow model the two sides share is the same file, so the work is a small wrapper plus a data reader plus a config, about three to four days to a checked first run. It does not put our numbers and Dongchen's on the same scale, because the two predict different things. Page: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/_preview/lightning_feasibility/index.html
- 2026-08-23 21:02 [lightning-scout] Two small closers from the review.
- 2026-08-23 21:02 [lightning-scout] Cleared the last stray dash from the page's wording and republished, then confirmed the published page has none left. The scratch working copy already sat on our project storage from the earlier move, so the old location now just holds a note saying where it went.
- 2026-08-23 21:02 [lightning-scout] Yes, our model can be trained under the team's framework, and the framework now demonstrably runs on our cluster from our own project storage. The flow model the two sides share is the same file, so the work is a small wrapper plus a data reader plus a config, about three to four days to a checked first run. It does not put our numbers and Dongchen's on the same scale, because the two predict different things. Page: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/_preview/lightning_feasibility/index.html
outcome: Yes, our model can be trained under the team's framework, and the framework now demonstrably runs on our cluster from our own project storage. The flow model the two sides share is the same file, so the work is a small wrapper plus a data reader plus a config, about three to four days to a checked first run. It does not put our numbers and Dongchen's on the same scale, because the two predict different things. Page: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/_preview/lightning_feasibility/index.html
