title: Overfit diagnostic, 10 shapes (pw5 cond/zero + pw1 control)
executor: overfit-test
track: research
status: done
started: 2026-08-08 21:38
updated: 2026-08-10 18:18
slurm: 242210 (single-shape run), 242211 (400-epoch run); earlier: 240857, 240858, 242172, 242142, 242178 all completed
link: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/_preview/overfit_condtest/
page: overfit_condtest
motivation: If the model cannot even memorize 10 training shapes, no 72k run can succeed; this isolates why training fails before spending more compute.
log:
- 2026-08-09 17:15 The model fails to memorize the 10 training shapes at either pos_weight setting (best IoU 0.42 and 0.28, against a 0.96 ceiling). So pos_weight 5 is not the sole cause of the training failures; something else in the current setup blocks memorization, or 10 shapes need far more epochs.
- 2026-08-09 17:15 The diagnostic page is complete: rendered examples for all 10 shapes plus both training curves. Two follow-up runs are training now. One trains on a single shape to test whether the current code can still memorize at all (in July it could). The other extends the pos_weight-1 run to 400 epochs so the two settings are compared at equal length. The page updates automatically when they finish.
- 2026-08-09 20:34 [overfit-test] The single-shape control on today's code reached 0.998 IoU, matching July's result, so the code itself is not broken. Extending the pos_weight-1 run to 400 epochs only moved its best score from 0.28 to 0.317, still far below the 0.96 ceiling, so more training time alone is not the fix either. The open question now is what specifically breaks when ten shapes are trained together instead of one.
- 2026-08-09 20:34 [overfit-test] The model cannot memorize ten training shapes together at either loss weighting, but a single shape memorizes cleanly on today's code and giving the ten-shape run more epochs barely helped. So the current code is not broken and pos_weight is not the sole cause; something specific to training multiple shapes together is the remaining suspect, possibly the pretrained shape model handling this family of objects poorly, which is being checked separately.
- 2026-08-09 20:35 [master] For your review: Final verdict page: the single-shape run memorizes to 0.998 while ten shapes together saturate at 0.317 even at 400 epochs. Read the verdict section and the two curves: do you agree the next probe is interference/capacity (e.g. 2-3 shapes, or a larger model) rather than more epochs or more data?
- 2026-08-10 18:18 [owner] I wonder if this is really an overfit? The prediction seems to be not random, it has clear patterns, but nothing reflect the gt, must be some bug.
- 2026-08-10 18:18 [master] Verdict recorded. The owner's reading reframed the page twice: the experiment is a memorization test that FAILS, so nothing on it is an overfit, and the failure pattern (predictions glowing in each shape's own albedo colors) points at a training-target bug rather than a capacity limit. That hypothesis now runs as its own workstream, the training-target decode check, which supersedes this page's next-probe question until it answers. Flag cleared.
- 2026-08-10 18:18 [master] Review complete; flag cleared.
outcome: The model cannot memorize ten training shapes together at either loss weighting, but a single shape memorizes cleanly on today's code and giving the ten-shape run more epochs barely helped. So the current code is not broken and pos_weight is not the sole cause; something specific to training multiple shapes together is the remaining suspect, possibly the pretrained shape model handling this family of objects poorly, which is being checked separately.
