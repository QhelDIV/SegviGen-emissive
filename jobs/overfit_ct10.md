title: Overfit diagnostic, 10 shapes (pw5 cond/zero + pw1 control)
executor: overfit-test
track: research
status: ongoing
started: 2026-08-08 21:38
updated: 2026-08-09 17:15
slurm: 242210 (single-shape run), 242211 (400-epoch run); earlier: 240857, 240858, 242172, 242142, 242178 all completed
link: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/_preview/overfit_condtest/
page: overfit_condtest
motivation: If the model cannot even memorize 10 training shapes, no 72k run can succeed; this isolates why training fails before spending more compute.
log:
- 2026-08-09 17:15 The model fails to memorize the 10 training shapes at either pos_weight setting (best IoU 0.42 and 0.28, against a 0.96 ceiling). So pos_weight 5 is not the sole cause of the training failures; something else in the current setup blocks memorization, or 10 shapes need far more epochs.
- 2026-08-09 17:15 The diagnostic page is complete: rendered examples for all 10 shapes plus both training curves. Two follow-up runs are training now. One trains on a single shape to test whether the current code can still memorize at all (in July it could). The other extends the pos_weight-1 run to 400 epochs so the two settings are compared at equal length. The page updates automatically when they finish.
outcome: 
