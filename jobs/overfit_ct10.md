title: Overfit diagnostic, 10 shapes (pw5 cond/zero + pw1 control)
executor: overfit-test
status: ongoing
started: 2026-08-08 21:38
updated: 2026-08-09 15:09
slurm: train pw5 cond 240857 (done), zero 240858 (done); pw1 control 242172 (running); maskxfer reshard array 242142 (running, resubmitted after 240862 timed out serial at case 29/30); render pending resubmit
link: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/_preview/overfit_condtest/
now: page live with pw5 verdict (plateau, does not overfit); pw1 control (--pos_weight 1.0) running to tell apart broken-pipeline vs pos_weight-5-specific failure; maskxfer resharded to a 30-task array after first serial attempt timed out and stranded the render job; render to be resubmitted by hand (no dependency chain) once maskxfer array is verified fully done
outcome: 
