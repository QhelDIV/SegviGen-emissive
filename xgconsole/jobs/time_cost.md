title: Where the pipeline's time goes, step by step
executor: time-cost
track: research
status: ongoing
started: 2026-08-11 16:37
updated: 2026-08-11 16:57
slurm: 
link: 
page: time_cost
needs: evaluation: The page names three concrete optimization candidates (double model load in inference jobs, conversion time limits that ignore material-slot count, a non-checkpointed render job that lost 27 core-hours); which do you want pursued first?
motivation: Owner request: a page showing the time cost of each pipeline step, from data processing to model inference to conversion to Blender rendering. The owner suspects some steps take excessive time that we can optimize; the page should show measured durations, not impressions, and point at the dominant costs.
log:
- 2026-08-11 16:37 [master] Registered at dispatch. A worker will mine the cluster accounting records and job logs for real per-step durations and build the page.
- 2026-08-11 16:37 [owner] [owner] Also, I want a page to show the time cost for each steps, from data processing, to model inference, to conversion, to blender rendering. I have the feeling some steps take excessive time that we can optimize
- 2026-08-11 16:56 [time-cost] Published the pipeline time cost page: rendering is the biggest cost by far, and three specific causes explain most of the waste elsewhere.
- 2026-08-11 16:57 [master] For your review: The page names three concrete optimization candidates (double model load in inference jobs, conversion time limits that ignore material-slot count, a non-checkpointed render job that lost 27 core-hours); which do you want pursued first?
outcome: 
