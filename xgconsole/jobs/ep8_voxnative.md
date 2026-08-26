title: Epoch-8 samples rendered voxel-native: the 56-shape gallery on the new default pipeline
executor: renderab-scout
track: research
status: ongoing
started: 2026-08-25 18:38
updated: 2026-08-25 21:00
slurm: 
link: 
page: ep8_voxnative
upstreams: render_ab_study,sample56_latest
motivation: Owner, after endorsing the voxel-native pipeline from the A/B study: a separate page showing the epoch-8 sampling results rendered voxel-native, the first full gallery on the newly ratified default renderer.
log:
- 2026-08-25 18:38 [master] Job started.
- 2026-08-25 18:47 [renderab-scout] Started the full gallery on the new default renderer: fifty five shapes, ground truth plus five sampled results each, three hundred and thirty pictures. Reading the epoch eight results straight from where the sampling job wrote them, after checking on three shapes that this reproduces the numbers the earlier page used exactly.
- 2026-08-25 19:18 [renderab-scout] Exports past halfway. One notable result already visible in the numbers: because this page shows all five results per shape rather than one, the variation between them is measurable, and it is larger than the average score. On thirty two of the fifty five shapes at least one of the five results finds no light at all where there should be some. That means showing a single result per shape reports which one was picked as much as it reports the model.
- 2026-08-25 20:53 [renderab-scout] All three hundred and thirty pictures are exported and every one passed an independent check that the exported surface sits where the prediction says it should, worst case ninety seven percent overlap. Two thirds are rendered; the rest are queued behind another job. Two problems handled along the way: a handful of shapes tripped my own checks, which turned out to be the checks being wrong rather than the shapes, and seven render batches died on machines whose shared disk was temporarily invisible, which now retries instead of failing.
- 2026-08-25 21:00 [renderab-scout] Gallery is live: fifty five shapes, each shown with the picture the model was given, the authored original, the ground truth, and all five sampled results, every one drawn by the new default renderer. https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/_preview/ep8_voxnative/index.html Two hundred and sixty of three hundred and thirty pictures are in; the rest are queued behind another job and show as clearly labelled gaps rather than blanks. Checked at five screen widths, no layout faults, no broken images.
outcome: 
