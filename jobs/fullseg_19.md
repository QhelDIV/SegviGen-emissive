title: Original SegviGen part segmentation, 19 shapes
executor: fullseg-19b
track: research
status: done
started: 2026-08-08 23:10
updated: 2026-08-09 16:20
slurm: 242177, 242192, 242193, 242194
link: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/_preview/fullseg_19/
page: fullseg_19
motivation: See what the original SegviGen part segmentation produces on our 19 benchmark shapes, as a reference for the emissive variant.
log:
- 2026-08-09 16:20 tiny-INPUT panel defect (34170054845344..., 0.01% of pixels) diagnosed (stray unrelated 'Icosphere' mesh in the source GLB inflating the auto-scale bbox), fixed, republished, confirmed by eye against its own SEG views
- 2026-08-09 16:20 19/19 shapes rendered, page live and content-reviewed. Two real findings disclosed: (1) predicted mesh's up-axis often differs from the source GLB's, so several SEG panels show the object tumbled relative to INPUT under the same nominal camera -- disclosed, not fixed; (2) one shape's INPUT panel was a near-invisible speck from a stray Icosphere object in its source GLB dominating the auto-scale bbox -- diagnosed and fixed (scoped to this one shape).
outcome: 19/19 shapes rendered, page live and content-reviewed. Two real findings disclosed: (1) predicted mesh's up-axis often differs from the source GLB's, so several SEG panels show the object tumbled relative to INPUT under the same nominal camera -- disclosed, not fixed; (2) one shape's INPUT panel was a near-invisible speck from a stray Icosphere object in its source GLB dominating the auto-scale bbox -- diagnosed and fixed (scoped to this one shape).
