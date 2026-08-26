title: Agentic QC filter: what was kept and what was dropped, as a gallery
executor: qc-gallery
track: research
status: done
started: 2026-08-22 17:13
updated: 2026-08-22 17:34
slurm: 
link: 
page: workspace/qc_filter_gallery
motivation: Dongchen finished the full-dataset agentic QC: 36,652 of 73,470 shapes survive. Owner asked where the filtered list lives and to see a gallery of kept versus dropped shapes, stratified by how each verdict was reached (agent, human, and the four rule-based rejects), drawn at random so the picture is representative.
log:
- 2026-08-22 17:13 [master] Job started.
- 2026-08-22 17:28 [qc-gallery] Built the gallery page: funnel and breakdown charts from the verdict table, plus a random 12-shape gallery per verdict path (agent, human, whole-object rule, albedo copy, zero emission, no atlas, images incomplete), using the same multiview images the annotators judged on. Registered in pages.yaml and published.
- 2026-08-22 17:34 [qc-gallery] Verified live: 288 evidence images all load, no horizontal overflow at 1600px or 390px, per-stratum PNG montage exports wired in and downloadable. Fixed one caption where I'd mislabeled the whole-object-rule gallery's sparse-looking rows.
- 2026-08-22 17:34 [qc-gallery] Published a gallery page (funnel + breakdown charts, 8 stratified 12-shape random-seed galleries plus a small images-incomplete gallery, a reading section, provenance) at workspace/qc_filter_gallery. All numbers re-derived from the verdict csv and cross-checked against the meta.json; all 288 evidence images verified live; no missing-pair resamples needed for the main 8 strata.
outcome: Published a gallery page (funnel + breakdown charts, 8 stratified 12-shape random-seed galleries plus a small images-incomplete gallery, a reading section, provenance) at workspace/qc_filter_gallery. All numbers re-derived from the verdict csv and cross-checked against the meta.json; all 288 evidence images verified live; no missing-pair resamples needed for the main 8 strata.
