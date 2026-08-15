# web/ — lightgen published content

This directory IS the project's web root:
  /project/3dlg-hcvc/omages/www/yanxg/lightgen → lightgen/web/

The console lives at the directory root (index.html). Published page directories
sit alongside it. In-progress pages live under _preview/ until promoted.

## Published pages

- dataset_gallery_v1/    — fine-tune dataset statistics + gallery
- finetune_binary_v1/    — fine-tune data + binary predictions
- training_curves_v1/    — loss + IoU curves for Phase 4 runs
- official_repro/        — SegviGen official example reproduction
- gt_vs_pred_canon10/    — GT parts vs full-seg predictions (mesh)
- fullseg_canon10_mesh/  — full-seg on decoded mesh surface
- fullseg_canon10/       — full-seg as voxels (superseded)
- fullseg_overfit10_adhoc/ — ad-hoc 10-shape set (deprecated)

## pages.yaml

The curated page registry (blurbs, tags, and since round-3 of the graph
rework: `upstreams:` — which page(s) motivated this one's creation, the
page-relationship graph's directed arrows — and an optional `shortname:`
display override for a long id). See pages.yaml's own header comment for
the full field contract and the `_preview/`-prefix gotcha (a registry
entry's own `name:` may carry the prefix; every `upstreams:`/graph_edges.yaml
reference to ANY page must use the bare id regardless).

## Rebuild console

    python tools/build_console.py --publish

Contents are NOT committed (generated + binary); only README.md and pages.yaml are tracked.
