# SegviGen-emissive: places and standing policy

(Agent-facing orientation. The lightgen-worker agent template carries the
full standing rules; this file records the location policy so any tool or
person landing in the repo inherits it.)

## Default target: FALAS (owner-ratified 2026-08-25)

Every NEW output, artifact, staging dir, or run out_dir goes under this
repo's tree on falas:

    workstation:   /cs/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/...
    compute nodes: /3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/...

The /cs prefix exists only on the workstation; never put it in an sbatch.

## Jupiter is legacy

/3dlg-jupiter-project/lightgen/segvigen_emissive/ is the old cluster deploy
and holds existing datasets (dataset_direct), live-run outputs, and the
canonical fir checkpoint mirror. LIVE RUNS KEEP THEIR JUPITER PATHS until
they end; do not create new artifacts there without a stated reason.
Consolidation of the deploy onto falas is a planned post-campaign move.

## Team data is read-only and stays put

/cs/3dlg-jupiter-project/lightgen/{uv_voxel_pipeline,trellis2_bw,annotate74k}
belong to the team (Dongchen); read where they live, never write, never move.

## Other placement rules

- sbatch logs: the run's own output dir or logs/, never a repo root
  (emissive/slurm/README_LOGS.md).
- Published web root: /project/3dlg-hcvc/omages/www/yanxg/lightgen
  (merge-copy, never delete).
- Console, pages, jobs board: xgconsole/ in this repo.
