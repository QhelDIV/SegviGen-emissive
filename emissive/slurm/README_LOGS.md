# Log placement convention (2026-08-20)

sbatch logs NEVER go to the deploy root. `--output` points at either:
- `/3dlg-jupiter-project/lightgen/segvigen_emissive/logs/<name>_%j.log` for
  repo-level utility jobs, or
- the job's own output directory (`outputs/<run>/<name>_%j.log`) for
  training/eval runs, beside the artifacts they describe.

Why: the deploy root accumulated 184 stray logs by 2026-08-20 (archived to
`logs/archive_pre20260820/`) because old templates defaulted `--output` to
the root. A messy root hides real structure; keep the root for code and
data directories only.

## See also

`README_RUN_PINNING.md` — how to know which trainer a run actually executed. SLURM
copies the sbatch at submit but not the code the script reads at start, so a queued
job runs whatever is committed when it begins. Each run records that fact in
`TRAINER_AT_START.txt`. The doc also records why copying the code into the run
directory was tried, failed in production, and reverted, and why a takeover job
must prove it can start before it cancels the job it is replacing.
