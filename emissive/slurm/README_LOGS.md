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
