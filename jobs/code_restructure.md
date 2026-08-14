title: One repo: survey and migrate the scattered codebase into SegviGen-emissive
executor: code-survey
track: tooling
status: ongoing
started: 2026-08-14 16:36
updated: 2026-08-14 16:53
slurm: 
link: 
page: none (the survey report and the migrated repo are the deliverables)
motivation: Owner call: the code is scattered over five locations with drifted duplicates and no stage structure, which is what let render bugs hide in parallel reimplementations. Target: SegviGen-emissive becomes the single project repo with the research code staged under emissive and the project console under xgconsole, each package with its own dependencies. First step is a read-only survey mapping every script to its location, pipeline stage, and duplicate set, resolving drift newest-wins with diffs shown, so the migration is safe rather than hopeful.
log:
- 2026-08-14 16:36 [master] Registered at dispatch. A fresh worker inventories all code locations and produces the migration map for owner ratification; no files move in this pass.
- 2026-08-14 16:36 [code-survey] Starting read-only survey of all code locations for the migration map.
- 2026-08-14 16:42 [code-survey] Read-only survey done across the ops repo, the fork, the cluster working copy, and the two debug scratch trees; migration map written, two real content drifts found and resolved in favor of the cluster debug copy.
- 2026-08-14 16:53 [code-survey] Consolidation done and pushed: ops history merged into SegviGen-emissive (QhelDIV/SegviGen-emissive main 1842e3b), tree sorted into emissive/ + xgconsole/ + attic/, mask_debug continuous-render fixes ported into emissive/render/, per-package pyproject files separate the torch stack from the console stack. This repo stays the live console until cutover.
outcome: 
