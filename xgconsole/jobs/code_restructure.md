title: One repo: survey and migrate the scattered codebase into SegviGen-emissive
executor: code-survey
track: tooling
status: done
started: 2026-08-14 16:36
updated: 2026-08-19 11:02
slurm: 
link: 
page: none (the survey report and the migrated repo are the deliverables)
upstreams: mask_debug,code_sync
motivation: Owner call: the code is scattered over five locations with drifted duplicates and no stage structure, which is what let render bugs hide in parallel reimplementations. Target: SegviGen-emissive becomes the single project repo with the research code staged under emissive and the project console under xgconsole, each package with its own dependencies. First step is a read-only survey mapping every script to its location, pipeline stage, and duplicate set, resolving drift newest-wins with diffs shown, so the migration is safe rather than hopeful.
log:
- 2026-08-14 16:36 [master] Registered at dispatch. A fresh worker inventories all code locations and produces the migration map for owner ratification; no files move in this pass.
- 2026-08-14 16:36 [code-survey] Starting read-only survey of all code locations for the migration map.
- 2026-08-14 16:42 [code-survey] Read-only survey done across the ops repo, the fork, the cluster working copy, and the two debug scratch trees; migration map written, two real content drifts found and resolved in favor of the cluster debug copy.
- 2026-08-14 16:53 [code-survey] Consolidation done and pushed: ops history merged into SegviGen-emissive (QhelDIV/SegviGen-emissive main 1842e3b), tree sorted into emissive/ + xgconsole/ + attic/, mask_debug continuous-render fixes ported into emissive/render/, per-package pyproject files separate the torch stack from the console stack. This repo stays the live console until cutover.
- 2026-08-14 20:59 [code-survey] SegviGen-emissive is the single project repo: ops history merged with full log continuity, tree sorted into emissive plus xgconsole plus attic, the two drifted mask_debug render fixes ported, per-package pyproject files separating the research and console stacks, all verified (editable installs, tools resolving from the new root, compile sweep) and pushed. Deliberately out of scope, pending the owner: cutting the live console over from the ops working directory, and deploying the fork tree to the cluster so the old flat code directory retires.
- 2026-08-17 15:23 [code-survey] Console cutover done: canonical checkout now lives at /cs/3dlg-falas/project/omages/lightgen/segvigen_emissive (falas NFS, 19T free; the omages share stayed over quota). Both cron sweeps, the board CLI, the graph, and the full console publish run from the new root; graph positions carried over; the old local folder carries a RETIRED marker.
- 2026-08-17 15:24 [code-survey] Console cutover done: canonical checkout now lives at /cs/3dlg-falas/project/omages/lightgen/segvigen_emissive (falas NFS, 19T free; the omages share stayed over quota). Both cron sweeps, the board CLI, the graph, and the full console publish run from the new root; graph positions carried over; the old local folder carries a RETIRED marker.
- 2026-08-19 11:02 [master] Cluster deploy done: the consolidated tree now lives at the jupiter root (emissive/ + xgconsole/ + the fork's own inference_full.py anchor), the old flat code/ retired to code_retired_20260819, smoke check passed in the trellis2 env. The two-day solar outage that delayed this never existed: solar's SSH listens on port 24 and the probes tested 22.
outcome: SegviGen-emissive is the single project repo: ops history merged with full log continuity, tree sorted into emissive plus xgconsole plus attic, the two drifted mask_debug render fixes ported, per-package pyproject files separating the research and console stacks, all verified (editable installs, tools resolving from the new root, compile sweep) and pushed. Deliberately out of scope, pending the owner: cutting the live console over from the ops working directory, and deploying the fork tree to the cluster so the old flat code directory retires.
