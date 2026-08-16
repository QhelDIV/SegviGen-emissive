title: Extract the citation system and other stranded somages machinery into xgpage
executor: bibcite-extract
track: tooling
status: done
started: 2026-08-12 14:33
updated: 2026-08-12 14:54
slurm: 
link: 
page: none (package work; the proof is the somages pages rebuilding unchanged plus the package tests)
upstreams: pkg_extract
motivation: Owner call: the package should be self-contained, and the bib citation system (one BibTeX file rendering to both web reference cards and the paper's natbib citations, with build-failing gates) lives only in somages tools. Lightgen is the second consumer. The extraction also sweeps somages tools for any other genre-level machinery the package is missing.
log:
- 2026-08-12 14:33 [master] Registered at dispatch. The design-system agent will extract the citation module following the established config-plus-driver pattern and survey the remaining somages tools for other candidates.
- 2026-08-12 14:52 [bibcite-extract] Extracted xgpage.bibcite (dual-render citations, 4 gates, hover cards, figure markers) from somages paper_zone.py; somages driver rebuilt both citation pages, live page verified, somages 3-layer suite passed in full; sweep of somages tools/ found publish_version.py has not adopted xgpage.versioning (flagged, not touched) and everything else already thin or genuinely project-owned.
- 2026-08-12 14:54 [master] The citation system is now a package module (xgpage.bibcite) with 21 tests, and somages' paper zone became a thin driver (537 lines removed) with all 128 of its own test assertions passing and both live pages verified serving identical citation counts after the republish. The sweep found most somages tools already thin; three follow-up candidates recorded: the versioning driver has not adopted the package module it inspired (deferred, production-critical mid-sprint), the pure JS test layer could join the package suite, and the daily-page mechanism awaits a second consumer. One somages commit is local-only pending the owner's push approval per that repo's policy.
outcome: The citation system is now a package module (xgpage.bibcite) with 21 tests, and somages' paper zone became a thin driver (537 lines removed) with all 128 of its own test assertions passing and both live pages verified serving identical citation counts after the republish. The sweep found most somages tools already thin; three follow-up candidates recorded: the versioning driver has not adopted the package module it inspired (deferred, production-critical mid-sprint), the pure JS test layer could join the package suite, and the daily-page mechanism awaits a second consumer. One somages commit is local-only pending the owner's push approval per that repo's policy.
