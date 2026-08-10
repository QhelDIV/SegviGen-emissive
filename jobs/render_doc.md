title: Name and document the project's rendering setups
executor: render-doc
status: done
started: 2026-08-09 18:05
updated: 2026-08-09 19:40
slurm: 
link: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/workspace/rendering/
motivation: Figures across the pages and the paper are made with several different lighting setups, and nobody can currently tell which one a given figure used or which one to reach for. Naming the five setups and writing down their verified parameters makes figure choices consistent and reviewable.
log:
- 2026-08-09 18:05 Started. Reading the render scripts to verify every parameter before writing anything down.
- 2026-08-09 19:40 All five setups are named and written up, in a repo document and on a workspace page. Every parameter was read off the scripts and cross-checked against the settings blocks that real published renders wrote alongside themselves, so nothing is quoted from memory. No new rendering was needed: one asset already had renders under three of the setups, so the comparison figure is a lookup rather than a job.
- 2026-08-09 19:40 Three things the scripts say that the agreed definitions did not. The bright studio setup covers two different renders, one of which is the actual image fed to the model and one of which is only a reference panel, and they share no lighting or camera. The part-colour setup likewise covers two, and the one currently in use is not flat-shaded but ordinary studio shading over the model's own colours. And three of the published settings are not the scripts' own defaults, so a command that leans on defaults will quietly produce something else.
- 2026-08-09 19:55 Re-checked the live state on request. The page and its deep link both load, all twelve images on it load, and the link the console tree points at is exactly the address that was published, so the earlier missing-page report was from before it went up. The workspace landing page was rebuilt as well so its own copy of the navigation lists the new page rather than relying on the runtime refresh.
outcome: RENDERING.md at the repo root plus a workspace page, both covering all five setups with verified parameters and a side-by-side figure of one asset under three of them. The page passed the width checks at six widths including phone width, and every image on it loads.
