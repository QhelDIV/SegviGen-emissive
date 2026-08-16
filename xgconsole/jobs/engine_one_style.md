title: One page style: v3 standalone mode, console membership by registration
executor: xgpage-designer
track: tooling
status: done
started: 2026-08-13 16:06
updated: 2026-08-13 16:37
slurm: 
link: 
page: none (engine work; the proof is the test suite, the QA matrix, and existing pages rebuilding unchanged)
upstreams: pkg_bibcite
motivation: Owner-ratified redesign: v3 is the completed language and becomes the only style for new pages, working standalone with its full right rail, while the left page tree appears exactly when a project's manifest registers the page. v2 demotes to legacy beside v1, kept byte-stable for existing pages. This makes every page independent, makes console membership a one-line registry decision instead of a rebuild, and keeps one URL per page so annotations never split.
log:
- 2026-08-13 16:06 [master] Registered at dispatch. The design-system agent implements the standalone shell mode, the runtime tree mount, the legacy demotion, and the documentation collapse, with the width-matrix QA over both modes.
- 2026-08-13 16:36 [xgpage-designer] v3 is now the single current page style with a standalone mode: page(theme="v3") with no tree_html renders standalone (content + right rail, no left tree), and tree_src= lets xg3.js mount a left tree at runtime once the project's manifest registers the page. v1/v2 verified byte-stable by diff; found and fixed a real off-center grid bug via the width matrix before landing. README/HISTORY/ENGINE_NOTES updated. Pushed to xgpage main at 0347f3f.
- 2026-08-13 16:37 [master] The engine now has one current page style: v3 works standalone with its full right rail, and the left page tree mounts at runtime exactly when the project's manifest lists the page, so console membership is a one-line registry entry instead of a rebuild. Old pages are untouched, proven by rebuilding fixed v1 and v2 content through the old and new package and diffing to zero bytes. The whole test suite passes, the width matrix ran clean over both shell modes against a live server, and the QA pass itself caught and fixed a reserved-track off-center bug plus a missing width in the documented matrix. Docs collapsed to the single current style with the old languages as a legacy note.
outcome: The engine now has one current page style: v3 works standalone with its full right rail, and the left page tree mounts at runtime exactly when the project's manifest lists the page, so console membership is a one-line registry entry instead of a rebuild. Old pages are untouched, proven by rebuilding fixed v1 and v2 content through the old and new package and diffing to zero bytes. The whole test suite passes, the width matrix ran clean over both shell modes against a live server, and the QA pass itself caught and fixed a reserved-track off-center bug plus a missing width in the documented matrix. Docs collapsed to the single current style with the old languages as a legacy note.
