title: Jobs board evolution (design and data model)
executor: jobs-redesign
status: ongoing
started: 2026-08-09 14:30
page: none (the board itself is the deliverable)
needs: evaluation: You are looking at it. Judge this board itself: are the pinned violet rows, the warm-to-gray freshness colors, and the click-open log timelines readable at a glance? Say good or what bothers you, in the CLI.
motivation: The owner needs one glance to know what is running, what finished, and where their attention is needed; the board is that instrument and has been evolving through owner feedback rounds.
log:
- 2026-08-09 15:16 The board became a sortable database like the pages list.
- 2026-08-09 16:15 Rebuilt as a console tab matching the pages tab, and the table layout was redesigned so nothing clips or misaligns at any width.
- 2026-08-09 17:20 Status became an append-only log per job: motivation under each title, latest update in the table, full history on click.
- 2026-08-09 17:40 Jobs and pages now point at each other: job titles link to their results pages, pages carry a chip back to their producing job.
- 2026-08-09 17:50 In progress: attention bands (jobs waiting on the owner pinned on top) and recency coloring so fresh updates stand out. Registered late by the master during a board reconciliation; this workstream predates its own entry because the board could not track itself while being built.
- 2026-08-09 17:55 The attention system is live and waiting on the owner: review-flagged jobs pin to the top in violet, fresh updates glow warm and cool to gray over two days. This entry is itself the first flagged job.
