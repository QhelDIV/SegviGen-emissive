title: Page-relationship graph console view
executor: xgpage-designer
status: done
started: 2026-08-09 16:30
updated: 2026-08-09 17:55
slurm:
link: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/graph.html
motivation: The owner asked for an Obsidian-style graph of the project's published pages, so the owner and future agents can see how pages relate and how the project developed, as a permanent tab on the console rather than a one-off page.
log:
- 2026-08-09 16:45 found that the page-relationship graph needed pages the console's own page inventory could not see: workspace/rendering, workspace/render_sweep, workspace/diagnostics, and workspace/paper_skeleton live one directory level under workspace/, and the existing scan only ever looked at the top level, so those four pages were invisible everywhere, not just in the graph. Added a "workspace" tier to the page scan so they show up in the Pages tab too, not only the graph.
- 2026-08-09 17:00 the node-and-edge builder is working: it reads each published page's real HTML and picks out only the links inside the article body, ignoring the page tree, the outline, and the theme toggle. Verified against a real page that has the same link once in the navigation and once in a citation, and only the citation counted.
- 2026-08-09 17:20 first working render of the graph page: pan, zoom, drag, hover-highlight, search, and a legend are all working, but the labels were unreadable because the whole graph had to shrink very small to fit a few widely scattered pages on screen.
- 2026-08-09 17:35 fixed the unreadable labels (node markers now stay a constant readable size no matter how far the view is zoomed out) and fixed a real bug where the initial view was cropping out most of the graph instead of framing it, found by checking every node's on-screen position rather than trusting a screenshot alone.
- 2026-08-09 17:50 full QA pass done: rebuilding the graph twice with no changes reproduces the exact same layout, adding a new linked page leaves every existing page's position untouched, all the interactions were driven and checked by script, and the page loads with zero requests to any external site. Verdict: done, live on the console as the new Graph tab.
outcome: The Graph tab is live on the console (41 pages, 24 real content links, 23 pages not yet cross-linked to anything). Positions are remembered between rebuilds so the owner's mental map of the graph stays stable; only genuinely new or newly-linked pages move.
