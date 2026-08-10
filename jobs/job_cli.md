title: xgjobs CLI (the board's command vocabulary)
executor: jobs-redesign
status: ongoing
started: 2026-08-09 18:05
updated: 2026-08-09 18:21
slurm: 
link: 
page: none (tooling; the board itself shows the result)
motivation: Agents keep getting the mechanical parts of board-keeping wrong (timestamps, registration, format, rebuilding). A single command tool owns all of that, so agents only state intent and the standards are enforced at write time.
log:
- 2026-08-09 18:05 Job started. Building the tools/job command: start, log, done, flag; automatic timestamps, locking, format checks, and board rebuild on every call.
- 2026-08-09 18:11 Master verification passed: help output, lint rejection without mutation, and the live rebuild path all checked directly.
- 2026-08-09 18:21 [master] Same refinement round on the tool side: every verb that appends a log line stamps the author automatically, and the flag verb requires a stated review ask.
outcome: 
