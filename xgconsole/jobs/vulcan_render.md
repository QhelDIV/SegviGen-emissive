title: Can Vulcan render our panels faster
executor: vulcan-test
track: tooling
status: done
started: 2026-08-11 19:34
updated: 2026-08-11 20:00
slurm: 
link: 
page: none (a feasibility test; results go into the time-cost page and the board rather than a page of its own)
upstreams: mask_debug
motivation: Owner request: test the same box rendering on the Vulcan cluster, which may have a large CPU pool. If Vulcan renders as fast or faster than Solar's 64-core jobs, we gain a second queue for the render fan-out and cut wall-clock further.
log:
- 2026-08-11 19:34 [master] Registered at dispatch. A dedicated runner will check Vulcan's partitions, storage reachability, and Blender availability, then time the same box render against Solar's numbers.
- 2026-08-11 19:34 [owner] [owner] create an agent to test doing the same kind of rendering on vulcan, maybe vulcan has a huge amount of cpus that can be used for fast rendering?
- 2026-08-11 19:35 [vulcan-test] Starting Vulcan reachability and rendering capability test
- 2026-08-11 19:54 [owner] [owner] I see the job for vulcan test is pending. I think there should be CPU partition? I have the assumption that the cpu partition is usually less crowded, maybe you should check the alliancecan's doc on this?
- 2026-08-11 19:57 [vulcan-test] Vulcan test render succeeded: warhammer GT box render matches Solar output, 1.66x faster per core (115s vs 191s at 64 cores each)
- 2026-08-11 19:58 [master] Vulcan runs our box renders 1.66 times faster per core than Solar (115 versus 191 seconds for the same warhammer panel at 64 cores) and the output is pixel-equivalent, verified by eye and by matching lit-fraction statistics. Answer to the partition question: dedicated CPU partitions exist but hold only two busy nodes each; the practical route is CPU-only allocations on the large GPU partitions, which had dozens of idle 64-core nodes. Costs: no shared filesystem so inputs stage in (about 9 MB per shape, seconds), a one-time Python environment built from Alliance's wheel mirror, and one small import patch. The staged environment and job template sit on Vulcan scratch, ready for a real fan-out.
- 2026-08-11 20:00 [vulcan-test] Checked Vulcan's CPU partition: it is only 2 shared physical nodes (128 cores total), already loaded by other jobs, too small to fit our 64-core render right now. Vulcan auto-assigns the partition from the resource request and overrides any manual partition choice; the render's earlier placement on the GPU-node pool with GPUs turned off was already the system's best choice, confirmed by a probe job.
outcome: Vulcan runs our box renders 1.66 times faster per core than Solar (115 versus 191 seconds for the same warhammer panel at 64 cores) and the output is pixel-equivalent, verified by eye and by matching lit-fraction statistics. Answer to the partition question: dedicated CPU partitions exist but hold only two busy nodes each; the practical route is CPU-only allocations on the large GPU partitions, which had dozens of idle 64-core nodes. Costs: no shared filesystem so inputs stage in (about 9 MB per shape, seconds), a one-time Python environment built from Alliance's wheel mirror, and one small import patch. The staged environment and job template sit on Vulcan scratch, ready for a real fan-out.
