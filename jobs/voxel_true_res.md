title: Voxel figures at true resolution
executor: overfit-test
track: research
status: done
started: 2026-08-11 16:11
updated: 2026-08-11 16:44
slurm: 
link: 
page: none (updates existing pages (paper_skeleton and a sweep of others), no new page of its own)
upstreams: mask_debug
motivation: The owner asked why earlier voxel figures do not show the full 512-cubed grid, since a subsampled render does not reflect the real output.
log:
- 2026-08-11 16:11 [overfit-test] Job started.
- 2026-08-11 16:20 [overfit-test] Traced the coarse-looking voxel cubes on the paper page to their source: they come from a display grid deliberately shrunk to 64 cells across for legibility in a small gallery tile, built from Dongchen's 256-resolution scan, which is itself already coarser than the checkpoint's real 512-resolution grid. Wrote a converter that reads the same 512-resolution files the checkpoint actually trains on for these twelve shapes, matched their occupied-cell counts against an earlier independent measurement on one of the shapes as a sanity check, and started the true-resolution render on the cluster with the exact same camera and style as the current figures, job 243323.
- 2026-08-11 16:44 [overfit-test] Traced the coarse cubes to a display grid that was deliberately shrunk for legibility, built from an older, coarser scan of the shapes. Re-rendered all eleven voxel figures on the paper page using the true, full-resolution grid the checkpoint actually trains on, same camera and style as before, and every figure now states its measured grid size and the number of cells that are actually filled. Published live.
outcome: Traced the coarse cubes to a display grid that was deliberately shrunk for legibility, built from an older, coarser scan of the shapes. Re-rendered all eleven voxel figures on the paper page using the true, full-resolution grid the checkpoint actually trains on, same camera and style as before, and every figure now states its measured grid size and the number of cells that are actually filled. Published live.
