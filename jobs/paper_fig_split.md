title: Separate figure tex and images in the paper source
executor: latex-split
track: paper
status: ongoing
started: 2026-08-12 17:25
updated: 2026-08-12 17:34
slurm: 
link: 
page: none (paper-source restructure; the proof is an identical compile)
motivation: Owner request: the paper's figure environments move to their own files under figs/, images move to imgs/, and the body includes each figure by input. This makes figure swaps single-file diffs, keeps sections readable, and separates binary churn from text history.
log:
- 2026-08-12 17:25 [master] Registered at dispatch. A LaTeX-focused agent extracts every figure environment into its own file, relocates the images, and verifies the compiled paper is unchanged before anything is committed.
- 2026-08-12 17:26 [latex-split] Baseline compile of the untouched paper tree is clean: 10 pages, no overfull or underfull boxes, no undefined references. Starting the figs and imgs split.
- 2026-08-12 17:27 [latex-split] Split done and committed as 699ebfd. Seven figures now live in figs/, all images moved to imgs/ with git mv, and the restructured tree compiles to a PDF that is pixel-identical to the baseline on all ten pages. Push to Overleaf is pending the owner. The qual_rendering image is now imgs/qual_rendering.pdf.
- 2026-08-12 17:34 [master] Pushed to Overleaf after resolving a crossing edit: Dongchen had changed the teaser caption on Overleaf while the split was in flight, so the caption edit was carried into the new teaser figure file and the rebased split compiles to the same ten pages with no errors. The paper source now has figure code under figs and images under imgs.
outcome: 
