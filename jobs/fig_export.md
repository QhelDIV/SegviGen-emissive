title: Figure export tool: rasterize page figures to real PNGs
executor: fig-ref
track: tooling
status: done
started: 2026-08-10 18:12
updated: 2026-08-10 18:29
slurm: 
link: 
page: none (judged on the rendering page's thumbnail strip)
needs: evaluation: Open the Rendering setups page and look at the FIGURES strip in the right rail: the first tile should be the whole five-panel teaser as one small image, and every other tile its full figure. Is this what you wanted?
motivation: Multi-panel figures exist only as page layout, so side thumbnails show just the first panel and export figures need hand compositing; rasterizing each figure as rendered gives faithful thumbnails and export images from one mechanism.
log:
- 2026-08-10 18:12 [fig-ref] Job started.
- 2026-08-10 18:28 [fig-ref] Built export_figs.js in the xgpage package (alongside the other Playwright QA scripts): it opens a built page headless, forces the light theme, screenshots every numbered figure at double resolution, and saves each one as its own PNG next to the page's other images. It also marks each figure element so the thumbnail strip knows a real image exists for it and should use that instead of guessing.
- 2026-08-10 18:28 [fig-ref] Wired the thumbnail strip to prefer these exported images when present, falling back to its old behavior (the figure's first picture) on any page that has not run the export step, so nothing breaks for pages that skip it.
- 2026-08-10 18:28 [fig-ref] Ran it on the rendering page and found the video panel in the five-setup figure came out solid black instead of showing its preview picture, because the browser had already started decoding the video by the time the screenshot was taken and its very first frame is a genuinely dark scene. Fixed by swapping every video for its own preview picture before taking any screenshot, so the composite always shows the picture the page author chose.
- 2026-08-10 18:28 [fig-ref] Opened the rendering page and confirmed the owner's test by eye: the side thumbnail for the opening figure now shows the whole five-panel comparison shrunk down, not just the first panel, and every other figure's thumbnail is correct too, including the video one after the frame fix. Checked at two screen widths and in both light and dark reading modes, the panel of thumbnails always shows every figure with no scrolling needed inside it, and clicking a thumbnail still jumps to and briefly highlights its figure. Also confirmed a page that has not run the export step still shows correct, if simpler, thumbnails, so nothing breaks for pages that skip this step. Published the updated rendering page live.
- 2026-08-10 18:29 [master] For your review: Open the Rendering setups page and look at the FIGURES strip in the right rail: the first tile should be the whole five-panel teaser as one small image, and every other tile its full figure. Is this what you wanted?
outcome: Opened the rendering page and confirmed the owner's test by eye: the side thumbnail for the opening figure now shows the whole five-panel comparison shrunk down, not just the first panel, and every other figure's thumbnail is correct too, including the video one after the frame fix. Checked at two screen widths and in both light and dark reading modes, the panel of thumbnails always shows every figure with no scrolling needed inside it, and clicking a thumbnail still jumps to and briefly highlights its figure. Also confirmed a page that has not run the export step still shows correct, if simpler, thumbnails, so nothing breaks for pages that skip this step. Published the updated rendering page live.
