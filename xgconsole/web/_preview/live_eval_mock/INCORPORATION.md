Incorporation spec: live_eval redesign -> the real build.py
=============================================================

For liveeval-builder, when idle. This mockup (web/_preview/live_eval_mock/)
is review material only; it does not touch web/_preview/live_eval/ or its
build.py. Everything below is what the real build.py needs to emit to carry
this design, mechanically, on every rebuild -- no hand-tuned per-content
tweaks, same discipline the current build.py already follows.

What changed, in one line each
-------------------------------
1. The hero statband + curve are promoted to the top of the page, minimal
   captions, no prose paragraphs above them.
2. The single-checkpoint wall (one `grid_figure()` per checkpoint, stacked
   down the page) becomes ONE fixed grid plus a checkpoint slider that swaps
   its images.
3. Method, per-checkpoint numbers, caveats, and provenance move into one
   collapsed zone at the bottom (`lp.expandable()`), still complete.

1. HTML structure the slider needs
-----------------------------------
Reuse `xgpage.grid_figure`'s markup exactly (same CSS classes: `grid-figure`,
`gf-cell`, `gf-rowhead`, `gf-colhead`, `gf-imgcell`, `gf-cap`,
`grid-figure-scroll`) so it inherits the engine's existing styling with zero
new CSS beyond the control bar. `grid_figure()` itself doesn't take
per-cell `data-*` attributes, so the wall has to be hand-assembled (see
`build_wall_and_index()` in `build_mock.py` -- copy that function, it is
already in the shape the real build needs) rather than called through
`lp.grid_figure()` directly. Two kinds of cell:

- Static cells (shape geometry, ground truth): plain `<img>`, never touched
  by JS. Same images at every checkpoint.
- Dynamic cells (the 5 draw columns): each `gf-imgcell` carries
  `data-sw-cell data-sw-sid="<sid>" data-sw-draw="<k>"`; its `<img src=...>`
  is baked at build time to the LATEST checkpoint's image, and its
  `.gf-cap` (also tagged `data-sw-cap`) is baked to that checkpoint's
  per-draw IoU. This is the no-JS fallback: a reader with JavaScript off
  sees a complete, correct wall for the newest checkpoint, identical to
  what the current live page's "newest epoch" wall shows today.
- Each row header (`gf-rowhead`) carries `data-sw-rowlabel="<sid>"` so its
  IoU number can update too -- it is NOT frozen to the checkpoint it was
  baked at; leaving it stale while draws update was rejected during design
  (a stale header number reads as evidence, and D-law reasoning must be
  stitched to real numbers, not leftover ones).
- Row order is fixed once, by IoU at the newest checkpoint, and never
  re-sorts on scrub -- reordering while scrubbing would defeat the point of
  watching one shape move.

The control bar (`prev` / range input / `next` / label span) sits above the
grid inside the same `data-sw-root data-src="ckpts.json"` wrapper; see
`build_wall_and_index()`'s `controls` string for the exact markup, including
the `<noscript>` fallback message and the epoch tick row.

2. The JSON index contract (`ckpts.json`)
-------------------------------------------
One file, written next to `index.html`, read by `slider_view.js` via
`fetch()`. Schema (see `build_mock.py`'s `build_wall_and_index()` for the
exact builder):

```json
{
  "run": "segvigen_256_bw_6gpu",
  "draws": 5,
  "img_dir_tmpl": "img/step%07d",
  "default_idx": 4,
  "shapes": [
    {"sid": "...", "geom": "img/<sid>_geom.png", "gt": "img/<sid>_gt.png",
     "gt_frac": 0.363}
  ],
  "checkpoints": [
    {"step": 2000, "epoch": "2.6", "iou": 0.0111, "n_scored": 10,
     "per_shape": {"<sid>": {"iou_mean": 0.0497,
                              "iou_per_draw": [0.078, 0.094, 0.076, 0.0, 0.001]}}}
  ]
}
```

- `shapes` order == the grid's row order (fixed, see above).
- `checkpoints` order == slider order (ascending step); `default_idx` is
  always the LAST index (the newest evaluated checkpoint) -- this is what
  makes "refresh always lands on latest" true without the page persisting
  any state.
- Draw image paths are NOT enumerated in the JSON; the JS derives them as
  `{img_dir_tmpl % step}/{sid}_d{k}.png`. A missing file falls back to the
  existing `.gf-placeholder` "no panel" tile via the `<img>`'s `onerror`
  handler (see `slider_view.js`'s `render()`), the same convention
  `wall_figure()`'s `{"placeholder": "no panel"}` already uses for a shape
  with no panel on disk. This means a partially-evaluated or in-flight
  checkpoint degrades cell by cell instead of failing the whole build or
  showing a broken-image icon.

3. Where the JS lives
-----------------------
`slider_view.js` + `slider_view.css`, page-local files loaded via
`page(extra_head=... '<link rel="stylesheet" href="slider_view.css">',
extra_body_end='<script src="slider_view.js"></script>')` -- the same
pattern `train_rungraph/build.py` already uses for `rungraph_view.js` /
`rungraph_view.css`. Copy both files verbatim from this mock directory; they
are self-contained (no dependency on this project's other JS) and read only
`ckpts.json` plus the DOM markup described above. No engine change needed:
this is a page-local component, not a new `xgpage.core` function, the same
way `compare_slider()` is reserved and this is deliberately NOT that
component (a before/after divider is the wrong shape for N checkpoints).

4. Data-retention decision the real build.py must make (flagging, not
   deciding -- outside design-agent scope)
-----------------------------------------------------------------------
The current build.py's `N_FULL_WALLS = 4` prunes thumbnail directories for
checkpoints older than the last 4 (`prune_images()`), because the old design
only ever showed one wall at a time. The slider's whole value is scrubbing
across MORE history than that. The evaluation store itself
(`outputs/live_eval/img/`) already keeps every checkpoint's images
indefinitely (nothing there gets pruned), so raising `N_FULL_WALLS` (or
changing its meaning to "every checkpoint the page's own `img/` mirrors")
is a page-directory disk-budget call, not a design call -- flag it to the
owner/liveeval-builder rather than silently widening retention. This
mockup's own `build_mock.py` keeps all 5 checkpoints currently on disk
precisely to make that budget question visible.

5. QA to carry forward
------------------------
`qa_slider_journeys.js` in this directory is the simulated-user journey
suite (6 checks: default-latest + no-JS-equivalent state, prev button,
keyboard arrows without focus, dragging to an extreme + boundary button
disabling, missing-image placeholder fallback, refresh-resets-to-latest).
Copy it alongside the real page and update its `DEFAULT_URL`. Run it after
any edit to `slider_view.js`, same discipline as
`qa_rungraph_journeys.js` for `rungraph_view.js`. On this workstation (no
nvm, no Node 20+ on PATH as of 2026-08-26) it was run with:

```
NODE_PATH=/localhome/xya120/.npm/_npx/9833c18b2d85bc59/node_modules \
/localhome/xya120/.vscode-server/cli/servers/Stable-08d4889f9ec4a1685d257b9b95de036c8e1ce1e5/server/node \
  qa_slider_journeys.js <url>
```

The layout-invariant check (`qa_widths.js`) passed clean at the full
8-width matrix (2560/1920/1600/1440/1275/1024/768/390), both themes spot
checked at 1600 and 390 -- run it again on the real page once wired in,
since new markup (the control bar) is new surface for D11/D12 symmetry
checks even though it reused existing engine components everywhere else.

A note on the data this mockup was built from
------------------------------------------------
While building this, `outputs/live_eval/` was caught mid-migration
(`records_legacy_quick/`, `bridged/`, `shards/` all appeared within
minutes, and `records/` transiently held stub/failed records while
checkpoints were being re-sharded and re-evaluated). That is liveeval-
builder's territory, not a bug this page's design should chase, so this
mockup reads a frozen one-time snapshot taken from `records_legacy_quick/`
into `_snapshot/` (excluded from publish) rather than the live store. The
real build.py should keep reading the live store as it does today; this
note exists only so nobody mistakes 5 checkpoints as some new plateau in
the run.
