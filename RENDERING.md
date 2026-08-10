# Rendering setups

Five named lighting setups produce every figure in this project. Naming them
means a figure can be specified by one word instead of eight flags, and two
figures can be compared only when they name the same setup.

**The default rule.** Any figure that shows emission output, ground truth or
predicted, on a page or in the paper, is a **box render**. The other four
setups are used only for their own specific purposes: **key-lit emission
render** when the shell's geometry has to stay legible alongside the emission,
**input render** for the panel showing what the model was given, **segmentation
render** for part decomposition, **emission sweep** for the strength ladder.

Rendered examples of all five, side by side on one asset:
https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/workspace/rendering/

Everything below is read off the scripts as they stand at commit `cc218ed`.
Where a script's argparse default differs from the setup, that is called out,
because a command that leans on defaults will not reproduce the published
panels.

---

## 1. Box render

**Purpose.** Show what the asset's emission *does*. The asset sits in a closed
neutral box with no environment light and no lamps; the only photons in the
scene come off the asset itself, so the pool on the floor and the wash up the
walls are its own light. This is the default for every emission-result figure.

**Script.** `segvigen_emissive/render/render_emissive.py`, `--mode box`.

**Invocation** (the shipped settings, every flag written out):

```bash
PY=/project/3dlg-hcvc/omages/omages_internal/.venv/bin/python
export PYTHONPATH=/project/3dlg-hcvc/omages/xgutils/src
$PY render_emissive.py \
    --manifest $WORK/manifest.json --glb_dir $WORK/glb --out $WORK/out \
    --only <sid> --mode box \
    --res 768 --samples 1024 \
    --view_transform Filmic --exposure 1.5 \
    --bloom 1 --bloom_size 7 --bloom_threshold 1.0 --bloom_mix -0.45 \
    --emit_strength 4.0 --export_glb 0 --overwrite 1
```

Runs on a CPU node on Solar. `bpy` is a pip package in the shared venv, so no
Blender application is involved.

**Key parameters.**

| | value | where |
|---|---|---|
| lighting | none: every lamp deleted, world strength 0, world colour black | `render_emissive.py:673-705` (`emission_only_box`) |
| the claim is asserted | build fails if any lamp survives or the world emits | `render_emissive.py:758-767` (`assert_emission_is_only_light`) |
| box geometry | five quads (floor, ceiling, back, two sides), open toward the camera; half width `max(r*2.0, span_z*0.55)`, top `max(span_z*1.7, half*1.5)`, back `max(r*1.8, half*0.9)`, opening at `-half*2.6` | `render_emissive.py:706-741`, defaults at `:1224-1226` |
| box rotation | rotated to the camera azimuth so the open face squares up with the viewer | `render_emissive.py:730-736` |
| wall material | albedo 0.80 (blue channel 0.99x), roughness 0.9, metallic 0, specular IOR level 0.15 | `render_emissive.py:745-752`, default at `:1219` |
| view transform | Filmic, look None | `render_emissive.py:958-959`; the argparse default is **AgX** (`:1197`) |
| exposure | +1.5 stops | `render_emissive.py:960`; the argparse default is **0.0** (`:1196`) |
| bounces | 32 total, 16 diffuse | `render_emissive.py:1022-1023`, defaults at `:1227-1228` |
| samples | 1024, denoising on | `render_emissive.py:943-944` |
| bloom | Fog Glow, size 7, threshold 1.0, mix -0.45, quality high | `render_emissive.py:787-805`, defaults at `:1198-1202` |
| background | none: the box fills the frame, film not transparent | `render_emissive.py:1033-1034` passes `transparent=False` |
| resolution | 768 x 768 | `--res`, default at `:1187` |
| camera | three-quarter product view, azimuth 38, elevation 17, 52 mm lens, distance solved so the eight bounding-box corners just fit with margin 1.06 | `render_emissive.py:894-935`, defaults at `:1190-1193` |
| emission strength | 4.0, the same for every shape | `render_emissive.py:58`, `:1264` |

**Two things worth knowing.** The 32/16 bounce count is not a quality setting:
in a box lit only by the asset, the ambient fill *is* multi-bounce diffuse
light, and Cycles' default 4 diffuse bounces truncates it. And the emission
strength of 4.0 is a look choice, not a measurement; the bake stores emission
as uint8 and drops `KHR_materials_emissive_strength`, so nothing in the data
says how brightly a surface emits (`render_emissive.py:50-58`).

**Why Filmic and +1.5.** Measured, not chosen by taste. At the exposure the dim
shapes need, Standard blows out 12.4 percent of the vending-machine frame and
still gives the dimmest shape a lower midtone than Filmic. Full evidence, 26
exposure renders and 33 bloom grades with every number recomputed at build
time: https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/workspace/render_sweep/

**Verification.** Every box render writes a sidecar JSON next to it recording
`samples`, `max_bounces`, `diffuse_bounces`, `wall`, `view_transform`,
`exposure` and the camera (`render_emissive.py:1038-1045`). Compare sidecars
between two sets of panels rather than trusting that two commands matched.

---

## 2. Key-lit emission render

**Purpose.** Show the asset glowing while its shell stays readable. One dim key
light gives the silhouette and surface detail; the emission does the rest. Use
this when the reader has to see *where on the object* the emission sits, which
a box render can leave ambiguous on a dark asset. It cannot show what the light
does to a room, because a lamp is also in the scene.

**Script.** `segvigen_emissive/render/render_emissive.py`, `--mode method`
(the default mode).

**Invocation:**

```bash
$PY render_emissive.py \
    --manifest $WORK/manifest.json --glb_dir $WORK/glb --out $WORK/out \
    --only <sid> \
    --res 768 --samples 256 --samples_lit 96 \
    --key 8 --bg 0.012 --view_transform AgX --exposure 0.0 \
    --bloom 1 --bloom_size 9 --bloom_threshold 1.0 --bloom_mix -0.15 \
    --emit_strength 4.0 --export_glb 0 --overwrite 1
```

Three PNGs come out: `<sid>_lit.png` (the input render, section 3),
`<sid>_true.png` (the asset's own emission) and `<sid>_glow.png` (emission
rebuilt as mask x albedo, which is the panel this setup exists for).

**Key parameters.**

| | value | where |
|---|---|---|
| world | near-black background, colour 0.012, strength 1.0; the preset's linked environment texture is replaced outright | `render_emissive.py:635-651`; the argparse default for `--bg` is **0.004** (`:1194`) |
| key light | first lamp of the preset at energy 8, the rest at 1.6 (0.2x), soft size raised to at least 0.6 | `render_emissive.py:652-656`, default at `:1195` |
| floor | matte dark plate, base colour (0.10, 0.10, 0.108), roughness 0.5, metallic 0, dropped 4 mm to sit under the asset | `render_emissive.py:657-670`, `:1053-1054` |
| view transform | AgX, look None, exposure 0 | `render_emissive.py:958-960`, defaults at `:1196-1197` |
| samples | 256 | `--samples`, default at `:1188` |
| bloom | Fog Glow, size 9, threshold 1.0, mix -0.15 | `render_emissive.py:787-805` with the flags above |
| background | opaque dark room, film not transparent | `render_emissive.py:1116-1117` passes `transparent=False` |
| resolution | 768 x 768 | `--res` |
| camera | identical to the box render (same function, same defaults) | `render_emissive.py:894-935` |
| emission strength | 4.0 | `render_emissive.py:58` |

**Why key 8.** The Glare node's threshold is 1.0 in linear space. At key 8 no
non-emissive surface reaches it, so the bloom fires only where something
actually emits. At key 20 the bloom fired on 108,865 pixels of a shape emitting
nothing against 123,948 pixels of real ground truth, meaning the glow was
reporting the lamp rather than the object; at key 8 it is 0 against 50,458.
Recorded in the run script that made the change,
`/project/3dlg-hcvc/omages/yanxg_scratch/paper_v3/k8_all.sh:14-24`.

**Where it is used.** The overfit and fbv1 galleries, and the secondary band of
the strength-ladder page.

---

## 3. Input render

**Purpose.** Show the asset as it is, under bright neutral studio lighting on a
white background. Two jobs: the panel that tells the reader what the shape
looks like, and the image actually fed to the model as its condition. Those two
jobs are served by two different implementations, and they are not the same
render.

### 3a. The condition image (`render_from_transforms`)

This is the one that goes into the model. It is SegviGen's own renderer, called
by `inference_full.inference()` to build the DINOv3 conditioning image, so a
panel made this way is literally the model's input.

**Script.** `segvigen_emissive/code/SegviGen/data_toolkit/bpy_render.py`
(vendored upstream SegviGen, not in git). Driven for the 19-shape gallery by
`/project/3dlg-hcvc/omages/yanxg_scratch/fullseg19/code/render_cond.py`:

```bash
python render_cond.py --shapes shapes.json --transforms transforms_v0.json --out cond_img
```

| | value | where |
|---|---|---|
| lighting | three lamps: a point light of energy 1000 at (4, 1, 6), an area light of energy 10000 at (0, 0, 10) scaled 100x, an area light of energy 1000 at (0, 0, -10) | `bpy_render.py:80-101` (`init_lighting`) |
| background | transparent film, so it composites as white on a light page | `bpy_render.py:38` |
| scene normalization | the asset is scaled to unit longest bounding-box edge and centred on the origin | `bpy_render.py:193-209` |
| camera | one canonical camera from `transforms.json`, 32 mm square sensor, lens `16 / tan(camera_angle_x / 2)` with `camera_angle_x = 0.6981` rad (40 degrees), giving 43.95 mm | `bpy_render.py:66-78`, `:243-247` |
| samples | 128, denoising on | `bpy_render.py:41`, `:48` |
| bounces | 1 diffuse, 1 glossy, 3 transmission, 3 transparent | `bpy_render.py:44-47` |
| filter | BOX, width 1 | `bpy_render.py:42-43` |
| resolution | 512 x 512 | `render_cond.py:25` passes `resolution=512` |

Because the scene is normalized before the camera is placed, this render frames
every asset the same way. A stray unrelated mesh in a source GLB will inflate
the bounding box and shrink the asset to a speck, which is exactly what
happened once on the 19-shape gallery.

### 3b. The studio reference panel (`_lit.png`)

The verification panel that `render_emissive.py` writes alongside every key-lit
render. Same camera as the box and key-lit panels of the same shape, so the
three line up; different lighting and a different renderer from 3a.

| | value | where |
|---|---|---|
| lighting | the `xgutils` `preset_glb` studio scene, unchanged | `render_emissive.py:1078-1084` |
| view transform | Khronos PBR Neutral | `render_emissive.py:1084` |
| background | transparent film, floor set as a shadow catcher so the contact shadow survives on alpha | `render_emissive.py:1082-1083` |
| samples | 96 (`--samples_lit`) | default at `:1189` |
| resolution | 768 x 768 | `--res` |
| camera | identical to the box and key-lit renders | `render_emissive.py:894-935` |

**Which to use.** If the panel's claim is "this is what the model saw", use 3a.
If the panel's claim is "this is what the object looks like, from the same
viewpoint as the emission panels beside it", use 3b.

---

## 4. Segmentation render

**Purpose.** Show how a model decomposes a shape into parts, as per-part
colours on a white background. As with the input render there are two
implementations, and they differ in more than plumbing.

### 4a. The model's own part colours (the 19-shape gallery)

`slat_to_glb` bakes the model's part colouring straight into the exported mesh,
and the mesh is then rendered by the same `render_from_transforms` used for the
condition image. So the part colours are the model's, and the shading is
ordinary three-lamp studio shading, not flat.

**Script.** `/project/3dlg-hcvc/omages/yanxg_scratch/fullseg19/code/render_seg_views.py`

```bash
python render_seg_views.py --shapes shapes.json --glb_dir pred_glb \
    --transforms_a transforms_v0.json --transforms_b transforms_v1.json --out seg_img
```

Parameters are those of section 3a exactly, with two additions: two cameras
rather than one, `transforms_v0.json` (camera A, shared with the input render)
and `transforms_v1.json` (camera B, about 140 degrees around); and the input is
the predicted GLB rather than the source asset.

**Known caveat 1, orientation.** The predicted mesh's up-axis often differs from
the source asset's, so a segmentation panel frequently shows the object tumbled
relative to the input panel beside it, under the same nominal camera. Of six
shapes from the 19-shape gallery inspected by eye, one was aligned. This is an
open issue in the segmentation pipeline, not a property of the setup. Do not
build a figure whose argument depends on the two panels registering unless you
have checked that particular shape.

**Known caveat 2, the emissive asset family.** Run zero-shot on the emissive
assets this project actually works with, the pretrained model does not
reconstruct the object at all: a jack-o'-lantern comes back as a smooth ball
with the carved face gone, a headphone stand as a barrel and a plate, three
candles as a single disc, a vending machine as a bare box. All five assets
tested returned a plausible part count anyway, so a metric alone does not
surface this. Rendered evidence is on the workspace page, section 05. A
segmentation render of an emissive asset is therefore not currently usable as a
figure that argues anything about parts.

### 4b. Flat palette colours (the earlier mesh and voxel pages)

Used on `gt_vs_pred_canon10` and `fullseg_canon10_mesh`. Part labels are
recoloured through a fixed 20-entry high-contrast palette and painted as vertex
colours, which reads as flat per-part colour.

**Scripts.** `segvigen_emissive/render_seg_mesh.py` (mesh view) and
`segvigen_emissive/render_seg.py` (voxel view).

| | value | where |
|---|---|---|
| palette | 20 fixed RGB triples, tab20-like, indexed by `label % 20` | `render_seg_mesh.py:17-24`, `render_seg.py:24-31` |
| colour assignment | each mesh vertex takes the label of its nearest segmentation voxel, after bounding-box alignment onto the voxel grid | `render_seg_mesh.py:26-43` |
| lighting | the `xgutils` `preset_glb` studio scene via `bpyutil.render_mesh` | `render_seg_mesh.py:63-66` |
| camera | fixed at (0, -2.6, 1.4), up (0, 0, 1) | `render_seg_mesh.py:65-66` |
| background | transparent; alpha composited over 0.07 grey on save | `render_seg_mesh.py:45-52` |
| samples | 32 | `render_seg_mesh.py:64` |
| resolution | 460 x 460 (mesh), 440 x 440 (voxels) | `render_seg_mesh.py:54`, `render_seg.py:72` |

**Which to use.** 4a when the point is what the model itself produced, colours
included. 4b when parts have to be told apart at a glance and the model's own
colours are too close together.

---

## 5. Emission sweep

**Purpose.** Hold everything fixed and raise emission strength across a ladder
of values, in the same view. It shows how a result depends on a parameter that
is a look choice rather than a measurement, and it is the basis of the user
study design.

**Script.** `segvigen_emissive/render/strength_ladder.py`, which shells out to
`render_emissive.py` once per rung.

```bash
$PY strength_ladder.py \
    --manifest $WORK/manifest.json --glb_dir $WORK/glb \
    --out $WORK/ladder --only <sid> \
    --strengths 0,1,4,8,16
```

Add `--pred_masks DIR` for the prediction band; run it twice, once with and
once without, to get both bands.

**Key parameters.**

| | value | where |
|---|---|---|
| rungs | 0, 1, 4, 8, 16 by default; each rung is a real render, because strength changes light transport | `strength_ladder.py:44-45`, `:69-92` |
| what varies | `--emit_strength` only; every other flag is pinned and passed through identically | `strength_ladder.py:74-85` |
| base setup | the key-lit render's flags, pinned rather than defaulted: key 8, bg 0.012, AgX, exposure 0, bloom 9 / 1.0 / -0.15, samples 256, resolution 768 | `strength_ladder.py:52-62` |
| box variant | the published ladder page runs the same rungs through `--mode box`, via its own shell driver, so the primary band is a box render at each strength | the page's `img/<sid>_box_mask_s*.png` |
| per-rung check | a rung that produced no file fails the run, rather than trusting exit status | `strength_ladder.py:93-95` |
| output | one PNG per rung plus a labelled montage and a JSON index | `strength_ladder.py:97-116` |

**As a video.** The workspace page shows the sweep as a short looping clip
rather than only as stills, which makes the growth of the light pool on the
floor readable in a way a row of thumbnails is not. It is built by rendering
the upward half of the ramp only (33 strengths, eased with a smoothstep) and
ping-ponging those frames to 64 at 24 fps, so the loop is exactly symmetric and
the cluster cost is halved. Encoder:
`web/_preview/rendering/make_video.py`, which writes VP9 WebM, H.264 MP4 and a
poster still, and fails if the encoded frame count does not match the sequence.

**What to expect, so a correct result is not read as a bug.** The bloom grows
as strength rises. The Glare threshold is 1.0 in linear space, so raising
strength moves emitters further above it. Do not retune the bloom to keep the
ladder even; the growth is part of what the ladder shows
(`strength_ladder.py:8-12`).

**What the ladder cannot show.** Our model's masks are exactly binary. Across
99 mask files from three checkpoints, 34,603,008 pixels, the only values present
are 0 and 255, so a surface the model did not select stays exactly black at
every rung and amplification cannot make this formulation leak
(`strength_ladder.py:14-18`).

---

## Traps that apply to more than one setup

1. **Defaults have moved.** `render_emissive.py`'s defaults changed three times
   during development (bloom size 9 to 7, mix -0.15 to -0.45, key 20 to 8).
   Anything rendered without pinned flags will silently differ from anything
   rendered earlier. Write every flag out. Three of the shipped values above
   are *not* the current argparse defaults: `--view_transform Filmic` and
   `--exposure 1.5` for the box render, and `--bg 0.012` for the key-lit
   render.
2. **Use the supplied cameras.** `segvigen_emissive/render/cameras/<sid>.json`
   holds the per-shape camera, verified pixel-identical to the solved one. Pass
   `--camera_json`. A camera solved from a remeshed prediction frames that
   prediction to its own bounds, and the difference between two panels then is
   a viewpoint rather than a method.
3. **Blender slot order is not glTF material index.** Key predictions by
   Blender slot and ship the slot-ordered `materials` name list; the loader
   refuses a prediction it cannot verify.
4. **Compute nodes cannot see the working tree.** The repository is on
   `/local-scratch2`, a local disk. Stage the code to `/project` first with
   `segvigen_emissive/render/example/stage.sh`, and copy GLBs rather than
   symlinking them, because TexVerse is mounted on the workstation only.

Full trap list, with the symptom each one actually produces:
`segvigen_emissive/render/README.md`.
