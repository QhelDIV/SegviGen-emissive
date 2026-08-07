# Emissive render code

Renders TexVerse GLBs with their emission replaced by a predicted or
ground-truth signal, in one fixed treatment, so panels from different methods
are comparable side by side.

**STATUS: this is the working code, copied from scratch so it is versioned and
readable. It is NOT yet packaged for outside use.** Proper packaging is in
progress: a runnable worked example, a mode that renders predicted emission RGB
directly, and an emission-strength sweep. Until then, read the traps below
before running anything. Every one of them produces silently wrong output
rather than an error.

Environment: the `trellis2` conda env at
`/3dlg-jupiter-project/lightgen/miniforge3`. `bpy` is a pip package, no Blender
app needed. Rendering goes to solar, never the workstation.

---

## TRAPS

Each is described by the symptom you would actually see.

### 1. Exit status means nothing

`render_emissive.py` ends in `os._exit(0)`. A job that rendered zero panels,
or hit an assertion on the first shape, still exits 0. Any assertion inside the
script is advisory.

**Count output files and check their sizes.** Do not trust job state.

### 2. There are two treatments in this one script, and the defaults have moved

| | key-lit path | box path |
|---|---|---|
| view transform | AgX | Filmic |
| exposure | 0 | +1.5 |
| key light | **8** | none, emission is the only light |
| bounces | Cycles default (12 / 4) | 32 total / 16 diffuse |
| wall albedo | n/a | 0.80 |
| bloom size / threshold / mix | 9 / 1.0 / -0.15 | 7 / 1.0 / -0.45 |

The script defaults changed twice during development (bloom size 9 to 7, mix
-0.15 to -0.45; key 20 to 8). **Anything rendered without pinned flags will
silently differ from anything rendered earlier.** Pass every flag explicitly
and record them; the `treatment` block in each render's JSON sidecar exists for
this.

### 3. `--bloom 0` was not read on the key-lit path

Fixed, but if you are working from an older copy: the bloom-off arm renders
with bloom ON and returns bit-identical output, which reads as "bloom has no
effect" rather than as a bug.

### 4. The preset camera crops tall objects

`xgutils`'s `preset_glb.blend` camera carries a TRACK_TO constraint evaluated
after `matrix_world`, so `set_camera_orientation`'s target is ignored and the
camera keeps looking at the origin. Anything taller than it is wide gets its
top cut off. Symptom: a street lamp rendered with its base mid-frame and the
lantern past the top edge.

Cleared in this copy. A stock `xgutils` checkout still has it.

### 5. Blender slot order is not glTF material index

The headphone stand's Blender slot 0 is glTF material 10. A range check
(`index < material_count`) passes and applies your data to the wrong material.
Symptom: panels that look like model errors.

Key everything by **Blender slot**, and pass a slot-ordered material-name list
so the loader can verify rather than assume. The loader refuses predictions it
cannot verify.

### 6. Use the supplied cameras, do not solve your own

`cameras/<sid>.json` holds the per-shape camera, verified pixel-identical to
the solved camera. If you solve your own, your panels will not align with
existing ones even under an identical treatment, and the difference is a
different viewpoint rather than a different method.

---

## Modes

```
--mode method        emission = albedo * binary mask   (OUR method, see below)
--mode random        random mask at the shape's own GT density
--mode allemissive   every surface texel emissive
--mode box           emission-only, object inside a Cornell box
```

### `--mode method` is not a general renderer

It ends in `out[..., :3] = alb * mask[..., None]`: the asset's own albedo
multiplied by a binary mask. **That multiply is our method's central
assumption**, not a neutral rendering step.

If your method predicts emission RGB directly, do NOT push it through this
path: it would apply our method to your prediction and render something
neither of us predicted. A separate mode for predicted emission RGB is being
added.

---

## Emission strength

`EMIT_STRENGTH = 4.0`, currently a module constant applied uniformly to every
shape. It is a look choice, not anything recovered from the assets: our bake
stores emission as uint8 and drops the `KHR_materials_emissive_strength`
extension (measured: 3 of 60 sampled source GLBs carry it).

Being exposed as a flag, with a sweep mode.

**If your predicted values already encode radiance, multiplying by 4 replaces
your brightness with ours.** Until this is settled, panels from different
methods are comparable in WHERE emission is, not in HOW BRIGHT.

---

## Why the treatment is what it is

Measured, not chosen by taste. Full evidence:
https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/workspace/render_sweep/

- **Key light 8.** The Glare threshold is 1.0 in linear space. At key 8 no
  non-emissive surface reaches it, so bloom fires only where something actually
  emits. At key 20 the bloom fired on 108,865 pixels of a shape emitting
  nothing, against 123,948 on real ground truth: the glow was reporting the
  lamp, not the object. At key 8 it is 0 against 50,458.
  **Raising emission strength moves emitters further above that threshold and
  will grow the bloom.** Expect that when sweeping strength.
- **Bloom size 7, mix -0.45.** Size is radius, mix is intensity.
  **Threshold is the wrong lever**: it keys off absolute brightness and cannot
  serve a range of shapes. At threshold 2.5 the dimmest shape loses its bloom
  entirely (0.0%) while the brightest keeps a sixth of frame (17.0%).
- **Filmic over Standard** on the box path. At the exposure dim shapes need,
  Standard blows out 12.4% of the vending machine frame and destroys the
  artwork the object is emitting, while giving the dimmest shape a *lower*
  midtone than Filmic.
- **32 / 16 bounces** on the box path against Cycles' 12 / 4. In a box lit only
  by the object, the ambient fill IS multi-bounce diffuse light; truncating at
  4 discards it. Worth +39% image mean on the dimmest shape.
