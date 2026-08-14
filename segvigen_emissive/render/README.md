# Emissive render code

Renders TexVerse GLBs with their emission replaced by a predicted or
ground-truth signal, in one fixed treatment, so panels from different methods
are comparable side by side.

Environment: the shared venv at
`/project/3dlg-hcvc/omages/omages_internal/.venv`. `bpy` is a pip package, so
no Blender application is needed. `xgutils` comes from
`PYTHONPATH=/project/3dlg-hcvc/omages/xgutils/src`. Rendering goes to solar,
never the workstation.

---

## Getting the code onto the cluster

**The repository working tree is on `/local-scratch2`, a local disk on the
workstation. Compute nodes cannot see it.** A job pointed at the repo path fails
with `can't open file '/local-scratch2/.../render_emissive.py'`. Stage it first:

```bash
bash example/stage.sh /project/3dlg-hcvc/omages/$USER/render
```

Everything else in the pipeline already reads from `/project`, so the code is
the only piece that has to move. Re-run `stage.sh` after pulling changes.

---

## Worked example

Renders one shape end to end and leaves three images you can look at. Copy and
paste it; nothing below needs editing except `WORK`.

**Step 1, on the workstation** (TexVerse is mounted here and nowhere else):

```bash
WORK=/project/3dlg-hcvc/omages/$USER/render_example
CODE=/project/3dlg-hcvc/omages/$USER/render

bash example/prepare.sh $WORK
bash example/stage.sh $CODE
```

`prepare.sh` copies one GLB out of TexVerse into `$WORK/glb/` and writes a
one-line manifest. It copies rather than links on purpose: see trap 7.

**Step 2, submit to solar:**

```bash
cat > $WORK/run.sh <<EOF
#!/bin/bash
#SBATCH --job-name=render_example
#SBATCH --account=3dlg-hcvc-lab
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --exclude=cs-venus-05,cs-venus-09,cs-venus-19
#SBATCH --gres=gpu:0 --cpus-per-task=16 --mem=48G --time=00:30:00
#SBATCH --output=$WORK/render_%j.out
export PYTHONPATH=/project/3dlg-hcvc/omages/xgutils/src
/project/3dlg-hcvc/omages/omages_internal/.venv/bin/python \\
    $CODE/render_emissive.py \\
    --manifest $WORK/manifest.json --glb_dir $WORK/glb --out $WORK/out \\
    --only 48af42db48c44cd9bfab32bbb057a39c \\
    --res 768 --samples 256 --samples_lit 96 \\
    --key 8 --bg 0.012 --view_transform AgX --exposure 0.0 \\
    --bloom 1 --bloom_size 9 --bloom_threshold 1.0 --bloom_mix -0.15 \\
    --emit_strength 4.0 --export_glb 0 --overwrite 1
EOF

sbatch $WORK/run.sh
```

**Step 3, check the output.** Three PNGs and a sidecar appear in `$WORK/out`:

```
48af42db..._lit.png     the object under a neutral studio light, no emission
48af42db..._true.png    dark room, the asset's own emission
48af42db..._glow.png    dark room, emission = albedo x the asset's mask
48af42db..._stats.json  per-material coverage, and the treatment it was rendered with
```

The shape is a jack-o'-lantern. In `_glow.png` the cut eyes and mouth glow and
the rest of the gourd is dark. If the whole pumpkin glows, or the top of the
object is cut off, read the traps.

Every flag above is written out rather than defaulted. That is the habit to
copy, for the reason in trap 2.

---

## Presets

Two treatments exist and they are not interchangeable. Cite one by name so
figures are comparable by reference rather than by transcribing eight flags.

| | `key-lit` | `box` |
|---|---|---|
| what it shows | the object, glowing | what the object's light does to a room |
| view transform | AgX | Filmic |
| exposure | 0 | +1.5 |
| key light | 8 | none, emission is the only light |
| world background | 0.012 | 0 |
| bounces | Cycles default (12 / 4) | 32 total / 16 diffuse |
| wall albedo | n/a | 0.80 |
| bloom size / threshold / mix | 9 / 1.0 / -0.15 | 7 / 1.0 / -0.45 |
| samples | 256 | 1024 |

`key-lit` is what the comparison figures use. Reproduce it with the flags in
the worked example above. For `box`, add `--mode box`, `--view_transform
Filmic`, `--exposure 1.5`, `--bloom_size 7`, `--bloom_mix -0.45`,
`--samples 1024`.

Each render writes its own settings into the `treatment` block of its
`_stats.json`. Compare those blocks between two sets of panels rather than
trusting that two commands matched.

---

## Modes

```
--mode method        emission = albedo x binary mask   (see below)
--mode random        random blocky mask at the shape's own ground-truth density
--mode allemissive   every surface texel emissive
--mode box           emission only, object inside a Cornell box
```

### If your method predicts a MASK: `--pred_masks`

```
--pred_masks DIR     replaces the asset's own mask with yours
```

`--mode method` ends in `out[..., :3] = alb * mask[..., None]`: the asset's own
albedo multiplied by a binary mask. **That multiply is our method's central
assumption**, not a neutral rendering step.

Two behaviours are deliberate. Every material is considered, not only the ones
the asset emits from, so a prediction can light a surface that is dark in the
ground truth. And a material the prediction does not select is switched off even
where the asset does emit, so a miss reads as a miss rather than as the asset
showing through.

### If your method predicts emission COLOUR: `--pred_emission`

```
--pred_emission DIR          your RGB values, written in unchanged
--emission_linear 0|1        0 (default) treats 8-bit PNGs as sRGB
--emission_strength_ours 0|1 0 (default) leaves your radiance alone
```

Use this for TEXGen and for the TRELLIS.2 emission variants. **Do not push RGB
predictions through `--pred_masks`**: it would apply our albedo multiply to your
output and render something neither method produced.

Two choices worth knowing, because both change the picture:

- **Colour space.** An 8-bit PNG of emission is almost always written through an
  sRGB transfer curve, so the default converts to linear on load. If your files
  are already linear, pass `--emission_linear 1`. Getting this wrong shifts the
  midtones by roughly a factor of two and looks like a brightness disagreement
  between methods rather than like a bug.
- **Strength.** This path defaults to strength 1.0, not our 4.0, because a
  predicted colour may already encode radiance and scaling it is a second
  opinion about brightness. Pass `--emission_strength_ours 1` to match our
  panels' look, knowing that is what you are doing.

### File convention, shared by both prediction paths

```
DIR/<sid>__mat<SLOT>__emis.png    one file per material slot
DIR/<sid>__stats.json             {"materials": [...], "uniform": {...}}
```

`materials` is one name per slot **in the asset's own material order**, and it
is required. The loader checks it against the loaded object and refuses a
prediction it cannot verify. `uniform` covers slots whose primitives carry no
UVs and therefore cannot hold a texture: a scalar for `--pred_masks`, an RGB
triple for `--pred_emission`. A slot with neither is a slot you predicted
nothing on. See trap 5 for why the name list exists.

---

## Emission strength

```
--emit_strength FLOAT     default 4.0
```

**4.0 is a look choice, not a measurement.** The bake stores emission as uint8
and drops the `KHR_materials_emissive_strength` extension, which 3 of 60 sampled
source GLBs carry, so nothing in the data says how brightly a surface emits.
Every panel published so far uses 4.0, and it stays the default so pinned
commands reproduce unchanged.

Panels from different methods are comparable in WHERE emission is. They are
comparable in HOW BRIGHT only if both sides used the same strength, which for
`--pred_emission` means deciding whether your values already encode radiance.

### Strength ladder

Renders one shape across a ladder and writes a contact sheet:

```bash
/project/3dlg-hcvc/omages/omages_internal/.venv/bin/python \
    $CODE/strength_ladder.py \
    --manifest $WORK/manifest.json --glb_dir $WORK/glb \
    --out $WORK/ladder --only 48af42db48c44cd9bfab32bbb057a39c \
    --strengths 0,1,4,8,16
```

Add `--pred_masks DIR` for the prediction band; run it twice, once with and
once without, to get both bands.

Each rung is a real render, because strength changes light transport. The script
checks that every rung produced a file rather than trusting exit status.

**The bloom will grow as strength rises, and that is correct.** The Glare
threshold is 1.0 in linear space, so raising strength moves emitters further
above it. Do not retune the bloom to keep the ladder looking even; the growth is
part of what the ladder shows.

**What a ladder over our own masks cannot show:** our model's masks are exactly
binary. Measured across 99 mask files from three checkpoints, 34,603,008 pixels,
the only values present are 0 and 255. A surface the model did not select stays
exactly black at every rung, so amplification cannot make this formulation leak.
That is a property of predicting a mask rather than a continuous texture, and it
is the opposite of the failure a continuous emission generator shows.

---

## TRAPS

Each is described by the symptom you would actually see. Every one produces
silently wrong output or a confusing error rather than a clear failure.

### 1. Exit status used to mean nothing

`render_emissive.py` ends in `os._exit(0)`, added to dodge a bpy crash during
interpreter teardown. It used to swallow real failures with it: a shape that
raised printed `FAIL` and the job still exited 0.

Fixed. A run with any failure now prints `FAILED_SHAPES <n>: <sids>` and exits
1. **But your sbatch wrapper can still throw that away**: piping the command
into `tail` or `grep`, or running an `echo` after it, makes `$?` the last
command's. Use `set -o pipefail` and let the python command be last.

Count output files anyway. That is the check that cannot be fooled.

### 2. There are two treatments in this one script, and the defaults have moved

See Presets above. The script's defaults changed three times during development
(bloom size 9 to 7, mix -0.15 to -0.45, key 20 to 8). **Anything rendered
without pinned flags will silently differ from anything rendered earlier.**
Pass every flag explicitly. The `treatment` block in each sidecar exists so two
sets of panels can be compared afterwards rather than assumed equal.

### 3. `--bloom 0` was not read on the key-lit path

Fixed. If you are working from an older copy, the bloom-off arm renders with
bloom ON and returns bit-identical output, which reads as "bloom has no effect"
rather than as a bug. That is how it was found.

Turning bloom off has to clear the Glare node, not merely skip adding it: the
compositor persists across renders within one scene.

### 4. The preset camera crops tall objects

`xgutils`'s `preset_glb.blend` camera carries a TRACK_TO constraint evaluated
after `matrix_world`, so `set_camera_orientation`'s target is ignored and the
camera keeps looking at the origin. Anything taller than it is wide loses its
top. Symptom: a street lamp with its base mid-frame and the lantern past the top
edge.

**`render_emissive.py` clears the constraint itself, at runtime.** A stock
`xgutils` checkout is therefore fine for this script. Anything else you write
against `preset_glb` needs the same two lines.

### 5. Blender slot order is not glTF material index

The headphone stand's Blender slot 0 is glTF material 10. A range check
(`index < material_count`) passes and applies your data to the wrong materials.
Symptom: panels that look like model errors.

Key everything by **Blender slot**, and ship the slot-ordered `materials` name
list so the loader can verify rather than assume.

### 6. Use the supplied cameras, do not solve your own

`cameras/<sid>.json` holds the per-shape camera, verified pixel-identical to the
solved camera. Pass it with `--camera_json`. If you solve your own, your panels
will not align with existing ones even under an identical treatment, and the
difference will be a viewpoint rather than a method.

### 7. TexVerse is mounted on the workstation only

`/cs/3dlg-falas` is not visible from solar login or compute nodes. **Copy the
GLBs to `/project` rather than symlinking them.** A symlink resolves where you
prepare it and dangles where the render runs, and Blender reports that as
`Error: Please select a file`, which reads like a bad argument rather than a
missing file. `example/prepare.sh` copies for this reason.

### 8. Working directories must be under `/project` or `/cs/3dlg-project`

Compute nodes see those and nothing else. A workdir on `/local-scratch2` or in a
home directory fails the same way as trap 7, with the same misleading message.

---

## Why the treatment is what it is

Measured, not chosen by taste. Full evidence:
https://aspis.cmpt.sfu.ca/projects/omages/yanxg/lightgen/workspace/render_sweep/

- **Key light 8.** The Glare threshold is 1.0 in linear space. At key 8 no
  non-emissive surface reaches it, so bloom fires only where something actually
  emits. At key 20 the bloom fired on 108,865 pixels of a shape emitting
  nothing, against 123,948 on real ground truth: the glow was reporting the
  lamp, not the object. At key 8 it is 0 against 50,458. Measured on the panels
  that had to be told apart, the ground-truth-over-prediction bright-pixel ratio
  on the hardest shape went from 1.2 to 6.2.
  **Raising emission strength moves emitters further above that threshold and
  will grow the bloom.** Expect that when sweeping strength.
- **Bloom size 7, mix -0.45** on the box path, 9 and -0.15 on the key-lit path.
  Size is radius, mix is intensity. **Threshold is the wrong lever**: it keys
  off absolute brightness and cannot serve a range of shapes. At threshold 2.5
  the dimmest shape loses its bloom entirely while the brightest keeps a sixth
  of frame.
- **Filmic over Standard** on the box path. At the exposure dim shapes need,
  Standard blows out 12.4% of the vending machine frame and destroys the
  artwork the object is emitting, while giving the dimmest shape a *lower*
  midtone than Filmic.
- **32 / 16 bounces** on the box path against Cycles' 12 / 4. In a box lit only
  by the object, the ambient fill IS multi-bounce diffuse light; truncating at
  4 discards it. Worth +39% image mean on the dimmest shape.
