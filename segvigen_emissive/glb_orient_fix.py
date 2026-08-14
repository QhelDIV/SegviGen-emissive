"""
Shared fix for a GLB axis-convention bug found while investigating owner feedback that the
Section A fish example looked "laid down": somage_to_glb-exported GLBs store Z-up vertex
data inside a nominally Y-up glTF container. Blender's glTF importer
(bpy.ops.import_scene.gltf, used by xgutils.bpyutil.load_glb) applies the standard
Y-up -> Z-up conversion assuming Y-up input — which is WRONG for these files, and rotates
the object ~90 deg off its true orientation (e.g. an asset's true dorsal-ventral axis ends
up horizontal instead of vertical).

Root-caused by direct vertex tracing (not just bounding-box extent matching, which only
proves magnitude, not axis identity/sign) on d5fb4f19d4164612b165caac5471555c
(finetune_examples): the raw pre-import vertex at maximum raw Z — (0.212, 0, 0.765) —
lands at Blender's *minimum Y* after xgutils' load_glb+normalize_mesh, not at max Z as
it should for an upright render. The raw/npz-voxel coordinate convention (used untouched
for the stage-3/4 voxel renders, which the owner confirmed look correct) has no such
import step, so it never suffers this rotation.

fix_glb_upright() applies a -90 deg rotation about world X directly via matrix
multiplication (bpy.ops.transform.rotate's sign convention did not match hand-derived
expectations in testing, so this bypasses the operator and is verified independently):
after applying it, the same raw max-Z vertex (0.212, ~0, 0.765) lands at Blender's max Z,
confirming the fix restores the correct up-axis.

Usage:
    obj = bpyutil.load_glb(path, ...)
    fix_glb_upright(obj)
    # ... then render as usual
"""
import math
import mathutils


def fix_glb_upright(obj):
    import bpy
    R = mathutils.Matrix.Rotation(math.radians(-90), 4, 'X')
    obj.matrix_world = R @ obj.matrix_world
    bpy.context.view_layer.update()
