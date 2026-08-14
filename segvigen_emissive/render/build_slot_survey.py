#!/usr/bin/env python3
"""Minimal material_survey.json for a handful of sids: Blender slot order only.

pred_mask_to_asset.py refuses to run without --survey, because Blender's material slot
order is not glTF material index order (see its own docstring) and the renderer indexes
masks by slot. paper_v3/material_survey.json already carries this for the 11 paper-figure
sids; the fbv1_repro 8 shapes are a different set, so this script builds the same
{sid: {"materials": [{"slot": i, "material": name}, ...]}} shape for just them, reading
the order straight off obj.material_slots after xgutils loads the GLB -- the same load
path render_emissive.py itself uses, so the slot order it produces IS the order the
renderer will apply masks in.

Run on a CPU node, shared venv, PYTHONPATH=<xgutils>/src.

  python build_slot_survey.py --manifest manifest.json --glb_dir glb --out survey.json
"""
import argparse
import json
import os

import bpy  # noqa: E402
from xgutils import bpyutil  # noqa: E402


def one(sid, glb_path):
    bpyutil.load_blend(bpyutil.preset_glb)
    bpyutil.clear_collection("workbench")
    obj = bpyutil.load_glb(glb_path, import_shading=None)
    names = [slot.material.name if slot.material else None for slot in obj.material_slots]
    bpyutil.purge_obj(obj)
    return {"materials": [{"slot": i, "material": n} for i, n in enumerate(names)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="json list of {sid: ...} rows")
    ap.add_argument("--glb_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = json.load(open(args.manifest))
    sids = [r["sid"] for r in rows]
    survey = {}
    for sid in sids:
        glb = os.path.join(args.glb_dir, f"{sid}.glb")
        try:
            survey[sid] = one(sid, glb)
            print(f"OK {sid} n_slots={len(survey[sid]['materials'])}", flush=True)
        except Exception as e:
            print(f"FAIL {sid}: {e}", flush=True)
    json.dump(survey, open(args.out, "w"), indent=1)
    print(f"WROTE {args.out} n={len(survey)}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
