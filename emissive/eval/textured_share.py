"""
Per-shape share of surface area carried by a TEXTURED material, for the 300-shape eval set.

Why it matters: on a constant-colour material there is nothing to segment spatially. The
model picks among a handful of discrete slots, so a shape with two materials is one binary
choice with a 50 percent prior, and guessing that bit right scores like a good segmentation.
Per-voxel IoU averages that categorical regime together with genuine spatial placement and
separates them nowhere. This computes the axis needed to split them.

Definition: a material is textured if its BASE COLOUR is fed by an image (glTF
`pbrMetallicRoughness.baseColorTexture`, or `KHR_materials_pbrSpecularGlossiness.diffuseTexture`
for spec-gloss assets). Normal and roughness maps do NOT count: they carry no information
about where a surface emits. Textured share is the area-weighted fraction, using true
world-space triangle area, not primitive or material counts.

The definition is CHECKED against paper-v3's independently produced `material_survey.json`
on the 11 shapes both cover, rather than assumed to match. `--verify_only` runs just that
check.

  python emissive/eval/textured_share.py --sids_json sids.json --glb_root /3dlg-falas/datasets/TexVerse-1K \
      --split_json .../data_splits_74k.json --out textured_share.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pred_mask_to_asset import read_glb, primitives


def material_textured(gltf, idx):
    """True if this material's BASE COLOUR comes from an image."""
    if idx is None or idx >= len(gltf.get("materials", [])):
        return False
    m = gltf["materials"][idx]
    if (m.get("pbrMetallicRoughness") or {}).get("baseColorTexture") is not None:
        return True
    sg = (m.get("extensions") or {}).get("KHR_materials_pbrSpecularGlossiness") or {}
    return sg.get("diffuseTexture") is not None


def shape_textured_share(glb_path):
    gltf, bins = read_glb(glb_path)
    prims = primitives(gltf, bins)
    area_by_mat, textured = {}, {}
    for p in prims:
        v = p["positions"][p["faces"]]
        a = float(np.abs(0.5 * np.linalg.norm(
            np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)).sum())
        mi = p["material"]
        area_by_mat[mi] = area_by_mat.get(mi, 0.0) + a
        textured[mi] = material_textured(gltf, mi)
    total = sum(area_by_mat.values())
    if total <= 0:
        return None
    tex_area = sum(a for mi, a in area_by_mat.items() if textured.get(mi))
    return {"textured_area_share": tex_area / total,
            "n_materials": len(area_by_mat),
            "n_textured_materials": int(sum(1 for v in textured.values() if v)),
            "total_area": total,
            "fully_constant": not any(textured.values()),
            "fully_textured": all(textured.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sids_json", default=None, help='{"sid": "split"} or a list of sids')
    ap.add_argument("--diagnostics", default=None,
                    help="diagnostics.json whose per_shape_diag3 keys are the 300 eval sids")
    ap.add_argument("--glb_root", default="/3dlg-falas/datasets/TexVerse-1K")
    ap.add_argument("--split_json",
                    default="/3dlg-jupiter-project/lightgen/diffusionnet_xg/data/data_splits_74k.json")
    ap.add_argument("--survey", default=None, help="paper-v3 material_survey.json, for the check")
    ap.add_argument("--verify_only", action="store_true", default=False)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # ---- definition check against paper-v3's survey, before trusting anything ----
    if args.survey and os.path.exists(args.survey):
        survey = json.load(open(args.survey))
        agree = disagree = 0
        for sid, sv in survey.items():
            glb = os.path.join(args.glb_root, resolve_one(sid, args.split_json))
            if not os.path.exists(glb):
                continue
            gltf, _ = read_glb(glb)
            mine = {}
            for i, m in enumerate(gltf.get("materials", [])):
                mine[m.get("name", f"material_{i}")] = material_textured(gltf, i)
            for m in sv["materials"]:
                theirs = (m.get("base_color") != "constant")
                if m["material"] in mine:
                    if mine[m["material"]] == theirs:
                        agree += 1
                    else:
                        disagree += 1
                        print(f"  DISAGREE {sid[:8]} {m['material']}: "
                              f"mine textured={mine[m['material']]} theirs={m.get('base_color')}")
        print(f"DEFINITION_CHECK agree={agree} disagree={disagree}", flush=True)
        if args.verify_only:
            return

    sids = []
    if args.diagnostics:
        sids = sorted(json.load(open(args.diagnostics))["per_shape_diag3"].keys())
    elif args.sids_json:
        j = json.load(open(args.sids_json))
        sids = sorted(j if isinstance(j, list) else j.keys())
    print(f"computing textured share for {len(sids)} shapes", flush=True)

    out, n_err, t0 = {}, 0, time.time()
    for i, sid in enumerate(sids):
        try:
            rel = resolve_one(sid, args.split_json)
            r = shape_textured_share(os.path.join(args.glb_root, rel))
            if r is not None:
                out[sid] = r
            else:
                n_err += 1
        except Exception as e:
            n_err += 1
            if n_err <= 5:
                print(f"  ERR {sid[:8]}: {repr(e)[:140]}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"PROGRESS {i + 1}/{len(sids)} errs={n_err} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)
    print(f"DONE n={len(out)} errs={n_err} elapsed={time.time() - t0:.0f}s", flush=True)

    if out:
        s = np.array([v["textured_area_share"] for v in out.values()])
        print(f"  fully constant (share==0) : {(s == 0).sum()} of {len(s)}")
        print(f"  fully textured (share==1) : {(s == 1).sum()}")
        print(f"  mixed                     : {((s > 0) & (s < 1)).sum()}")
        print(f"  share < 0.5               : {(s < 0.5).sum()}")
        print(f"  median material count     : {np.median([v['n_materials'] for v in out.values()])}")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"WROTE {args.out}")


_SPLIT_CACHE = {}
# same table and column build_dataset_direct.resolve_glb_rel uses, so a shape resolves to
# the identical file the dataset was built from
PARQUET = "/3dlg-falas/project/omages/datasets/TexVerse/lightgen/emissive_thumbnails_obj_ids_df.parquet"


def resolve_one(sid, split_json=None):
    """sid -> glb path relative to GLB_ROOT."""
    if "df" not in _SPLIT_CACHE:
        import pandas as pd
        _SPLIT_CACHE["df"] = pd.read_parquet(PARQUET)
    df = _SPLIT_CACHE["df"]
    if sid not in df.index:
        raise KeyError(f"{sid} not in {PARQUET}")
    return df.loc[sid, "glb_1k_path"]


if __name__ == "__main__":
    main()
