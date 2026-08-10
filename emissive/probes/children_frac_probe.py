"""
Children-count diagnostic, requested by team-lead re: the 256->512 emissive_frac
inflation seen on the two sparse smoke samples (raw 0.0114 -> built 0.0143,
raw 0.0711 -> built 0.0882, both roughly +25% relative). That inflation was
reported but never explained -- the semantic check (roundtrip_probe_v2.py)
only tests LABEL correctness (does an emissive 256 parent's built children
come out emissive), which would pass identically whether or not emissive
parents produce systematically more occupied 512 children than dark parents.

This script answers the geometry question directly, no encoder/decoder
involved: for each occupied 256 parent voxel, count how many of its 8
candidate 512 children are OCCUPIED in the built 512 voxelization (not
whether they're emissive -- just occupied at all), split by whether the
parent itself is emissive or dark at 256-res.

If mean-children(emissive parents) > mean-children(dark parents): the
built target really is denser near emissive regions for a geometric reason
(surface curvature/thin-structure effects at the 256->512 boundary), the
frac shift is real and benign.

If the two means are ~equal: something in the upsample/gap-fill step is
disproportionately keeping/creating positive-class children, which would be
a bug worth finding before the 72k build.

Runs on all 3 existing round-trip-probe samples for completeness, but the
two sparse ones (smoke1 and smoke20/94124b53) are the ones that showed the
+25% inflation and are the ones this question is actually about; the
near-full sample (frac 0.957) showed a built frac slightly BELOW raw
(0.9585->0.9569), i.e. no inflation there, included as a contrast case.
"""
import os
import json
import numpy as np
import o_voxel

DATASET = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct"
OVOX_ROOT = "/3dlg-jupiter-project/lightgen/uv_voxel_pipeline/out_uv_voxel_74k"
SAMPLES = [
    ("smoke1", "294095f9c38d48f39b6f9b7162b963d7"),   # low frac 0.0143, raw 0.0114
    ("smoke20", "94124b539e714bd29d7889c3cb4c5325"),   # median frac 0.088, raw 0.0711
    ("smoke20", "9acd6bd8c0c1453d9d2bea771ee3941f"),   # near-full frac 0.957, raw 0.9585 (contrast, no inflation)
]

for split, sid in SAMPLES:
    d = os.path.join(DATASET, split, sid)
    meta = json.load(open(os.path.join(d, "meta.json")))

    # raw 256 emission labels (which parents are emissive vs dark)
    emis_vxz = os.path.join(OVOX_ROOT, sid, "emission_voxels_256", f"{sid}.vxz")
    coords256, emis256 = o_voxel.io.read(emis_vxz)
    is_emis_256 = (emis256["emissive"] > 0).any(dim=1).numpy()
    c256 = coords256.numpy()
    raw_frac = float(is_emis_256.mean())

    # built 512 occupancy (all occupied voxels, regardless of emissive label)
    output_vxz = os.path.join(d, "output.vxz")
    coords512, _ = o_voxel.io.read(output_vxz)
    c512 = coords512.numpy()
    occ512 = set(map(tuple, c512.tolist()))
    built_frac = meta["emissive_frac"]

    n_emis_parents = int(is_emis_256.sum())
    n_dark_parents = int((~is_emis_256).sum())

    child_counts_emis = []
    child_counts_dark = []
    for i in range(len(c256)):
        parent = c256[i]
        children = [(parent[0]*2+dx, parent[1]*2+dy, parent[2]*2+dz)
                    for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
        n_present = sum(1 for c in children if c in occ512)
        if is_emis_256[i]:
            child_counts_emis.append(n_present)
        else:
            child_counts_dark.append(n_present)

    ce = np.array(child_counts_emis)
    cd = np.array(child_counts_dark)
    print(f"\n=== {split}/{sid} (raw_frac={raw_frac:.4f} built_frac={built_frac:.4f}) ===", flush=True)
    print(f"[PARENTS] n_emis_256={n_emis_parents} n_dark_256={n_dark_parents}", flush=True)
    print(f"[CHILDREN/PARENT] emissive parents: mean={ce.mean():.4f} std={ce.std():.4f} "
          f"median={np.median(ce):.1f} (n={len(ce)})", flush=True)
    print(f"[CHILDREN/PARENT] dark parents:     mean={cd.mean():.4f} std={cd.std():.4f} "
          f"median={np.median(cd):.1f} (n={len(cd)})", flush=True)
    ratio = ce.mean() / cd.mean() if cd.mean() > 0 else float("inf")
    print(f"[RATIO] emis_mean_children / dark_mean_children = {ratio:.4f}", flush=True)

print("\nDONE children_frac_probe", flush=True)
