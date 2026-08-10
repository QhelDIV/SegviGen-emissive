import json, numpy as np, torch, trimesh, o_voxel, pandas as pd

PARQUET = "/3dlg-falas/project/omages/datasets/TexVerse/lightgen/emissive_thumbnails_obj_ids_df.parquet"
GLB_ROOT = "/3dlg-falas/datasets/TexVerse-1K"
OVOX_ROOT = "/3dlg-jupiter-project/lightgen/uv_voxel_pipeline/out_uv_voxel_74k"

with open("/3dlg-jupiter-project/lightgen/segvigen_emissive/smoke20_shas.txt") as f:
    sids = [l.strip() for l in f if l.strip()]

df = pd.read_parquet(PARQUET)

# Step 1: find bbox aspect ratio for each smoke20 shape, pick the most asymmetric.
cands = []
for sid in sids:
    glb_path = f"{GLB_ROOT}/{df.loc[sid, 'glb_1k_path']}"
    try:
        asset = trimesh.load(glb_path, force="scene")
    except Exception as e:
        print(f"[skip] {sid}: {e}")
        continue
    aabb = asset.bounding_box.bounds
    extent = aabb[1] - aabb[0]
    ratio = extent.max() / extent.min()
    cands.append((ratio, sid, extent.tolist()))

cands.sort(reverse=True)
print("Top 5 most bbox-asymmetric shapes in smoke20:")
for ratio, sid, extent in cands[:5]:
    print(f"  ratio={ratio:.2f} sid={sid} extent={extent}")

SHA = cands[0][1]
print(f"\n=== verifying frame on {SHA} (bbox aspect ratio {cands[0][0]:.2f}) ===")

glb_path = f"{GLB_ROOT}/{df.loc[SHA, 'glb_1k_path']}"
asset = trimesh.load(glb_path, force="scene")
aabb = asset.bounding_box.bounds
center = (aabb[0] + aabb[1]) / 2
scale = 0.99999 / (aabb[1] - aabb[0]).max()
print("center:", center, "scale:", scale, "extent:", (aabb[1]-aabb[0]).tolist())
asset.apply_translation(-center)
asset.apply_scale(scale)

coord256, attr256 = o_voxel.convert.textured_mesh_to_volumetric_attr(
    asset, grid_size=256, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    mip_level_offset=-1e6, verbose=False, timing=False,
)
print("re-derived coord256:", coord256.shape, coord256.dtype)

pbr_coords, pbr_data = o_voxel.io.read(f"{OVOX_ROOT}/{SHA}/pbr_voxels_256/{SHA}.vxz")

def coord_set(t):
    t = t.numpy() if torch.is_tensor(t) else t
    return set(map(tuple, t.tolist()))

my_set = coord_set(coord256)
dc_set = coord_set(pbr_coords)
inter = my_set & dc_set
print(f"my N={len(my_set)} dongchen N={len(dc_set)} intersection={len(inter)}")
print(f"IoU = {len(inter) / len(my_set | dc_set):.4f}")

# also sanity-check a permuted-axis hypothesis would NOT also pass, to prove asymmetry discriminates
def permuted(t, perm, flip):
    t = t.numpy().astype(np.int64)
    t = t[:, perm]
    for ax, fl in enumerate(flip):
        if fl:
            t[:, ax] = 255 - t[:, ax]
    return set(map(tuple, t.tolist()))

for name, perm, flip in [
    ("identity", [0,1,2], [0,0,0]),
    ("dong=(x,z,1-y)", [0,2,1], [0,0,1]),
    ("swap yz", [0,2,1], [0,0,0]),
]:
    pset = permuted(coord256, perm, flip)
    iou = len(pset & dc_set) / len(pset | dc_set)
    print(f"  hypothesis {name:20s}: IoU={iou:.4f}")
