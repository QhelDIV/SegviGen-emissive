import time, os
import o_voxel

SHAPE = "294095f9c38d48f39b6f9b7162b963d7"
SRC = f"/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct/smoke1/{SHAPE}/input.vxz"
OUT_DIR = "/3dlg-jupiter-project/lightgen/segvigen_emissive/vxz_compress_probe"
os.makedirs(OUT_DIR, exist_ok=True)

coord, attr = o_voxel.io.read_vxz(SRC, num_threads=1)
print("n_voxels:", coord.shape[0])

configs = [
    ("lzma", None),      # current default (level 9)
    ("zstd", None),      # zstd default level 22
    ("zstd", 3),
    ("zstd", 9),
    ("deflate", None),   # deflate default level 9
    ("deflate", 1),
    ("none", None),
]

for algo, level in configs:
    out = os.path.join(OUT_DIR, f"probe_{algo}_{level}.vxz")
    t0 = time.perf_counter()
    o_voxel.io.write(out, coord, attr, compression=algo, compression_level=level)
    t1 = time.perf_counter()
    size = os.path.getsize(out)
    print(f"algo={algo:8s} level={str(level):5s} write_s={t1-t0:6.3f} size_bytes={size}")
