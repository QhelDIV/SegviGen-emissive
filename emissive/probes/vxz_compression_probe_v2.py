"""
Extends vxz_compression_probe.py: measure DECOMPRESS (read) time, not just compress
(write) time, per codec. The dataset is read every epoch of every training run for the
life of the project, so read cost matters at least as much as write cost. Same source
fixture (input.vxz, 315335 voxels) as the original probe.

Each config: write once, then read 3x (report min/mean -- first read may pay NFS cold-
cache cost, later reads are page-cache warm; report both so we don't understate a truly
NFS-bound difference).
"""
import time
import os
import o_voxel

SHAPE = "294095f9c38d48f39b6f9b7162b963d7"
SRC = f"/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct/smoke1/{SHAPE}/input.vxz"
OUT_DIR = "/3dlg-jupiter-project/lightgen/segvigen_emissive/vxz_compress_probe"
os.makedirs(OUT_DIR, exist_ok=True)

coord, attr = o_voxel.io.read_vxz(SRC, num_threads=1)
print("n_voxels:", coord.shape[0], flush=True)

configs = [
    ("lzma", None),      # current project-wide default (level 9)
    ("zstd", None),       # zstd default level 22
    ("zstd", 9),
    ("zstd", 3),
    ("deflate", None),    # deflate default level 9
    ("deflate", 1),
    ("none", None),
]

N_READS = 3
for algo, level in configs:
    out = os.path.join(OUT_DIR, f"probe_{algo}_{level}.vxz")
    t0 = time.perf_counter()
    o_voxel.io.write(out, coord, attr, compression=algo, compression_level=level)
    t_write = time.perf_counter() - t0
    size = os.path.getsize(out)

    read_times = []
    for k in range(N_READS):
        t0 = time.perf_counter()
        o_voxel.io.read_vxz(out, num_threads=1)
        read_times.append(time.perf_counter() - t0)

    print(f"algo={algo:8s} level={str(level):5s} write_s={t_write:6.3f} "
          f"read_s(1st)={read_times[0]:6.3f} read_s(min_of_3)={min(read_times):6.3f} "
          f"read_s(mean_of_3)={sum(read_times)/len(read_times):6.3f} size_bytes={size}", flush=True)

print("COMPRESSION_PROBE_V2_DONE", flush=True)
