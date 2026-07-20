"""
Render a voxel emissive mask as a 3D figure: emissive voxels in accent orange,
non-emissive surface voxels in light grey, on a light backdrop. Small cube per
(subsampled) voxel via xgutils.bpyutil.render_mesh.

Reads either:
  - a direct.vxz (emissive = attr 'emissive' max-channel lum > thr), or
  - a somage output.vxz (emissive = base_color mean > 127.5 white), or
  - a precomputed emis_gt_direct.npz (coords + mask).

Runs in omages_internal/.venv2 (bpy). Coords are @512 z-up internal; we normalize
to a unit box and render a canonical three-quarter view.
"""
import os, sys, types, importlib, importlib.util
import numpy as np


def _cpu_io():
    if "o_voxel" not in sys.modules:
        spec = importlib.util.find_spec("o_voxel")
        pkg = types.ModuleType("o_voxel"); pkg.__path__ = spec.submodule_search_locations; pkg.__spec__ = spec
        sys.modules["o_voxel"] = pkg
    return importlib.import_module("o_voxel.io")


def load_mask(path, thr=0.04):
    """-> coords (M,3) float, emissive (M,) bool."""
    if path.endswith(".npz"):
        d = np.load(path)
        return d["coords"].astype(np.float64), d["mask"].astype(bool)
    IO = _cpu_io()
    import torch
    coords, data = IO.read_vxz(path, num_threads=1)
    coords = coords.cpu().numpy().astype(np.float64)
    if "emissive" in data:
        em = data["emissive"].float() / 255.0
        mask = (em.max(dim=-1).values > thr).cpu().numpy()
    else:  # somage output.vxz white = emissive
        mask = (data["base_color"].float().mean(dim=-1) > 127.5).cpu().numpy()
    return coords, mask


# unit cube (8 verts, 12 tris)
_CV = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], np.float64) - 0.5
_CF = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[1,5,6],[1,6,2],
                [2,6,7],[2,7,3],[3,7,4],[3,4,0]], np.int64)

ACCENT = np.array([0.79, 0.36, 0.20, 1.0])   # terracotta
GREY   = np.array([0.72, 0.72, 0.74, 1.0])


def build_cube_mesh(coords, mask, max_grey=45000, max_emis=60000, seed=0):
    """coords @512 -> normalized [-.5,.5], one small cube per (subsampled) voxel."""
    rng = np.random.default_rng(seed)
    c = coords / 512.0 - 0.5
    # z-up internal -> keep as-is; render camera handles view
    emis_idx = np.where(mask)[0]
    grey_idx = np.where(~mask)[0]
    if len(emis_idx) > max_emis:
        emis_idx = rng.choice(emis_idx, max_emis, replace=False)
    if len(grey_idx) > max_grey:
        grey_idx = rng.choice(grey_idx, max_grey, replace=False)
    keep = np.concatenate([grey_idx, emis_idx])
    is_emis = np.concatenate([np.zeros(len(grey_idx), bool), np.ones(len(emis_idx), bool)])
    centers = c[keep]
    cube = 1.4 / 512.0  # cube edge ~ voxel pitch
    V = (centers[:, None, :] + _CV[None] * cube).reshape(-1, 3)
    F = (_CF[None] + (np.arange(len(centers))[:, None, None] * 8)).reshape(-1, 3)
    col = np.where(is_emis[:, None], ACCENT[None], GREY[None])
    VC = np.repeat(col, 8, axis=0)
    return V, F, VC, int(mask.sum()), int(len(coords))


def render(coords, mask, out_png, cam=(1.9, -2.3, 1.5)):
    from xgutils import bpyutil
    from xgutils.vis import visutil
    V, F, VC, n_emis, n_tot = build_cube_mesh(coords, mask)
    # normalize to unit box centered (render_mesh does not autoscale)
    lo, hi = V.min(0), V.max(0)
    V = (V - (lo + hi) / 2) / (hi - lo).max()
    img = bpyutil.render_mesh(V, F, vert_color=VC, resolution=(720, 720),
                              samples=24, shadow_catcher=True, camera_position=cam)
    visutil.saveImg(out_png, img)
    return n_emis, n_tot


if __name__ == "__main__":
    # argv: out_png thr path
    out_png, thr, path = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    coords, mask = load_mask(path, thr)
    n_emis, n_tot = render(coords, mask, out_png)
    print(f"RENDERED {out_png} n_emis={n_emis} n_tot={n_tot} frac={n_emis/max(1,n_tot):.4f}")
