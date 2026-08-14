"""Convert a draw's npz (coords + pred_bc + gt_e) into the voxel-viewer format
render_voxels.py expects, using either gt_e or pred_bc>0.5 as the marked-lit
set. Same convention as convert_voxel_arbitration.py (grey occupied, uniform
color 180/180/180), just parameterized over which field to threshold."""
import json
import sys
import numpy as np

GRID = 512


def convert(npz_in, out_path, sid, mode):
    z = np.load(npz_in)
    coords = z["coords"].astype(np.int32)
    if mode == "gt":
        lit = z["gt_e"].astype(bool)
    elif mode == "pred":
        lit = (z["pred_bc"] > 0.5)
    else:
        raise ValueError(mode)
    color = np.full((coords.shape[0], 3), 180, dtype=np.uint8)
    meta = json.dumps({"res_display": GRID, "source": npz_in, "sid": sid, "mode": mode})
    np.savez_compressed(out_path, cells=coords, color=color, lit=lit, meta=meta)
    print(f"CONVERTED {sid} ({mode}): n_cells={coords.shape[0]} n_lit={int(lit.sum())} -> {out_path}", flush=True)


if __name__ == "__main__":
    npz_in, out_path, sid, mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    convert(npz_in, out_path, sid, mode)
