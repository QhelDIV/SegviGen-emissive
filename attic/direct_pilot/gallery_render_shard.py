import sys, os, json, types, importlib, importlib.util
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import software_voxrender as sw
def cpuio():
    if "o_voxel" not in sys.modules:
        spec=importlib.util.find_spec("o_voxel"); pkg=types.ModuleType("o_voxel")
        pkg.__path__=spec.submodule_search_locations; pkg.__spec__=spec; sys.modules["o_voxel"]=pkg
    return importlib.import_module("o_voxel.io")
IO=cpuio()
PD="/3dlg-jupiter-project/lightgen/segvigen_emissive/direct_pilot"
B="/3dlg-jupiter-project/lightgen/uv_voxel_pipeline/out_uv_voxel_74k"
OUT=f"{PD}/uvvox_gallery"; SH=f"{PD}/uvvox_gallery_shards"
os.makedirs(OUT,exist_ok=True); os.makedirs(SH,exist_ok=True)
THR=1.0/255.0   # emissive iff normalized max-channel luminance > 1/255 (any authored emission; drops the value<=1 encoding-noise floor)
shard=int(sys.argv[1]); nshards=int(sys.argv[2])
rows=json.load(open(f"{PD}/gallery_sids.json"))
mine=[r for i,r in enumerate(rows) if i%nshards==shard]
out=[]; nfail=0
for r in mine:
    sid=r["sid"]; png=f"{OUT}/{sid}.png"
    try:
        coords,data=IO.read_vxz(f"{B}/{sid}/emission_voxels_256/{sid}.vxz",num_threads=1)
        coords=coords.cpu().numpy().astype(np.int32)
        # emissive iff normalized max-channel lum > 1/255  <=>  uint8 max-channel > 1
        # (integer compare avoids float32(1/255) rounding; drops the 0-1 encoding floor)
        mask=(data["emissive"].max(dim=-1).values.to("cpu").int()>1).numpy()
        ne,nt=sw.render(coords,mask,png,px=300,fill=0.86)
        out.append(dict(sid=sid,stratum=r["stratum"],emissive_frac=round(float(ne/max(1,nt)),5),n_voxels=int(nt)))
    except Exception as e:
        nfail+=1; print(f"[fail] {sid}: {repr(e)[:150]}",flush=True)
json.dump(out, open(f"{SH}/shard_{shard}.json","w"))
print(f"SHARD {shard} done={len(out)} fail={nfail}",flush=True)
