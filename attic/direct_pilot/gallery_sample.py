"""Stratified sample ~700 shapes by emission_cov_voxel from classification.parquet."""
import pandas as pd, numpy as np, json, os
D="/3dlg-jupiter-project/lightgen/uv_voxel_pipeline"
B=f"{D}/out_uv_voxel_74k"
df=pd.read_parquet(f"{D}/out_uv_voxel_74k_verification/classification.parquet")
df=df[df["classification"]=="complete"].copy()
df=df[df["emission_cov_voxel"].notna()]
cov=df["emission_cov_voxel"].values; sids=df["sha"].values
rng=np.random.default_rng(0)
def pick(mask,n):
    idx=np.where(mask)[0]; rng.shuffle(idx); return idx[:n]
targets=[("glow", cov>0.05, 385), ("tiny", (cov>0)&(cov<=0.05), 210), ("zero", cov==0, 105)]
rows=[]
for name,m,n in targets:
    for i in pick(m,n):
        rows.append({"sid":sids[i],"stratum":name,"cov":float(cov[i])})
# verify vxz exists; drop + backfill from same stratum if missing
have=[]; missing_by={"glow":[],"tiny":[],"zero":[]}
for r in rows:
    if os.path.exists(f"{B}/{r['sid']}/emission_voxels_256/{r['sid']}.vxz"): have.append(r)
    else: missing_by[r["stratum"]].append(r)
nmiss=sum(len(v) for v in missing_by.values())
print(f"sampled {len(rows)}, vxz-present {len(have)}, missing {nmiss}")
json.dump(have, open("gallery_sids.json","w"))
from collections import Counter
print("strata:", Counter(r["stratum"] for r in have))
