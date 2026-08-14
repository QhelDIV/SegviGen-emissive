import json, glob, os
from collections import Counter
PD="/3dlg-jupiter-project/lightgen/segvigen_emissive/direct_pilot"
rows=[]
for f in sorted(glob.glob(f"{PD}/uvvox_gallery_shards/shard_*.json")):
    rows += json.load(open(f))
# keep only those whose png exists
rows=[r for r in rows if os.path.exists(f"{PD}/uvvox_gallery/{r['sid']}.png")]
rows.sort(key=lambda r: -r["emissive_frac"])
json.dump(rows, open(f"{PD}/uvvox_gallery/gallery_manifest.json","w"), indent=1)
print("total rendered:", len(rows))
print("strata:", dict(Counter(r["stratum"] for r in rows)))
# sanity: zero-stratum should be ~0 frac, glow real
import numpy as np
for s in ["zero","tiny","glow"]:
    fr=[r["emissive_frac"] for r in rows if r["stratum"]==s]
    if fr: print(f"  {s}: n={len(fr)} frac mean={np.mean(fr):.3f} max={np.max(fr):.3f} n(frac>0.05)={sum(f>0.05 for f in fr)}")
print("MANIFEST_DONE")
