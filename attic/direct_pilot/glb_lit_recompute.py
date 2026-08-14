import numpy as np, pandas as pd, os, datetime
R=np.load("/tmp/glb_scan_res.npy",allow_pickle=True)  # sha,pbr,status,n_mat,n_emit,n_copy,has_real,lit
sha=np.array([str(x) for x in R[:,0]])
pbr=np.array([("metalness" if str(x)=="metalness" else "specular" if str(x)=="specular" else "NA") for x in R[:,1]])
n_emit=R[:,4].astype(int); n_copy=R[:,5].astype(int); lit=R[:,7].astype(int).astype(bool)
N=len(R)
# baked-nonzero from crosstab (old_max=somage emission max)
cx=pd.read_parquet("/cs/3dlg-jupiter-project/lightgen/uv_voxel_pipeline/emission_bake_crosstab_74k.parquet")
om=dict(zip(cx.uuid.astype(str),cx.old_max)); nm=dict(zip(cx.uuid.astype(str),cx.new_max))
som_nz=np.array([om.get(s,0)>0 for s in sha])
in_cx=np.array([s in om for s in sha])
print(f"corpus={N}  in_crosstab={int(in_cx.sum())}  somage_nonzero={int(som_nz.sum())}")

print("\n=== SELF-LIT by pbrType (FIXED detector, corr>0.9 & resid<0.05) ===")
for c in ["metalness","specular","NA"]:
    m=pbr==c; l=m&lit
    print(f"  {c:9} n={int(m.sum()):6d}  lit={int(l.sum()):6d} ({100*l.sum()/max(1,m.sum()):.1f}%)  keep={int((m&~lit).sum()):6d}")
print(f"  TOTAL      n={N}  lit={int(lit.sum())} ({100*lit.mean():.1f}%)")

print("\n=== TRAINABLE (nonzero = somage old_max>0) ===")
pbr_ms=np.isin(pbr,["metalness","specular"])
old_tr = pbr_ms & som_nz
print(f"OLD (metal+spec & nonzero) = {int(old_tr.sum())}  [owner ~26,989]")
print(f"  of which self-lit (contamination in old set): {int((old_tr&lit).sum())} ({100*(old_tr&lit).sum()/max(1,old_tr.sum()):.1f}%)")
new_pbr_keep = pbr_ms & som_nz & ~lit
na_rescue    = (pbr=="NA") & som_nz & ~lit
new_tr       = som_nz & ~lit
print(f"NEW (any pbrType & nonzero & NOT self-lit):")
print(f"  pbr-kept-clean : {int(new_pbr_keep.sum())}")
print(f"  NA-rescued     : {int(na_rescue.sum())}")
print(f"  = NEW TRAINABLE: {int(new_tr.sum())}   (net {int(new_tr.sum()-old_tr.sum()):+d} vs old)")
print(f"  self-lit dropped (nonzero, all pbr): {int((som_nz&lit).sum())}")

print("\n=== multi-material rule sensitivity (drop count, nonzero) ===")
base=som_nz&(n_emit>0)
r_all=int((base&lit).sum()); r_maj=int((base&(n_copy>n_emit/2.0)).sum()); r_any=int((base&(n_copy>0)).sum())
print(f"  ALL emitting copy (default): {r_all}")
print(f"  MAJORITY copy:               {r_maj}")
print(f"  ANY copy:                    {r_any}")

# SAVE lit list (nonzero)
OUT="/local-scratch2/xya120/studio/misc/lightgen/segvigen_emissive/direct_pilot/lit_shadeless_shas.txt"
sel=np.where(som_nz&lit)[0]; sel=sel[np.argsort(pbr[sel])]
with open(OUT,"w") as f:
    f.write("# lit_shadeless_shas -- FILTER STAGE 3 (STRUCTURAL, FIXED detector): drop de-facto shadeless/fullbright shapes\n")
    f.write("# Detector: a glb material is a COPY if its emissiveTexture image == baseColor/diffuse image by\n")
    f.write("#   same glTF image index, OR content: corr(emission,albedo brightness)>0.9 AND scale-fit residual<0.05.\n")
    f.write("#   (corr gate added to kill the k->0 false-positive on near-black localized emitters.)\n")
    f.write("#   Shape is SELF-LIT if it has emission AND every emitting material is a copy. Cross-cuts pbrType.\n")
    f.write("# This list = self-lit shapes with somage-nonzero emission (the ones dropped from training).\n")
    f.write(f"# Generated {datetime.date.today().isoformat()}. cols: sha\tpbrType\tn_emit\tn_copy\n")
    for i in sel: f.write(f"{sha[i]}\t{pbr[i]}\t{int(n_emit[i])}\t{int(n_copy[i])}\n")
print(f"\nSAVED {len(sel)} self-lit(nonzero) shas -> {OUT}")

# regenerate REAL flagged-copy sample: 20 metalness self-lit
ml=np.where((pbr=="metalness")&lit&som_nz)[0]
rng=np.random.RandomState(3); pick=rng.choice(ml,min(20,len(ml)),replace=False)
S2="/local-scratch2/xya120/studio/misc/lightgen/segvigen_emissive/direct_pilot/sanity_metal_flagged_lit.txt"
with open(S2,"w") as f:
    f.write("# SANITY (REGEN, fixed detector): 20 pbrType=metalness shapes flagged SELF-LIT (real copies)\n")
    f.write("# cols: sha\tn_emit\tn_copy  (all emitting materials are albedo-copies: same-index or corr>0.9&resid<0.05)\n")
    for i in pick: f.write(f"{sha[i]}\t{int(n_emit[i])}\t{int(n_copy[i])}\n")
print(f"regenerated {len(pick)} real metal-lit examples -> {S2}")
print("METAL_LIT_SAMPLE:", " ".join(sha[i] for i in pick[:20]))
print("DONE_MARKER")
