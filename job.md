Target:
I would like to train diffusionnet for mesh segmentation (trained on emissive mask)
So at a larger picture, the input is a triangle mesh. Output is a point cloud / mesh mask hinting emissive region.

Context: latex paper: /localhome/xya120/studio/misc/lightgen/lightgen_overleaf
Dongchen is working on using TEXGen(/localhome/xya120/studio/misc/lightgen/TEXGen) and TRELLIS to predict emissive textures.
Where my (XG) goal is to train diffusionnet as a baseline.


Dongchen: (slack message)
emission mask的datalloader在这里，就是加了个threshold https://github.com/dongchen-yang/TEXGen/blob/e980afee4e95e3131f24ae90f2e06744fe44c936/spuv/data/lightgen_uv.py#L247

1k sample的parquet:
/cs/3dlg-jupiter-project/lightgen/data/baked_uv_local_subset/df_SomgProc_emission_filtered.parquet

overfit 1 sample 的split:
/cs/3dlg-jupiter-project/lightgen/data/baked_uv_local_subset/overfit_split_single.jsonoverfit 10 sample的split:
/cs/3dlg-jupiter-project/lightgen/data/baked_uv_local_subset/overfit_split_10.json1k sample 的split:
/cs/3dlg-jupiter-project/lightgen/data/baked_uv_local_subset/data_splits_emission_filtered.json
