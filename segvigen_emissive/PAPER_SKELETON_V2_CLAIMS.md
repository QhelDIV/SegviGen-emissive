# LightGen paper skeleton v2: the claim chain
# Written 2026-08-06. Format follows somages' `.skel` component: one atomic
# claim per line, numbered, grouped under uppercase heads.
#
# RULES FOR THE BUILDER
#  - Each numbered line is ONE sentence and becomes one sentence or one short
#    paragraph in the paper. Do not merge, split, or expand them.
#  - Do not add claims. Do not add hedges that are not written here.
#  - Claims are stated as the paper's TARGET. The owner has decided this
#    explicitly; results catch up later. Do not soften them.
#  - `def` = bold, a definition or the central claim of its section.
#  - `open` = italic, uncounted, a question we have not settled.
#  - Numbers appear only where they are verified. Anything absent is absent
#    on purpose.

---

HEAD: Motivation

1. TRELLIS.2 and other 3D generation models produce geometry and PBR material, and none of them produce emission.
2. A generated lamp therefore reflects light correctly and emits nothing.
3. Emission is what lets an object light a scene rather than only be lit by it, and emissive assets are the ones that carry a scene.
DEF: We generate emission for 3D assets, as a stage that composes with an existing image-to-3D pipeline.

HEAD: Why it is hard

4. Emissive assets are rare relative to general 3D data, and the labels that exist are unreliable.
5. In our dataset a large group carries an emissive texture identical to its base color, so the whole object is labeled emissive while looking ordinary.
6. Median emissive coverage is 0.025 against a mean of 0.244, and 22.9 percent of shapes are more than half emissive.
7. Emission is not only a texture, because emitted radiance is unbounded while reflectance is not.
8. Predicting where an object emits and in what color is insufficient without predicting how brightly.
9. Evaluation cannot judge the object alone, because a texture compared in isolation does not show whether the object lights a room correctly.
10. An emissive asset has to be placed in an environment and judged by what it illuminates.

HEAD: What we do

11. We solve these problems and achieve emission generation on top of TRELLIS.2.
12. Given an image, we generate a 3D shape with emission.
13. Given an existing 3D shape, we generate emission for it.
14. The second mode requires no reference image of the object glowing, which is the only setting available inside a generation pipeline.

HEAD: Relation to concurrent work

15. EmissionGen addresses emission texture generation, so we do not claim the problem is untouched.
16. Its input reference image already shows the object glowing, while we predict emission from geometry and material alone.
17. It works multi-view and fuses to UV, while we operate natively in TRELLIS.2's sparse 3D latent.
18. It textures an existing mesh from a photograph, while we add a stage to image-to-3D generation.

HEAD: Dataset

19. We build from TexVerse, with 74,503 shapes in the split and 72,546 built.
20. Of the 1,957 not built, 1,036 never had source data, 584 came from a rebake that produced no output, and 337 remain buildable.
21. Emissive coverage is bimodal, and the gap between its median and its mean is the label noise rather than a property of emission.
22. A filter removing full-bright and imperceptible cases retains 32,121 shapes, of which 32,050 are built.
23. We train unfiltered first so the baseline covers everything, and hold filtering as an ablation.
24. Geometry agrees with an independently produced bake to 0.119 voxel widths in the mean over 72,092,657 vertex pairs.

HEAD: Method

DEF: We predict a per-voxel binary emissive mask and take the emissive texture to be the input albedo restricted to that mask.
25. The model never generates emission color, because the color is already present in the PBR input.
26. It only has to decide where the object emits.
27. Generating emission as a continuous texture in this latent fails in a documented way: sparse regions reconstruct to black, colors collapse toward orange, and nonzero emission leaks across surfaces that should be exactly zero.
28. That leak persists when a single sample is memorized for 500 steps, which makes it representational rather than an optimization shortfall.
29. Predicting a mask and reusing the input albedo avoids the failure entirely.
OPEN: The mask times albedo premise is not yet validated against ground-truth emission.

HEAD: Model

30. We fine-tune TRELLIS.2's slat_flow_imgshape2tex, warm started from SegviGen's full_seg checkpoint.
31. That checkpoint is already fine-tuned for 3D part segmentation in the same latent space, so we begin from a segmentation prior rather than a texture-generation prior.
32. The binary target is written into the pretrained encoder's base color slot with the remaining material slots pinned to constants.
33. Input and output shapes match the pretrained model exactly, so the encoder is reused unchanged and no new VAE is trained.
34. Emission is a third stage, after geometry and after PBR.
35. It is separate because emissive assets are too rare to retrain the PBR stage on, because emission must be exactly zero almost everywhere while reflectance is dense, and because a separate stage drops into an existing pipeline without regenerating anything.
OPEN: A joint PBR and emission model has not been run, and it is the ablation a reviewer will ask for.
OPEN: Current runs use zero image conditioning while the baselines receive a thumbnail, so the comparison is not yet like for like.

HEAD: Evaluation

36. We report mask IoU as the mean over K independent draws with the draw standard deviation.
37. A single draw is unreliable, because our own per-epoch validation moved between 0.0008 and 0.1499 on adjacent epochs.
38. Best of K appears only as a labeled oracle bound, because without a selector it is not obtainable at inference and it rewards variance.
39. The threshold is fixed on validation and never swept on the test set.
40. Every metric is stratified by emissive coverage, because a flat mean over a bimodal distribution is decided by its degenerate cases.
41. Emission quality is judged by rendering the asset as the only light source and comparing what it illuminates.

HEAD: Results

42. We compare against TEXGen in UV space, and against TRELLIS.2 with albedo replaced by emission and with all PBR replaced by a single emission channel.
43. DiffusionNet is excluded because it is too restricted by mesh topology.
OPEN: The quantitative table is not yet written, because all four current models sit near 0.1 IoU, which points at a cause shared upstream of every architecture rather than at a comparison between them.
OPEN: The qualitative figure is decided in form, object thumbnail then ground truth then prediction then difference across the coverage range, and not yet in content.
