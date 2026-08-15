title: Decode the training targets: is the model being taught albedo instead of emission
executor: target-check
track: research
status: done
started: 2026-08-10 16:50
updated: 2026-08-10 19:27
slurm: 
link: 
page: target_check
upstreams: overfit_ct10
motivation: The owner noticed overfit predictions glow in each shape's own albedo colors while matching nothing in the emission ground truth, which points at the dataset builder feeding the wrong target; if true, the entire multi-shape training failure is a data bug, not a capacity limit.
log:
- 2026-08-10 16:50 [overfit-test] Job started.
- 2026-08-10 19:03 [master] Reassigned to a dedicated runner: the original executor kept it queued behind the epoch 8 evaluation and asked for the handoff. Its urgency has grown: the epoch 8 page shows the false glow strengthening with training, which is what a model faithfully learning a wrong target would do.
- 2026-08-10 19:17 [target-check] REFUTED via GPU decode (job 242949, cs-venus-08): output_tex_slat.pth decodes to the binary emissive mask (corr 0.94-0.999 vs raw GT mask) not albedo (corr low/inconsistent, flat R=G=B pattern = achromatic signature) on all 5 probed shapes (both ct10 red cars, single-shape control, 2x train_72k). Positive control (decode input_tex_slat.pth vs raw albedo) hits 0.94-0.998, proving the decode path is not masking a real leak. Code reading (build_dataset_direct.py, vxz_to_slat.py, train_emissive.py) independently confirms no swap: builder writes binary mask to output.vxz, unmodified encoder writes it to output_tex_slat.pth, trainer uses otx (not itx) as the flow target. NOT a data bug. The red-car-glowing-in-own-albedo symptom must be downstream: model behavior (cond/input_tex_slat appearance leak) or an eval/viz attribution bug, not chased further here.
- 2026-08-10 19:17 [target-check] Target-swap hypothesis REFUTED by direct GPU decode of on-disk output_tex_slat.pth against raw pre-encode ground truth (5 shapes: ct10 x2, single-shape control, train_72k x2). Dataset builder, encoder, and trainer all correctly use the binary emission mask as the training target, not albedo. The albedo-glow symptom is real but is not a data bug; next step (not chased here) is the model's learned behavior or the eval/viz pipeline.
- 2026-08-10 19:27 [master] Addendum closing the last ambiguity: an epoch 8 prediction file for the zero-emission hot dog was checked directly. The prediction is a single scalar mask value per voxel with no color channels, so the albedo-colored glow in renders is the baseline's own mask-times-albedo compositing, not a visualization bug and not appearance leaking into the output. The model's error is purely where the mask fires: on this shape it marks essentially the whole surface emissive at full confidence. The symptom is model behavior, fully attributed.
outcome: Target-swap hypothesis REFUTED by direct GPU decode of on-disk output_tex_slat.pth against raw pre-encode ground truth (5 shapes: ct10 x2, single-shape control, train_72k x2). Dataset builder, encoder, and trainer all correctly use the binary emission mask as the training target, not albedo. The albedo-glow symptom is real but is not a data bug; next step (not chased here) is the model's learned behavior or the eval/viz pipeline.
