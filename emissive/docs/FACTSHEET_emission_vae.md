# Fact sheet: Dongchen's two emission VAEs
# Assembled 2026-08-06 by the master session, read directly from his configs,
# checkpoints, and his own visualization PNGs.
# EVERY number below was read off a file or an image. Do not invent, re-round,
# or extrapolate. Where something is unverified it says so; keep those caveats
# on the page rather than resolving them.

## 0. Why this page exists

Our emissive segmentation eval decodes ground truth through TRELLIS.2's
PRETRAINED tex decoder. Dongchen separately fine-tuned two VAEs specifically for
emission. If those fine-tuned VAEs still cannot reconstruct emission, then no
model trained in that latent space can produce it, and the ~0.1 IoU seen across
four unrelated architectures (TEXGen, two TRELLIS.2 variants, SegviGen) has a
common cause upstream of every one of them.

## 1. The two designs

Both live at `/3dlg-jupiter-project/lightgen/trellis2_bw/latents_v2/vae_ckpts/`.
Both use `SparseUnetVaeEncoder` / `SparseUnetVaeDecoder`, latent_channels 32,
model_channels [32, 64, 128, 256, 512], SparseConvNeXtBlock3d.

### A. `albedo2emission`  (the channel-substitution design)
- in_channels 6, out_channels 6
- Trainer `EmissionPbrFinetuneTrainer`, max_steps 35,800
- **Initializes BOTH encoder and decoder** from TRELLIS.2:
  `tex_enc_next_dc_f16c32_fp16.safetensors` and
  `tex_dec_next_dc_f16c32_fp16.safetensors`
- lr_scheduler CosineAnnealingLR, T_max 10,000, eta_min 1e-6
- checkpoint `step0034800-0.0056.ckpt`, 13,258,637,734 bytes (~13.3 GB)
- This is the design §4's `sec:vae_finetune` describes: emission RGB occupies
  the albedo slot, other material slots pinned, so input and output shapes match
  the pretrained model exactly.

### B. `pbr2emission`  (the emission-only design)
- in_channels 3, out_channels 3
- Trainer `EmissionVaeTrainer`, max_steps 160,000
- Encoder from `pretrained_pbr_encoder` (same TRELLIS.2 tex encoder); no
  decoder init listed in the config
- No lr_scheduler in the config
- checkpoint `step0154600-0.0088.ckpt`, 2,058,711,934 bytes (~2.1 GB)

### Shared training settings (identical in both configs)
- AdamW lr 1e-4, weight_decay 0.0
- loss_type l1, lambda_kl 1.0e-07
- ema_rate 0.9999, fp16_mode inflat_all
- AdaptiveGradClipper max_norm 1.0, clip_percentile 95
- batch_size_per_gpu 32, dataloader_num_workers 8
- **use_balanced_sampler: true**
- monitor_metric val/loss, save_top_k 1
- dataset `LightGenSLatEmission`, data_root `data/lightgen_74k_newbake`,
  **resolution 256**, pbr_attrs [base_color, metallic, roughness, alpha],
  split json `data_splits_emissive_74k_stratified_newbake_vae.json`

## 2. Results, read off his own visualizations

### 2A. Ten validation samples (`emission_vae_10sample_vis.png`)
Columns: GT | VAE Recon | Difference | per-channel L1. Per-sample L1, in order:

  0.0066, 0.0277, 0.0092, 0.5031, 0.0154, 0.1247, 0.0082, 0.0153, 1.3856, 0.0477

Three failure modes are visible:

- **The emissive region vanishes entirely (5 of 10).** A thin white bar
  (L1 0.0277), a small red ellipse (0.0092), a blue speck (0.0082), a white bar
  (0.0153), and a grey rectangle (0.0477) all reconstruct to essentially pure
  black. Note these are among the LOWEST L1 values in the set.
- **Color collapses toward orange/brown.** A white, green and pink flower
  reconstructs as an orange starburst (0.5031). A white and blue diagonal shape
  becomes a small orange blob in the wrong location (0.0154). A yellow blob
  becomes dim orange (0.1247).
- **The worst case is a large bright region.** A solid white disc reconstructs
  as an orange-brown disc with concentric rings (L1 1.3856).

One clean success: a ring of orange dots (0.0066).

### 2B. Single-sample overfit (`emission_vae_overfit_vis.png`)
Sample `0007deb6...`, 500 steps, lr 0.001, final L1 **0.0097**.
Panel text: Voxels 303,573 | GT emission mean 0.0506 | Rec emission mean 0.0588.
Shows XY, XZ, YZ projections, columns Input (GT) | Reconstructed | |Error| |
Overlay (G=GT, R=Rec).

**This is the most important panel on the page.** The subject is a birthday cake
with lit candles. The bright candle flames ARE reconstructed. But the
reconstruction also spreads a dark red wash across the ENTIRE cake body, which
is pure black in the ground truth, and the |Error| panel shows error over the
whole object silhouette rather than only at the emissive parts. Reconstructed
emission mean is 16% above ground truth (0.0588 vs 0.0506), consistent with that
leak.

The significance: this is a SINGLE sample memorized for 500 steps, the easiest
possible setting. The VAE still cannot output exact zeros.

### 2C. End-to-end test (`emission_vae_optB_test.png`, titled "Emission VAE E2E Test")
Same sample. L1 recon **0.006644**. Panel text:
  Emission voxels 303,573
  Emission latent torch.Size([1009, 32]) at res ~16
  PBR latent      torch.Size([1009, 32]) at res ~16
Columns: GT Emission | Reconstructed | |Error| | Emission Latent (mean over C) |
PBR Material Latent (mean over C). The |Error| panel again shows error spread
across the whole object, not localized to emissive regions.

## 3. A resolution question this ANSWERS (worth stating on the page)

Our notes disagreed on whether the SLAT latent is spatially 16^3 or 32^3. Both
are correct, for different pipelines: **the latent grid is input_resolution / 16**.
Dongchen's VAE runs at resolution 256, so his latent is ~16^3 (confirmed by the
"at res ~16" annotation in 2C). Our direct-ovoxel pipeline runs at 512, so ours
is 32^3. The discrepancy was two pipelines, not an error.

## 4. Interpretation (LABEL THIS AS INTERPRETATION, not measurement)

- **The reported losses are misleading because emission is sparse.** Emission is
  roughly 97.5% black by voxel count (our own sample: median emissive_frac
  0.025). A reconstruction that outputs black everywhere earns a very low L1.
  The clearest evidence is internal to his own figure: the five samples that
  reconstruct to nothing carry L1 values of 0.0082 to 0.0477, at or below the
  checkpoint-name losses of 0.0056 and 0.0088.
- **The failure is representational, not an optimization shortfall.** The
  single-sample overfit shows it, and overfitting one sample removes
  generalization from the picture entirely.
- **This plausibly caps every downstream model.** If the round trip destroys
  sparse emission, a DiT trained in that latent space cannot recover it, which
  would explain four unrelated architectures landing near 0.1 IoU.

## 5. What is NOT verified (keep these on the page)

- **Which VAE produced which figure.** The filenames do not say, and neither do
  the panel titles. Do not attribute a figure to `albedo2emission` or
  `pbr2emission` on the page.
- **Whether the figures come from the stored checkpoints.** Most per-sample L1
  values in 2A exceed both checkpoint-name losses (0.0056, 0.0088), so the
  panels may predate the saved checkpoints.
- **The exact projection reduction** (max or mean along each axis) in 2B and 2C.
- **The color space** of the visualizations, and whether any tone mapping is
  applied before display.
- Whether our own eval could use either fine-tuned decoder in place of the
  pretrained one. That is the open question this page should end on, not answer.

## 6. Source files (for the provenance block)

Configs and checkpoints:
  trellis2_bw/latents_v2/vae_ckpts/albedo2emission/{config.yaml,step0034800-0.0056.ckpt}
  trellis2_bw/latents_v2/vae_ckpts/pbr2emission/{config.yaml,step0154600-0.0088.ckpt}
Figures (all under trellis2_bw/code_snapshot/):
  emission_vae_10sample_vis.png, emission_vae_overfit_vis.png,
  emission_vae_optB_test.png, emission_vae_e2e_test.png, vae_channel_vis.png
Related training code under trellis2_bw/code_snapshot/data_toolkit/:
  vis_emission_vae.py, test_pbr_vae.py, encode_emission_pbrfinetune_10samples.py,
  predict_emission_voxels_twostream.py
