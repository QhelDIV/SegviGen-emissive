# Binary emissive segmentation — the new mission (2026-07-01)

Status: active
TL;DR: Fine-tune SegviGen full-seg for binary black/white emissive masks, instead of the earlier multi-color/zero-cond emissive experiments — plan still being drafted.

After the team discussion on 2026-06-25, the assignment for this project
changed focus. Instead of continuing the autonomous zero-cond emissive
fine-tune as an open-ended ablation, the task is now concrete:

**Fine-tune the SegviGen full-segmentation model for binary segmentation** —
prepare our data into SegviGen's expected format, but collapse the target to
only two colors: **white = emissive region, black = non-emissive**. This
narrows scope from "predict emissive-vs-not as one part among many possible
segmentations" to a single, fixed binary task.

This does not throw away the autonomous run — it's the evidence base that
motivated the pivot. From `segvigen_emissive/WORKLOG.md`'s summary:

- The zero-cond (no DINOv3 image conditioning) ablation topped out around
  **0.176–0.203 val IoU**, below the DiffusionNet geometry-only baseline of
  **0.259**.
- The drop between ep4 (0.176) and ep8 (0.063) on the 512-sample run is **not
  classical overfitting** — train IoU dropped too (0.179→0.145) — it's
  **majority-class collapse**: with only ~11% of voxels emissive, continued
  training pushes the model toward predicting "nothing is emissive."
- This was visually confirmed: on a sample that is 55% emissive by ground
  truth, the model predicted almost entirely non-emissive.
- Architecture is not the limiter — the flow already conditions on both the
  shape latent and the material/PBR latent (base color, metallic, roughness),
  so appearance information reaches the model; DiffusionNet lacked this.
- The pretrained `full_seg` checkpoint's zero-shot outputs (see the published
  visual pages) get the coarse part structure right but under-segment versus
  the artist decomposition — reasonable prior to fine-tune from.

**Status of the plan:** under discussion. The concrete data-prep spec (where
the binary target lives in SegviGen's input/output format) and training
config (data scale, class-imbalance handling, real vs. zero image-cond) are
not yet decided — see the "Open decisions waiting on you" list in `AGENTS.md`.
