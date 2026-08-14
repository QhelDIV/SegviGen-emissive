---
title: Team update — TEMPLATE (not a real update)
date: 2026-07-05
tldr: This is the placeholder update used to smoke-test tools/build_update.py — demonstrates a results table, an embedded figure from an existing published page, a figs/ screenshot, and a "what's next" list.
template: true
hero: figs/placeholder_screenshot.png
related:
  - finetune_binary_v1: Fine-tune data + predictions (binary emissive v1)
  - dataset_gallery_v1: Dataset statistics + gallery
---

## What happened this week

This is placeholder body text standing in for a real weekly update. A real
update would summarize what shipped, what the numbers say, and what's
blocking. It's written in plain markdown — headings, lists, tables, and
images all work the same way they would in any other markdown file.

## Results snapshot

| run | data | epoch | val IoU@0.2 | notes |
|---|---|---|---|---|
| W5 (eager) | train_1k | ep18 | 0.176 | 5x loss weight on emissive voxels |
| W1 (control) | train_1k | ep18 | 0.142 | plain loss, same schedule |
| zero-shot oracle | — | — | ~0.235 | frozen pretrained full_seg, the bar to beat |

## A figure pulled from an existing page

This image is **not copied** — it's a relative link straight into the already
published `finetune_binary_v1` page, so there is exactly one copy of the file
on disk.

![W5-EMA prediction on a real val_96 shape](../../finetune_binary_v1/d5fb4f19d4164612b165caac5471555c_pred_w5ema.png)

## A figure from this update's own figs/

This image *is* copied alongside the published page (it lives in this
update's own `figs/` directory and nowhere else).

![placeholder screenshot](figs/placeholder_screenshot.png)

## What's next

- Wire the real owner-authored update into this same convention.
- Confirm the console home lists updates newest-first with their tldr.
- Retire this template once a real update has been published successfully.
