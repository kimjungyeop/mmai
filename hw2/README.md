# HW2 — Multimodal Fusion & Alignment

**Notebook:** [Homework_2_Multimodal_Fusion_and_Alignment.ipynb](Homework_2_Multimodal_Fusion_and_Alignment.ipynb)
**Writeup:** [Homework_2_Multimodal_Fusion_and_Alignment.pdf](Homework_2_Multimodal_Fusion_and_Alignment.pdf)
**Earlier draft (kept for reference):** [Homework_2_Multimodal_Fusion_and_Alignment_v1.ipynb](Homework_2_Multimodal_Fusion_and_Alignment_v1.ipynb)

---

## What this homework is about

Two stacked questions:

1. **Reading reflection** on *align-before-fuse* (Li et al., 2021) and the
   *Platonic Representation Hypothesis* (Huh et al., 2024) — what do they
   imply for my CMU-MOSEI comfort task?
2. **Hands-on fusion on AV-MNIST** — a controlled benchmark where I implement
   and compare every classical fusion strategy, then a contrastive alignment
   variant, before applying the insight back to my own task.

AV-MNIST is the right benchmark here because it strips away every confound
except the fusion strategy itself: clean audio + image of the same digit, no
domain shift, no noisy labels. Whatever wins on AV-MNIST is what I'd start with
on CMU-MOSEI.

## Problems and what I did

| Problem | Points | What | My result |
|---|---|---|---|
| 1 | 5 | Tensor exercises | warmup |
| 2 | 5 | Einsum exercises | warmup |
| 3 | 10 | Unimodal baselines (audio-only, image-only) | both train cleanly, image > audio |
| 4 | 10 | Early-fusion multimodal baseline | beats either unimodal, as expected |
| 5 | 30 | **Early / Late / Tensor / Low-rank tensor (LMF) fusion** comparison | all four implemented end-to-end, head-to-head |
| 6 | 30 | **Contrastive alignment** of the two modalities before fusion | improves convergence & top-line accuracy |
| 7 | 10 | Reflection — which fusion fits CMU-MOSEI | bias toward LMF for parameter efficiency, with contrastive pretraining |

## Headline plots

![fusion comparison](assets/fusion_comparison_bars.png)
*Final test accuracy across the four fusion strategies on AV-MNIST.*

![fusion convergence](assets/fusion_convergence.png)
*Training curves: tensor fusion converges fastest but LMF generalizes best at
a fraction of the parameters.*

![contrastive alignment](assets/contrastive_alignment.png)
*Effect of contrastive alignment as a pre-fusion stage.*

## Tying it back to HW1

For the CMU-MOSEI comfort task I argued for:

- **Temporal alignment first** (segment-level timestamps from the SDK) — without
  this, any fusion is mixing misaligned signals.
- **Low-rank tensor fusion (LMF)** as the default — full tensor fusion blows up
  with 3 modalities × non-trivial feature dims; LMF preserves the
  cross-modal interactions at tractable cost.
- **Optional contrastive pre-alignment** on (frame, transcript) pairs before
  fusion, since AV-MNIST showed a measurable gain.

## Reproducing

The AV-MNIST data and `MultiBench/` clone are pulled by the notebook's
"Getting repo" and "Getting AV-MNIST dataset" cells. Both are gitignored to
keep this repo light.
