# HW1 — Multimodal Data Preprocessing

> *"Before we start directly processing data, let's think about a project
> objective… extract a set of modalities from the dataset of your choice that is
> rich and contains unique information from other modalities."* — assignment

**Notebook:** [mmai_HW1.ipynb](mmai_HW1.ipynb) · **Writeup:** [mmai_HW1.pdf](mmai_HW1.pdf)

---

## The task I picked (and stuck with all semester)

**Predict how comfortable a person appears in a video** — a binary classification
between *Comfortable* and *Uncomfortable*. Motivation: human-robot interaction.
A robot that has to behave appropriately around a person needs a fast, noisy
read on whether the person is at ease.

I map CMU-MOSEI sentiment (-3..+3) to:

- positive sentiment → **Comfortable**
- negative sentiment → **Uncomfortable**

Sentiment is an imperfect proxy for comfort, but it gave me thousands of labeled
multimodal segments to work with — the alternative (collecting my own
robot-interaction footage) wasn't realistic for HW1.

## Datasets considered

- **CMU-MOSEI** (chosen) — 23,500+ YouTube monologue segments with sentiment
  scores, plus per-frame visual features and per-word text/audio alignment.
- **CMU-MOSI** — smaller, similar format, kept as a backup.
- **IEMOCAP** — dyadic conversations, closer to robot-style interaction but
  harder to access.

## Modalities I extracted

1. **Visual** — frames sampled at 1 fps from each video segment, downsampled to
   64×64 grayscale (4,096-d).
2. **Text** — segment transcripts, encoded as the mean of GloVe-300 word vectors.
3. **Audio** — COVAREP features (74-d per timestep), mean-pooled over each segment.

All three modalities are aligned at the **segment** level via the CMU-Multimodal
SDK timestamps.

## What was hard

- **Video availability** — many original CMU-MOSEI YouTube IDs have been deleted
  since 2018; I had to iterate through ~30 candidates to pull ~20 usable videos
  with `yt-dlp`.
- **Resolution trade-off** — 64×64 grayscale keeps features tractable but throws
  away color and texture; a real model would want a CNN feature extractor on
  larger crops.
- **Temporal misalignment** — frames at ~30 fps vs word-level text. Averaging
  GloVe vectors per segment loses word order; I noted this as a thing to fix in
  HW2/HW3.

## Visuals

| | |
|---|---|
| ![sample frames](assets/sample_frames.png) | ![data distribution](assets/data_distribution.png) |
| Sampled frames per video at 1 fps | Class balance after sentiment→comfort mapping |

![input distribution](assets/input_distribution.png)
*Distribution of per-modality feature vectors after extraction.*

## Reproducing

The 30 GB `cmumosei_data/` H5 dump and 388 MB `cmumosei_videos/` directory are
**gitignored**. To rebuild:

1. Download the CMU-Multimodal SDK and the CMU-MOSEI standard fold from
   `https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK`.
2. Run the data-download cells in `mmai_HW1.ipynb` (uses `yt-dlp`).
3. Frame extraction is the loop in the "Frame extraction" section.

The MultiBench checkout (`MultiBench/`) is used starting in HW2 and is also
gitignored — clone from `https://github.com/pliang279/MultiBench`.
