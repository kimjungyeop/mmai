# HW3 — Multimodal LLMs (Qwen2.5-VL LoRA fine-tune)

**Notebook:** [Homework_3_Multimodal_LLMs.ipynb](Homework_3_Multimodal_LLMs.ipynb) · **Writeup:** [Homework_3_Multimodal_LLMs.pdf](Homework_3_Multimodal_LLMs.pdf)

## Task

Fine-tune `Qwen/Qwen2.5-VL-3B-Instruct` on the HW1 comfort task. Held-out set:
4 frames × 3 question types (sentiment, emotion, comfort) = 12 QA pairs.
Train/test split is by video (5 of 26 videos held out) so frames from the same
speaker don't leak between splits.

## Problems

| Problem | Points | What |
|---|---|---|
| 1 | — | GPU + secret word |
| 2 | 20 | Build the (image, question, answer) dataset |
| 3 | 10 | Baseline inference on the held-out images |
| 4 | 15 | Prompt engineering — 5 strategies |
| 5 | 20 | LoRA fine-tuning |
| 6 | 30 | Post-training evaluation |
| 7 | 10 | Final reflection |

## Dataset

[`mmai-data/`](mmai-data/) — committed (~4 MB total):

- `images/` — frames from HW1, cropped to single faces.
- `data.jsonl` — train split with `{image, question, answer}` records.
- `test_data.jsonl` — held-out test split.

## Headline numbers

| Model | Held-out acc |
|---|---|
| Pre-trained baseline (Problem 3) | 4 / 12 (33%) |
| Prompt-engineered (best of 5 strategies) | 50% on the 2 tested images |
| LoRA, LR = 2e-4 (**selected**) | 6 / 12 (50%) |
| LoRA, LR = 5e-4 | 4 / 12 (33%) |

Chain-of-thought prompting scored 0% — when asked to describe before
classifying, the model confabulated details about the low-resolution image and
reasoned from them.

LoRA hyperparameters (both runs): `r=8`, `alpha=16`, `dropout=0.05`,
`target=q_proj/k_proj/v_proj/o_proj`, `num_epochs=10`, `batch_size=2`,
`grad_accum=4` (effective batch 8), `shortest_edge=288`. Trainable params:
3.69 M (0.098% of the base model). Each run took ~35 minutes.

## Headline result

![LoRA comparison](assets/lora_comparison.png)
*Pre/post comparison. The LR=2e-4 fine-tune improved on the baseline by
correcting one positive-sentiment video, gaining one comfort prediction, and
eliminating freeform emotion labels in favor of the trained vocabulary. Both
baseline and fine-tuned still miss the two negative-sentiment images — the
training set has only 48/462 negative samples.*

## Reproducing

Adapter folders (`qwen2_5_vl_lora_lr2e-4/`, `qwen2_5_vl_lora_lr5e-4/`,
symlink `qwen2_5_vl_lora_best`) are gitignored. Re-run Problem 5 to regenerate.
