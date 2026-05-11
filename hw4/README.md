# HW4 — GRPO for Vision-Language Models

**Notebook:** [Homework_4_GRPO_VLMs.ipynb](Homework_4_GRPO_VLMs.ipynb) · **Writeup:** [Homework_4_GRPO_VLMs.pdf](Homework_4_GRPO_VLMs.pdf)

## Task

Train `Qwen/Qwen3-VL-2B-Instruct` with **GRPO** (Group Relative Policy
Optimization, DeepSeekMath 2024) using two rule-based binary rewards.

> Note: HW4 uses a different base model than HW3 (`Qwen3-VL-2B` vs
> `Qwen2.5-VL-3B`). The HW3 LoRA adapter is not reused.

## Problems

| Problem | Points | What |
|---|---|---|
| 1 | — | GPU + secret word |
| 2 | 10 | Reuse the HW3 dataset (`mmai-data/`) |
| 3 | 15 | Walk through GRPO conceptually |
| 4 | 25 | Implement the GRPO advantage from scratch |
| 5 | 15 | Define reward functions |
| 6 | 10 | Build the GRPO training dataset |
| 7 | 20 | Train with TRL's `GRPOTrainer` |
| 8 | 20 | Post-training evaluation |

## Rewards

| Reward | What it grades |
|---|---|
| `accuracy_reward` | Does the text after `Answer:` match ground truth? (binary) |
| `format_reward` | Did the output contain the literal `Answer:` marker? (binary) |

## Training settings

`num_generations=2`, `max_completion_length=256`, `lr=1e-5`, `100 steps`,
`batch_size=1`, `epsilon=0.2`, `temperature=0.9`, `beta=0.0`. LoRA: `r=16`,
`alpha=32`, target `q/k/v/o_proj`. bf16 + gradient checkpointing.

## Training dynamics

`format_reward` saturated near 1.0 within ~10–20 steps (the marker is easy to
emit). `accuracy_reward` climbed more slowly. With `G=2` the advantage is 0
whenever both rollouts get the same reward, so many early steps produce
near-zero gradients.

## Held-out evaluation

2 COCO samples (cat, truck). The notebook reports **100% format compliance,
50% accuracy** — cat correct, truck labelled as "highway sign" because the
test image includes a highway sign overhead and the small VL backbone latched
onto it.

![GRPO held-out example](assets/grpo_eval_example.png)
*The GRPO-trained model produces numbered chain-of-thought reasoning before
emitting `Answer:`.*

## Reproducing

[`grpo-output/`](grpo-output/) (LoRA adapter + rollout completions + a
mid-training `checkpoint-100/`) is gitignored. Re-run Problem 7 to regenerate.
The training dataset is reused from [`../hw3/mmai-data/`](../hw3/mmai-data/).
