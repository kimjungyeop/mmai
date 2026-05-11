# HW3 — Multimodal LLMs (Qwen2.5-VL LoRA fine-tuning)

**Notebook:** [Homework_3_Multimodal_LLMs.ipynb](Homework_3_Multimodal_LLMs.ipynb)
**Writeup:** [Homework_3_Multimodal_LLMs.pdf](Homework_3_Multimodal_LLMs.pdf)

---

## What this homework is about

Take a real vision-language model (Qwen2.5-VL), point it at my CMU-MOSEI
comfort task, and:

1. Build an instruction-tuning dataset out of segment frames + transcripts.
2. Run **baseline inference** with the off-the-shelf model on held-out frames.
3. Try **prompt engineering** to lift baseline performance.
4. **LoRA fine-tune** on my dataset, sweeping learning rate.
5. **Re-evaluate** post-training and compare against the prompt-engineered baseline.

This is the first homework where I'm working with a model that already "knows"
things about images and language — the question shifts from *can the model
learn at all?* (HW2) to *how do I push an already-capable model toward my
specific task without wrecking it?*

## Problems

| Problem | Points | What |
|---|---|---|
| 1 | — | GPU verification + secret word |
| 2 | 20 | Build the (image, instruction, label) dataset from CMU-MOSEI frames |
| 3 | 10 | Baseline inference on 4 held-out images |
| 4 | 15 | Prompt engineering on the same held-out set |
| 5 | 20 | LoRA fine-tuning (the main event) |
| 6 | 30 | Post-training evaluation vs the baseline |
| 7 | 10 | Final reflection |

## Dataset shape

[`mmai-data/`](mmai-data/) — the small (≈4 MB) image+JSONL dataset I generated
from CMU-MOSEI:

- `mmai-data/images/` — ≈thousand frames cropped from segments where the
  sentiment signal was strong (|sentiment| ≥ 1).
- `mmai-data/data.jsonl` — train split, one record per frame:
  `{image, prompt, response}` triples.
- `mmai-data/test_data.jsonl` — held-out test split.

Keeping this in git (rather than gitignored) makes the notebook reproducible
end-to-end without re-downloading CMU-MOSEI.

## LR sweep — what's in each checkpoint folder

| Folder | LR | Notes |
|---|---|---|
| `qwen2_5_vl_lora_lr2e-4/` | 2e-4 | **Best** — cleaner convergence, picked as `qwen2_5_vl_lora_best` symlink |
| `qwen2_5_vl_lora_lr5e-4/` | 5e-4 | More aggressive; faster early loss drop but overshot on eval |

Both are **gitignored** (LoRA adapters are ~70 MB each). To regenerate, run
Problem 5 of the notebook on an A100.

## What was interesting

- The off-the-shelf Qwen2.5-VL is already surprisingly OK at coarse facial
  affect — the baseline isn't terrible even with no fine-tuning.
- Prompt engineering bought a meaningful jump *and* exposed how brittle the
  framing is: small wording changes flip predictions.
- After LoRA, the model gets noticeably more consistent on borderline frames,
  but I had to be careful not to push LR high enough to over-confidence it on
  the training distribution.

## Reproducing

- Adapter folders are gitignored. Re-run Problem 5 with the LR you want; the
  notebook writes to a `qwen2_5_vl_lora_lr{LR}/` directory.
- The fine-tuned adapter from this homework is the **starting checkpoint for
  HW4 (GRPO)** — see [../hw4/README.md](../hw4/README.md).
