# Multimodal AI — Course Portfolio

**MAS.S60 / 6.S985 · MIT · Spring 2026**
**Author:** Jungyeop Kim (`kimjungyeop@gmail.com`, `jungyeop@mit.edu`)

This repo holds all five homework assignments and (later) the final project for
the *Multimodal AI* class. Each `hwN/` folder has its own README with the
details for that assignment.

## Highlights

<table>
<tr>
<td width="50%"><img src="hw1/assets/sample_frames.png" alt="HW1 sampled frames"/><br/><b>HW1 — CMU-MOSEI preprocessing.</b> 1 fps frames from 26 YouTube videos, used as the visual modality for a binary <i>Comfortable / Uncomfortable</i> task.</td>
<td width="50%"><img src="hw2/assets/fusion_comparison_bars.png" alt="HW2 fusion comparison"/><br/><b>HW2 — Fusion comparison on the HW1 dataset.</b> All multimodal fusions reached 100% because the text-only model was already 100% (26 unique videos = 26 unique GloVe-averaged text embeddings).</td>
</tr>
<tr>
<td width="50%"><img src="hw3/assets/lora_comparison.png" alt="HW3 LoRA results"/><br/><b>HW3 — LoRA fine-tune of Qwen2.5-VL-3B.</b> Held-out accuracy: baseline 4/12 (33%) → LoRA @ LR=2e-4 6/12 (50%). LR=5e-4 dropped back to 4/12.</td>
<td width="50%"><img src="hw4/assets/grpo_eval_example.png" alt="HW4 GRPO output"/><br/><b>HW4 — GRPO on Qwen3-VL-2B.</b> After 100 steps the format reward saturated near 1.0; the held-out test (2 COCO samples) shows 100% format compliance, 50% accuracy.</td>
</tr>
<tr>
<td width="50%"><img src="hw5/assets/part4_architecture.png" alt="HW5 agent architecture"/><br/><b>HW5 — RestaurantFeatureBot.</b> A smolagents agent that answers "does <i>X</i> in Cambridge have <i>Y</i>?" using web search + image search + a Qwen2.5-VL-3B verification tool.</td>
<td width="50%"><img src="hw5/assets/discord_interaction.png" alt="HW5 Discord interaction"/><br/><b>HW5 — Live in the class Discord.</b> Bot triggers on @mention or the keyword <code>restaurant:</code>.</td>
</tr>
</table>

## Layout

```
hw1/   CMU-MOSEI preprocessing — pick a task, extract modalities
hw2/   Fusion & alignment — fusion variants on the HW1 dataset
hw3/   Multimodal LLMs — Qwen2.5-VL-3B LoRA fine-tune
hw4/   GRPO for VLMs — RL fine-tune of Qwen3-VL-2B with rule-based rewards
hw5/   AI Agents — smolagents + Langfuse + Discord
final-project/   placeholder
```

HW1–HW3 share the same dataset (CMU-MOSEI comfort task). HW4 reuses the HW1
dataset for training but switches to a different base model and is evaluated
on COCO probes. HW5 is a separate task (restaurant feature lookup), not a
continuation of the comfort task.

Large artifacts (the 30 GB CMU-MOSEI dump, 8 GB MultiBench checkout, LoRA / GRPO
adapters, `.venv`s) are not tracked — see `.gitignore`. Each homework README
notes what to re-download.
