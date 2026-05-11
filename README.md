# Multimodal AI — Course Portfolio

**MAS.S60 / 6.S985 · MIT · Spring 2026**
**Author:** Jungyeop (`jungyeop@mit.edu`)

This repository is my living lab notebook for the *Multimodal AI* class: five homework
assignments plus the final project, all developed around a single threaded idea —
**predicting how comfortable a person appears in a video** — and progressively
layered with new techniques each week.

I picked one task and one dataset (CMU-MOSEI, binary
*Comfortable* vs *Uncomfortable*) and carried it through every homework so I could
watch the same problem under increasingly capable models: raw modality extraction
→ fusion & alignment → vision-language fine-tuning → reinforcement learning →
agentic behavior. The result is a portfolio that reads as a continuous story
rather than five disconnected exercises.

---

## Repository layout

```
.
├── hw1/   Multimodal data preprocessing — CMU-MOSEI ingestion + modality extraction
├── hw2/   Fusion & alignment — AV-MNIST, early/late/tensor/LMF fusion, contrastive
├── hw3/   Multimodal LLMs — Qwen2.5-VL LoRA fine-tuning on my comfort task
├── hw4/   GRPO for VLMs — RL-tuning the HW3 model with rule-based rewards
├── hw5/   AI Agents in the Wild — smolagents + vision + Discord + Langfuse traces
├── final-project/   (in progress — see folder README)
└── README.md
```

Large artifacts (the 30 GB CMU-MOSEI dump, 8 GB MultiBench checkout, LoRA / GRPO
adapters, `.venv`s) are not tracked — see `.gitignore`. Each homework README lists
exactly what to re-download to reproduce.

---

## The assignments at a glance

| # | Topic | What I built | Key artifact |
|---|---|---|---|
| 1 | Data preprocessing | Hand-built CMU-MOSEI pipeline: 20 videos → frames + GloVe text + COVAREP audio, all aligned to segment-level labels | [hw1/assets/sample_frames.png](hw1/assets/sample_frames.png) |
| 2 | Fusion & alignment | AV-MNIST with early / late / tensor / LMF fusion + contrastive alignment study | [hw2/assets/fusion_comparison_bars.png](hw2/assets/fusion_comparison_bars.png) |
| 3 | Multimodal LLMs | LoRA fine-tune of Qwen2.5-VL on my comfort task; LR sweep (2e-4 vs 5e-4) | [hw3/Homework_3_Multimodal_LLMs.pdf](hw3/Homework_3_Multimodal_LLMs.pdf) |
| 4 | GRPO / RL | Implemented GRPO advantage from scratch, trained the HW3 adapter further with rule-based rewards | [hw4/Homework_4_GRPO_VLMs.pdf](hw4/Homework_4_GRPO_VLMs.pdf) |
| 5 | AI Agents | smolagents-based vision agent, Discord integration, Langfuse traces, online eval | [hw5/assets/discord_interaction.png](hw5/assets/discord_interaction.png) |

Each `hwN/` folder has its own README that gives the full context, results,
and reproduction notes for that assignment.

---

## Through-line: the comfort-prediction task

I committed to one concrete task in HW1 and never changed it. The thread:

- **HW1** — frame the problem, ingest CMU-MOSEI, define *Comfortable* vs
  *Uncomfortable* from sentiment polarity, extract three modalities.
- **HW2** — try AV-MNIST as a controlled fusion playground; reason about which
  fusion strategy would best suit my real task.
- **HW3** — build a VLM-style image+text instruction dataset from CMU-MOSEI
  frames, fine-tune Qwen2.5-VL with LoRA, sweep learning rate.
- **HW4** — keep the same Qwen2.5-VL LoRA, post-train with GRPO using a binary
  correctness reward, compare against SFT-only.
- **HW5** — wrap the same model into a goal-directed agent, give it tools, log
  every trace in Langfuse, and let it interact in the class Discord.

The final project will likely extend HW5: a multimodal agent that does the
comfort-sensing in a loop with a human, exposed through a richer interface.

---

## Running things

- Each homework was originally executed on Colab Pro / A100. Where I ran locally
  on RTX hardware, the per-hw README notes that.
- Datasets and trained weights live outside git; see each homework's README for
  download/regenerate instructions.
- Python environments are pinned per-homework inside `hwN/.venv/` (gitignored).

---

## Course context

> *"This repository should bring everything together in one place: past
> assignments, ongoing explorations, and final projects. […] A well-crafted
> repository tells the story of how you think, what you've explored, and how
> your ideas evolve."*
> — course handout, repo requirement (20 pts)

I'm using this repo as my technical lab notebook for the semester. Diagrams and
images live alongside each notebook in `hwN/assets/` so the visual story
survives even when the data and weights don't.
