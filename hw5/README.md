# HW5 — AI Agents in the Wild

**Notebook:** [Homework_5_AI_Agents.ipynb](Homework_5_AI_Agents.ipynb)
**Writeup:** [Homework_5_AI_Agents.pdf](Homework_5_AI_Agents.pdf)
**Rendered HTML (easier to skim):** [Homework_5_AI_Agents_submission.html](Homework_5_AI_Agents_submission.html)

---

## What this homework is about

Build a real **goal-directed agent** — observation loop, tools, termination
conditions — and then evaluate it as honestly as possible. Open-ended: I picked
my own domain (continuing the comfort-sensing thread) and my own evaluation
metrics. The notebook walks through six parts:

| Part | Points | What |
|---|---|---|
| 1 | 20 | Reading reflection on agents vs chatbots, sequential decision framing, agent architectures, evaluation challenges |
| 2 | 10 | Design metrics, grading rules, and trace schema *before* building anything |
| 3 | 30 | Build the baseline agent with **smolagents** + custom tool integration |
| 4 | 30 | **Multimodal language agent** — vision implementation + controlled comparison + safety/policy eval |
| 5 | 20 | **Observability** — set up [Langfuse](https://langfuse.com), record traces, online eval |
| 6 | 10 | **Discord integration** — agent runs in the class MMAI Discord world |
| — | +10 | Optional: try OpenClaw |

## Architecture (Part 4)

![part 4 architecture](assets/part4_architecture.png)

The agent is a smolagents `CodeAgent` wrapping a vision-language backbone. Tools
include a frame-fetcher, the comfort classifier from HW3/HW4, and a Discord
sender (see [`utils.py`](utils.py) for the `@mention` hydration logic that lets
the agent tag real users correctly).

## Observability with Langfuse

Every agent run logs a trace: inputs, tool calls, intermediate thoughts, final
output, latency, cost. This is what makes "online evaluation" tractable —
you can replay any trace and re-score it later.

![langfuse dashboard](assets/langfuse_dashboard.png)

## Discord integration (Part 6)

The agent lives in the class Discord server and responds to `@`-mentions.
[`utils.py`](utils.py) handles the surprisingly fiddly job of converting LLM
output like `"@alice, the frame looks calm"` into a real Discord mention
(`<@123456789>`) by looking up the member in the guild cache — including a
fallback to the message author so a user can always be tagged in a reply.

![discord interaction](assets/discord_interaction.png)

## Reading reflection (Part 1)

Three readings — Anthropic's *Building Effective Agents* (2024) on the
workflow↔agent spectrum and evaluator-optimizer patterns, plus two surveys
of multimodal LLM agents and agent evaluation. The four answers in the
notebook walk through:

1. What makes a system an *agent* rather than a chatbot or tool-using model.
2. My system written out as a sequential decision problem (S, A, T, R, π).
3. Comparison of two architectures from the literature applied to my task.
4. Evaluation challenges specific to multimodal agents (trace ground-truthing,
   tool-call grading, the difference between *correct answer* and *correct
   process*).

## Files in this folder

- `Homework_5_AI_Agents.ipynb` — full notebook, executed.
- `Homework_5_AI_Agents.pdf` — writeup for submission.
- `Homework_5_AI_Agents_submission.html` — rendered version, faster to read
  on GitHub than scrolling the .ipynb.
- `utils.py` — Discord helpers (mention hydration).
- `assets/` — the three figures embedded above.

## Reproducing

Requires accounts/keys for: a Claude or OpenAI endpoint for the LLM, Langfuse
(free tier is fine), and a Discord bot token. None of these are committed —
set them as environment variables before running the notebook.
