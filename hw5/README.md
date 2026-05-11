# HW5 — AI Agents (RestaurantFeatureBot)

**Notebook:** [Homework_5_AI_Agents.ipynb](Homework_5_AI_Agents.ipynb) · **Writeup:** [Homework_5_AI_Agents.pdf](Homework_5_AI_Agents.pdf) · **Rendered HTML:** [Homework_5_AI_Agents_submission.html](Homework_5_AI_Agents_submission.html)

## Task

**RestaurantFeatureBot** — given a question of the form *"does &lt;X&gt; in Cambridge
have &lt;feature Y&gt;?"*, return one of `yes`, `no`, or `uncertain`. The benchmark
has 12 tasks split into normal / edge / adversarial.

> This homework is on a different task than HW1–HW4. The comfort-prediction
> thread does not continue into HW5.

## Parts and points

| Part | Points | What |
|---|---|---|
| 1 | 20 | Reading reflection (3 papers) |
| 2 | 10 | Design metrics + trace schema *before* implementing |
| 3 | 30 | smolagents baseline (Stage A: built-in tools; Stage B: custom tools) |
| 4 | 30 | Multimodal vision agent + safety/policy evaluation |
| 5 | 20 | Langfuse observability + online evaluation |
| 6 | 10 | Discord integration |
| Optional | +10 | OpenClaw |

## Results across configurations

12-task benchmark, single ground-truth `yes`/`no`/`uncertain` per task.

| Configuration | Verdict acc | Trajectory pass | Mean latency | Total tokens |
|---|---|---|---|---|
| Part 3 baseline (built-in tools only) | 58.3% | 66.7% | 20.4 s | 121 K |
| Part 3 + custom tools | 41.7% | 100% | 78.5 s | 490 K |
| Part 4 vision-enhanced (8-task subset) | 37.5% | 100% | 5.7 s | — |
| Part 5 Config A (ToolCallingAgent, max_steps=4) | 0% | 0% | ~9 s | — |
| Part 5 Config B (CodeAgent, 4 tools, max_steps=8) | 40% | 100% | ~85 s | — |

The custom-tool and vision-enhanced agents add `image_search` and
`verify_feature_in_image` (a Qwen2.5-VL-3B call) to the toolset. Trajectory
pass-rate jumps to 100% because the agent now uses the perception channel,
but raw verdict accuracy drops — the VL backbone returns "uncertain" when an
image is partially cropped, and that uncertainty propagates up. The
text-only baseline was sometimes more accurate because it confidently said
"yes" or "no" from a review snippet, which inflated accuracy at the cost of
calibration.

## Safety eval (Part 4 Problem 2)

Three prompts: PII request, prompt-injection with a transactional ask, and
face identification. The pre-mitigation Qwen system prompt already refused
face ID cleanly. Adding a `GuardedTool` keyword wrapper as defense-in-depth
*regressed* the surface behavior — the tool layer blocked the call silently,
the agent looped, and returned `uncertain` instead of a clear refusal
sentence. The lesson in the writeup: keep the system-prompt rules as the
primary refusal mechanism, layer tool guardrails on top, but require an
explicit refusal sentence whenever a guardrail fires.

## Observability (Part 5)

Five instrumented runs landed in Langfuse: 5 traces, 97 observations
(≈19 spans per trace). The dominant span is `verify_feature_in_image` —
each VLM forward pass takes 6–12 s and the agent calls it multiple times
before committing.

![Langfuse dashboard](assets/langfuse_dashboard.png)

## Discord (Part 6)

Bot username: **RestaurantFeatureBot#3140**, deployed to *MMAI Agents World*.
Trigger: @mention **or** the keyword `restaurant:` (compound trigger chosen
over always-on or LLM-decides — always-on burns A100 cycles on irrelevant
chatter, LLM-decides adds a model call per inbound message).
[`utils.py`](utils.py) handles converting `@alice` strings emitted by the LLM
into real Discord mentions by looking up the member in the guild cache.

![Discord interaction](assets/discord_interaction.png)

## Architecture

![Part 4 architecture](assets/part4_architecture.png)

## Reproducing

Requires accounts/keys for: an LLM endpoint, Langfuse (free tier is fine),
and a Discord bot token. None are committed — set them as environment
variables before running the notebook.
