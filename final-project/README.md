# Final Project — (in progress)

This folder will hold my copy of the team final project, per the course
requirement that *"each team member should include their own copy of the
project in their repository."*

## Working idea

Extending the thread that runs through HW1–HW5: a **multimodal comfort-sensing
agent** that closes the loop with a human user. The HW5 agent answered queries
*about* a person on demand; the final project wants the agent to be present
during an interaction and decide on its own when to intervene.

Concretely the plan is roughly:

- Reuse the HW3+HW4 fine-tuned Qwen2.5-VL adapter for the comfort signal.
- Build a longer-horizon agent loop (HW5 agent scaffold) that observes a video
  stream, maintains a state estimate, and chooses actions from a small toolbox
  (e.g. *ask a clarifying question*, *change topic*, *stay silent*).
- Evaluate with both offline traces and a small live user study.

## Status

- [ ] Team formed
- [ ] Proposal draft
- [ ] Data + eval protocol
- [ ] Prototype
- [ ] Final write-up + demo

I'll update this README as the project firms up and drop figures/notes into
this folder along the way.
