---
title: RecallTrace OpenEnv
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# RecallTrace: Causal Inference via Adversarial Self-Play

An RL agent that doesn't just learn to detect contamination — it learns to infer the hidden causal intervention behind it. 

Trained via adversarial self-play, where an adversary learns to hide better as the investigator learns to reason better.

---

## 🚀 Run in one command

```bash
python run_selfplay.py
```

*(No API keys, no GPUs, runs in <2 seconds on CPU)*

---

## 🎥 What you'll see

- Agent improves from random (spray-and-pray) to precise, belief-calibrated quarantine.
- F1 score increases to ~1.0 over 200 episodes.
- Nodes quarantined drops from 8.3/episode to 3.1/episode.
- Adversary adapts to agent weaknesses dynamically.

---

## 📊 Proof of Learning

### 1. The Learning Curves
*(Generated automatically when you run the script)*

![Training Curves](plots/selfplay_training.png)

### 2. Before vs After Behavior
*(Untrained vs Trained Agent Comparison)*

![Before vs After](plots/before_after_demo.png)

---

## 🧠 Why This Is Unique

1. **Causal Inference (not Graph Traversal)**: 30-50% of the graph edges are hidden. The agent must perform abductive reasoning to identify *which* hidden causal intervention (relabeling, mixing, record deletion) produced the observed contamination pattern.
2. **Partial Observability**: The agent relies on a probabilistic belief state (`P(contaminated)` per node) and tool calls to reduce entropy.
3. **Adversarial Self-Play (Theme 4)**: The environment's difficulty is not static. An adversary agent chooses where to place interventions, adapting its curriculum based on the investigator's failure modes.
4. **Belief-Based Decisions (Theme 3.1)**: Quarantines are only rewarded if the agent is confident (`P > 0.8`). Uncalibrated guesses are heavily penalized.

---

## ⚙️ How It Works

- **The Environment**: A procedural generator builds a unique contamination propagation graph every episode with decoys, false positives, and hidden interventions.
- **The Investigator (Agent 1)**: Inspects nodes, traces lineages, and cross-references data to find contamination and quarantine it. Rewarded for precision and recall (+2.0 for correct, -1.5 for incorrect).
- **The Adversary (Agent 2)**: Chooses intervention types and placements. Rewarded exclusively when the Investigator fails.

---

## 🧪 Reproducibility

- **Runs in <2 seconds on CPU.**
- **No external APIs or heavy models required.**
- **Deterministic seeds used** for exact evaluation and metric reproducibility.

---

## 📦 Project Structure
```text
recalltrace-openenv/
├── run_selfplay.py        # ENTRY POINT
├── app.py                 # Hugging Face Gradio UI
├── README.md              # Project Story
├── PITCH.md               # 3-Minute Mentor Pitch Script
├── MENTOR_PREP.md         # Fast-prep for live judging
├── PITCH_LANGUAGE.md      # Language guidelines
├── architecture.html      # Visual Flow Diagram
│
├── selfplay/              # Core Logic (Investigator, Adversary, Tracker)
├── env/                   # Original OpenEnv Environment definition
│
├── plots/                 # Auto-generated Demo Imagery
│   ├── selfplay_training.png
│   ├── before_after_demo.png
│   └── episode_comparison.png
```
