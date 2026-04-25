# RecallTrace — Training Guide

How to train the adversarial self-play RL model and understand what's happening.

---

## Quick Start (2 seconds on CPU)

```bash
python run_selfplay.py
```

This runs **200 episodes** of Investigator vs Adversary training and generates 3 plots:
- `plots/selfplay_training.png` — 4-panel training curves
- `plots/episode_comparison.png` — early vs late episode comparison
- `plots/before_after_demo.png` — side-by-side graph replay

---

## Understanding the Training Loop

Each episode follows this cycle:

1. **Graph Generation**: A random supply-chain DAG is created
2. **Adversary Chooses**: Picks an intervention type (relabel, mixing, deletion) and placement
3. **Intervention Applied**: Contamination is hidden using the chosen strategy + decoys added
4. **Investigator Acts**: Inspects nodes, traces lineages, quarantines suspicious stock
5. **Both Update**: Investigator adjusts thresholds, Adversary updates its strategy table

### What the Investigator Learns

| Parameter | Start | After Training | What it does |
|---|---|---|---|
| `quarantine_threshold` | 0.0 | ~0.55 | Min evidence to quarantine (0 = quarantine everything) |
| `suspect_trust` | 1.0 | ~0.05 | How much to trust "suspect" evidence (decoys!) |
| `mixed_trust` | 0.95 | ~0.3 | Trust in "mixed" evidence |
| `exploration_rate` | 0.95 | ~0.05 | Probability of visiting non-traced nodes |

### What the Adversary Learns

The adversary maintains a **3×3 score table** over (intervention_type × graph_region). It uses a softmax policy with temperature annealing to pick strategies that make the investigator fail most.

---

## Extended Training (Longer Runs)

For more thorough training:

```python
from selfplay.trainer import SelfPlayTrainer

trainer = SelfPlayTrainer(num_nodes=20)        # Larger graphs
stats = trainer.train(num_episodes=2000)       # More episodes
```

### Scaling Parameters

| Parameter | Default | Extended | Effect |
|---|---|---|---|
| `num_episodes` | 200 | 2000-5000 | More training iterations |
| `num_nodes` | 10 | 15-25 | Larger, harder graphs |
| `threshold_lr` | 0.004 | 0.002 | Slower, more stable learning |
| `temperature` | 2.0 | 3.0 | More adversary exploration |

A 2000-episode run with 20 nodes takes approximately **30-60 seconds** on CPU.

---

## Upgrading to Neural RL (PyTorch)

To train with neural network policies (like your friend's 2-hour training), you would:

### 1. Install Dependencies
```bash
pip install torch stable-baselines3 gymnasium
```

### 2. Wrap as Gym Environment
```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RecallTraceGymEnv(gym.Env):
    def __init__(self, num_nodes=10):
        super().__init__()
        self.num_nodes = num_nodes
        # Observation: belief state vector + graph features
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_nodes * 4,))
        # Actions: inspect(N), quarantine(N), trace, finalize
        self.action_space = spaces.Discrete(num_nodes * 2 + 2)

    def reset(self, seed=None, options=None):
        # Generate new scenario, return observation
        ...

    def step(self, action):
        # Execute action, return obs, reward, done, truncated, info
        ...
```

### 3. Train with PPO
```python
from stable_baselines3 import PPO

env = RecallTraceGymEnv(num_nodes=15)
model = PPO("MlpPolicy", env, verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10)
model.learn(total_timesteps=500_000)  # ~2 hours on CPU
model.save("recalltrace_ppo")
```

---

## Reading the Training Output

### F1 Score
- **Early (ep 1-20)**: ~0.3-0.5 — agent quarantines too aggressively (spray & pray)
- **Late (ep 180-200)**: ~0.85-1.0 — agent quarantines precisely

### Adversary Reward
- **Positive**: Adversary is winning (investigator failing)
- **Negative**: Investigator is winning (adversary's tricks aren't working)
- **Should trend negative** over training

### Nodes Quarantined
- **Early**: 6-8 per episode (quarantining everything)
- **Late**: 2-3 per episode (surgical precision)

---

## Hyperparameter Tuning

Key knobs to adjust:

```python
# In selfplay/investigator.py
threshold_lr = 0.004    # How fast the quarantine threshold adapts
trust_lr = 0.005        # How fast evidence trust parameters adapt

# In selfplay/adversary.py  
temperature = 2.0       # Exploration vs exploitation (higher = more random)
min_temperature = 0.3   # Minimum temperature (exploitation floor)
```

**Tips:**
- If F1 plateaus below 0.7: increase `threshold_lr` to learn faster
- If F1 oscillates wildly: decrease both learning rates
- If adversary always picks the same strategy: increase `temperature`
