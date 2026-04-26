import os
import matplotlib.pyplot as plt

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

losses = [
    2.405, 1.927, 1.184, 0.3884, 0.09162, 0.03675, 0.02496, 0.01895, 0.01838, 0.01794,
    0.01691, 0.01584, 0.01471, 0.01471, 0.0138, 0.01404, 0.01404, 0.01315, 0.01271, 0.01221,
    0.01145, 0.01035, 0.009906, 0.01096, 0.009928, 0.01093, 0.01076, 0.009659, 0.01026, 0.009521,
    0.00914, 0.008566, 0.008741, 0.008682, 0.008574, 0.008453, 0.008783, 0.008452, 0.00854, 0.008325,
    0.008671, 0.00839, 0.008425, 0.008395, 0.008689, 0.008234, 0.008654, 0.008448, 0.008507, 0.008681,
    0.008344, 0.008281, 0.008645, 0.00853, 0.00857, 0.008191, 0.008447, 0.008351, 0.008434, 0.008516,
    0.008106, 0.008195, 0.008332, 0.008627, 0.008091
]
steps = [10 * (i + 1) for i in range(len(losses))]

eval_results = {
    "Random": {"avg_score": 0.1552},
    "Heuristic": {"avg_score": 0.9677},
    "Trained LLM": {"avg_score": 0.9677}
}

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(steps, losses, color="#ff6f3c", linewidth=2, label="SFT Training Loss")
ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("RecallTrace — SFT Training Loss (Unsloth + TRL)", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "trl_training_loss.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
names = list(eval_results.keys())
avgs = [eval_results[n]["avg_score"] for n in names]
colors = ["#8b949e", "#f0c040", "#2ea043"][:len(names)]
bars = ax.bar(names, avgs, color=colors, width=0.5, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, avgs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("Average Episode Score", fontsize=12)
ax.set_title("RecallTrace — Baseline vs Trained Agent", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "trl_evaluation_comparison.png"), dpi=150)
plt.close()

print("Plots successfully recovered locally!")
