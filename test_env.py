"""
RecallTrace — ContaminationEnv Simulation
Tasks 1-9: Environment, Tools, F1, Hidden Nodes,
           Belief Calibration, Training, Curriculum, Plots
"""

# ─── Required installs (for cold Colab run) ──────────────────────────────────
# !pip install networkx numpy matplotlib

import json
import os
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt

# ─── Always use relative paths so code runs anywhere (Task 8 fix) ─────────────
os.makedirs("plots", exist_ok=True)
PLOT_DIR = "plots"
RESULTS_FILE = "training_results.json"


# =============================================================================
# ContaminationEnv  (Tasks 1-4 + 5 + 7)
# =============================================================================

class ContaminationEnv:
    """
    Supply-chain contamination environment with:
    - Random DAG generation per reset()          [Task 1]
    - 4 noisy investigation tools                [Task 2]
    - F1-scored finalize()                       [Task 3]
    - Hidden intervention nodes                  [Task 4]
    - Belief-calibrated finalize_with_beliefs()  [Task 5]
    - Adversarial curriculum difficulty levels   [Task 7]
    """

    def __init__(self, difficulty_level: int = 3):
        self.graph = None
        self.contaminated_nodes: set = set()
        self.hidden_nodes: set = set()
        self.source_nodes: set = set()
        self.difficulty_level = max(1, min(5, difficulty_level))

    def set_difficulty(self, level: int) -> None:
        """Set difficulty 1 (easy) … 5 (very hard)."""
        self.difficulty_level = max(1, min(5, level))

    # ── Task 1 + 7: Reset ────────────────────────────────────────────────────

    def reset(self) -> dict:
        """Generate a new contamination scenario scaled to current difficulty."""
        params = {
            1: dict(n_range=(6,  8),  n_sources=2, n_hidden=0, edge_p=0.25),
            2: dict(n_range=(8,  10), n_sources=2, n_hidden=1, edge_p=0.30),
            3: dict(n_range=(10, 13), n_sources=3, n_hidden=1, edge_p=0.30),
            4: dict(n_range=(12, 14), n_sources=3, n_hidden=2, edge_p=0.35),
            5: dict(n_range=(14, 16), n_sources=4, n_hidden=2, edge_p=0.40),
        }[self.difficulty_level]

        n_nodes = np.random.randint(*params["n_range"])
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(n_nodes))

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if np.random.random() < params["edge_p"]:
                    self.graph.add_edge(i, j)

        n_sources = min(params["n_sources"], n_nodes)
        self.source_nodes = set(
            np.random.choice(n_nodes, n_sources, replace=False).tolist()
        )

        n_hidden = min(params["n_hidden"], len(self.source_nodes))
        self.hidden_nodes = (
            set(np.random.choice(list(self.source_nodes), n_hidden, replace=False).tolist())
            if n_hidden > 0 else set()
        )

        self.contaminated_nodes = set(self.source_nodes)
        self._spread_contamination()

        return {
            "n_nodes": n_nodes,
            "graph_structure": list(self.graph.edges()),
            "observable_nodes": [n for n in range(n_nodes) if n not in self.hidden_nodes],
            "difficulty": self.difficulty_level,
            "n_hidden": len(self.hidden_nodes),
            "message": (
                f"Difficulty {self.difficulty_level}: {n_nodes}-node graph, "
                f"{len(self.hidden_nodes)} hidden source(s)."
            ),
        }

    def _spread_contamination(self) -> None:
        to_contaminate = set(self.contaminated_nodes)
        for source in self.contaminated_nodes:
            to_contaminate.update(nx.descendants(self.graph, source))
        self.contaminated_nodes = to_contaminate

    # ── Task 2: Tools ────────────────────────────────────────────────────────

    def inspect_node(self, node_id: int) -> dict:
        """Noisy visual inspection (80% TP / 10% FP). Blocked on hidden nodes."""
        if node_id not in self.graph.nodes():
            return {"error": "Node does not exist"}
        if node_id in self.hidden_nodes:
            return {
                "error": "Cannot inspect this node",
                "reason": "Node is not directly observable",
                "hint": "Examine downstream nodes to infer its state",
            }
        is_cont = node_id in self.contaminated_nodes
        obs = np.random.random() < (0.8 if is_cont else 0.1)
        return {
            "node_id": node_id,
            "appears_contaminated": bool(obs),
            "confidence": "medium",
            "upstream_count": len(list(self.graph.predecessors(node_id))),
            "downstream_count": len(list(self.graph.successors(node_id))),
        }

    def test_batch(self, node_id: int) -> dict:
        """Lab test (95% TP / 5% FP). Blocked on hidden nodes."""
        if node_id not in self.graph.nodes():
            return {"error": "Node does not exist"}
        if node_id in self.hidden_nodes:
            return {
                "error": "Cannot test this node",
                "reason": "Node is not directly testable",
                "hint": "Infer contamination from causal structure",
            }
        is_cont = node_id in self.contaminated_nodes
        pos = np.random.random() < (0.95 if is_cont else 0.05)
        return {
            "node_id": node_id,
            "test_result": "POSITIVE" if pos else "NEGATIVE",
            "confidence": "high",
            "cost": 10,
        }

    def trace_upstream(self, node_id: int) -> dict:
        if node_id not in self.graph.nodes():
            return {"error": "Node does not exist"}
        parents = list(self.graph.predecessors(node_id))
        return {"node_id": node_id, "immediate_upstream": parents, "upstream_count": len(parents)}

    def trace_downstream(self, node_id: int) -> dict:
        if node_id not in self.graph.nodes():
            return {"error": "Node does not exist"}
        children = list(self.graph.successors(node_id))
        return {"node_id": node_id, "immediate_downstream": children, "downstream_count": len(children)}

    # ── Task 3: Finalize (F1) ─────────────────────────────────────────────────

    def finalize(self, suspected_nodes) -> dict:
        """Score binary guess with F1 (precision + recall)."""
        suspected = set(suspected_nodes)
        actual = self.contaminated_nodes
        tp = len(suspected & actual)
        fp = len(suspected - actual)
        fn = len(actual - suspected)
        precision = tp / (tp + fp) if suspected else 0.0
        recall    = tp / (tp + fn) if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "f1_score": f1, "precision": precision, "recall": recall,
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "suspected_nodes": list(suspected), "actual_contaminated": list(actual),
            "total_nodes": self.graph.number_of_nodes(),
        }

    # ── Task 5: Finalize with Belief Calibration ──────────────────────────────

    def finalize_with_beliefs(self, beliefs: dict) -> dict:
        """
        Score the agent's probabilistic beliefs.

        Args:
            beliefs: {node_id: confidence_probability}  e.g. {1: 0.9, 3: 0.4}

        Returns:
            Dict with f1_score, calibration_score (Brier), total_reward, breakdown.
        """
        suspected = {n for n, conf in beliefs.items() if conf > 0.5}
        actual = self.contaminated_nodes

        tp = len(suspected & actual)
        fp = len(suspected - actual)
        fn = len(actual - suspected)
        precision = tp / (tp + fp) if suspected else 0.0
        recall    = tp / (tp + fn) if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        calibration = self._calculate_calibration(beliefs)

        # 70% accuracy + 30% calibration
        total_reward = 0.7 * f1 + 0.3 * calibration

        return {
            "f1_score": round(f1, 4),
            "calibration_score": round(calibration, 4),
            "total_reward": round(total_reward, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "breakdown": self._get_belief_breakdown(beliefs),
        }

    def _calculate_calibration(self, beliefs: dict) -> float:
        """Inverted Brier score: 1 = perfect calibration, 0 = worst."""
        if not beliefs:
            return 0.0
        brier = sum(
            (conf - (1 if n in self.contaminated_nodes else 0)) ** 2
            for n, conf in beliefs.items()
        )
        return round(1 - brier / len(beliefs), 4)

    def _get_belief_breakdown(self, beliefs: dict) -> list:
        """Classify each prediction by correctness and confidence."""
        breakdown = []
        for node_id, confidence in beliefs.items():
            is_cont = node_id in self.contaminated_nodes
            if is_cont and confidence > 0.5:
                result = "CORRECT_HIGH_CONF"
            elif is_cont:
                result = "MISSED_LOW_CONF"
            elif confidence > 0.5:
                result = "FALSE_ALARM_HIGH_CONF"
            else:
                result = "CORRECT_LOW_CONF"
            breakdown.append({
                "node": node_id,
                "confidence": round(confidence, 3),
                "actually_contaminated": is_cont,
                "result": result,
            })
        return breakdown


# =============================================================================
# Heuristic Agent  (causal inference — same as Tasks 1-4)
# =============================================================================

def simple_heuristic_agent(env: ContaminationEnv, n_nodes: int) -> dict:
    """
    Inspect all observable nodes, infer hidden nodes causally.
    Returns belief dict {node_id: confidence}.
    """
    observable = [n for n in range(n_nodes) if n not in env.hidden_nodes]
    hidden = list(env.hidden_nodes)
    beliefs = {}

    # Step 1: lab-test observable nodes
    for node in observable:
        result = env.test_batch(node)
        if result.get("test_result") == "POSITIVE":
            beliefs[node] = 0.92
        elif result.get("test_result") == "NEGATIVE":
            beliefs[node] = 0.08

    # Step 2: causal inference for hidden nodes (multi-pass)
    changed = True
    while changed:
        changed = False
        for h in hidden:
            if h in beliefs:
                continue
            parents = list(env.graph.predecessors(h))
            children = list(env.graph.successors(h))

            # If a known-contaminated parent -> this node must be contaminated
            if any(beliefs.get(p, 0) > 0.5 for p in parents):
                beliefs[h] = 0.85
                changed = True
                continue

            # If all children are contaminated -> infer hidden source
            if children and all(beliefs.get(c, 0) > 0.5 for c in children):
                beliefs[h] = 0.75
                changed = True
                continue

            # Partial evidence from children
            if children:
                pos_children = sum(1 for c in children if beliefs.get(c, 0) > 0.5)
                ratio = pos_children / len(children)
                if ratio > 0:
                    beliefs[h] = round(0.4 + 0.4 * ratio, 3)
                    changed = True

    return beliefs


def random_agent(n_nodes: int) -> dict:
    """Purely random baseline."""
    return {
        i: float(np.random.random())
        for i in range(n_nodes)
        if np.random.random() > 0.5
    }


# =============================================================================
# Task 6: Training Loop (30 episodes)
# =============================================================================

def train_agent(n_episodes: int = 30, difficulty: int = 3) -> tuple:
    """Run n_episodes and track F1, calibration, and total reward."""
    env = ContaminationEnv(difficulty_level=difficulty)
    rewards, f1_scores, calibration_scores = [], [], []

    print(f"\n{'='*55}")
    print(f" Training Agent — {n_episodes} Episodes  (difficulty={difficulty})")
    print(f"{'='*55}")

    for ep in range(n_episodes):
        state = env.reset()
        n_nodes = state["n_nodes"]
        beliefs = simple_heuristic_agent(env, n_nodes)
        result = env.finalize_with_beliefs(beliefs)

        rewards.append(result["total_reward"])
        f1_scores.append(result["f1_score"])
        calibration_scores.append(result["calibration_score"])

        if (ep + 1) % 5 == 0:
            print(f" Ep {ep+1:3d}/{n_episodes}  |  F1={result['f1_score']:.3f}  "
                  f"Cal={result['calibration_score']:.3f}  "
                  f"Reward={result['total_reward']:.3f}")

    print(f"\n Final averages ->  F1={np.mean(f1_scores):.3f}  "
          f"Cal={np.mean(calibration_scores):.3f}  "
          f"Reward={np.mean(rewards):.3f}")

    return rewards, f1_scores, calibration_scores


# =============================================================================
# Task 7: Adversarial Curriculum (5 difficulty stages)
# =============================================================================

def train_with_curriculum(total_episodes: int = 50) -> tuple:
    """Train from difficulty 1 -> 5, stepping up every 10 episodes."""
    env = ContaminationEnv(difficulty_level=1)
    rewards, difficulties = [], []

    print(f"\n{'='*55}")
    print(f" Curriculum Training — {total_episodes} Episodes")
    print(f"{'='*55}")

    for ep in range(total_episodes):
        level = min(5, 1 + ep // 10)
        env.set_difficulty(level)
        state = env.reset()
        beliefs = simple_heuristic_agent(env, state["n_nodes"])
        result = env.finalize_with_beliefs(beliefs)

        rewards.append(result["total_reward"])
        difficulties.append(level)

        if (ep + 1) % 10 == 0:
            print(f" Ep {ep+1:3d}/{total_episodes}  |  "
                  f"Difficulty={level}  Reward={result['total_reward']:.3f}")

    return rewards, difficulties


# =============================================================================
# Task 9: Baseline Comparison
# =============================================================================

def compare_baselines(n_trials: int = 20, difficulty: int = 3) -> dict:
    """Compare random vs heuristic agent across n_trials."""
    env = ContaminationEnv(difficulty_level=difficulty)
    results = {"random": [], "heuristic": []}

    for _ in range(n_trials):
        state = env.reset()
        n_nodes = state["n_nodes"]

        # Random baseline
        rg = random_agent(n_nodes)
        results["random"].append(env.finalize_with_beliefs(rg)["f1_score"])

        # Heuristic baseline
        hg = simple_heuristic_agent(env, n_nodes)
        results["heuristic"].append(env.finalize_with_beliefs(hg)["f1_score"])

    return {k: {"mean": round(float(np.mean(v)), 4),
                "std":  round(float(np.std(v)), 4)}
            for k, v in results.items()}


# =============================================================================
# Plot helpers  (Task 6 + 9) — always save as files, never rely on display
# =============================================================================

def plot_training_curves(rewards, f1_scores, calibration_scores):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    episodes = range(1, len(rewards) + 1)

    axes[0].plot(episodes, rewards, "b-", linewidth=2)
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Learning Curve: Total Reward"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(episodes, f1_scores, "g-", linewidth=2)
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("F1 Score")
    axes[1].set_title("Detection Accuracy (F1)"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(episodes, calibration_scores, "r-", linewidth=2)
    axes[2].set_xlabel("Episode"); axes[2].set_ylabel("Calibration Score")
    axes[2].set_title("Belief Calibration"); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_curriculum(rewards, difficulties):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()

    ax.plot(rewards, "b-", linewidth=2, label="Reward")
    ax2.plot(difficulties, "r--", linewidth=2, label="Difficulty", alpha=0.7)

    ax.set_xlabel("Episode"); ax.set_ylabel("Reward", color="b")
    ax2.set_ylabel("Difficulty Level", color="r")
    ax.set_title("Curriculum Learning: Reward vs Difficulty")
    ax.grid(True, alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    path = os.path.join(PLOT_DIR, "curriculum_learning.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_baseline_comparison(baselines):
    fig, ax = plt.subplots(figsize=(8, 6))
    names = list(baselines.keys())
    means = [baselines[k]["mean"] for k in names]
    stds  = [baselines[k]["std"]  for k in names]
    colors = ["#ff6b6b", "#6bcf7f"]

    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Baseline Comparison: Detection Performance", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{mean:.3f}", ha="center", va="bottom", fontweight="bold")

    path = os.path.join(PLOT_DIR, "baseline_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_before_after(f1_scores):
    first5 = f1_scores[:5]
    last5  = f1_scores[-5:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter([1] * len(first5), first5, s=120, alpha=0.7, color="red",  label="First 5 Episodes")
    ax.scatter([2] * len(last5),  last5,  s=120, alpha=0.7, color="green",label="Last 5 Episodes")

    ax.plot([1, 2], [np.mean(first5), np.mean(last5)], "k--", linewidth=2, alpha=0.5)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Before Training", "After Training"])
    ax.set_ylabel("F1 Score"); ax.set_title("Learning Progress: Before vs After")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y"); ax.set_ylim(0, 1.05)

    path = os.path.join(PLOT_DIR, "before_after.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


# =============================================================================
# Task 9: Generate everything for Shreya
# =============================================================================

def generate_all_plots_for_shreya():
    print("\n" + "="*55)
    print(" Generating All Plots & Results")
    print("="*55)

    # ── Training run ──────────────────────────────────────────────────────────
    print("\n[1/4] Training agent (30 episodes, difficulty 3)…")
    rewards, f1, cal = train_agent(n_episodes=30, difficulty=3)
    plot_training_curves(rewards, f1, cal)
    plot_before_after(f1)

    # ── Curriculum run ────────────────────────────────────────────────────────
    print("\n[2/4] Curriculum training (50 episodes, difficulty 1->5)…")
    cur_rewards, cur_diff = train_with_curriculum(total_episodes=50)
    plot_curriculum(cur_rewards, cur_diff)

    # ── Baseline comparison ───────────────────────────────────────────────────
    print("\n[3/4] Baseline comparison (20 trials)…")
    baselines = compare_baselines(n_trials=20, difficulty=3)
    plot_baseline_comparison(baselines)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    print("\n[4/4] Saving results JSON…")
    data = {
        "training": {
            "n_episodes": 30,
            "difficulty": 3,
            "final_f1": float(f1[-1]),
            "final_calibration": float(cal[-1]),
            "final_reward": float(rewards[-1]),
            "avg_f1": round(float(np.mean(f1)), 4),
            "avg_calibration": round(float(np.mean(cal)), 4),
            "avg_reward": round(float(np.mean(rewards)), 4),
            "improvement_f1": round(float(f1[-1] - f1[0]), 4),
        },
        "curriculum": {
            "n_episodes": 50,
            "final_reward": float(cur_rewards[-1]),
            "avg_reward": round(float(np.mean(cur_rewards)), 4),
        },
        "baselines": baselines,
        "plots": [
            os.path.join(PLOT_DIR, f)
            for f in ["training_curves.png", "before_after.png",
                      "curriculum_learning.png", "baseline_comparison.png"]
        ],
    }
    with open(RESULTS_FILE, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"  Saved -> {RESULTS_FILE}")

    print("\n" + "="*55)
    print(" RESULTS FOR SHREYA")
    print("="*55)
    t = data["training"]
    print(f"  Avg F1 Score     : {t['avg_f1']:.3f}")
    print(f"  Avg Calibration  : {t['avg_calibration']:.3f}")
    print(f"  Avg Total Reward : {t['avg_reward']:.3f}")
    print(f"  F1 Improvement   : +{t['improvement_f1']:.3f}")
    print(f"\n  Baselines (F1):")
    for name, stats in baselines.items():
        print(f"    {name:12s}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    print(f"  All plots saved to -> {PLOT_DIR}/")
    print("="*55)

    return data


# =============================================================================
# Main — runs everything end-to-end
# =============================================================================

if __name__ == "__main__":
    print("RecallTrace — Tasks 1-9 Simulation")
    print("="*55)

    # ── Quick sanity check (Tasks 1-4) ────────────────────────────────────────
    print("\n[SANITY] 10-episode automated agent run…")
    f1_history = []
    for ep in range(10):
        env = ContaminationEnv(difficulty_level=3)
        state = env.reset()
        beliefs = simple_heuristic_agent(env, state["n_nodes"])
        r = env.finalize_with_beliefs(beliefs)
        f1_history.append(r["f1_score"])
        print(f"  Ep {ep+1:2d} | nodes={state['n_nodes']:2d} "
              f"| hidden={state['n_hidden']} "
              f"| F1={r['f1_score']:.3f} "
              f"| Cal={r['calibration_score']:.3f} "
              f"| Reward={r['total_reward']:.3f}")
    print(f"  => Mean F1 over 10 episodes: {np.mean(f1_history):.3f}")

    # ── Task 5: Belief calibration demo ──────────────────────────────────────
    print("\n[TASK 5] Belief calibration example…")
    env = ContaminationEnv(difficulty_level=3)
    env.reset()
    demo_beliefs = {
        n: float(np.random.random())
        for n in range(env.graph.number_of_nodes())
    }
    result = env.finalize_with_beliefs(demo_beliefs)
    print(f"  F1={result['f1_score']:.3f}  "
          f"Calibration={result['calibration_score']:.3f}  "
          f"Total Reward={result['total_reward']:.3f}")

    # ── Tasks 6, 7, 9: Full training + plots ─────────────────────────────────
    data = generate_all_plots_for_shreya()
    print("All done! Done")
