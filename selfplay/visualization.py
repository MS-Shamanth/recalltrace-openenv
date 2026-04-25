"""Visualization for RecallTrace adversarial self-play training.

Two main functions:
  - show_training_curves(): 2x2 panel with F1, adversary reward, quarantined, steps
  - show_episode_comparison(): side-by-side early vs late episode comparison
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np


def _rolling_average(data: List[float], window: int = 20) -> List[float]:
    """Compute rolling average with the given window size."""
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result


def show_training_curves(
    stats: List[Dict[str, Any]],
    save_path: str = "plots/selfplay_training.png",
) -> None:
    """Create a 2x2 publication-quality training curves figure.

    Top left:     Investigator F1 over episodes (raw + rolling avg)
    Top right:    Adversary reward over episodes
    Bottom left:  Nodes quarantined over episodes
    Bottom right: Steps to finalize over episodes

    Uses a dark theme for hackathon-ready visuals.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    episodes = [s["episode"] for s in stats]
    f1_scores = [s["investigator_f1"] for s in stats]
    adv_rewards = [s["adversary_reward"] for s in stats]
    quarantined = [s["num_quarantined"] for s in stats]
    steps = [s["steps_taken"] for s in stats]

    f1_rolling = _rolling_average(f1_scores)
    adv_rolling = _rolling_average(adv_rewards)
    q_rolling = _rolling_average(quarantined)
    s_rolling = _rolling_average(steps)

    # --- Dark theme setup ---
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#0d1117")

    colors = {
        "f1_raw": "#3b82f6",       # blue
        "f1_avg": "#60a5fa",       # light blue
        "adv_raw": "#ef4444",      # red
        "adv_avg": "#f87171",      # light red
        "q_raw": "#22c55e",        # green
        "q_avg": "#4ade80",        # light green
        "s_raw": "#f59e0b",        # amber
        "s_avg": "#fbbf24",        # light amber
    }
    bg_color = "#161b22"
    grid_color = "#30363d"
    text_color = "#e6edf3"

    for ax in axes.flat:
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=text_color, labelsize=10)
        ax.spines["bottom"].set_color(grid_color)
        ax.spines["left"].set_color(grid_color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, color=grid_color)

    # --- Top Left: Investigator F1 ---
    ax = axes[0, 0]
    ax.scatter(episodes, f1_scores, c=colors["f1_raw"], alpha=0.15, s=8, zorder=2)
    ax.plot(episodes, f1_rolling, color=colors["f1_avg"], linewidth=2.5, zorder=3, label="20-ep rolling avg")
    ax.axhline(y=0.5, color="#ef4444", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(y=0.8, color="#22c55e", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_title("Investigator F1 Score", fontsize=14, color=text_color, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", color=text_color, fontsize=11)
    ax.set_ylabel("F1 Score", color=text_color, fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=9, facecolor=bg_color, edgecolor=grid_color)
    # Add annotations
    ax.text(0.02, 0.95, "Adversary wins ↓", transform=ax.transAxes,
            fontsize=8, color="#ef4444", alpha=0.7, va="top")
    ax.text(0.02, 0.05, "Investigator wins ↑", transform=ax.transAxes,
            fontsize=8, color="#22c55e", alpha=0.7, va="bottom")

    # --- Top Right: Adversary Reward ---
    ax = axes[0, 1]
    ax.scatter(episodes, adv_rewards, c=colors["adv_raw"], alpha=0.15, s=8, zorder=2)
    ax.plot(episodes, adv_rolling, color=colors["adv_avg"], linewidth=2.5, zorder=3, label="20-ep rolling avg")
    ax.axhline(y=0, color=text_color, linestyle="-", alpha=0.2, linewidth=1)
    ax.set_title("Adversary Reward", fontsize=14, color=text_color, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", color=text_color, fontsize=11)
    ax.set_ylabel("Reward", color=text_color, fontsize=11)
    ax.set_ylim(-1.3, 1.3)
    ax.legend(loc="upper right", fontsize=9, facecolor=bg_color, edgecolor=grid_color)

    # --- Bottom Left: Nodes Quarantined ---
    ax = axes[1, 0]
    ax.scatter(episodes, quarantined, c=colors["q_raw"], alpha=0.15, s=8, zorder=2)
    ax.plot(episodes, q_rolling, color=colors["q_avg"], linewidth=2.5, zorder=3, label="20-ep rolling avg")
    ax.set_title("Nodes Quarantined per Episode", fontsize=14, color=text_color, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", color=text_color, fontsize=11)
    ax.set_ylabel("Count", color=text_color, fontsize=11)
    ax.legend(loc="upper right", fontsize=9, facecolor=bg_color, edgecolor=grid_color)

    # --- Bottom Right: Steps Taken ---
    ax = axes[1, 1]
    ax.scatter(episodes, steps, c=colors["s_raw"], alpha=0.15, s=8, zorder=2)
    ax.plot(episodes, s_rolling, color=colors["s_avg"], linewidth=2.5, zorder=3, label="20-ep rolling avg")
    ax.set_title("Steps to Finalize", fontsize=14, color=text_color, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", color=text_color, fontsize=11)
    ax.set_ylabel("Steps", color=text_color, fontsize=11)
    ax.legend(loc="upper right", fontsize=9, facecolor=bg_color, edgecolor=grid_color)

    # --- Main title ---
    fig.suptitle(
        "RecallTrace — Adversarial Self-Play Training",
        fontsize=18, color=text_color, fontweight="bold", y=0.98,
    )
    fig.text(
        0.5, 0.935,
        "Investigator vs Adversary co-evolution over 200 episodes",
        ha="center", fontsize=11, color="#8b949e",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved training curves to {save_path}")


def show_episode_comparison(
    early_stats: Dict[str, Any],
    late_stats: Dict[str, Any],
    save_path: str = "plots/episode_comparison.png",
) -> None:
    """Create a side-by-side comparison of early vs late episode behavior.

    Shows: nodes visited, nodes quarantined, F1 score, belief confidence,
    intervention type, correctly identified or not.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, (ax_early, ax_late) = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor("#0d1117")

    bg_color = "#161b22"
    text_color = "#e6edf3"
    dim_color = "#8b949e"

    def _draw_episode_card(ax, stats, title, is_good):
        ax.set_facecolor(bg_color)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        # Title bar
        border_color = "#22c55e" if is_good else "#ef4444"
        title_bg = "#1a3a2a" if is_good else "#3a1a1a"

        rect = FancyBboxPatch(
            (0.3, 8.5), 9.4, 1.2,
            boxstyle="round,pad=0.15",
            facecolor=title_bg, edgecolor=border_color, linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(5, 9.1, title, fontsize=16, fontweight="bold",
                color=text_color, ha="center", va="center")

        # F1 Score (large)
        f1 = stats["investigator_f1"]
        f1_color = "#22c55e" if f1 > 0.7 else "#f59e0b" if f1 > 0.4 else "#ef4444"
        ax.text(5, 7.5, f"F1 Score: {f1:.3f}", fontsize=28, fontweight="bold",
                color=f1_color, ha="center", va="center")

        # Stats grid
        info_lines = [
            ("Nodes Visited", str(len(stats.get("nodes_visited", [])))),
            ("Nodes Quarantined", str(stats["num_quarantined"])),
            ("Steps Taken", str(stats["steps_taken"])),
            ("Belief Confidence", f"{stats['belief_confidence']:.2f}"),
            ("Intervention Type", stats["intervention_type"]),
            ("Correctly Identified", "YES" if stats["intervention_correctly_identified"] else "NO"),
            ("Quarantine Threshold", f"{stats['quarantine_threshold']:.3f}"),
            ("Exploration Rate", f"{stats['exploration_rate']:.3f}"),
        ]

        y_pos = 6.2
        for label, value in info_lines:
            # Label
            ax.text(1.0, y_pos, label + ":", fontsize=11, color=dim_color,
                    ha="left", va="center", fontfamily="monospace")
            # Value
            v_color = text_color
            if label == "Correctly Identified":
                v_color = "#22c55e" if value == "YES" else "#ef4444"
            ax.text(9.0, y_pos, value, fontsize=12, fontweight="bold",
                    color=v_color, ha="right", va="center", fontfamily="monospace")
            y_pos -= 0.7

        # Quarantined nodes list
        q_nodes = stats.get("nodes_quarantined_list", [])
        if q_nodes:
            ax.text(1.0, y_pos - 0.3, "Quarantined:", fontsize=10, color=dim_color,
                    ha="left", va="center")
            node_text = ", ".join(q_nodes[:6])
            if len(q_nodes) > 6:
                node_text += f" +{len(q_nodes)-6} more"
            ax.text(1.0, y_pos - 0.9, node_text, fontsize=9, color="#f59e0b",
                    ha="left", va="center", fontfamily="monospace")

    _draw_episode_card(ax_early, early_stats,
                       f"Episode {early_stats['episode']} (Early)", is_good=False)
    _draw_episode_card(ax_late, late_stats,
                       f"Episode {late_stats['episode']} (Late)", is_good=True)

    # Arrow between cards
    fig.text(0.5, 0.5, "→", fontsize=48, color="#8b949e",
             ha="center", va="center", fontweight="bold")

    fig.suptitle(
        "RecallTrace — Before / After Self-Play Training",
        fontsize=18, color=text_color, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.92,
        "Investigator behavior change: spray & pray → precision targeting",
        ha="center", fontsize=12, color=dim_color,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved episode comparison to {save_path}")
