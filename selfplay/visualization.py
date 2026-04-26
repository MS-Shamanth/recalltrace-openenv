"""Visualization for RecallTrace adversarial self-play training.

Generates:
  - show_training_curves():       2x2 combined panel
  - save_individual_plots():      4 separate metric PNGs
  - show_episode_comparison():    side-by-side early vs late
  - save_coevolution_plot():      adversary + investigator rewards together
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


# --- Dark theme constants ---
_BG_DARK = "#0d1117"
_BG_PANEL = "#161b22"
_GRID_COLOR = "#30363d"
_TEXT_COLOR = "#e6edf3"
_DIM_COLOR = "#8b949e"

_COLORS = {
    "f1_raw": "#3b82f6", "f1_avg": "#60a5fa",
    "adv_raw": "#ef4444", "adv_avg": "#f87171",
    "q_raw": "#22c55e", "q_avg": "#4ade80",
    "s_raw": "#f59e0b", "s_avg": "#fbbf24",
    "cal_raw": "#a78bfa", "cal_avg": "#c4b5fd",
    "inv_raw": "#38bdf8", "inv_avg": "#7dd3fc",
}


def _style_ax(ax):
    """Apply dark theme styling to an axis."""
    ax.set_facecolor(_BG_PANEL)
    ax.tick_params(colors=_TEXT_COLOR, labelsize=10)
    ax.spines["bottom"].set_color(_GRID_COLOR)
    ax.spines["left"].set_color(_GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, color=_GRID_COLOR)


def show_training_curves(
    stats: List[Dict[str, Any]],
    save_path: str = "plots/selfplay_training.png",
) -> None:
    """Create a 2x2 publication-quality training curves figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = [s["episode"] for s in stats]
    f1_scores = [s["investigator_f1"] for s in stats]
    adv_rewards = [s["adversary_reward"] for s in stats]
    quarantined = [s["num_quarantined"] for s in stats]
    steps = [s["steps_taken"] for s in stats]

    f1_rolling = _rolling_average(f1_scores)
    adv_rolling = _rolling_average(adv_rewards)
    q_rolling = _rolling_average(quarantined)
    s_rolling = _rolling_average(steps)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(_BG_DARK)

    for ax in axes.flat:
        _style_ax(ax)

    # --- Top Left: Investigator F1 ---
    ax = axes[0, 0]
    ax.scatter(episodes, f1_scores, c=_COLORS["f1_raw"], alpha=0.15, s=8, zorder=2)
    ax.plot(episodes, f1_rolling, color=_COLORS["f1_avg"], linewidth=2.5, zorder=3, label="20-ep rolling avg")
    ax.axhline(y=0.5, color="#ef4444", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(y=0.8, color="#22c55e", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_title("Investigator F1 Score", fontsize=14, color=_TEXT_COLOR, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", color=_TEXT_COLOR, fontsize=11)
    ax.set_ylabel("F1 Score\n[unsafe caught vs safe blocked]", color=_TEXT_COLOR, fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=9, facecolor=_BG_PANEL, edgecolor=_GRID_COLOR)
    ax.text(0.02, 0.95, "Adversary wins ↓", transform=ax.transAxes,
            fontsize=8, color="#ef4444", alpha=0.7, va="top")
    ax.text(0.02, 0.05, "Investigator wins ↑", transform=ax.transAxes,
            fontsize=8, color="#22c55e", alpha=0.7, va="bottom")

    # --- Top Right: Adversary Reward ---
    ax = axes[0, 1]
    ax.scatter(episodes, adv_rewards, c=_COLORS["adv_raw"], alpha=0.15, s=8, zorder=2)
    ax.plot(episodes, adv_rolling, color=_COLORS["adv_avg"], linewidth=2.5, zorder=3, label="20-ep rolling avg")
    ax.axhline(y=0, color=_TEXT_COLOR, linestyle="-", alpha=0.2, linewidth=1)
    ax.set_title("Adversary Reward", fontsize=14, color=_TEXT_COLOR, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", color=_TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Reward", color=_TEXT_COLOR, fontsize=11)
    ax.set_ylim(-1.3, 1.3)
    ax.legend(loc="upper right", fontsize=9, facecolor=_BG_PANEL, edgecolor=_GRID_COLOR)

    # --- Bottom Left: Nodes Quarantined ---
    ax = axes[1, 0]
    ax.scatter(episodes, quarantined, c=_COLORS["q_raw"], alpha=0.15, s=8, zorder=2)
    ax.plot(episodes, q_rolling, color=_COLORS["q_avg"], linewidth=2.5, zorder=3, label="20-ep rolling avg")
    ax.set_title("Nodes Quarantined per Episode", fontsize=14, color=_TEXT_COLOR, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", color=_TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Count", color=_TEXT_COLOR, fontsize=11)
    ax.legend(loc="upper right", fontsize=9, facecolor=_BG_PANEL, edgecolor=_GRID_COLOR)

    # --- Bottom Right: Steps Taken ---
    ax = axes[1, 1]
    ax.scatter(episodes, steps, c=_COLORS["s_raw"], alpha=0.15, s=8, zorder=2)
    ax.plot(episodes, s_rolling, color=_COLORS["s_avg"], linewidth=2.5, zorder=3, label="20-ep rolling avg")
    ax.set_title("Steps to Finalize", fontsize=14, color=_TEXT_COLOR, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", color=_TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Steps", color=_TEXT_COLOR, fontsize=11)
    ax.legend(loc="upper right", fontsize=9, facecolor=_BG_PANEL, edgecolor=_GRID_COLOR)

    fig.suptitle(
        "RecallTrace — Adversarial Self-Play Training",
        fontsize=18, color=_TEXT_COLOR, fontweight="bold", y=0.98,
    )
    fig.text(
        0.5, 0.935,
        "Investigator vs Adversary co-evolution over 200 episodes",
        ha="center", fontsize=11, color=_DIM_COLOR,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved training curves to {save_path}")


def save_individual_plots(stats: List[Dict[str, Any]], plots_dir: str = "plots") -> None:
    """Save individual metric plots as separate PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)
    episodes = [s["episode"] for s in stats]

    def _save_single(data, rolling, title, ylabel, filename, raw_color, avg_color, hlines=None):
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(_BG_DARK)
        _style_ax(ax)
        ax.scatter(episodes, data, c=raw_color, alpha=0.15, s=8, zorder=2)
        ax.plot(episodes, rolling, color=avg_color, linewidth=2.5, zorder=3, label="20-ep rolling avg")
        if hlines:
            for y, color, label in hlines:
                ax.axhline(y=y, color=color, linestyle="--", alpha=0.4, linewidth=1, label=label)
        ax.set_title(title, fontsize=14, color=_TEXT_COLOR, fontweight="bold", pad=12)
        ax.set_xlabel("Episode", color=_TEXT_COLOR, fontsize=11)
        ax.set_ylabel(ylabel, color=_TEXT_COLOR, fontsize=11)
        ax.legend(loc="best", fontsize=9, facecolor=_BG_PANEL, edgecolor=_GRID_COLOR)
        plt.tight_layout()
        path = os.path.join(plots_dir, filename)
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved {path}")

    # 1. F1 curve
    f1 = [s["investigator_f1"] for s in stats]
    _save_single(f1, _rolling_average(f1), "Investigator F1 Score Over Training",
                 "F1 Score [unsafe caught vs safe blocked]", "f1_curve.png",
                 _COLORS["f1_raw"], _COLORS["f1_avg"],
                 [(0.5, "#ef4444", "F1=0.5"), (0.8, "#22c55e", "F1=0.8")])

    # 2. Nodes quarantined
    q = [s["num_quarantined"] for s in stats]
    _save_single(q, _rolling_average(q), "Nodes Quarantined Per Episode",
                 "Count", "nodes_quarantined.png",
                 _COLORS["q_raw"], _COLORS["q_avg"])

    # 3. Steps to finalize
    st = [s["steps_taken"] for s in stats]
    _save_single(st, _rolling_average(st), "Steps to Finalize Per Episode",
                 "Steps", "steps_to_finalize.png",
                 _COLORS["s_raw"], _COLORS["s_avg"])

    # 4. Belief calibration
    cal = [s.get("belief_calibration", 0) for s in stats]
    _save_single(cal, _rolling_average(cal),
                 "Belief Calibration Score Over Training",
                 "Avg P(contaminated) at quarantine time", "belief_calibration.png",
                 _COLORS["cal_raw"], _COLORS["cal_avg"],
                 [(0.85, "#f97583", "Quarantine threshold")])


def save_coevolution_plot(stats: List[Dict[str, Any]], save_path: str = "plots/coevolution.png") -> None:
    """Show investigator and adversary rewards on the same plot — co-evolution proof."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = [s["episode"] for s in stats]
    inv_rewards = [s["investigator_reward"] for s in stats]
    adv_rewards = [s["adversary_reward"] for s in stats]

    inv_rolling = _rolling_average(inv_rewards)
    adv_rolling = _rolling_average(adv_rewards)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(_BG_DARK)
    _style_ax(ax1)

    ax1.plot(episodes, inv_rolling, color=_COLORS["f1_avg"], linewidth=2.5, label="Investigator reward", zorder=3)
    ax1.plot(episodes, adv_rolling, color=_COLORS["adv_avg"], linewidth=2.5, label="Adversary reward", zorder=3)
    ax1.axhline(y=0, color=_TEXT_COLOR, linestyle="-", alpha=0.2, linewidth=1)

    ax1.set_title("Co-Evolution: Both Agents Improving Simultaneously",
                   fontsize=14, color=_TEXT_COLOR, fontweight="bold", pad=12)
    ax1.set_xlabel("Episode", color=_TEXT_COLOR, fontsize=11)
    ax1.set_ylabel("Reward (20-ep rolling avg)", color=_TEXT_COLOR, fontsize=11)
    ax1.legend(loc="best", fontsize=10, facecolor=_BG_PANEL, edgecolor=_GRID_COLOR)

    # Annotation showing the cross-over
    ax1.text(0.5, 0.02,
             "The adversary keeps the problem hard → the investigator never stagnates",
             transform=ax1.transAxes, fontsize=9, color=_DIM_COLOR,
             ha="center", va="bottom", style="italic")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved co-evolution plot to {save_path}")


def show_episode_comparison(
    early_stats: Dict[str, Any],
    late_stats: Dict[str, Any],
    save_path: str = "plots/episode_comparison.png",
) -> None:
    """Create a side-by-side comparison of early vs late episode behavior."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, (ax_early, ax_late) = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor(_BG_DARK)

    def _draw_episode_card(ax, stats, title, is_good):
        ax.set_facecolor(_BG_PANEL)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        border_color = "#22c55e" if is_good else "#ef4444"
        title_bg = "#1a3a2a" if is_good else "#3a1a1a"

        rect = FancyBboxPatch(
            (0.3, 8.5), 9.4, 1.2,
            boxstyle="round,pad=0.15",
            facecolor=title_bg, edgecolor=border_color, linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(5, 9.1, title, fontsize=16, fontweight="bold",
                color=_TEXT_COLOR, ha="center", va="center")

        f1 = stats["investigator_f1"]
        f1_color = "#22c55e" if f1 > 0.7 else "#f59e0b" if f1 > 0.4 else "#ef4444"
        ax.text(5, 7.5, f"F1 Score: {f1:.3f}", fontsize=28, fontweight="bold",
                color=f1_color, ha="center", va="center")

        info_lines = [
            ("Nodes Visited", str(len(stats.get("nodes_visited", [])))),
            ("Nodes Quarantined", str(stats["num_quarantined"])),
            ("Steps Taken", str(stats["steps_taken"])),
            ("Belief Calibration", f"{stats.get('belief_calibration', stats.get('belief_confidence', 0)):.2f}"),
            ("Intervention Type", stats["intervention_type"]),
            ("Correctly Identified", "YES" if stats["intervention_correctly_identified"] else "NO"),
            ("Quarantine Threshold", f"{stats['quarantine_threshold']:.3f}"),
            ("Exploration Rate", f"{stats['exploration_rate']:.3f}"),
        ]

        y_pos = 6.2
        for label, value in info_lines:
            ax.text(1.0, y_pos, label + ":", fontsize=11, color=_DIM_COLOR,
                    ha="left", va="center", fontfamily="monospace")
            v_color = _TEXT_COLOR
            if label == "Correctly Identified":
                v_color = "#22c55e" if value == "YES" else "#ef4444"
            ax.text(9.0, y_pos, value, fontsize=12, fontweight="bold",
                    color=v_color, ha="right", va="center", fontfamily="monospace")
            y_pos -= 0.7

        q_nodes = stats.get("nodes_quarantined_list", [])
        if q_nodes:
            ax.text(1.0, y_pos - 0.3, "Quarantined:", fontsize=10, color=_DIM_COLOR,
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

    fig.text(0.5, 0.5, "→", fontsize=48, color=_DIM_COLOR,
             ha="center", va="center", fontweight="bold")

    fig.suptitle(
        "RecallTrace — Before / After Self-Play Training",
        fontsize=18, color=_TEXT_COLOR, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.92,
        "Investigator behavior change: spray & pray → precision targeting",
        ha="center", fontsize=12, color=_DIM_COLOR,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved episode comparison to {save_path}")
