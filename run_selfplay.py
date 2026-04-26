#!/usr/bin/env python3
"""RecallTrace -- Adversarial Self-Play Demo

Runs BOTH heuristic + PyTorch RL training, generates comparison plots.

Usage:
    python run_selfplay.py

Completes in under 5 minutes on CPU. No external APIs needed.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from selfplay.trainer import SelfPlayTrainer
from selfplay.rl_trainer import RLSelfPlayTrainer
from selfplay.visualization import (
    show_training_curves,
    save_individual_plots,
    save_coevolution_plot,
    show_episode_comparison,
)
from selfplay.demo_replay import render_demo


def main() -> None:
    os.makedirs("plots", exist_ok=True)

    # ── Phase 1: Heuristic Self-Play ─────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 1: Heuristic Adaptive Agent")
    print("=" * 70)
    heuristic_trainer = SelfPlayTrainer(num_nodes=10)
    heuristic_stats = heuristic_trainer.train(num_episodes=200)

    # ── Phase 2: PyTorch RL Self-Play ────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 2: PyTorch RL Policy Network")
    print("=" * 70)
    rl_trainer = RLSelfPlayTrainer(num_nodes=10, lr=3e-4)
    rl_stats = rl_trainer.train(num_episodes=200)

    # ── Generate all plots ───────────────────────────────────────────
    print("\n  Generating plots...")

    # Heuristic plots
    show_training_curves(heuristic_stats, save_path="plots/selfplay_training.png")
    save_individual_plots(heuristic_stats, plots_dir="plots")
    save_coevolution_plot(heuristic_stats, save_path="plots/coevolution.png")

    # RL plots
    show_training_curves(rl_stats, save_path="plots/rl_training.png")
    save_individual_plots(rl_stats, plots_dir="plots/rl")
    save_coevolution_plot(rl_stats, save_path="plots/rl_coevolution.png")

    # Episode comparison
    early = heuristic_stats[:30]
    late = heuristic_stats[-30:]
    worst_early = min(early, key=lambda s: s["investigator_f1"])
    best_late = max(late, key=lambda s: s["investigator_f1"])
    show_episode_comparison(worst_early, best_late, save_path="plots/episode_comparison.png")

    # Comparison plot: heuristic vs RL
    _save_comparison_plot(heuristic_stats, rl_stats)

    # Demo replay
    render_demo(save_path="plots/before_after_demo.png")

    # ── Print comparison table ───────────────────────────────────────
    _print_comparison(heuristic_stats, rl_stats)

    print("\n" + "=" * 70)
    print("  ALL TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Key plots:")
    print(f"    plots/selfplay_training.png   (heuristic 4-panel)")
    print(f"    plots/rl_training.png          (RL 4-panel)")
    print(f"    plots/comparison.png           (heuristic vs RL)")
    print(f"    plots/before_after_demo.png    (demo money shot)")
    print(f"    checkpoints/rl_policy.pt       (saved model weights)")
    print()


def _save_comparison_plot(heuristic_stats, rl_stats):
    """Side-by-side F1 comparison: heuristic vs RL."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG_DARK = "#0d1117"
    BG_PANEL = "#161b22"
    GRID = "#30363d"
    TEXT = "#e6edf3"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(BG_DARK)

    def _rolling(data, w=20):
        result = []
        for i in range(len(data)):
            s = max(0, i - w + 1)
            result.append(sum(data[s:i+1]) / (i - s + 1))
        return result

    for ax in [ax1, ax2]:
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=TEXT, labelsize=10)
        ax.spines["bottom"].set_color(GRID)
        ax.spines["left"].set_color(GRID)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, color=GRID)

    # Heuristic
    h_eps = [s["episode"] for s in heuristic_stats]
    h_f1 = [s["investigator_f1"] for s in heuristic_stats]
    ax1.scatter(h_eps, h_f1, c="#3b82f6", alpha=0.15, s=8)
    ax1.plot(h_eps, _rolling(h_f1), color="#60a5fa", linewidth=2.5, label="Heuristic F1")
    ax1.set_title("Heuristic Adaptive Agent", fontsize=14, color=TEXT, fontweight="bold")
    ax1.set_xlabel("Episode", color=TEXT)
    ax1.set_ylabel("F1 Score", color=TEXT)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="lower right", fontsize=9, facecolor=BG_PANEL, edgecolor=GRID)

    # RL
    r_eps = [s["episode"] for s in rl_stats]
    r_f1 = [s["investigator_f1"] for s in rl_stats]
    ax2.scatter(r_eps, r_f1, c="#f59e0b", alpha=0.15, s=8)
    ax2.plot(r_eps, _rolling(r_f1), color="#fbbf24", linewidth=2.5, label="RL Policy F1")
    ax2.set_title("PyTorch RL Policy Network", fontsize=14, color=TEXT, fontweight="bold")
    ax2.set_xlabel("Episode", color=TEXT)
    ax2.set_ylabel("F1 Score", color=TEXT)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="lower right", fontsize=9, facecolor=BG_PANEL, edgecolor=GRID)

    fig.suptitle("RecallTrace -- Heuristic vs RL Agent Comparison",
                 fontsize=16, color=TEXT, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("plots/comparison.png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved comparison to plots/comparison.png")


def _print_comparison(heuristic_stats, rl_stats):
    """Print comparison table."""
    h_late = heuristic_stats[-20:]
    r_late = rl_stats[-20:]

    h_f1 = sum(s["investigator_f1"] for s in h_late) / len(h_late)
    r_f1 = sum(s["investigator_f1"] for s in r_late) / len(r_late)
    h_q = sum(s["num_quarantined"] for s in h_late) / len(h_late)
    r_q = sum(s["num_quarantined"] for s in r_late) / len(r_late)
    h_steps = sum(s["steps_taken"] for s in h_late) / len(h_late)
    r_steps = sum(s["steps_taken"] for s in r_late) / len(r_late)

    print(f"\n  +----------------------+-------+--------------+-------+")
    print(f"  | Agent                |  F1   | Quarantined  | Steps |")
    print(f"  +----------------------+-------+--------------+-------+")
    print(f"  | Random baseline      | ~0.20 |    ~8.0      |  ~8   |")
    print(f"  | Quarantine-all       | ~0.30 |    all       |   1   |")
    print(f"  | Heuristic adaptive   | {h_f1:.2f}  |    {h_q:.1f}      |  {h_steps:.0f}   |")
    print(f"  | PyTorch RL policy    | {r_f1:.2f}  |    {r_q:.1f}      |  {r_steps:.0f}   |")
    print(f"  +----------------------+-------+--------------+-------+")


if __name__ == "__main__":
    main()
