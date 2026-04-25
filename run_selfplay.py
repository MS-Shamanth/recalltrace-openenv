#!/usr/bin/env python3
"""RecallTrace — Adversarial Self-Play Demo

Run 200 episodes of Investigator vs Adversary training, then generate:
  1. plots/selfplay_training.png  -- 4-panel training curves
  2. plots/episode_comparison.png -- before/after behavior comparison
  3. plots/before_after_demo.png  -- side-by-side graph replay (the money shot)

Usage:
    python run_selfplay.py

Designed to be Colab-runnable. No RL libraries needed.
Completes 200 episodes in under 5 minutes on CPU.
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from selfplay.trainer import SelfPlayTrainer
from selfplay.visualization import show_training_curves, show_episode_comparison
from selfplay.demo_replay import render_demo


def main() -> None:
    # --- Train ---
    trainer = SelfPlayTrainer(num_nodes=10)
    stats = trainer.train(num_episodes=200)

    # --- Plot training curves ---
    show_training_curves(stats, save_path="plots/selfplay_training.png")

    # --- Episode comparison: worst early vs best late ---
    # Find the episode with lowest F1 in first 30 episodes
    early_candidates = stats[:30]
    worst_early = min(early_candidates, key=lambda s: s["investigator_f1"])
    # Find the episode with highest F1 in last 30 episodes
    late_candidates = stats[-30:]
    best_late = max(late_candidates, key=lambda s: s["investigator_f1"])
    show_episode_comparison(
        worst_early,
        best_late,
        save_path="plots/episode_comparison.png",
    )

    # --- Demo replay visualization (the money shot) ---
    render_demo(save_path="plots/before_after_demo.png")

    # --- Print final summary ---
    print("\n" + "=" * 70)
    print("  SELF-PLAY TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Plots saved to:")
    print(f"    - plots/selfplay_training.png")
    print(f"    - plots/episode_comparison.png")
    print(f"    - plots/before_after_demo.png  (demo money shot)")

    early_stats = stats[:20]
    late_stats = stats[-20:]
    print(f"\n  Performance Summary:")
    print(f"    Early F1 (ep 1-20):   {sum(s['investigator_f1'] for s in early_stats)/len(early_stats):.3f}")
    print(f"    Late F1 (ep 181-200): {sum(s['investigator_f1'] for s in late_stats)/len(late_stats):.3f}")
    print(f"    Early quarantined:    {sum(s['num_quarantined'] for s in early_stats)/len(early_stats):.1f} nodes/ep")
    print(f"    Late quarantined:     {sum(s['num_quarantined'] for s in late_stats)/len(late_stats):.1f} nodes/ep")
    print(f"    Early steps:          {sum(s['steps_taken'] for s in early_stats)/len(early_stats):.1f} steps/ep")
    print(f"    Late steps:           {sum(s['steps_taken'] for s in late_stats)/len(late_stats):.1f} steps/ep")

    # Adversary evolution
    early_types = [s["intervention_type"] for s in early_stats]
    late_types = [s["intervention_type"] for s in late_stats]
    print(f"\n  Adversary Evolution:")
    for t in ["lot_relabel", "mixing_event", "record_deletion"]:
        early_pct = early_types.count(t) / len(early_types) * 100
        late_pct = late_types.count(t) / len(late_types) * 100
        print(f"    {t:20s}: {early_pct:5.1f}% (early) -> {late_pct:5.1f}% (late)")
    print()


if __name__ == "__main__":
    main()
