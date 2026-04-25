"""Adversarial self-play module for RecallTrace.

Two agents co-evolve in a shared environment:
  - InvestigatorAgent: finds and quarantines contaminated nodes.
  - AdversaryAgent: chooses where and how to hide contamination.
"""

from selfplay.adversary import AdversaryAgent
from selfplay.investigator import InvestigatorAgent
from selfplay.trainer import SelfPlayTrainer
from selfplay.visualization import show_training_curves, show_episode_comparison
from selfplay.demo_replay import render_demo
from selfplay.belief_tracker import BeliefStateTracker

__all__ = [
    "AdversaryAgent",
    "InvestigatorAgent",
    "SelfPlayTrainer",
    "show_training_curves",
    "show_episode_comparison",
    "render_demo",
    "BeliefStateTracker",
]
