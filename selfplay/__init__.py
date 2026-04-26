"""RecallTrace adversarial self-play system.

Core components:
  - AdversaryAgent: chooses intervention placement to maximize investigator failure
  - InvestigatorAgent: learns to identify hidden interventions via tool calls
  - SelfPlayTrainer: orchestrates co-evolutionary training loop
  - BeliefStateTracker: tracks and visualizes P(contaminated) per node
"""

from selfplay.adversary import AdversaryAgent
from selfplay.investigator import InvestigatorAgent
from selfplay.trainer import SelfPlayTrainer
from selfplay.belief_tracker import BeliefStateTracker

__all__ = [
    "AdversaryAgent",
    "InvestigatorAgent",
    "SelfPlayTrainer",
    "BeliefStateTracker",
]
