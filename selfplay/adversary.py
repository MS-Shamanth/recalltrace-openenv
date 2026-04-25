"""Adversary agent for adversarial self-play.

The Adversary chooses WHAT hidden intervention to apply and WHERE to
apply it in the supply-chain graph, trying to make the Investigator fail.

Policy: softmax score table over (intervention_type x graph_region).
Lower Investigator F1 = higher probability of picking that cell.
Temperature decays from exploration to exploitation over training.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import numpy as np


INTERVENTION_TYPES = ["lot_relabel", "mixing_event", "record_deletion"]
GRAPH_REGIONS = ["source", "midstream", "downstream"]

DEFAULT_HOPS = {
    "lot_relabel": 2,
    "mixing_event": 2,
    "record_deletion": 1,
}


class AdversaryAgent:
    """Chooses intervention placement to maximize Investigator failure."""

    def __init__(self, temperature: float = 2.0, min_temperature: float = 0.3):
        self.score_table = np.full((3, 3), 0.5, dtype=np.float64)
        self.update_counts = np.zeros_like(self.score_table, dtype=np.int32)
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.initial_temperature = temperature
        self.total_updates = 0
        self.history: List[Dict[str, Any]] = []

    def choose_intervention(
        self, scenario: Dict[str, Any], rng: random.Random | None = None,
    ) -> Tuple[str, str, int]:
        """Pick (intervention_type, target_node, num_hops)."""
        rng = rng or random.Random()
        logits = -self.score_table / max(self.temperature, 0.01)
        flat = logits.flatten()
        flat -= flat.max()
        probs = np.exp(flat)
        probs /= probs.sum()

        cell = rng.choices(range(len(probs)), weights=probs.tolist(), k=1)[0]
        t_idx, r_idx = divmod(cell, 3)
        intervention_type = INTERVENTION_TYPES[t_idx]
        target_region = GRAPH_REGIONS[r_idx]

        region_nodes = [
            n for n, r in scenario.get("_node_regions", {}).items() if r == target_region
        ]
        if not region_nodes:
            region_nodes = scenario.get("_all_node_ids", list(scenario["nodes"].keys()))
        target_node = rng.choice(region_nodes)
        num_hops = DEFAULT_HOPS.get(intervention_type, 1) + rng.randint(0, 1)
        return intervention_type, target_node, num_hops

    def update(self, intervention_type: str, graph_region: str, investigator_f1: float) -> float:
        """EMA update of score table. Returns adversary reward."""
        ti = INTERVENTION_TYPES.index(intervention_type)
        ri = GRAPH_REGIONS.index(graph_region)
        self.score_table[ti, ri] = 0.85 * self.score_table[ti, ri] + 0.15 * investigator_f1
        self.update_counts[ti, ri] += 1
        self.total_updates += 1
        self.temperature = max(self.min_temperature, self.initial_temperature * (0.985 ** self.total_updates))
        reward = self._compute_reward(investigator_f1)
        self.history.append({
            "intervention_type": intervention_type, "graph_region": graph_region,
            "investigator_f1": round(investigator_f1, 4), "adversary_reward": round(reward, 4),
        })
        return reward

    @staticmethod
    def _compute_reward(f1: float) -> float:
        if f1 < 0.5:
            return 1.0
        elif f1 > 0.8:
            return -1.0
        return 1.0 - 2.0 * (f1 - 0.5) / 0.3

    def get_strategy_summary(self) -> Dict[str, Any]:
        best = np.unravel_index(np.argmin(self.score_table), self.score_table.shape)
        return {
            "preferred_intervention": INTERVENTION_TYPES[best[0]],
            "preferred_region": GRAPH_REGIONS[best[1]],
            "temperature": round(self.temperature, 4),
            "total_updates": self.total_updates,
        }
