"""Adversary agent for adversarial self-play.

The Adversary chooses WHAT hidden intervention to apply and WHERE to
apply it in the contamination propagation graph, trying to make the
Investigator fail.

Policy: softmax score table over (intervention_type x graph_region x density).
Lower Investigator F1 = higher probability of picking that cell.
Temperature decays from exploration to exploitation over training.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import numpy as np


INTERVENTION_TYPES = ["lot_relabel", "mixing_event", "record_deletion"]
GRAPH_REGIONS = ["source", "midstream", "downstream"]
DENSITY_BUCKETS = ["sparse", "dense"]

DEFAULT_HOPS = {
    "lot_relabel": 2,
    "mixing_event": 2,
    "record_deletion": 1,
}


class AdversaryAgent:
    """Chooses intervention placement to maximize Investigator failure.

    Score table has 3 dimensions: intervention_type x graph_region x density_bucket
    = 3 x 3 x 2 = 18 cells. Softmax over these 18 entries drives the adversary
    toward placements that historically caused the investigator to struggle.
    """

    def __init__(self, temperature: float = 2.0, min_temperature: float = 0.3):
        # 3D score table: intervention_type x graph_region x density_bucket
        self.score_table = np.full((3, 3, 2), 0.5, dtype=np.float64)
        self.update_counts = np.zeros_like(self.score_table, dtype=np.int32)
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.initial_temperature = temperature
        self.total_updates = 0
        self.history: List[Dict[str, Any]] = []

    def choose_intervention(
        self, scenario: Dict[str, Any], rng: random.Random | None = None,
    ) -> Tuple[str, str, int, str]:
        """Pick (intervention_type, target_node, num_hops, density_bucket)."""
        rng = rng or random.Random()

        # Compute density bucket for this graph
        density = self._compute_density(scenario)

        logits = -self.score_table / max(self.temperature, 0.01)
        flat = logits.flatten()
        flat -= flat.max()
        probs = np.exp(flat)
        probs /= probs.sum()

        cell = rng.choices(range(len(probs)), weights=probs.tolist(), k=1)[0]
        t_idx = cell // 6
        remainder = cell % 6
        r_idx = remainder // 2
        d_idx = remainder % 2

        intervention_type = INTERVENTION_TYPES[t_idx]
        target_region = GRAPH_REGIONS[r_idx]
        density_bucket = DENSITY_BUCKETS[d_idx]

        region_nodes = [
            n for n, r in scenario.get("_node_regions", {}).items() if r == target_region
        ]
        if not region_nodes:
            region_nodes = scenario.get("_all_node_ids", list(scenario["nodes"].keys()))
        target_node = rng.choice(region_nodes)
        num_hops = DEFAULT_HOPS.get(intervention_type, 1) + rng.randint(0, 1)
        return intervention_type, target_node, num_hops, density_bucket

    def update(self, intervention_type: str, graph_region: str, investigator_f1: float, density_bucket: str = "sparse") -> float:
        """EMA update of score table. Returns adversary reward."""
        ti = INTERVENTION_TYPES.index(intervention_type)
        ri = GRAPH_REGIONS.index(graph_region)
        di = DENSITY_BUCKETS.index(density_bucket) if density_bucket in DENSITY_BUCKETS else 0

        self.score_table[ti, ri, di] = 0.85 * self.score_table[ti, ri, di] + 0.15 * investigator_f1
        self.update_counts[ti, ri, di] += 1
        self.total_updates += 1
        self.temperature = max(self.min_temperature, self.initial_temperature * (0.985 ** self.total_updates))
        reward = self._compute_reward(investigator_f1)
        self.history.append({
            "intervention_type": intervention_type, "graph_region": graph_region,
            "density_bucket": density_bucket,
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

    @staticmethod
    def _compute_density(scenario: Dict[str, Any]) -> str:
        """Classify graph as sparse or dense based on edge density."""
        graph = scenario.get("shipment_graph", {})
        num_nodes = len(graph)
        num_edges = sum(len(targets) for targets in graph.values())
        if num_nodes <= 1:
            return "sparse"
        density = num_edges / (num_nodes * (num_nodes - 1))
        return "dense" if density > 0.15 else "sparse"

    def get_strategy_summary(self) -> Dict[str, Any]:
        best = np.unravel_index(np.argmin(self.score_table), self.score_table.shape)
        return {
            "preferred_intervention": INTERVENTION_TYPES[best[0]],
            "preferred_region": GRAPH_REGIONS[best[1]],
            "preferred_density": DENSITY_BUCKETS[best[2]],
            "temperature": round(self.temperature, 4),
            "total_updates": self.total_updates,
            "score_table_shape": list(self.score_table.shape),
        }
