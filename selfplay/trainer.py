"""Self-play training loop for RecallTrace.

Runs episodes where the Adversary picks intervention placements and the
Investigator tries to find them. Both agents update after each episode.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List

from env.env import RecallTraceEnv
from selfplay.adversary import AdversaryAgent, GRAPH_REGIONS
from selfplay.investigator import InvestigatorAgent
from selfplay.scenario_gen import apply_intervention, compute_f1, generate_graph


class SelfPlayTrainer:
    """Orchestrates adversarial self-play between Investigator and Adversary."""

    def __init__(self, num_nodes: int = 12):
        self.num_nodes = num_nodes
        self.adversary = AdversaryAgent(temperature=2.0, min_temperature=0.3)
        self.investigator = InvestigatorAgent()
        self.all_stats: List[Dict[str, Any]] = []

    def run_episode(self, episode_num: int, seed: int | None = None) -> Dict[str, Any]:
        """Run a single self-play episode. Returns episode stats dict."""
        rng = random.Random(seed)

        # 1) Generate a fresh supply-chain graph
        graph_scenario = generate_graph(num_nodes=self.num_nodes, seed=seed)

        # 2) Adversary picks intervention
        intervention_type, target_node, num_hops = self.adversary.choose_intervention(
            graph_scenario, rng=rng,
        )

        # Determine graph region of target node
        graph_region = graph_scenario.get("_node_regions", {}).get(target_node, "downstream")

        # 3) Apply intervention to scenario
        scenario = apply_intervention(
            graph_scenario, intervention_type, target_node, num_hops, rng=rng,
        )

        # 4) Create environment and reset
        env = RecallTraceEnv(scenario_data=scenario)
        observation = env.reset()

        # 5) Investigator runs the episode
        self.investigator.reset_episode()
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < scenario["max_steps"]:
            action = self.investigator.act(observation, rng=rng)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

        # Force finalize if not done
        if not done:
            action = self.investigator.act(observation, rng=rng)
            if action.type.value != "finalize":
                from env.models import RecallAction
                action = RecallAction(type="finalize", rationale="Budget exhausted.")
            observation, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

        # 6) Compute F1 from quarantine results
        quarantined_nodes = list(set(self.investigator.nodes_quarantined))
        # Also check env state for quarantined inventory
        env_state = env.state()
        for node_id, node_data in env_state.state_data.get("nodes", {}).items():
            q_inv = node_data.get("quarantined_inventory", {})
            if q_inv and node_id not in quarantined_nodes:
                quarantined_nodes.append(node_id)

        f1, f1_details = compute_f1(scenario, quarantined_nodes)
        quarantine_match = info.get("quarantine_match", {}) if isinstance(info, dict) else {}
        if not quarantine_match:
            quarantine_match = env._compute_quarantine_match()
        remaining_contaminated_nodes = len(quarantine_match.get("missing_quantities", {}))
        total_contaminated_nodes = len(env_state.ground_truth.get("affected_nodes", []))

        # 7) Compute investigator reward with the specified reward structure
        inv_reward = 0.0
        tp = f1_details["tp"]
        fp = f1_details["fp"]
        inv_reward += tp * 2.0       # +2.0 per correctly quarantined unsafe node
        inv_reward += fp * (-1.5)    # -1.5 per safe node wrongly blocked
        inv_reward += steps * (-0.05)  # -0.05 per step
        # Belief calibration bonus
        if f1 > 0.6:
            inv_reward += 0.3

        # 8) Update both agents
        adversary_reward = self.adversary.update(intervention_type, graph_region, f1)
        self.investigator.update(inv_reward, f1, steps)

        # 9) Build stats dict
        inv_summary = self.investigator.get_episode_summary()
        correctly_identified = (
            inv_summary["intervention_guess"] == intervention_type
            if inv_summary["intervention_guess"] is not None
            else False
        )

        stats = {
            "episode": episode_num,
            "investigator_f1": round(f1, 4),
            "adversary_reward": round(adversary_reward, 4),
            "investigator_reward": round(inv_reward, 4),
            "num_quarantined": len(quarantined_nodes),
            "remaining_contaminated_nodes": remaining_contaminated_nodes,
            "total_contaminated_nodes": total_contaminated_nodes,
            "contamination_reduction_rate": round(
                max(0.0, 1.0 - remaining_contaminated_nodes / max(total_contaminated_nodes, 1)), 4
            ),
            "root_cause_accuracy": 1.0 if correctly_identified else 0.0,
            "intervention_type": intervention_type,
            "graph_region": graph_region,
            "target_node": target_node,
            "num_hops": num_hops,
            "steps_taken": steps,
            "nodes_visited": inv_summary["nodes_visited"],
            "nodes_quarantined_list": sorted(set(quarantined_nodes)),
            "belief_confidence": inv_summary["belief_confidence"],
            "quarantine_threshold": inv_summary["quarantine_threshold"],
            "exploration_rate": inv_summary["exploration_rate"],
            "intervention_guess": inv_summary["intervention_guess"],
            "intervention_correctly_identified": correctly_identified,
            "f1_details": f1_details,
        }
        return stats

    def train(self, num_episodes: int = 200) -> List[Dict[str, Any]]:
        """Run the full self-play training loop."""
        print(f"\n{'='*70}")
        print(f"  RecallTrace — Adversarial Self-Play Training")
        print(f"  Episodes: {num_episodes} | Nodes per graph: {self.num_nodes}")
        print(f"{'='*70}\n")

        self.all_stats = []
        start_time = time.time()

        for ep in range(1, num_episodes + 1):
            stats = self.run_episode(episode_num=ep, seed=ep * 42)
            self.all_stats.append(stats)

            # Progress logging every 20 episodes
            if ep % 20 == 0 or ep == 1:
                recent = self.all_stats[-20:] if len(self.all_stats) >= 20 else self.all_stats
                avg_f1 = sum(s["investigator_f1"] for s in recent) / len(recent)
                avg_adv = sum(s["adversary_reward"] for s in recent) / len(recent)
                avg_q = sum(s["num_quarantined"] for s in recent) / len(recent)
                avg_steps = sum(s["steps_taken"] for s in recent) / len(recent)
                elapsed = time.time() - start_time

                print(
                    f"  Episode {ep:>4d} | "
                    f"F1: {avg_f1:.3f} | "
                    f"Adv Reward: {avg_adv:+.3f} | "
                    f"Quarantined: {avg_q:.1f} | "
                    f"Steps: {avg_steps:.1f} | "
                    f"Time: {elapsed:.1f}s"
                )

        elapsed = time.time() - start_time
        print(f"\n  Training complete in {elapsed:.1f}s")
        print(f"  Adversary strategy: {self.adversary.get_strategy_summary()}")

        # Print summary
        early = self.all_stats[:20]
        late = self.all_stats[-20:]
        print(f"\n  Early avg F1:  {sum(s['investigator_f1'] for s in early)/len(early):.3f}")
        print(f"  Late avg F1:   {sum(s['investigator_f1'] for s in late)/len(late):.3f}")
        print(f"  Early avg quarantined: {sum(s['num_quarantined'] for s in early)/len(early):.1f}")
        print(f"  Late avg quarantined:  {sum(s['num_quarantined'] for s in late)/len(late):.1f}")
        print()

        return self.all_stats

    @staticmethod
    def get_training_curves(stats: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Extract plottable series from training stats."""
        return {
            "episodes": [s["episode"] for s in stats],
            "investigator_f1": [s["investigator_f1"] for s in stats],
            "adversary_reward": [s["adversary_reward"] for s in stats],
            "num_quarantined": [s["num_quarantined"] for s in stats],
            "steps_taken": [s["steps_taken"] for s in stats],
            "quarantine_threshold": [s["quarantine_threshold"] for s in stats],
            "exploration_rate": [s["exploration_rate"] for s in stats],
            "belief_confidence": [s["belief_confidence"] for s in stats],
            "remaining_contaminated_nodes": [s.get("remaining_contaminated_nodes", 0) for s in stats],
        }
