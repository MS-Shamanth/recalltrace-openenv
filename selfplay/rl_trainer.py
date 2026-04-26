"""PyTorch RL Self-Play Trainer for RecallTrace.

Trains a neural policy network against the adversary using REINFORCE.
Tracks all metrics for visualization and comparison with heuristic baseline.
This is commented.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List

from env.env import RecallTraceEnv
from env.models import RecallAction
from selfplay.adversary import AdversaryAgent
from selfplay.rl_model import RLInvestigatorAgent
from selfplay.scenario_gen import apply_intervention, compute_f1, generate_graph


class RLSelfPlayTrainer:
    """Trains a PyTorch RL policy via adversarial self-play."""

    def __init__(self, num_nodes: int = 10, lr: float = 3e-4):
        self.num_nodes = num_nodes
        self.adversary = AdversaryAgent(temperature=2.0, min_temperature=0.3)
        self.agent = RLInvestigatorAgent(lr=lr, gamma=0.99, entropy_coef=0.02)
        self.all_stats: List[Dict[str, Any]] = []

    def run_episode(self, episode_num: int, seed: int | None = None) -> Dict[str, Any]:
        """Run one RL episode."""
        rng = random.Random(seed)

        # 1) Generate graph
        graph_scenario = generate_graph(num_nodes=self.num_nodes, seed=seed)

        # 2) Adversary picks intervention
        intervention_type, target_node, num_hops, density_bucket = self.adversary.choose_intervention(
            graph_scenario, rng=rng,
        )
        graph_region = graph_scenario.get("_node_regions", {}).get(target_node, "downstream")

        # 3) Apply intervention
        scenario = apply_intervention(graph_scenario, intervention_type, target_node, num_hops, rng=rng)

        # 4) Create env
        env = RecallTraceEnv(scenario_data=scenario)
        observation = env.reset()

        # 5) RL agent runs episode
        self.agent.reset_episode()
        steps = 0
        done = False
        total_reward = 0.0

        while not done and steps < scenario["max_steps"]:
            action_dict = self.agent.act(observation, rng=rng)
            try:
                observation, reward, done, info = env.step(action_dict)
                self.agent.store_reward(reward)
                total_reward += reward
            except Exception:
                # Invalid action — penalize and skip
                self.agent.store_reward(-0.1)
                total_reward -= 0.1
            steps += 1

        # Force finalize
        if not done:
            try:
                observation, reward, done, info = env.step({"type": "finalize"})
                self.agent.store_reward(reward)
                total_reward += reward
            except Exception:
                self.agent.store_reward(0.0)
            steps += 1

        # 6) Compute F1
        quarantined_nodes = list(set(self.agent.nodes_quarantined))
        env_state = env.state()
        for node_id, node_data in env_state.state_data.get("nodes", {}).items():
            q_inv = node_data.get("quarantined_inventory", {})
            if q_inv and node_id not in quarantined_nodes:
                quarantined_nodes.append(node_id)

        f1, f1_details = compute_f1(scenario, quarantined_nodes)

        # 7) RL policy gradient update
        update_info = self.agent.update()

        # 8) Update adversary
        adversary_reward = self.adversary.update(intervention_type, graph_region, f1, density_bucket)

        # 9) Stats
        inv_summary = self.agent.get_episode_summary()
        correctly_identified = (
            inv_summary["intervention_guess"] == intervention_type
            if inv_summary["intervention_guess"] is not None
            else False
        )

        stats = {
            "episode": episode_num,
            "investigator_f1": round(f1, 4),
            "adversary_reward": round(adversary_reward, 4),
            "investigator_reward": round(total_reward, 4),
            "num_quarantined": len(quarantined_nodes),
            "intervention_type": intervention_type,
            "graph_region": graph_region,
            "density_bucket": density_bucket,
            "target_node": target_node,
            "num_hops": num_hops,
            "steps_taken": steps,
            "nodes_visited": [],
            "nodes_quarantined_list": sorted(set(quarantined_nodes)),
            "belief_confidence": inv_summary["belief_confidence"],
            "belief_calibration": inv_summary["belief_calibration"],
            "quarantine_threshold": 0.0,
            "exploration_rate": 0.0,
            "intervention_guess": inv_summary["intervention_guess"],
            "intervention_correctly_identified": correctly_identified,
            "f1_details": f1_details,
            "reward_components": {},
            "rl_metrics": update_info,
        }
        return stats

    def train(self, num_episodes: int = 200) -> List[Dict[str, Any]]:
        """Full RL training loop."""
        print(f"\n{'='*70}")
        print(f"  RecallTrace -- PyTorch RL Self-Play Training")
        print(f"  Episodes: {num_episodes} | Nodes: {self.num_nodes} | Model: PolicyNetwork(128)")
        print(f"{'='*70}\n")

        self.all_stats = []
        start_time = time.time()

        for ep in range(1, num_episodes + 1):
            stats = self.run_episode(episode_num=ep, seed=ep * 42)
            self.all_stats.append(stats)

            if ep % 20 == 0 or ep == 1:
                recent = self.all_stats[-20:] if len(self.all_stats) >= 20 else self.all_stats
                avg_f1 = sum(s["investigator_f1"] for s in recent) / len(recent)
                avg_adv = sum(s["adversary_reward"] for s in recent) / len(recent)
                avg_q = sum(s["num_quarantined"] for s in recent) / len(recent)
                avg_steps = sum(s["steps_taken"] for s in recent) / len(recent)
                avg_loss = sum(s["rl_metrics"].get("total_loss", 0) for s in recent) / len(recent)
                elapsed = time.time() - start_time

                print(
                    f"  Ep {ep:>4d} | "
                    f"F1: {avg_f1:.3f} | "
                    f"Adv: {avg_adv:+.3f} | "
                    f"Q: {avg_q:.1f} | "
                    f"Steps: {avg_steps:.1f} | "
                    f"Loss: {avg_loss:.3f} | "
                    f"Time: {elapsed:.1f}s"
                )

        elapsed = time.time() - start_time

        # Save model
        os.makedirs("checkpoints", exist_ok=True)
        self.agent.save("checkpoints/rl_policy.pt")

        print(f"\n  RL Training complete in {elapsed:.1f}s")
        print(f"  Model saved to checkpoints/rl_policy.pt")
        print(f"  Adversary strategy: {self.adversary.get_strategy_summary()}")

        # Performance summary
        early = self.all_stats[:20]
        late = self.all_stats[-20:]
        print(f"\n  Performance Summary (RL Agent):")
        print(f"    Early avg F1:          {sum(s['investigator_f1'] for s in early)/len(early):.3f}")
        print(f"    Late avg F1:           {sum(s['investigator_f1'] for s in late)/len(late):.3f}")
        print(f"    Early avg quarantined: {sum(s['num_quarantined'] for s in early)/len(early):.1f}")
        print(f"    Late avg quarantined:  {sum(s['num_quarantined'] for s in late)/len(late):.1f}")
        print()

        return self.all_stats
