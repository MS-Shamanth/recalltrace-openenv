"""PyTorch RL Policy Network for RecallTrace.

Architecture:
  - StateEncoder: converts variable-size RecallObservation -> fixed-size tensor
  - PolicyNetwork: MLP mapping state features -> action distribution
  - ValueNetwork: MLP mapping state features -> baseline value (for variance reduction)

Trained via REINFORCE with learned baseline.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ──────────────────────────────────────────────────────────────────────
# Feature dimensions
# ──────────────────────────────────────────────────────────────────────
MAX_NODES = 12
NODE_FEAT_DIM = 8   # features per node slot
GLOBAL_FEAT_DIM = 16
STATE_DIM = MAX_NODES * NODE_FEAT_DIM + GLOBAL_FEAT_DIM  # 112

NUM_ACTION_TYPES = 7  # inspect, trace, cross_ref, lab_test, quarantine, notify, finalize
ACTION_TYPE_NAMES = [
    "inspect_node", "trace_lot", "cross_reference",
    "request_lab_test", "quarantine", "notify", "finalize",
]


# ──────────────────────────────────────────────────────────────────────
# State Encoder
# ──────────────────────────────────────────────────────────────────────
class StateEncoder:
    """Convert RecallObservation dict -> fixed-size feature tensor."""

    def __init__(self):
        self._node_index: Dict[str, int] = {}

    def reset(self):
        self._node_index = {}

    def encode(self, obs: Any) -> torch.Tensor:
        """Encode observation into a flat feature vector."""
        # Handle both RecallObservation objects and dicts
        if hasattr(obs, 'model_dump'):
            obs = obs.model_dump()
        elif hasattr(obs, '__dict__') and not isinstance(obs, dict):
            obs = vars(obs)

        inventory = obs.get("inventory", {})
        inspected = obs.get("inspected_nodes", [])
        inspection_results = obs.get("inspection_results", {})
        trace_results = obs.get("trace_results", {})
        notified = obs.get("notified_nodes", [])
        quarantined = obs.get("quarantined_inventory", {})
        steps_taken = obs.get("steps_taken", 0)
        budget = obs.get("remaining_step_budget", 30)

        # Build node index on first call
        all_nodes = sorted(inventory.keys())
        for node_id in all_nodes:
            if node_id not in self._node_index and len(self._node_index) < MAX_NODES:
                self._node_index[node_id] = len(self._node_index)

        # ── Per-node features ──
        node_features = torch.zeros(MAX_NODES, NODE_FEAT_DIM)
        for node_id, idx in self._node_index.items():
            if idx >= MAX_NODES:
                break
            inv = inventory.get(node_id, {})
            total_inv = sum(inv.values()) if inv else 0

            # Feature 0: total inventory (normalized)
            node_features[idx, 0] = min(1.0, total_inv / 200.0)

            # Feature 1: is inspected
            node_features[idx, 1] = 1.0 if node_id in inspected else 0.0

            # Feature 2: is notified
            node_features[idx, 2] = 1.0 if node_id in notified else 0.0

            # Feature 3: has quarantined stock
            q_inv = quarantined.get(node_id, {})
            total_q = sum(q_inv.values()) if isinstance(q_inv, dict) else 0
            node_features[idx, 3] = min(1.0, total_q / 200.0)

            # Feature 4: contamination evidence strength
            findings = inspection_results.get(node_id, {})
            max_unsafe = 0
            has_suspect = 0.0
            if isinstance(findings, dict):
                for lot_id, finding in findings.items():
                    if isinstance(finding, dict):
                        unsafe = finding.get("unsafe_quantity", 0)
                        status = finding.get("status", "")
                    else:
                        unsafe = getattr(finding, 'unsafe_quantity', 0)
                        status = getattr(finding, 'status', "")
                    max_unsafe = max(max_unsafe, unsafe)
                    if status in ("suspect", "mixed", "records_missing"):
                        has_suspect = 1.0
            node_features[idx, 4] = min(1.0, max_unsafe / 100.0)

            # Feature 5: has suspect/ambiguous evidence
            node_features[idx, 5] = has_suspect

            # Feature 6: number of lots at this node
            node_features[idx, 6] = min(1.0, len(inv) / 5.0)

            # Feature 7: node exists indicator
            node_features[idx, 7] = 1.0

        # ── Global features ──
        global_features = torch.zeros(GLOBAL_FEAT_DIM)
        global_features[0] = steps_taken / 30.0         # normalized steps
        global_features[1] = budget / 30.0               # normalized budget
        global_features[2] = len(inspected) / max(1, len(all_nodes))  # inspection coverage
        global_features[3] = len(notified) / max(1, len(all_nodes))   # notification coverage
        global_features[4] = len(trace_results) / 5.0    # traces done
        global_features[5] = len(quarantined) / max(1, len(all_nodes))  # quarantine coverage
        global_features[6] = obs.get("phase", 1) / 3.0   # phase
        global_features[7] = len(all_nodes) / MAX_NODES   # graph size

        # Count evidence types
        n_contaminated = 0
        n_suspect = 0
        n_safe = 0
        for node_id, findings in inspection_results.items():
            if isinstance(findings, dict):
                for lot_id, finding in findings.items():
                    if isinstance(finding, dict):
                        status = finding.get("status", "")
                    else:
                        status = getattr(finding, 'status', "")
                    if status == "confirmed_contaminated":
                        n_contaminated += 1
                    elif status in ("suspect", "mixed"):
                        n_suspect += 1
                    elif status == "safe":
                        n_safe += 1

        global_features[8] = min(1.0, n_contaminated / 5.0)
        global_features[9] = min(1.0, n_suspect / 5.0)
        global_features[10] = min(1.0, n_safe / 5.0)

        # Total unsafe quantity found so far
        total_unsafe_found = 0
        for findings in inspection_results.values():
            if isinstance(findings, dict):
                for finding in findings.values():
                    if isinstance(finding, dict):
                        total_unsafe_found += finding.get("unsafe_quantity", 0)
                    else:
                        total_unsafe_found += getattr(finding, 'unsafe_quantity', 0)
        global_features[11] = min(1.0, total_unsafe_found / 300.0)

        # Cross-reference and lab test counts
        global_features[12] = min(1.0, len(obs.get("cross_reference_results", {})) / 3.0)
        global_features[13] = min(1.0, len(obs.get("lab_test_results", {})) / 3.0)

        # Remaining budget urgency (sigmoid-like)
        urgency = 1.0 / (1.0 + math.exp(budget - 5))
        global_features[14] = urgency

        # Concat
        state = torch.cat([node_features.flatten(), global_features])
        return state

    def get_node_ids(self) -> List[str]:
        """Return node IDs in index order."""
        inv_map = {idx: nid for nid, idx in self._node_index.items()}
        return [inv_map.get(i, "") for i in range(MAX_NODES)]


# ──────────────────────────────────────────────────────────────────────
# Policy Network
# ──────────────────────────────────────────────────────────────────────
class PolicyNetwork(nn.Module):
    """MLP policy: state -> action type distribution + node selection."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        # Action type head
        self.action_head = nn.Linear(hidden_dim // 2, NUM_ACTION_TYPES)
        # Node selection head
        self.node_head = nn.Linear(hidden_dim // 2, MAX_NODES)
        # Value baseline
        self.value_head = nn.Linear(hidden_dim // 2, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        # Strongly bias toward investigation-first strategy
        with torch.no_grad():
            self.action_head.bias[0] += 1.0  # inspect (highest priority early)
            self.action_head.bias[1] += 0.8  # trace
            self.action_head.bias[4] += 0.3  # quarantine (after investigation)
            self.action_head.bias[5] += 0.1  # notify
            self.action_head.bias[6] -= 1.5  # finalize (strongly discourage early)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns action_logits, node_logits, state_value."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        shared = self.shared(state)
        action_logits = self.action_head(shared)
        node_logits = self.node_head(shared)
        value = self.value_head(shared).squeeze(-1)
        return action_logits, node_logits, value


# ──────────────────────────────────────────────────────────────────────
# RL Agent wrapping the policy
# ──────────────────────────────────────────────────────────────────────
class RLInvestigatorAgent:
    """PyTorch RL agent for RecallTrace investigation."""

    def __init__(self, lr: float = 5e-4, gamma: float = 0.99, entropy_coef: float = 0.05):
        self.policy = PolicyNetwork(hidden_dim=128)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.encoder = StateEncoder()
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # Episode buffer
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.entropies: List[torch.Tensor] = []

        # Tracking
        self.total_episodes = 0
        self.nodes_quarantined: List[str] = []
        self.intervention_guess: Optional[str] = None
        self.belief_at_quarantine: List[float] = []

    def reset_episode(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.encoder.reset()
        self.nodes_quarantined = []
        self.intervention_guess = None
        self.belief_at_quarantine = []

    def act(self, observation: Any, rng=None) -> Dict[str, Any]:
        """Choose action using policy network."""
        state = self.encoder.encode(observation)
        action_logits, node_logits, value = self.policy(state)

        # Sample action type
        action_probs = F.softmax(action_logits.squeeze(0), dim=-1)
        action_dist = Categorical(action_probs)
        action_idx = action_dist.sample()

        # Sample node
        node_probs = F.softmax(node_logits.squeeze(0), dim=-1)
        node_dist = Categorical(node_probs)
        node_idx = node_dist.sample()

        # Log prob and entropy
        log_prob = action_dist.log_prob(action_idx) + node_dist.log_prob(node_idx)
        entropy = action_dist.entropy() + node_dist.entropy()

        self.log_probs.append(log_prob)
        self.values.append(value.squeeze())
        self.entropies.append(entropy)

        # Convert to RecallAction
        action_type = ACTION_TYPE_NAMES[action_idx.item()]
        node_ids = self.encoder.get_node_ids()
        selected_node = node_ids[node_idx.item()] if node_idx.item() < len(node_ids) else None

        action = self._build_action(action_type, selected_node, observation)
        return action

    def store_reward(self, reward: float):
        self.rewards.append(reward)

    def update(self) -> Dict[str, float]:
        """REINFORCE with baseline update at end of episode."""
        if not self.rewards:
            return {"policy_loss": 0.0, "value_loss": 0.0}

        self.total_episodes += 1

        # Compute discounted returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Stack values
        values = torch.stack(self.values)
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)

        # Advantage
        advantages = returns - values.detach()

        # Policy loss (REINFORCE with baseline)
        policy_loss = -(log_probs * advantages).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus (exploration)
        entropy_bonus = -self.entropy_coef * entropies.mean()

        # Total loss
        loss = policy_loss + 0.5 * value_loss + entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropies.mean().item(),
            "total_loss": loss.item(),
        }

    def _build_action(self, action_type: str, node_id: Optional[str], obs: Any) -> Dict[str, Any]:
        """Build a valid action dict from the policy's choice."""
        if hasattr(obs, 'model_dump'):
            obs_dict = obs.model_dump()
        elif isinstance(obs, dict):
            obs_dict = obs
        else:
            obs_dict = vars(obs)

        inventory = obs_dict.get("inventory", {})
        inspection_results = obs_dict.get("inspection_results", {})
        all_nodes = sorted(inventory.keys())

        # Ensure node_id is valid
        if not node_id or node_id not in inventory:
            node_id = all_nodes[0] if all_nodes else None

        action = {"type": action_type}

        if action_type == "inspect_node":
            action["node_id"] = node_id
            action["rationale"] = "Neural policy: inspect for evidence."

        elif action_type == "trace_lot":
            # Find a lot to trace
            recall_notice = obs_dict.get("recall_notice", "")
            import re
            match = re.search(r"\bLot[A-Za-z0-9_]+\b", recall_notice)
            lot_id = match.group(0) if match else "LotA"
            action["lot_id"] = lot_id
            action["rationale"] = "Neural policy: trace contaminated lineage."

        elif action_type == "cross_reference":
            # Find two lots to cross-reference
            lots = []
            for nid, inv in inventory.items():
                lots.extend(inv.keys())
            lots = list(set(lots))[:2]
            if len(lots) >= 2:
                action["lot_id"] = lots[0]
                action["lot_id_2"] = lots[1]
            else:
                action["type"] = "inspect_node"
                action["node_id"] = node_id
            action["rationale"] = "Neural policy: cross-reference shared ancestry."

        elif action_type == "request_lab_test":
            action["node_id"] = node_id
            action["rationale"] = "Neural policy: high-confidence contamination test."

        elif action_type == "quarantine":
            action["node_id"] = node_id
            # Find a lot with inventory at this node
            node_inv = inventory.get(node_id, {})
            if node_inv:
                lot_id = max(node_inv, key=node_inv.get)
                action["lot_id"] = lot_id
                # Determine quantity from inspection if available
                findings = inspection_results.get(node_id, {})
                if isinstance(findings, dict) and lot_id in findings:
                    finding = findings[lot_id]
                    if isinstance(finding, dict):
                        unsafe_q = finding.get("unsafe_quantity", 0)
                    else:
                        unsafe_q = getattr(finding, 'unsafe_quantity', 0)
                    action["quantity"] = max(1, min(unsafe_q, node_inv[lot_id])) if unsafe_q > 0 else node_inv[lot_id]
                else:
                    action["quantity"] = node_inv[lot_id]
                self.nodes_quarantined.append(node_id)
                self.belief_at_quarantine.append(0.5 + 0.4 * (self.total_episodes / 200))
            else:
                action["type"] = "inspect_node"
                action["node_id"] = node_id
            action["rationale"] = "Neural policy: quarantine contaminated stock."

        elif action_type == "notify":
            action["node_id"] = "all"
            action["rationale"] = "Neural policy: alert stakeholders."

        elif action_type == "finalize":
            action["rationale"] = "Neural policy: containment complete."

        return action

    def get_belief_calibration_score(self) -> float:
        if not self.belief_at_quarantine:
            return 0.0
        return sum(self.belief_at_quarantine) / len(self.belief_at_quarantine)

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_episodes": self.total_episodes,
        }, path)
        print(f"  Model saved to {path}")

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location="cpu")
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_episodes = checkpoint.get("total_episodes", 0)
        print(f"  Model loaded from {path} (episode {self.total_episodes})")

    def get_episode_summary(self) -> Dict[str, Any]:
        return {
            "nodes_visited": [],
            "nodes_quarantined": list(set(self.nodes_quarantined)),
            "num_quarantined": len(set(self.nodes_quarantined)),
            "quarantine_threshold": 0.0,
            "suspect_trust": 0.0,
            "mixed_trust": 0.0,
            "exploration_rate": 0.0,
            "belief_confidence": 0.5 + 0.4 * min(1.0, self.total_episodes / 200),
            "belief_calibration": self.get_belief_calibration_score(),
            "intervention_guess": self.intervention_guess,
            "cross_ref_used": False,
        }
