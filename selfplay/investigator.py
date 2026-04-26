"""Investigator agent for adversarial self-play.

Wraps the heuristic baseline with LEARNABLE parameters that determine
how the agent interprets ambiguous evidence. Early on it trusts everything
and quarantines aggressively (spray & pray -> F1 ~0.3). Over training
it learns to distinguish real contamination from decoys.

Key learning parameters:
  - quarantine_threshold: min evidence strength needed to quarantine
  - suspect_trust: how much to trust "suspect" evidence (starts HIGH -> learns LOW)
  - mixed_trust: how much to trust "mixed" evidence (starts HIGH -> learns optimal)
  - exploration_rate: probability of inspecting non-traced nodes
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from env.models import RecallAction, RecallObservation


class InvestigatorAgent:
    """Investigator that learns from episode rewards over self-play."""

    def __init__(self):
        # Learnable parameters
        self.quarantine_threshold = 0.0    # starts at 0: quarantine EVERYTHING
        self.suspect_trust = 1.0           # starts at MAX: treats all suspects as guilty
        self.mixed_trust = 0.95            # starts near max: quarantines all mixed lots
        self.exploration_rate = 0.95       # starts very high — visits every node
        self.belief_confidence = 0.1

        # Learning rates
        self.threshold_lr = 0.004
        self.trust_lr = 0.005

        # Episode tracking
        self.nodes_visited: List[str] = []
        self.nodes_quarantined: List[str] = []
        self.quarantine_decisions: List[Dict[str, Any]] = []
        self.intervention_guess: Optional[str] = None
        self.total_episodes = 0
        self.cross_ref_done: bool = False

        # Belief calibration tracking
        self.belief_at_quarantine: List[float] = []

        # Adaptation history
        self._f1_history: List[float] = []

    def reset_episode(self) -> None:
        """Reset per-episode state."""
        self.nodes_visited = []
        self.nodes_quarantined = []
        self.quarantine_decisions = []
        self.intervention_guess = None
        self.cross_ref_done = False
        self.belief_at_quarantine = []
        self.belief_confidence = max(0.1, min(0.95, 0.1 + self.total_episodes * 0.004))

    def act(self, observation: RecallObservation, rng: random.Random | None = None) -> RecallAction:
        """Choose the next action based on observation and learned parameters."""
        rng = rng or random.Random()

        root_lot = self._extract_root_lot(observation)
        trace_result = observation.trace_results.get(root_lot)

        # Step 1: Trace the contaminated lot first
        if trace_result is None:
            return RecallAction(type="trace_lot", lot_id=root_lot,
                                rationale="Map the recall lineage first.")

        affected_nodes = trace_result.get("affected_nodes", [])

        # Step 2: Inspect affected nodes
        for node_id in affected_nodes:
            if node_id not in observation.inspected_nodes:
                self.nodes_visited.append(node_id)
                return RecallAction(type="inspect_node", node_id=node_id,
                                    rationale="Collect evidence.")

        # Step 2.5: Cross-reference suspect lots if we haven't yet
        # This is the causal reasoning step — learned behavior
        if not self.cross_ref_done and self.total_episodes > 15:
            suspect_lots = self._find_suspect_lots(observation)
            if len(suspect_lots) >= 2:
                lot_a, lot_b = suspect_lots[0], suspect_lots[1]
                self.cross_ref_done = True
                return RecallAction(
                    type="cross_reference", lot_id=lot_a, lot_id_2=lot_b,
                    rationale=f"Cross-reference {lot_a} and {lot_b} to check shared ancestry.",
                )

        # Step 3: Exploration — inspect non-traced nodes (high early, low late)
        if rng.random() < min(self.exploration_rate, 0.95):
            all_nodes = list(observation.inventory.keys())
            uninspected = [n for n in all_nodes if n not in observation.inspected_nodes]
            if uninspected:
                node_id = rng.choice(uninspected)
                self.nodes_visited.append(node_id)
                return RecallAction(type="inspect_node", node_id=node_id,
                                    rationale="Exploring non-traced node.")

        # Step 4: Quarantine decisions — THIS IS WHERE LEARNING MATTERS
        # Scan ALL findings and decide what to quarantine based on learned trust
        for node_id, findings in observation.inspection_results.items():
            for lot_id, finding in findings.items():
                unsafe_qty = finding.unsafe_quantity
                quarantined_qty = observation.quarantined_inventory.get(node_id, {}).get(lot_id, 0)
                available_qty = observation.inventory.get(node_id, {}).get(lot_id, 0)

                if available_qty <= 0:
                    continue

                # Assess evidence using LEARNED trust parameters
                evidence_score = self._assess_evidence(finding)

                # Skip if below threshold
                if evidence_score < self.quarantine_threshold:
                    continue

                # Decide quantity to quarantine
                if unsafe_qty > 0:
                    remaining = unsafe_qty - quarantined_qty
                    if remaining <= 0:
                        continue
                    qty = min(remaining, available_qty)
                elif evidence_score >= 0.5:
                    # No stated unsafe_qty but evidence looks suspicious
                    # Early agent: quarantines these (FPs on decoys!)
                    # Late agent: threshold filters these out
                    qty = available_qty
                else:
                    continue

                # Track belief at quarantine time for calibration
                self.belief_at_quarantine.append(evidence_score)

                self.nodes_quarantined.append(node_id)
                self.quarantine_decisions.append({
                    "node_id": node_id, "lot_id": lot_id,
                    "quantity": qty, "confidence": evidence_score,
                })
                self._update_intervention_guess(finding)
                return RecallAction(
                    type="quarantine", node_id=node_id,
                    lot_id=lot_id, quantity=qty,
                    rationale=f"Quarantining (conf={evidence_score:.2f})",
                )

        # Step 5: Notify and finalize
        if affected_nodes:
            missing = [n for n in affected_nodes if n not in observation.notified_nodes]
            if missing:
                return RecallAction(type="notify", node_id="all",
                                    rationale="Alert all stakeholders.")

        return RecallAction(type="finalize", rationale="Containment complete.")

    def update(self, episode_reward: float, f1: float, steps_taken: int) -> None:
        """Update learned parameters after an episode."""
        self.total_episodes += 1
        self._f1_history.append(f1)

        num_q = len(set(self.nodes_quarantined))

        # --- Adapt quarantine threshold ---
        if f1 < 0.4:
            if num_q > 3:
                # Too many FPs (spray & pray). Raise threshold to filter decoys.
                self.quarantine_threshold = min(0.85, self.quarantine_threshold + self.threshold_lr * 3)
            else:
                # Missing things, lower threshold
                self.quarantine_threshold = max(0.0, self.quarantine_threshold - self.threshold_lr)
        elif f1 < 0.65:
            # Improving but still noisy, keep nudging threshold up
            self.quarantine_threshold = min(0.85, self.quarantine_threshold + self.threshold_lr * 1.5)
        elif f1 < 0.8:
            self.quarantine_threshold = min(0.85, self.quarantine_threshold + self.threshold_lr * 0.5)
        else:
            # Good F1 — fine-tune
            target = 0.55
            self.quarantine_threshold += self.threshold_lr * 0.3 * (target - self.quarantine_threshold)

        # --- Adapt trust in ambiguous evidence ---
        if f1 < 0.5 and num_q > 3:
            # Trusting too much ambiguous evidence
            self.suspect_trust = max(0.05, self.suspect_trust - self.trust_lr * 3)
            self.mixed_trust = max(0.2, self.mixed_trust - self.trust_lr * 1.5)
        elif f1 < 0.7:
            self.suspect_trust = max(0.05, self.suspect_trust - self.trust_lr * 1.5)
            self.mixed_trust = max(0.3, self.mixed_trust - self.trust_lr * 0.5)
        elif f1 > 0.8:
            # Good performance, small adjustments only
            pass

        # --- Decay exploration very slowly ---
        self.exploration_rate = max(0.05, self.exploration_rate - 0.004)

        # --- Decay learning rates over time ---
        if self.total_episodes > 80:
            self.threshold_lr = max(0.002, self.threshold_lr * 0.995)
            self.trust_lr = max(0.002, self.trust_lr * 0.995)

    def _assess_evidence(self, finding: Any) -> float:
        """Score evidence strength using LEARNED trust parameters.

        This is the core of the agent's decision-making. Early on:
          - suspect_trust = 0.95 -> suspects score 0.95 -> above threshold (0.0)
          - Agent quarantines decoys (FPs) -> low F1

        After learning:
          - suspect_trust = 0.05 -> suspects score 0.05 -> below threshold (0.6)
          - Agent ignores decoys -> high F1
        """
        status = finding.status if hasattr(finding, 'status') else str(finding.get("status", ""))
        unsafe_qty = finding.unsafe_quantity if hasattr(finding, 'unsafe_quantity') else finding.get("unsafe_quantity", 0)

        if status == "confirmed_contaminated":
            return 0.95
        elif status == "suspect":
            # DECOYS live here. Early agent trusts them. Late agent doesn't.
            return self.suspect_trust
        elif status == "mixed":
            if unsafe_qty > 0:
                return 0.5 + 0.4 * self.mixed_trust
            else:
                # Mixed but no unsafe qty = likely a red herring
                return 0.3 * self.mixed_trust
        elif status == "records_missing":
            if unsafe_qty > 0:
                return 0.6
            return 0.35 * self.suspect_trust
        elif status == "safe":
            return 0.0
        elif unsafe_qty > 0:
            return 0.7
        return 0.05

    def _find_suspect_lots(self, observation: RecallObservation) -> List[str]:
        """Find lots worth cross-referencing — suspect or mixed status."""
        suspect_lots = []
        for node_id, findings in observation.inspection_results.items():
            for lot_id, finding in findings.items():
                status = finding.status if hasattr(finding, 'status') else str(finding.get("status", ""))
                if status in ("suspect", "mixed", "records_missing"):
                    suspect_lots.append(lot_id)
        return suspect_lots[:4]  # limit to 4 to keep tool calls bounded

    def _update_intervention_guess(self, finding: Any) -> None:
        """Try to identify the intervention type from evidence patterns."""
        status = finding.status if hasattr(finding, 'status') else str(finding.get("status", ""))
        evidence = ""
        if hasattr(finding, 'evidence'):
            evidence = finding.evidence
        elif isinstance(finding, dict):
            evidence = finding.get("evidence", "")

        if status == "mixed":
            self.intervention_guess = "mixing_event"
        elif status == "records_missing":
            self.intervention_guess = "record_deletion"
        elif "relabel" in evidence.lower() or "repack" in evidence.lower():
            self.intervention_guess = "lot_relabel"

    @staticmethod
    def _extract_root_lot(observation: RecallObservation) -> str:
        import re
        match = re.search(r"\bLot[A-Za-z0-9_]+\b", observation.recall_notice)
        return match.group(0) if match else "LotA"

    def get_belief_calibration_score(self) -> float:
        """Average belief confidence at quarantine time.

        Returns 0.0 if no quarantine decisions were made.
        """
        if not self.belief_at_quarantine:
            return 0.0
        return sum(self.belief_at_quarantine) / len(self.belief_at_quarantine)

    def get_episode_summary(self) -> Dict[str, Any]:
        return {
            "nodes_visited": list(set(self.nodes_visited)),
            "nodes_quarantined": list(set(self.nodes_quarantined)),
            "num_quarantined": len(set(self.nodes_quarantined)),
            "quarantine_threshold": round(self.quarantine_threshold, 4),
            "suspect_trust": round(self.suspect_trust, 4),
            "mixed_trust": round(self.mixed_trust, 4),
            "exploration_rate": round(self.exploration_rate, 4),
            "belief_confidence": round(self.belief_confidence, 4),
            "belief_calibration": round(self.get_belief_calibration_score(), 4),
            "intervention_guess": self.intervention_guess,
            "cross_ref_used": self.cross_ref_done,
        }
