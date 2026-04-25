"""Core RecallTrace environment with deterministic action execution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Tuple

from env.models import EnvironmentState, InspectionEvidence, RecallAction, RecallObservation, RewardSignal, StepInfo, TaskDefinition
from scenario.scenario import build_scenario, list_task_specs


class RecallTraceEnv:
    """Deterministic OpenEnv-style environment for product recall containment."""

    ACTIONS = [
        "inspect_node",
        "trace_lot",
        "quarantine",
        "notify",
        "finalize",
    ]

    def __init__(
        self,
        scenario_data: Dict[str, Any] | None = None,
        task_id: str | None = None,
        phase: int | None = 1,
    ):
        self._scenario_template = deepcopy(scenario_data) if scenario_data is not None else build_scenario(task_id=task_id, phase=phase)
        self.task = self._build_task_definition(self._scenario_template)
        self.state_data: Dict[str, Any] = {}
        self.ground_truth: Dict[str, Any] = {}
        self.done = False
        self.last_reward = RewardSignal(value=0.0, reason="Environment initialized.", components={})

    @classmethod
    def available_tasks(cls) -> list[TaskDefinition]:
        return [TaskDefinition(**task_spec) for task_spec in list_task_specs()]

    def reset(self, task_id: str | None = None, phase: int | None = None) -> RecallObservation:
        """Start a new deterministic scenario and recompute ground truth."""
        if task_id is not None or phase is not None:
            self._scenario_template = build_scenario(task_id=task_id, phase=phase)
            self.task = self._build_task_definition(self._scenario_template)

        self.done = False
        self.last_reward = RewardSignal(value=0.0, reason="Episode reset.", components={})

        scenario = deepcopy(self._scenario_template)
        self.state_data = {
            "task_id": scenario["task_id"],
            "phase": scenario["phase"],
            "recall_notice": scenario["recall_notice"],
            "contaminated_lot_hint": scenario["contaminated_lot"],
            "shipment_graph": scenario["shipment_graph"],
            "lot_catalog": scenario["lot_catalog"],
            "nodes": scenario["nodes"],
            "history": [],
            "discovered_shipments": {},
            "inspected_nodes": set(),
            "inspection_results": {},
            "traced_lots": {},
            "notified_nodes": set(),
            "quarantine_log": [],
            "steps_taken": 0,
            "max_steps": scenario["max_steps"],
        }
        self.ground_truth = self._build_ground_truth(scenario)
        return self._get_observation()

    def step(self, action: RecallAction | Dict[str, Any]) -> Tuple[RecallObservation, float, bool, Dict[str, Any]]:
        """Execute an action and return observation, reward, done, info."""
        if self.done:
            return self._get_observation(), 0.0, True, {
                "message": "Environment already finalized.",
                "action_type": "noop",
                "reward_breakdown": {},
            }

        validated_action = action if isinstance(action, RecallAction) else RecallAction.model_validate(action)
        self.state_data["steps_taken"] += 1

        handler = getattr(self, f"_handle_{validated_action.type.value}")
        reward_signal, info = handler(validated_action)
        self.last_reward = reward_signal

        if not self.done and self.state_data["steps_taken"] >= self.state_data["max_steps"]:
            self.done = True
            timeout_penalty = -0.25
            reward_signal = RewardSignal(
                value=max(-1.0, reward_signal.value + timeout_penalty),
                reason="Step budget exhausted before finalizing containment.",
                components={**reward_signal.components, "timeout_penalty": timeout_penalty},
            )
            info = {
                **info,
                "message": "Step budget exhausted before finalizing containment.",
                "reward_breakdown": reward_signal.components,
            }
            self._record_history("Episode terminated after exhausting the step budget")
            self.last_reward = reward_signal

        return self._get_observation(), reward_signal.value, self.done, info

    def state(self) -> EnvironmentState:
        """Return the full internal state for debugging and graders."""
        return EnvironmentState(
            done=self.done,
            task=self.task,
            steps_taken=self.state_data.get("steps_taken", 0),
            state_data=deepcopy(self._serialize_state(self.state_data)),
            ground_truth=deepcopy(self.ground_truth),
        )

    def _get_observation(self) -> RecallObservation:
        return RecallObservation(
            task_id=self.state_data["task_id"],
            phase=self.state_data["phase"],
            recall_notice=self.state_data["recall_notice"],
            available_actions=list(self.ACTIONS),
            inventory=self._inventory_snapshot(),
            discovered_shipments=deepcopy(self.state_data["discovered_shipments"]),
            inspected_nodes=sorted(self.state_data["inspected_nodes"]),
            inspection_results=deepcopy(self.state_data["inspection_results"]),
            trace_results=deepcopy(self.state_data["traced_lots"]),
            notified_nodes=sorted(self.state_data["notified_nodes"]),
            quarantined_inventory=self._quarantine_snapshot(),
            history=list(self.state_data["history"]),
            steps_taken=self.state_data["steps_taken"],
            remaining_step_budget=max(0, self.state_data["max_steps"] - self.state_data["steps_taken"]),
        )

    def _handle_inspect_node(self, action: RecallAction) -> tuple[RewardSignal, Dict[str, Any]]:
        node_id = self._require_node(action.node_id)
        node = self.state_data["nodes"][node_id]
        repeated = node_id in self.state_data["inspected_nodes"]

        self.state_data["inspected_nodes"].add(node_id)
        self.state_data["discovered_shipments"][node_id] = list(self.state_data["shipment_graph"].get(node_id, []))
        findings = {
            lot_id: InspectionEvidence.model_validate(payload)
            for lot_id, payload in node.get("inspection_findings", {}).items()
        }
        self.state_data["inspection_results"][node_id] = findings
        self._record_history(f"Inspected node {node_id}")

        unsafe_total = sum(item.unsafe_quantity for item in findings.values())
        value = -0.03 if repeated else 0.08 + min(0.12, unsafe_total / 500.0)
        reason = "Repeated inspection provided no new information." if repeated else "Inspection revealed inventory evidence."
        reward = RewardSignal(
            value=round(value, 4),
            reason=reason,
            components={
                "inspection_value": round(value, 4),
            },
        )
        info = StepInfo(
            message=f"Inspected node {node_id} and collected node evidence.",
            action_type=action.type.value,
            reward_breakdown=reward.components,
        ).model_dump()
        info.update(
            {
                "node_id": node_id,
                "inventory": deepcopy(node["inventory"]),
                "quarantined_inventory": deepcopy(node["quarantined_inventory"]),
                "outbound_shipments": list(self.state_data["shipment_graph"].get(node_id, [])),
                "inspection_findings": {lot_id: item.model_dump() for lot_id, item in findings.items()},
            }
        )
        return reward, info

    def _handle_trace_lot(self, action: RecallAction) -> tuple[RewardSignal, Dict[str, Any]]:
        lot_id = action.lot_id
        if not lot_id:
            raise ValueError("trace_lot action requires 'lot_id'.")

        traced_lots = self._resolve_related_lots(lot_id)
        impacted_nodes = []
        impacted_quantities = {}
        impacted_lots = {}
        discovered_nodes = 0

        for node_id, node_data in self.state_data["nodes"].items():
            node_total = 0
            node_lots = []
            for candidate_lot in traced_lots:
                available_qty = node_data["inventory"].get(candidate_lot, 0)
                quarantined_qty = node_data["quarantined_inventory"].get(candidate_lot, 0)
                total_qty = available_qty + quarantined_qty
                if total_qty > 0:
                    node_total += total_qty
                    node_lots.append(candidate_lot)
            if node_total > 0:
                impacted_nodes.append(node_id)
                impacted_quantities[node_id] = node_total
                impacted_lots[node_id] = node_lots
                if node_id not in self.state_data["discovered_shipments"]:
                    discovered_nodes += 1

        self.state_data["traced_lots"][lot_id] = {
            "root_lot": self._root_lot_for(lot_id),
            "matched_lots": sorted(traced_lots),
            "affected_nodes": impacted_nodes,
            "lots_by_node": impacted_lots,
            "quantities_by_node": impacted_quantities,
        }
        self._record_history(f"Traced lot {lot_id} across {', '.join(sorted(traced_lots))}")

        if not impacted_nodes:
            reward_value = -0.1
            reason = "Trace returned no impacted nodes."
        elif self._root_lot_for(lot_id) in self.ground_truth["affected_roots"]:
            reward_value = 0.12 + min(0.13, discovered_nodes * 0.03 + len(traced_lots) * 0.02)
            reason = "Trace identified the affected lineage across the network."
        else:
            reward_value = 0.02
            reason = "Trace ran, but the lot is outside the affected lineage."

        reward = RewardSignal(
            value=round(reward_value, 4),
            reason=reason,
            components={
                "trace_value": round(reward_value, 4),
            },
        )
        info = StepInfo(
            message=f"Traced lot {lot_id} across the shipment network.",
            action_type=action.type.value,
            reward_breakdown=reward.components,
        ).model_dump()
        info.update(
            {
                "lot_id": lot_id,
                "root_lot": self._root_lot_for(lot_id),
                "matched_lots": sorted(traced_lots),
                "affected_nodes": impacted_nodes,
                "lots_by_node": impacted_lots,
                "quantities_by_node": impacted_quantities,
                "total_quantity": sum(impacted_quantities.values()),
            }
        )
        return reward, info

    def _handle_quarantine(self, action: RecallAction) -> tuple[RewardSignal, Dict[str, Any]]:
        node_id = self._require_node(action.node_id)
        lot_id = action.lot_id
        if not lot_id:
            raise ValueError("quarantine action requires 'lot_id'.")

        node = self.state_data["nodes"][node_id]
        available_qty = node["inventory"].get(lot_id, 0)
        if available_qty <= 0:
            reward = RewardSignal(
                value=-0.2,
                reason="Attempted to quarantine stock that is not available.",
                components={"invalid_quarantine": -0.2},
            )
            self._record_history(f"Failed quarantine for {lot_id} at {node_id}: no available stock")
            info = StepInfo(
                message="No available stock to quarantine.",
                action_type=action.type.value,
                reward_breakdown=reward.components,
            ).model_dump()
            info.update({"node_id": node_id, "lot_id": lot_id})
            return reward, info

        requested_qty = action.quantity or available_qty
        quarantined_qty = min(requested_qty, available_qty)
        node["inventory"][lot_id] = available_qty - quarantined_qty
        if node["inventory"][lot_id] == 0:
            del node["inventory"][lot_id]
        node["quarantined_inventory"][lot_id] = node["quarantined_inventory"].get(lot_id, 0) + quarantined_qty

        self.state_data["quarantine_log"].append({"node_id": node_id, "lot_id": lot_id, "quantity": quarantined_qty})
        self._record_history(f"Quarantined {quarantined_qty} units of {lot_id} at {node_id}")

        correct_qty = self.ground_truth["correct_quantities"].get(node_id, {}).get(lot_id, 0)
        cumulative_quarantined = node["quarantined_inventory"].get(lot_id, 0)
        delta = cumulative_quarantined - correct_qty

        if correct_qty == 0:
            reward_value = -0.35
            reason = "Quarantined safe inventory outside the recall scope."
        elif delta == 0:
            reward_value = 0.28
            reason = "Quarantine exactly matched the unsafe quantity."
        elif delta < 0:
            reward_value = max(0.05, 0.22 * (cumulative_quarantined / correct_qty))
            reason = "Quarantine made partial progress but missed some unsafe stock."
        else:
            reward_value = max(-0.25, -0.08 * delta)
            reason = "Quarantine overreached and blocked safe inventory."

        reward = RewardSignal(
            value=round(reward_value, 4),
            reason=reason,
            components={
                "quarantine_value": round(reward_value, 4),
                "target_quantity": float(correct_qty),
                "quarantined_quantity": float(cumulative_quarantined),
            },
        )
        info = StepInfo(
            message=f"Updated quarantine for {lot_id} at {node_id}.",
            action_type=action.type.value,
            reward_breakdown=reward.components,
        ).model_dump()
        info.update(
            {
                "node_id": node_id,
                "lot_id": lot_id,
                "quarantined_quantity": quarantined_qty,
                "remaining_inventory": node["inventory"].get(lot_id, 0),
                "cumulative_quarantined": cumulative_quarantined,
                "target_contaminated_quantity": correct_qty,
            }
        )
        return reward, info

    def _handle_notify(self, action: RecallAction) -> tuple[RewardSignal, Dict[str, Any]]:
        requested_target = action.node_id or "all"
        if requested_target in ("all", "all_nodes"):
            targets = list(self.state_data["nodes"].keys())
        else:
            targets = [self._require_node(requested_target)]

        newly_notified = []
        for node_id in targets:
            if node_id not in self.state_data["notified_nodes"]:
                self.state_data["notified_nodes"].add(node_id)
                newly_notified.append(node_id)

        affected_newly_notified = sum(1 for node_id in newly_notified if node_id in self.ground_truth["affected_nodes"])
        unaffected_newly_notified = len(newly_notified) - affected_newly_notified

        if not newly_notified:
            reward_value = -0.05
            reason = "Notification repeated without adding new recipients."
        else:
            reward_value = min(0.18, affected_newly_notified * 0.04) - unaffected_newly_notified * 0.01
            reason = "Notifications dispatched to downstream stakeholders."

        reward = RewardSignal(
            value=round(reward_value, 4),
            reason=reason,
            components={
                "notification_value": round(reward_value, 4),
            },
        )
        if newly_notified:
            self._record_history(f"Sent notifications to {', '.join(newly_notified)}")
        else:
            self._record_history("Notification action repeated without new recipients")

        info = StepInfo(
            message="Processed notification action.",
            action_type=action.type.value,
            reward_breakdown=reward.components,
        ).model_dump()
        info.update({"notified_nodes": targets, "newly_notified": newly_notified})
        return reward, info

    def _handle_finalize(self, action: RecallAction) -> tuple[RewardSignal, Dict[str, Any]]:
        del action
        self.done = True
        quarantine_match = self._compute_quarantine_match()

        missing_quantity_total = sum(
            quantity
            for lot_quantities in quarantine_match["missing_quantities"].values()
            for quantity in lot_quantities.values()
        )
        over_quantity_total = sum(
            quantity
            for lot_quantities in quarantine_match["over_quarantined_quantities"].values()
            for quantity in lot_quantities.values()
        )
        total_affected_quantity = self.ground_truth["total_affected_quantity"] or 1
        quarantine_score = max(0.0, 1.0 - ((missing_quantity_total + (1.25 * over_quantity_total)) / total_affected_quantity))

        notified_affected_nodes = set(self.ground_truth["affected_nodes"]).intersection(self.state_data["notified_nodes"])
        affected_node_total = len(self.ground_truth["affected_nodes"]) or 1
        notification_score = len(notified_affected_nodes) / affected_node_total

        investigated_nodes = set(self.state_data["inspected_nodes"]).intersection(self.ground_truth["affected_nodes"])
        investigation_score = len(investigated_nodes) / affected_node_total

        efficiency_penalty_steps = max(0, self.state_data["steps_taken"] - max(4, affected_node_total + 3))
        efficiency_score = max(0.0, 1.0 - (efficiency_penalty_steps / self.state_data["max_steps"]))

        score = round(
            (0.55 * quarantine_score) + (0.2 * notification_score) + (0.15 * investigation_score) + (0.1 * efficiency_score),
            4,
        )

        reward = RewardSignal(
            value=score,
            reason="Final recall response scored.",
            components={
                "quarantine_score": round(quarantine_score, 4),
                "notification_score": round(notification_score, 4),
                "investigation_score": round(investigation_score, 4),
                "efficiency_score": round(efficiency_score, 4),
            },
        )
        self._record_history("Finalized recall response")

        info = StepInfo(
            message="Finalized recall response.",
            action_type="finalize",
            score=score,
            reward_breakdown=reward.components,
        ).model_dump()
        info.update(
            {
                "score": score,
                "quarantine_score": round(quarantine_score, 4),
                "notification_score": round(notification_score, 4),
                "investigation_score": round(investigation_score, 4),
                "efficiency_score": round(efficiency_score, 4),
                "all_affected_nodes_notified": notification_score == 1.0,
                "all_affected_stock_quarantined": missing_quantity_total == 0 and over_quantity_total == 0,
                "quarantine_match": quarantine_match,
            }
        )
        return reward, info

    def _build_ground_truth(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        contaminated_roots = {
            self._root_lot_for(lot_id, scenario["lot_catalog"])
            for lot_id, lot_data in scenario["lot_catalog"].items()
            if lot_data.get("contaminated", False)
        }

        correct_quantities: Dict[str, Dict[str, int]] = {}
        affected_nodes = set()
        affected_lots = set()

        for node_id, node_data in scenario["nodes"].items():
            for lot_id, finding in node_data.get("inspection_findings", {}).items():
                unsafe_quantity = int(finding.get("unsafe_quantity", 0))
                if unsafe_quantity > 0:
                    affected_nodes.add(node_id)
                    affected_lots.add(lot_id)
                    correct_quantities.setdefault(node_id, {})[lot_id] = unsafe_quantity

        total_affected_quantity = sum(
            quantity
            for node_quantities in correct_quantities.values()
            for quantity in node_quantities.values()
        )
        return {
            "affected_lots": sorted(affected_lots),
            "affected_nodes": sorted(affected_nodes),
            "affected_roots": sorted(contaminated_roots),
            "correct_quantities": correct_quantities,
            "total_affected_quantity": total_affected_quantity,
        }

    def _compute_quarantine_match(self) -> Dict[str, Any]:
        missing_quantities: Dict[str, Dict[str, int]] = {}
        over_quarantined_quantities: Dict[str, Dict[str, int]] = {}

        for node_id, node_data in self.state_data["nodes"].items():
            expected = self.ground_truth["correct_quantities"].get(node_id, {})
            actual = node_data["quarantined_inventory"]
            relevant_lots = set(expected) | set(actual)

            for lot_id in relevant_lots:
                expected_qty = expected.get(lot_id, 0)
                actual_qty = actual.get(lot_id, 0)
                if actual_qty < expected_qty:
                    missing_quantities.setdefault(node_id, {})[lot_id] = expected_qty - actual_qty
                elif actual_qty > expected_qty:
                    over_quarantined_quantities.setdefault(node_id, {})[lot_id] = actual_qty - expected_qty

        return {
            "missing_quantities": missing_quantities,
            "over_quarantined_quantities": over_quarantined_quantities,
        }

    def _inventory_snapshot(self) -> Dict[str, Dict[str, int]]:
        return {node_id: deepcopy(node_data["inventory"]) for node_id, node_data in self.state_data["nodes"].items()}

    def _quarantine_snapshot(self) -> Dict[str, Dict[str, int]]:
        return {
            node_id: deepcopy(node_data["quarantined_inventory"])
            for node_id, node_data in self.state_data["nodes"].items()
            if node_data["quarantined_inventory"]
        }

    def _resolve_related_lots(self, lot_id: str) -> set[str]:
        root_lot = self._root_lot_for(lot_id)
        return {
            candidate_lot
            for candidate_lot in self.state_data["lot_catalog"].keys()
            if self._root_lot_for(candidate_lot) == root_lot or candidate_lot == lot_id
        }

    def _root_lot_for(self, lot_id: str, lot_catalog: Dict[str, Dict[str, Any]] | None = None) -> str:
        catalog = lot_catalog or self.state_data.get("lot_catalog", {})
        if lot_id not in catalog:
            return lot_id
        return catalog[lot_id].get("root_lot", lot_id)

    def _build_task_definition(self, scenario: Dict[str, Any]) -> TaskDefinition:
        return TaskDefinition(
            task_id=scenario["task_id"],
            name=scenario["name"],
            difficulty=scenario["difficulty"],
            objective=scenario["objective"],
            max_steps=scenario["max_steps"],
        )

    def _require_node(self, node_id: str | None) -> str:
        if not node_id:
            raise ValueError("Action requires 'node_id'.")
        if node_id not in self.state_data["nodes"]:
            raise ValueError(f"Unknown node_id '{node_id}'.")
        return node_id

    def _record_history(self, message: str) -> None:
        self.state_data["history"].append(message)

    def _serialize_state(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._serialize_state(item) for key, item in value.items()}
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, list):
            return [self._serialize_state(item) for item in value]
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return value
