"""Core RecallTrace environment with deterministic action execution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple

from env.models import EnvironmentState, InspectionEvidence, RecallAction, RecallObservation, RewardSignal, StepInfo, TaskDefinition
from scenario.scenario import build_scenario, list_task_specs


class RecallTraceEnv:
    """Deterministic OpenEnv-style environment for product recall containment."""

    ACTIONS = [
        "inspect_node",
        "trace_lot",
        "cross_reference",
        "request_lab_test",
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
        self._root_lot_index: Dict[str, str] = {}
        self._related_lots_index: Dict[str, set[str]] = {}
        self._lot_nodes_index: Dict[str, List[str]] = {}
        self._affected_nodes_set: set[str] = set()
        self._affected_roots_set: set[str] = set()
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
            "cross_references": {},
            "lab_results": {},
            "notified_nodes": set(),
            "quarantine_log": [],
            "belief_state": {},
            "root_cause_candidates": [],
            "steps_taken": 0,
            "max_steps": scenario["max_steps"],
        }
        self.ground_truth = self._build_ground_truth(scenario)
        self._rebuild_indexes()
        self._refresh_belief_state()
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

        self._refresh_belief_state()
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
            belief_state=deepcopy(self.state_data["belief_state"]),
            risk_summary=self._risk_summary(),
            root_cause_candidates=list(self.state_data["root_cause_candidates"]),
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
        for lot_id, finding in findings.items():
            if finding.unsafe_quantity > 0:
                self._remember_root_cause(self._derive_root_cause(lot_id, finding.model_dump()))
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

        candidate_nodes = sorted({
            node_id
            for candidate_lot in traced_lots
            for node_id in self._lot_nodes_index.get(candidate_lot, [])
        })
        for node_id in candidate_nodes:
            node_data = self.state_data["nodes"][node_id]
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
                for candidate_lot in node_lots:
                    finding = node_data.get("inspection_findings", {}).get(candidate_lot)
                    if finding and int(finding.get("unsafe_quantity", 0)) > 0:
                        self._remember_root_cause(self._derive_root_cause(candidate_lot, finding))

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
                "root_cause_candidates": list(self.state_data["root_cause_candidates"]),
            }
        )
        return reward, info

    def _handle_cross_reference(self, action: RecallAction) -> tuple[RewardSignal, Dict[str, Any]]:
        lot_id = action.lot_id or self.state_data["contaminated_lot_hint"]
        root_lot = self._root_lot_for(lot_id)
        matched_lots = sorted(self._resolve_related_lots(lot_id))
        affected_nodes = sorted({
            node_id
            for matched_lot in matched_lots
            for node_id in self._lot_nodes_index.get(matched_lot, [])
        })

        node_id = action.node_id
        if node_id:
            node_id = self._require_node(node_id)
            affected_nodes = [candidate for candidate in affected_nodes if candidate == node_id]

        evidence_statuses: Dict[str, int] = {}
        root_causes: set[str] = set()
        for candidate_node in affected_nodes or self._lot_nodes_index.get(lot_id, []):
            findings = self.state_data["nodes"][candidate_node].get("inspection_findings", {})
            for matched_lot in matched_lots:
                finding = findings.get(matched_lot)
                if not finding:
                    continue
                status = str(finding.get("status", "unknown"))
                evidence_statuses[status] = evidence_statuses.get(status, 0) + 1
                if int(finding.get("unsafe_quantity", 0)) > 0:
                    root_causes.add(self._derive_root_cause(matched_lot, finding))

        for cause in sorted(root_causes):
            self._remember_root_cause(cause)

        repeated = lot_id in self.state_data["cross_references"]
        self.state_data["cross_references"][lot_id] = {
            "root_lot": root_lot,
            "matched_lots": matched_lots,
            "affected_nodes": affected_nodes,
            "evidence_statuses": evidence_statuses,
            "root_cause_candidates": sorted(root_causes),
        }
        self._record_history(f"Cross-referenced {lot_id} against lot lineage and inspection evidence")

        is_recall_lineage = root_lot in self._affected_roots_set
        value = (0.14 if is_recall_lineage else 0.02) + min(0.1, 0.02 * len(affected_nodes))
        if repeated:
            value -= 0.08
        reward = RewardSignal(
            value=round(max(-0.05, min(0.28, value)), 4),
            reason="Cross-reference connected lot lineage, graph placement, and root-cause evidence.",
            components={"cross_reference_value": round(max(-0.05, min(0.28, value)), 4)},
        )
        info = StepInfo(
            message=f"Cross-referenced {lot_id} across lineage and graph records.",
            action_type=action.type.value,
            reward_breakdown=reward.components,
        ).model_dump()
        info.update(self.state_data["cross_references"][lot_id])
        info.update({"lot_id": lot_id})
        return reward, info

    def _handle_request_lab_test(self, action: RecallAction) -> tuple[RewardSignal, Dict[str, Any]]:
        node_id = self._require_node(action.node_id)
        node = self.state_data["nodes"][node_id]
        lot_id = action.lot_id
        if not lot_id:
            candidate_lots = list(node.get("inspection_findings", {}).keys()) or list(node["inventory"].keys())
            if not candidate_lots:
                raise ValueError("request_lab_test requires 'lot_id' when the node has no inventory.")
            lot_id = max(
                candidate_lots,
                key=lambda candidate: node.get("inspection_findings", {}).get(candidate, {}).get("unsafe_quantity", 0),
            )
        if lot_id not in node["inventory"] and lot_id not in node.get("inspection_findings", {}):
            raise ValueError(f"Lot '{lot_id}' is not present in node '{node_id}'.")

        finding_payload = node.get("inspection_findings", {}).get(
            lot_id,
            {
                "status": "not_detected",
                "unsafe_quantity": 0,
                "evidence": "Lab panel found no matching recall signal for this lot at this node.",
            },
        )
        finding = InspectionEvidence.model_validate(finding_payload)
        self.state_data["lab_results"].setdefault(node_id, {})[lot_id] = finding
        self.state_data["inspection_results"].setdefault(node_id, {})[lot_id] = finding

        if finding.unsafe_quantity > 0:
            cause = self._derive_root_cause(lot_id, finding.model_dump())
            self._remember_root_cause(cause)
            reward_value = 0.2
            reason = "Lab test confirmed unsafe stock and strengthened root-cause evidence."
        else:
            reward_value = 0.03
            reason = "Lab test ruled out a candidate lot and reduced false-positive risk."

        self._record_history(f"Requested lab test for {lot_id} at {node_id}")
        reward = RewardSignal(
            value=round(reward_value, 4),
            reason=reason,
            components={"lab_test_value": round(reward_value, 4)},
        )
        info = StepInfo(
            message=f"Lab test completed for {lot_id} at {node_id}.",
            action_type=action.type.value,
            reward_breakdown=reward.components,
        ).model_dump()
        info.update(
            {
                "node_id": node_id,
                "lot_id": lot_id,
                "lab_result": finding.model_dump(),
                "root_cause_candidates": list(self.state_data["root_cause_candidates"]),
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
                "containment_progress": self._risk_summary()["containment_progress"],
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

    def _rebuild_indexes(self) -> None:
        lot_catalog = self.state_data.get("lot_catalog", {})
        self._root_lot_index = {
            lot_id: payload.get("root_lot", lot_id)
            for lot_id, payload in lot_catalog.items()
        }
        self._related_lots_index = {}
        for lot_id, root_lot in self._root_lot_index.items():
            self._related_lots_index.setdefault(root_lot, set()).add(lot_id)
            self._related_lots_index[lot_id] = self._related_lots_index[root_lot]

        lot_nodes: Dict[str, set[str]] = {}
        for node_id, node_data in self.state_data.get("nodes", {}).items():
            lots = set(node_data.get("inventory", {})) | set(node_data.get("quarantined_inventory", {}))
            lots |= set(node_data.get("inspection_findings", {}))
            for lot_id in lots:
                lot_nodes.setdefault(lot_id, set()).add(node_id)
        self._lot_nodes_index = {
            lot_id: sorted(nodes)
            for lot_id, nodes in lot_nodes.items()
        }
        self._affected_nodes_set = set(self.ground_truth.get("affected_nodes", []))
        self._affected_roots_set = set(self.ground_truth.get("affected_roots", []))

    def _refresh_belief_state(self) -> None:
        recall_root = self._root_lot_for(self.state_data.get("contaminated_lot_hint", ""))
        traced_nodes = {
            node_id
            for trace in self.state_data.get("traced_lots", {}).values()
            for node_id in trace.get("affected_nodes", [])
        }
        beliefs: Dict[str, float] = {}

        for node_id, node_data in self.state_data.get("nodes", {}).items():
            inventory_lots = set(node_data.get("inventory", {})) | set(node_data.get("quarantined_inventory", {}))
            score = 0.05
            if any(self._root_lot_for(lot_id) == recall_root for lot_id in inventory_lots):
                score = max(score, 0.35)
            if node_id in traced_nodes:
                score = max(score, 0.55)

            findings = self.state_data.get("inspection_results", {}).get(node_id, {})
            if findings:
                unsafe_score = 0.0
                safe_only = True
                for finding in findings.values():
                    unsafe_qty = finding.unsafe_quantity if hasattr(finding, "unsafe_quantity") else int(finding.get("unsafe_quantity", 0))
                    status = finding.status if hasattr(finding, "status") else str(finding.get("status", ""))
                    if unsafe_qty > 0:
                        safe_only = False
                        if status == "mixed":
                            unsafe_score = max(unsafe_score, 0.82)
                        else:
                            unsafe_score = max(unsafe_score, 0.95)
                    elif status not in {"safe", "not_detected"}:
                        safe_only = False
                        unsafe_score = max(unsafe_score, 0.3)
                if unsafe_score:
                    score = max(score, unsafe_score)
                elif safe_only:
                    score = min(score, 0.1)

            expected = self.ground_truth.get("correct_quantities", {}).get(node_id, {})
            if expected:
                actual = node_data.get("quarantined_inventory", {})
                covered = sum(min(actual.get(lot_id, 0), qty) for lot_id, qty in expected.items())
                total = sum(expected.values()) or 1
                score *= max(0.05, 1.0 - (covered / total))

            beliefs[node_id] = round(max(0.0, min(0.99, score)), 4)

        self.state_data["belief_state"] = beliefs

    def _risk_summary(self) -> Dict[str, Any]:
        beliefs = self.state_data.get("belief_state", {})
        high_risk_nodes = [node_id for node_id, score in sorted(beliefs.items(), key=lambda item: item[1], reverse=True) if score >= 0.5]
        inspected_unsafe_nodes = sorted(
            node_id
            for node_id, findings in self.state_data.get("inspection_results", {}).items()
            if any(finding.unsafe_quantity > 0 for finding in findings.values())
        )
        quarantine_match = self._compute_quarantine_match()
        remaining_nodes = sorted(quarantine_match["missing_quantities"].keys())
        total_affected = len(self.ground_truth.get("affected_nodes", [])) or 1
        contained_nodes = total_affected - len(remaining_nodes)
        return {
            "high_risk_nodes": high_risk_nodes,
            "inspected_unsafe_nodes": inspected_unsafe_nodes,
            "remaining_suspected_nodes": len(high_risk_nodes),
            "containment_progress": round(max(0.0, contained_nodes / total_affected), 4),
            "root_cause_candidates": list(self.state_data.get("root_cause_candidates", [])),
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
        return set(self._related_lots_index.get(lot_id) or self._related_lots_index.get(root_lot) or {lot_id})

    def _root_lot_for(self, lot_id: str, lot_catalog: Dict[str, Dict[str, Any]] | None = None) -> str:
        if lot_catalog is None and lot_id in self._root_lot_index:
            return self._root_lot_index[lot_id]
        catalog = lot_catalog or self.state_data.get("lot_catalog", {})
        if lot_id not in catalog:
            return lot_id
        return catalog[lot_id].get("root_lot", lot_id)

    def _derive_root_cause(self, lot_id: str, finding: Dict[str, Any]) -> str:
        lot_data = self.state_data.get("lot_catalog", {}).get(lot_id, {})
        status = str(finding.get("status", ""))
        evidence = str(finding.get("evidence", "")).lower()
        if status == "mixed" or lot_data.get("mixed_from"):
            return "mixing_event"
        if status == "records_missing" or "missing" in evidence or "deleted" in evidence:
            return "record_deletion"
        if lot_data.get("relabeled_from") or "relabel" in evidence or "repack" in evidence:
            return "lot_relabel"
        return "source_contamination"

    def _remember_root_cause(self, cause: str) -> None:
        candidates = self.state_data.setdefault("root_cause_candidates", [])
        if cause and cause not in candidates:
            candidates.append(cause)
            candidates.sort()

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
