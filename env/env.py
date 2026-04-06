"""Core RecallTrace environment with deterministic action execution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Tuple

from scenario.scenario import build_phase1_scenario

ACTIONS = [
    "inspect_node",
    "trace_lot",
    "quarantine",
    "notify",
    "finalize",
]

OBSERVATION = {
    "recall_notice": str,
    "inventory": dict,
    "history": list,
    "discovered_shipments": dict,
    "inspected_nodes": list,
    "notified_nodes": list,
    "quarantined_inventory": dict,
}


class RecallTraceEnv:
    """Deterministic recall environment with hidden ground truth."""

    def __init__(self, scenario_data: Dict[str, Any] | None = None):
        self._scenario_template = (
            deepcopy(scenario_data) if scenario_data is not None else build_phase1_scenario()
        )
        self.state_data: Dict[str, Any] = {}
        self.ground_truth: Dict[str, Any] = {}
        self.done = False

    def reset(self) -> Dict[str, Any]:
        """Start a new deterministic scenario and recompute ground truth."""
        self.done = False

        scenario = deepcopy(self._scenario_template)
        self.state_data = {
            "scenario_id": scenario["scenario_id"],
            "recall_notice": scenario["recall_notice"],
            "contaminated_lot_hint": scenario["contaminated_lot"],
            "shipment_graph": scenario["shipment_graph"],
            "nodes": scenario["nodes"],
            "history": [],
            "discovered_shipments": {},
            "inspected_nodes": set(),
            "traced_lots": {},
            "notified_nodes": set(),
            "quarantine_log": [],
        }
        self.ground_truth = self._build_ground_truth()
        return self._get_observation()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute an action, update state, and return an OpenAI Gym style tuple."""
        if self.done:
            return self._get_observation(), 0.0, True, {"message": "Environment already finalized."}

        if not isinstance(action, dict):
            return self._get_observation(), -0.25, False, {"error": "Action must be a dictionary."}

        action_type = action.get("type")
        if action_type not in ACTIONS:
            self._record_history(f"Rejected unknown action: {action_type}")
            return self._get_observation(), -0.25, False, {"error": "Unknown action type."}

        handler = getattr(self, f"_handle_{action_type}")
        try:
            reward, info = handler(action)
        except (TypeError, ValueError) as exc:
            self._record_history(f"Rejected action {action_type}: {exc}")
            return self._get_observation(), -0.2, False, {"error": str(exc), "action_type": action_type}
        return self._get_observation(), reward, self.done, info

    def state(self) -> Dict[str, Any]:
        """Return full internal state, including hidden ground truth, for debugging."""
        return {
            "done": self.done,
            "state_data": deepcopy(self._serialize_state(self.state_data)),
            "ground_truth": deepcopy(self.ground_truth),
        }

    def _get_observation(self) -> Dict[str, Any]:
        """Return the observable state only, excluding the hidden oracle."""
        return {
            "recall_notice": self.state_data["recall_notice"],
            "inventory": self._inventory_snapshot(),
            "history": list(self.state_data["history"]),
            "discovered_shipments": deepcopy(self.state_data["discovered_shipments"]),
            "inspected_nodes": sorted(self.state_data["inspected_nodes"]),
            "notified_nodes": sorted(self.state_data["notified_nodes"]),
            "quarantined_inventory": self._quarantine_snapshot(),
        }

    def _handle_inspect_node(self, action: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        node_id = action.get("node_id")
        node = self._get_node(node_id)

        self.state_data["inspected_nodes"].add(node_id)
        self.state_data["discovered_shipments"][node_id] = list(
            self.state_data["shipment_graph"].get(node_id, [])
        )
        self._record_history(f"Inspected node {node_id}")

        info = {
            "node_id": node_id,
            "inventory": deepcopy(node["inventory"]),
            "quarantined_inventory": deepcopy(node["quarantined_inventory"]),
            "outbound_shipments": list(self.state_data["shipment_graph"].get(node_id, [])),
        }
        return 0.15, info

    def _handle_trace_lot(self, action: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        lot_id = action.get("lot_id")
        if not lot_id:
            raise ValueError("trace_lot action requires 'lot_id'.")

        impacted_nodes = []
        impacted_quantities = {}

        for node_id, node_data in self.state_data["nodes"].items():
            available_qty = node_data["inventory"].get(lot_id, 0)
            quarantined_qty = node_data["quarantined_inventory"].get(lot_id, 0)
            total_qty = available_qty + quarantined_qty
            if total_qty > 0:
                impacted_nodes.append(node_id)
                impacted_quantities[node_id] = total_qty

        self.state_data["traced_lots"][lot_id] = {
            "nodes": impacted_nodes,
            "quantities": impacted_quantities,
        }
        self._record_history(f"Traced lot {lot_id}")

        if not impacted_nodes:
            reward = -0.1
        elif lot_id in self.ground_truth["affected_lots"]:
            reward = 0.2
        else:
            reward = 0.05
        info = {
            "lot_id": lot_id,
            "affected_nodes": impacted_nodes,
            "quantities": impacted_quantities,
            "total_quantity": sum(impacted_quantities.values()),
        }
        return reward, info

    def _handle_quarantine(self, action: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        node_id = action.get("node_id")
        lot_id = action.get("lot_id")
        node = self._get_node(node_id)

        if not lot_id:
            raise ValueError("quarantine action requires 'lot_id'.")

        available_qty = node["inventory"].get(lot_id, 0)
        if available_qty <= 0:
            self._record_history(f"Failed quarantine for {lot_id} at {node_id}: no available stock")
            return -0.2, {
                "error": "No available stock to quarantine.",
                "node_id": node_id,
                "lot_id": lot_id,
            }

        requested_qty = action.get("quantity", available_qty)
        if requested_qty <= 0:
            raise ValueError("quarantine quantity must be positive.")

        quarantined_qty = min(requested_qty, available_qty)
        node["inventory"][lot_id] = available_qty - quarantined_qty
        if node["inventory"][lot_id] == 0:
            del node["inventory"][lot_id]

        node["quarantined_inventory"][lot_id] = (
            node["quarantined_inventory"].get(lot_id, 0) + quarantined_qty
        )

        log_entry = {
            "node_id": node_id,
            "lot_id": lot_id,
            "quantity": quarantined_qty,
        }
        self.state_data["quarantine_log"].append(log_entry)
        self._record_history(f"Quarantined {quarantined_qty} units of {lot_id} at {node_id}")

        correct_qty = self.ground_truth["correct_quantities"].get(node_id, {}).get(lot_id, 0)
        cumulative_quarantined = node["quarantined_inventory"].get(lot_id, 0)

        if lot_id not in self.ground_truth["affected_lots"]:
            reward = -0.3
        elif cumulative_quarantined == correct_qty:
            reward = 0.4
        elif cumulative_quarantined < correct_qty:
            reward = 0.2
        else:
            reward = -0.15

        info = {
            "node_id": node_id,
            "lot_id": lot_id,
            "quarantined_quantity": quarantined_qty,
            "remaining_inventory": node["inventory"].get(lot_id, 0),
            "cumulative_quarantined": cumulative_quarantined,
        }
        return reward, info

    def _handle_notify(self, action: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        requested_target = action.get("node_id", "all")
        if requested_target in ("all", "all_nodes", None):
            targets = list(self.state_data["nodes"].keys())
        else:
            self._get_node(requested_target)
            targets = [requested_target]

        newly_notified = []
        for node_id in targets:
            if node_id not in self.state_data["notified_nodes"]:
                self.state_data["notified_nodes"].add(node_id)
                newly_notified.append(node_id)

        if newly_notified:
            self._record_history(f"Sent notifications to {', '.join(newly_notified)}")
        else:
            self._record_history("Notification action repeated without new recipients")

        affected_newly_notified = sum(
            1 for node_id in newly_notified if node_id in self.ground_truth["affected_nodes"]
        )
        reward = 0.1 * affected_newly_notified
        if not newly_notified:
            reward = -0.05
        elif (
            requested_target not in ("all", "all_nodes", None)
            and requested_target not in self.ground_truth["affected_nodes"]
        ):
            reward -= 0.05

        return reward, {"notified_nodes": targets, "newly_notified": newly_notified}

    def _handle_finalize(self, action: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        del action
        self.done = True

        quarantine_match = self._compute_quarantine_match()
        notified_affected_nodes = sorted(
            set(self.ground_truth["affected_nodes"]).intersection(self.state_data["notified_nodes"])
        )
        affected_node_total = len(self.ground_truth["affected_nodes"])
        notification_score = (
            len(notified_affected_nodes) / affected_node_total if affected_node_total else 1.0
        )
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
        total_affected_quantity = self.ground_truth["total_affected_quantity"]
        quarantine_score = (
            max(0.0, 1.0 - ((missing_quantity_total + over_quantity_total) / total_affected_quantity))
            if total_affected_quantity
            else 1.0
        )
        score = round((notification_score + quarantine_score) / 2.0, 4)

        all_affected_nodes_notified = notification_score == 1.0
        all_affected_stock_quarantined = missing_quantity_total == 0 and over_quantity_total == 0

        status = {
            "score": score,
            "notification_score": round(notification_score, 4),
            "quarantine_score": round(quarantine_score, 4),
            "all_affected_nodes_notified": all_affected_nodes_notified,
            "all_affected_stock_quarantined": all_affected_stock_quarantined,
            "quarantine_match": quarantine_match,
        }
        self._record_history("Finalized recall response")

        reward = score
        return reward, status

    def _build_ground_truth(self) -> Dict[str, Any]:
        affected_lots = {
            lot_id
            for lot_id, lot_data in self._scenario_template["lot_catalog"].items()
            if lot_data.get("contaminated", False)
        }

        affected_nodes = set()
        correct_quantities: Dict[str, Dict[str, int]] = {}

        for node_id, node_data in self.state_data["nodes"].items():
            for lot_id, quantity in node_data["inventory"].items():
                if lot_id in affected_lots and quantity > 0:
                    affected_nodes.add(node_id)
                    correct_quantities.setdefault(node_id, {})[lot_id] = quantity

        total_affected_quantity = sum(
            quantity
            for node_quantities in correct_quantities.values()
            for quantity in node_quantities.values()
        )

        return {
            "affected_lots": sorted(affected_lots),
            "affected_nodes": sorted(affected_nodes),
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
        return {
            node_id: deepcopy(node_data["inventory"])
            for node_id, node_data in self.state_data["nodes"].items()
        }

    def _quarantine_snapshot(self) -> Dict[str, Dict[str, int]]:
        return {
            node_id: deepcopy(node_data["quarantined_inventory"])
            for node_id, node_data in self.state_data["nodes"].items()
            if node_data["quarantined_inventory"]
        }

    def _get_node(self, node_id: str) -> Dict[str, Any]:
        if not node_id:
            raise ValueError("Action requires 'node_id'.")
        if node_id not in self.state_data["nodes"]:
            raise ValueError(f"Unknown node_id '{node_id}'.")
        return self.state_data["nodes"][node_id]

    def _record_history(self, message: str) -> None:
        self.state_data["history"].append(message)

    def _serialize_state(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._serialize_state(item) for key, item in value.items()}
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, list):
            return [self._serialize_state(item) for item in value]
        return value
