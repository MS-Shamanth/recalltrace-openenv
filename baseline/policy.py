"""Heuristic baseline policy for RecallTrace."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from env.models import RecallAction, RecallObservation

# openai is imported lazily inside choose_llm_action to avoid import errors
# in environments where it is not installed (e.g. HF Jobs training runs).


LOT_PATTERN = re.compile(r"\bLot[A-Za-z0-9_]+\b")


def _extract_root_lot(observation: RecallObservation) -> str:
    match = LOT_PATTERN.search(observation.recall_notice)
    return match.group(0) if match else "LotA"


def choose_heuristic_action(observation: RecallObservation) -> RecallAction:
    """Choose the next deterministic action using only observable state."""
    root_lot = _extract_root_lot(observation)
    trace_result = observation.trace_results.get(root_lot)

    if trace_result is None:
        return RecallAction(type="trace_lot", lot_id=root_lot, rationale="Map the recall lineage first.")

    if not observation.root_cause_candidates and observation.remaining_step_budget > 2:
        return RecallAction(
            type="cross_reference",
            lot_id=root_lot,
            rationale="Connect lot lineage, graph placement, and evidence before quarantining.",
        )

    affected_nodes = trace_result.get("affected_nodes", [])
    for node_id in affected_nodes:
        if node_id not in observation.inspected_nodes:
            return RecallAction(type="inspect_node", node_id=node_id, rationale="Collect local evidence before quarantining.")

    for node_id, findings in observation.inspection_results.items():
        for lot_id, finding in findings.items():
            unsafe_quantity = finding.unsafe_quantity
            quarantined_quantity = observation.quarantined_inventory.get(node_id, {}).get(lot_id, 0)
            available_quantity = observation.inventory.get(node_id, {}).get(lot_id, 0)
            remaining_target = unsafe_quantity - quarantined_quantity
            if remaining_target > 0 and available_quantity > 0:
                return RecallAction(
                    type="quarantine",
                    node_id=node_id,
                    lot_id=lot_id,
                    quantity=min(remaining_target, available_quantity),
                    rationale="Isolate the exact unsafe quantity discovered during inspection.",
                )

    missing_notifications = [node_id for node_id in affected_nodes if node_id not in observation.notified_nodes]
    if missing_notifications:
        return RecallAction(type="notify", node_id="all", rationale="Alert every impacted stakeholder before closing the incident.")

    return RecallAction(type="finalize", rationale="Containment actions are complete.")


def choose_llm_action(
    client: Optional[Any],
    model_name: str,
    observation: RecallObservation,
    history: list[dict[str, Any]],
) -> Optional[RecallAction]:
    """Ask an LLM for the next action, returning None on failure."""
    try:
        from openai import OpenAI  # noqa: F401 — lazy import, openai may not be installed
    except ImportError:
        return None

    if client is None:
        return None

    prompt = {
        "task_id": observation.task_id,
        "phase": observation.phase,
        "notice": observation.recall_notice,
        "inventory": observation.inventory,
        "inspection_results": {
            node_id: {lot_id: evidence.model_dump() for lot_id, evidence in findings.items()}
            for node_id, findings in observation.inspection_results.items()
        },
        "trace_results": observation.trace_results,
        "notified_nodes": observation.notified_nodes,
        "quarantined_inventory": observation.quarantined_inventory,
        "steps_taken": observation.steps_taken,
        "remaining_step_budget": observation.remaining_step_budget,
        "history": history[-6:],
        "instruction": "Return only compact JSON with keys type,node_id,lot_id,quantity,rationale. Use one valid action.",
    }

    try:
        completion = client.chat.completions.create(
            model=model_name,
            temperature=0,
            max_tokens=180,
            messages=[
                {"role": "system", "content": "You are operating a deterministic product recall environment. Respond with only valid JSON for the next action."},
                {"role": "user", "content": json.dumps(prompt, sort_keys=True)},
            ],
        )
        text = (completion.choices[0].message.content or "").strip()
        if not text:
            return None
        return RecallAction.model_validate_json(text)
    except Exception:
        return None
