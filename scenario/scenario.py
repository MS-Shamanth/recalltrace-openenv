"""Deterministic scenario definitions for RecallTrace."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


SIMPLE_SCENARIO: Dict[str, Any] = {
    "scenario_id": "phase1_direct_recall",
    "recall_notice": "Immediate recall: contaminated LotA detected in the cold-chain network.",
    "contaminated_lot": "LotA",
    "shipment_graph": {
        "warehouse": ["store1", "store2"],
        "store1": ["store2"],
        "store2": [],
    },
    "lot_catalog": {
        "LotA": {"contaminated": True, "product": "ready_meal"},
        "LotB": {"contaminated": False, "product": "ready_meal"},
    },
    "nodes": {
        "warehouse": {
            "inventory": {"LotA": 100},
            "quarantined_inventory": {},
        },
        "store1": {
            "inventory": {"LotA": 50},
            "quarantined_inventory": {},
        },
        "store2": {
            "inventory": {"LotA": 20, "LotB": 30},
            "quarantined_inventory": {},
        },
    },
}


def build_phase1_scenario() -> Dict[str, Any]:
    """Return a fresh copy of the deterministic Phase 1 scenario."""
    return deepcopy(SIMPLE_SCENARIO)
