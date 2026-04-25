"""Deterministic scenario catalog for RecallTrace."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


PHASE1_SCENARIO: Dict[str, Any] = {
    "task_id": "phase1_direct_recall",
    "phase": 1,
    "difficulty": "easy",
    "name": "Direct Recall Containment",
    "objective": "Identify every location holding the recalled lot and quarantine all contaminated stock.",
    "max_steps": 10,
    "recall_notice": "Immediate recall: contaminated LotA detected in the cold-chain network.",
    "contaminated_lot": "LotA",
    "shipment_graph": {
        "warehouse": ["store1", "store2"],
        "store1": ["store2"],
        "store2": [],
    },
    "lot_catalog": {
        "LotA": {
            "contaminated": True,
            "product": "ready_meal",
            "root_lot": "LotA",
            "notes": "Original contaminated production batch.",
        },
        "LotB": {
            "contaminated": False,
            "product": "ready_meal",
            "root_lot": "LotB",
            "notes": "Safe control batch.",
        },
    },
    "nodes": {
        "warehouse": {
            "inventory": {"LotA": 100},
            "quarantined_inventory": {},
            "inspection_findings": {
                "LotA": {
                    "status": "confirmed_contaminated",
                    "unsafe_quantity": 100,
                    "evidence": "QA retained sample matched the recall notice for LotA.",
                }
            },
        },
        "store1": {
            "inventory": {"LotA": 50},
            "quarantined_inventory": {},
            "inspection_findings": {
                "LotA": {
                    "status": "confirmed_contaminated",
                    "unsafe_quantity": 50,
                    "evidence": "Receiving records show unopened cases from LotA.",
                }
            },
        },
        "store2": {
            "inventory": {"LotA": 20, "LotB": 30},
            "quarantined_inventory": {},
            "inspection_findings": {
                "LotA": {
                    "status": "confirmed_contaminated",
                    "unsafe_quantity": 20,
                    "evidence": "Backroom scan confirms LotA units remain unsold.",
                },
                "LotB": {
                    "status": "safe",
                    "unsafe_quantity": 0,
                    "evidence": "LotB is outside the recall scope.",
                },
            },
        },
    },
}

PHASE2_SCENARIO: Dict[str, Any] = {
    "task_id": "phase2_relabel_recall",
    "phase": 2,
    "difficulty": "medium",
    "name": "Relabeled Inventory Investigation",
    "objective": "Follow relabeled lots back to the source batch and quarantine every derived label precisely.",
    "max_steps": 14,
    "recall_notice": "Urgent recall: source LotA was relabeled during repacking and must be traced across derived labels.",
    "contaminated_lot": "LotA",
    "shipment_graph": {
        "warehouse": ["repack", "store1"],
        "repack": ["store2", "store3"],
        "store1": [],
        "store2": [],
        "store3": [],
    },
    "lot_catalog": {
        "LotA": {
            "contaminated": True,
            "product": "ready_meal",
            "root_lot": "LotA",
            "notes": "Original contaminated batch.",
        },
        "LotA_R1": {
            "contaminated": True,
            "product": "ready_meal",
            "root_lot": "LotA",
            "relabeled_from": "LotA",
            "notes": "Repacked under an internal secondary label.",
        },
        "LotA_R2": {
            "contaminated": True,
            "product": "ready_meal",
            "root_lot": "LotA",
            "relabeled_from": "LotA_R1",
            "notes": "Retail-ready relabel shipped after repacking.",
        },
        "LotB": {
            "contaminated": False,
            "product": "ready_meal",
            "root_lot": "LotB",
            "notes": "Safe control batch.",
        },
    },
    "nodes": {
        "warehouse": {
            "inventory": {"LotA": 40, "LotB": 30},
            "quarantined_inventory": {},
            "inspection_findings": {
                "LotA": {
                    "status": "confirmed_contaminated",
                    "unsafe_quantity": 40,
                    "evidence": "Source pallet labels match the recalled production run.",
                },
                "LotB": {
                    "status": "safe",
                    "unsafe_quantity": 0,
                    "evidence": "LotB remains outside the repacking stream.",
                },
            },
        },
        "repack": {
            "inventory": {"LotA_R1": 45},
            "quarantined_inventory": {},
            "inspection_findings": {
                "LotA_R1": {
                    "status": "confirmed_contaminated",
                    "unsafe_quantity": 45,
                    "evidence": "Repacking worksheet maps LotA directly to LotA_R1.",
                }
            },
        },
        "store1": {
            "inventory": {"LotA": 15, "LotB": 20},
            "quarantined_inventory": {},
            "inspection_findings": {
                "LotA": {
                    "status": "confirmed_contaminated",
                    "unsafe_quantity": 15,
                    "evidence": "Store retains cases with original LotA stickers.",
                },
                "LotB": {
                    "status": "safe",
                    "unsafe_quantity": 0,
                    "evidence": "LotB SKUs are unaffected.",
                },
            },
        },
        "store2": {
            "inventory": {"LotA_R1": 25},
            "quarantined_inventory": {},
            "inspection_findings": {
                "LotA_R1": {
                    "status": "confirmed_contaminated",
                    "unsafe_quantity": 25,
                    "evidence": "Receiving scan ties LotA_R1 to the repack facility transfer.",
                }
            },
        },
        "store3": {
            "inventory": {"LotA_R2": 20, "LotB": 10},
            "quarantined_inventory": {},
            "inspection_findings": {
                "LotA_R2": {
                    "status": "confirmed_contaminated",
                    "unsafe_quantity": 20,
                    "evidence": "Shelf tags reference the LotA_R2 relabel lineage.",
                },
                "LotB": {
                    "status": "safe",
                    "unsafe_quantity": 0,
                    "evidence": "LotB is a later safe shipment.",
                },
            },
        },
    },
}

PHASE3_SCENARIO: Dict[str, Any] = {
    "task_id": "phase3_mixed_shipments",
    "phase": 3,
    "difficulty": "hard",
    "name": "Mixed Inventory Precision Containment",
    "objective": "Contain only the unsafe quantity after contaminated stock was mixed with safe inventory during cross-docking.",
    "max_steps": 16,
    "recall_notice": "Critical recall: contaminated LotA was mixed with safe stock during cross-docking. Quarantine only the unsafe quantity.",
    "contaminated_lot": "LotA",
    "shipment_graph": {
        "warehouse": ["crossdock", "store1"],
        "crossdock": ["store2", "store3"],
        "store1": [],
        "store2": [],
        "store3": [],
    },
    "lot_catalog": {
        "LotA": {
            "contaminated": True,
            "product": "ready_meal",
            "root_lot": "LotA",
            "notes": "Contaminated upstream batch.",
        },
        "LotBlend": {
            "contaminated": True,
            "product": "ready_meal",
            "root_lot": "LotA",
            "mixed_from": ["LotA", "LotB"],
            "notes": "Cross-docked mixed lot containing both safe and unsafe units.",
        },
        "LotB": {
            "contaminated": False,
            "product": "ready_meal",
            "root_lot": "LotB",
            "notes": "Safe batch mixed into downstream palletization.",
        },
    },
    "nodes": {
        "warehouse": {
            "inventory": {"LotA": 30, "LotB": 25},
            "quarantined_inventory": {},
            "inspection_findings": {
                "LotA": {
                    "status": "confirmed_contaminated",
                    "unsafe_quantity": 30,
                    "evidence": "Source batch LotA remains fully unsafe at origin.",
                },
                "LotB": {
                    "status": "safe",
                    "unsafe_quantity": 0,
                    "evidence": "LotB remains unaffected at origin.",
                },
            },
        },
        "crossdock": {
            "inventory": {"LotBlend": 35, "LotB": 10},
            "quarantined_inventory": {},
            "inspection_findings": {
                "LotBlend": {
                    "status": "mixed",
                    "unsafe_quantity": 12,
                    "safe_quantity": 23,
                    "evidence": "Cross-dock exception log shows 12 unsafe units merged into LotBlend.",
                },
                "LotB": {
                    "status": "safe",
                    "unsafe_quantity": 0,
                    "evidence": "Standalone LotB pallet is outside the recall.",
                },
            },
        },
        "store1": {
            "inventory": {"LotA": 10, "LotB": 20},
            "quarantined_inventory": {},
            "inspection_findings": {
                "LotA": {
                    "status": "confirmed_contaminated",
                    "unsafe_quantity": 10,
                    "evidence": "Original LotA cases shipped directly before blending.",
                },
                "LotB": {
                    "status": "safe",
                    "unsafe_quantity": 0,
                    "evidence": "Store LotB stock is unaffected.",
                },
            },
        },
        "store2": {
            "inventory": {"LotBlend": 15},
            "quarantined_inventory": {},
            "inspection_findings": {
                "LotBlend": {
                    "status": "mixed",
                    "unsafe_quantity": 8,
                    "safe_quantity": 7,
                    "evidence": "Receiving variance report allocates 8 unsafe units to store2.",
                }
            },
        },
        "store3": {
            "inventory": {"LotBlend": 20, "LotB": 5},
            "quarantined_inventory": {},
            "inspection_findings": {
                "LotBlend": {
                    "status": "mixed",
                    "unsafe_quantity": 4,
                    "safe_quantity": 16,
                    "evidence": "Inventory reconciliation isolates 4 unsafe units in store3's mixed lot.",
                },
                "LotB": {
                    "status": "safe",
                    "unsafe_quantity": 0,
                    "evidence": "Separate LotB shelf stock is unaffected.",
                },
            },
        },
    },
}

SCENARIOS: Dict[str, Dict[str, Any]] = {
    PHASE1_SCENARIO["task_id"]: PHASE1_SCENARIO,
    PHASE2_SCENARIO["task_id"]: PHASE2_SCENARIO,
    PHASE3_SCENARIO["task_id"]: PHASE3_SCENARIO,
}

PHASE_LOOKUP: Dict[int, str] = {
    1: PHASE1_SCENARIO["task_id"],
    2: PHASE2_SCENARIO["task_id"],
    3: PHASE3_SCENARIO["task_id"],
}


def build_scenario(task_id: str | None = None, phase: int | None = None) -> Dict[str, Any]:
    """Return a fresh copy of the deterministic scenario for the requested task or phase."""
    if task_id is None:
        if phase is None:
            phase = 1
        task_id = PHASE_LOOKUP[phase]
    if task_id not in SCENARIOS:
        raise ValueError(f"Unknown task_id '{task_id}'. Expected one of {sorted(SCENARIOS)}.")
    return deepcopy(SCENARIOS[task_id])


def build_phase1_scenario() -> Dict[str, Any]:
    return build_scenario(task_id=PHASE1_SCENARIO["task_id"])


def build_phase2_scenario() -> Dict[str, Any]:
    return build_scenario(task_id=PHASE2_SCENARIO["task_id"])


def build_phase3_scenario() -> Dict[str, Any]:
    return build_scenario(task_id=PHASE3_SCENARIO["task_id"])


def list_task_specs() -> List[Dict[str, Any]]:
    """Return lightweight metadata for all tasks."""
    return [
        {
            "task_id": scenario["task_id"],
            "name": scenario["name"],
            "difficulty": scenario["difficulty"],
            "objective": scenario["objective"],
            "max_steps": scenario["max_steps"],
        }
        for scenario in SCENARIOS.values()
    ]
