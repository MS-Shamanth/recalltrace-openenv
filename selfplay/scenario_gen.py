"""Parametric scenario generator for adversarial self-play.

Generates random supply-chain DAGs and applies adversary-chosen
interventions. Interventions create GENUINE ambiguity — some nodes
look contaminated but aren't, and some truly contaminated nodes have
their evidence obscured.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Dict, List, Tuple


NODE_ROLES = ["warehouse", "crossdock", "store"]


def _make_node_id(role: str, index: int) -> str:
    return f"{role}_{index}"


def generate_graph(num_nodes: int = 10, seed: int | None = None) -> Dict[str, Any]:
    """Create a random supply-chain DAG with inventory at every node.

    Returns a scenario dict compatible with RecallTraceEnv(scenario_data=...).
    Contamination is placed at a single source warehouse by default.
    """
    rng = random.Random(seed)

    num_warehouses = min(2, max(1, num_nodes // 5))
    num_crossdocks = min(3, max(1, (num_nodes - num_warehouses) // 3))
    num_stores = max(2, num_nodes - num_warehouses - num_crossdocks)

    warehouses = [_make_node_id("warehouse", i) for i in range(num_warehouses)]
    crossdocks = [_make_node_id("crossdock", i) for i in range(num_crossdocks)]
    stores = [_make_node_id("store", i) for i in range(num_stores)]
    all_nodes: List[str] = warehouses + crossdocks + stores

    # Build directed edges
    shipment_graph: Dict[str, List[str]] = {n: [] for n in all_nodes}
    for wh in warehouses:
        for t in crossdocks + stores[:2]:
            if rng.random() < 0.7:
                shipment_graph[wh].append(t)
        if not shipment_graph[wh]:
            shipment_graph[wh].append(rng.choice(crossdocks or stores))
    for cd in crossdocks:
        for s in stores:
            if rng.random() < 0.5:
                shipment_graph[cd].append(s)
        if not shipment_graph[cd]:
            shipment_graph[cd].append(rng.choice(stores))

    contaminated_lot = "LotA"
    safe_lot = "LotB"

    lot_catalog = {
        contaminated_lot: {
            "contaminated": True, "product": "ready_meal",
            "root_lot": contaminated_lot,
            "notes": "Original contaminated production batch.",
        },
        safe_lot: {
            "contaminated": False, "product": "ready_meal",
            "root_lot": safe_lot,
            "notes": "Safe control batch.",
        },
    }

    nodes: Dict[str, Dict[str, Any]] = {}
    source_wh = warehouses[0]

    for node_id in all_nodes:
        inv: Dict[str, int] = {}
        findings: Dict[str, Dict[str, Any]] = {}

        safe_qty = rng.randint(10, 40)
        inv[safe_lot] = safe_qty
        findings[safe_lot] = {
            "status": "safe", "unsafe_quantity": 0,
            "evidence": f"{safe_lot} is outside the recall scope.",
        }

        is_source = node_id == source_wh
        # Only ONE downstream node gets contaminated (not all)
        first_downstream = shipment_graph.get(source_wh, [None])[0]
        is_downstream = node_id == first_downstream
        if is_source or is_downstream:
            unsafe_qty = rng.randint(15, 60)
            inv[contaminated_lot] = unsafe_qty
            findings[contaminated_lot] = {
                "status": "confirmed_contaminated",
                "unsafe_quantity": unsafe_qty,
                "evidence": f"QA testing confirms {contaminated_lot} contamination at {node_id}.",
            }

        # Add ambient suspicious lots at most nodes (safe but look fishy)
        if rng.random() < 0.6 and node_id != source_wh:
            suspect_lot = f"LotX_{node_id}"
            s_qty = rng.randint(5, 20)
            inv[suspect_lot] = s_qty
            findings[suspect_lot] = {
                "status": "suspect",
                "unsafe_quantity": 0,
                "evidence": f"Lot {suspect_lot} flagged during routine scan. Possibly contaminated.",
            }
            lot_catalog[suspect_lot] = {
                "contaminated": False, "product": "ready_meal",
                "root_lot": f"LotX_{node_id}",
                "notes": "Ambient suspect lot — actually safe.",
            }

        nodes[node_id] = {
            "inventory": inv,
            "quarantined_inventory": {},
            "inspection_findings": findings,
        }

    node_regions = {}
    for n in warehouses:
        node_regions[n] = "source"
    for n in crossdocks:
        node_regions[n] = "midstream"
    for n in stores:
        node_regions[n] = "downstream"

    return {
        "task_id": "selfplay_adversarial",
        "phase": 3,
        "difficulty": "adversarial",
        "name": "Adversarial Self-Play Episode",
        "objective": "Find and quarantine contaminated nodes under adversarial intervention.",
        "max_steps": 30,
        "recall_notice": f"Immediate recall: contaminated {contaminated_lot} detected in the supply chain.",
        "contaminated_lot": contaminated_lot,
        "shipment_graph": shipment_graph,
        "lot_catalog": lot_catalog,
        "nodes": nodes,
        "_node_regions": node_regions,
        "_all_node_ids": all_nodes,
        "_warehouses": warehouses,
        "_crossdocks": crossdocks,
        "_stores": stores,
    }


# ---------------------------------------------------------------------------
# Intervention application
# ---------------------------------------------------------------------------

def apply_intervention(
    scenario: Dict[str, Any],
    intervention_type: str,
    target_node: str,
    num_hops: int,
    rng: random.Random | None = None,
) -> Dict[str, Any]:
    """Apply an adversary-chosen intervention to the scenario.

    Each intervention creates genuine ambiguity:
      - lot_relabel: hides contamination behind new labels + adds decoy labels
      - mixing_event: mixes unsafe with safe, varies proportions across nodes
      - record_deletion: removes evidence + plants misleading evidence elsewhere
    """
    sc = deepcopy(scenario)
    rng = rng or random.Random()
    if target_node not in sc["nodes"]:
        target_node = list(sc["nodes"].keys())[0]

    if intervention_type == "lot_relabel":
        _apply_relabel(sc, target_node, num_hops, rng)
    elif intervention_type == "mixing_event":
        _apply_mixing(sc, target_node, num_hops, rng)
    elif intervention_type == "record_deletion":
        _apply_deletion(sc, target_node, num_hops, rng)
    return sc


def _apply_relabel(sc, target_node, num_hops, rng):
    """Relabel contamination AND add decoy relabeled lots that are safe."""
    nodes = sc["nodes"]
    catalog = sc["lot_catalog"]
    graph = sc["shipment_graph"]
    clot = sc["contaminated_lot"]

    node_data = nodes[target_node]
    original_qty = node_data["inventory"].pop(clot, 0) or rng.randint(15, 40)
    node_data["inspection_findings"].pop(clot, None)

    downstream = graph.get(target_node, [])
    current_lot = clot

    # Real relabeled contaminated lots
    for hop in range(num_hops):
        new_lot = f"LotA_R{hop + 1}"
        catalog[new_lot] = {
            "contaminated": True, "product": "ready_meal",
            "root_lot": clot, "relabeled_from": current_lot,
            "notes": f"Relabeled from {current_lot}.",
        }
        place = target_node if hop == 0 else (downstream[hop % len(downstream)] if downstream else target_node)
        qty = max(5, original_qty // (hop + 1))
        nodes[place]["inventory"][new_lot] = qty
        nodes[place]["inspection_findings"][new_lot] = {
            "status": "confirmed_contaminated", "unsafe_quantity": qty,
            "evidence": f"Relabeled lot {new_lot} traced to contaminated lineage.",
        }
        current_lot = new_lot

    # DECOY: add MANY fake relabeled lots that look suspicious but are safe
    num_decoys = rng.randint(3, 5 + num_hops)
    for d in range(num_decoys):
        decoy_lot = f"LotC_R{d + 1}"
        catalog[decoy_lot] = {
            "contaminated": False, "product": "ready_meal",
            "root_lot": "LotC",
            "relabeled_from": "LotC" if d == 0 else f"LotC_R{d}",
            "notes": "Relabeled lot from a separate safe batch.",
        }
        decoy_node = rng.choice(list(nodes.keys()))
        decoy_qty = rng.randint(8, 30)
        nodes[decoy_node]["inventory"][decoy_lot] = decoy_qty
        nodes[decoy_node]["inspection_findings"][decoy_lot] = {
            "status": "suspect",
            "unsafe_quantity": 0,
            "evidence": f"Relabeled lot {decoy_lot} found -- origin unclear, possibly contaminated.",
        }
    if "LotC" not in catalog:
        catalog["LotC"] = {
            "contaminated": False, "product": "ready_meal",
            "root_lot": "LotC",
            "notes": "Safe batch used in decoy relabeling.",
        }


def _apply_mixing(sc, target_node, num_hops, rng):
    """Mix contaminated with safe stock, vary unsafe proportions, add red herrings."""
    nodes = sc["nodes"]
    catalog = sc["lot_catalog"]
    graph = sc["shipment_graph"]
    clot = sc["contaminated_lot"]

    node_data = nodes[target_node]
    original_qty = node_data["inventory"].pop(clot, 0) or rng.randint(15, 40)
    node_data["inspection_findings"].pop(clot, None)

    blend_lot = "LotBlend"
    safe_qty = rng.randint(10, 30)
    total_qty = original_qty + safe_qty

    catalog[blend_lot] = {
        "contaminated": True, "product": "ready_meal",
        "root_lot": clot, "mixed_from": [clot, "LotB"],
        "notes": "Mixed lot containing both safe and unsafe units.",
    }

    downstream = graph.get(target_node, [])
    distribute_to = [target_node] + downstream[:num_hops]

    for i, place in enumerate(distribute_to):
        if i == 0:
            blend_qty = total_qty // 2 + rng.randint(0, 5)
            unsafe_in = max(1, original_qty // 2)
        else:
            blend_qty = max(5, total_qty // (len(distribute_to) * 2))
            unsafe_in = max(1, original_qty // (len(distribute_to) * 2))

        nodes[place]["inventory"][blend_lot] = blend_qty
        nodes[place]["inspection_findings"][blend_lot] = {
            "status": "mixed", "unsafe_quantity": unsafe_in,
            "safe_quantity": blend_qty - unsafe_in,
            "evidence": f"Cross-dock log: {unsafe_in} unsafe units in blend at {place}.",
        }

    # RED HERRING: add MANY safe-but-suspicious nodes that LOOK mixed
    herring_count = rng.randint(3, 5 + num_hops)
    for h in range(herring_count):
        herring_lot = f"LotBlend_H{h}"
        herring_node = rng.choice(list(nodes.keys()))
        herring_qty = rng.randint(10, 25)
        catalog[herring_lot] = {
            "contaminated": False, "product": "ready_meal",
            "root_lot": "LotB",
            "notes": "Safe blend mistakenly flagged.",
        }
        nodes[herring_node]["inventory"][herring_lot] = herring_qty
        nodes[herring_node]["inspection_findings"][herring_lot] = {
            "status": "mixed", "unsafe_quantity": 0,
            "safe_quantity": herring_qty,
            "evidence": f"Blend at {herring_node} flagged for review. Likely safe but unconfirmed.",
        }


def _apply_deletion(sc, target_node, num_hops, rng):
    """Remove evidence at target + neighbors AND plant false positives elsewhere."""
    nodes = sc["nodes"]
    graph = sc["shipment_graph"]
    clot = sc["contaminated_lot"]

    to_censor = [target_node]
    neighbors = graph.get(target_node, [])
    to_censor.extend(neighbors[:max(0, num_hops - 1)])

    for node_id in to_censor:
        if node_id not in nodes:
            continue
        findings = nodes[node_id].get("inspection_findings", {})
        for lot_id in list(findings.keys()):
            lot_data = sc["lot_catalog"].get(lot_id, {})
            if lot_data.get("contaminated") or lot_data.get("root_lot") == clot:
                # Hide the evidence — make it ambiguous
                findings[lot_id] = {
                    "status": "records_missing",
                    "unsafe_quantity": findings[lot_id].get("unsafe_quantity", 0),
                    "evidence": "Inspection records unavailable. Status unclear.",
                }

    # FALSE POSITIVE: plant MANY fake contamination evidence at safe nodes
    false_count = rng.randint(3, 5 + num_hops)
    safe_nodes = [n for n in nodes if n not in to_censor]
    for fp_idx in range(min(false_count, len(safe_nodes))):
        fp_node = rng.choice(safe_nodes)
        safe_nodes.remove(fp_node)
        fp_lot = f"LotA_phantom_{rng.randint(100, 999)}"
        fp_qty = rng.randint(5, 20)
        sc["lot_catalog"][fp_lot] = {
            "contaminated": False, "product": "ready_meal",
            "root_lot": "LotA_phantom",
            "notes": "Phantom lot -- actually safe despite suspicious name.",
        }
        nodes[fp_node]["inventory"][fp_lot] = fp_qty
        nodes[fp_node]["inspection_findings"][fp_lot] = {
            "status": "suspect",
            "unsafe_quantity": 0,
            "evidence": f"Lot {fp_lot} flagged as potentially contaminated. Pending verification.",
        }


# ---------------------------------------------------------------------------
# F1 computation
# ---------------------------------------------------------------------------

def compute_f1(
    scenario: Dict[str, Any],
    quarantined_nodes: List[str],
) -> Tuple[float, Dict[str, Any]]:
    """Compute node-level F1 for quarantine decisions.

    A node is truly unsafe if it holds inventory with unsafe_quantity > 0
    AND the lot is genuinely contaminated (catalog says contaminated=True).
    """
    nodes = scenario["nodes"]
    clot = scenario["contaminated_lot"]
    catalog = scenario["lot_catalog"]

    truly_unsafe = set()
    for node_id, node_data in nodes.items():
        for lot_id, finding in node_data.get("inspection_findings", {}).items():
            lot_info = catalog.get(lot_id, {})
            is_contaminated = lot_info.get("contaminated", False)
            has_unsafe = finding.get("unsafe_quantity", 0) > 0
            if is_contaminated and has_unsafe:
                truly_unsafe.add(node_id)

    quarantined_set = set(quarantined_nodes)
    tp = len(truly_unsafe & quarantined_set)
    fp = len(quarantined_set - truly_unsafe)
    fn = len(truly_unsafe - quarantined_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1, {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "truly_unsafe_nodes": sorted(truly_unsafe),
        "quarantined_nodes": sorted(quarantined_set),
    }
