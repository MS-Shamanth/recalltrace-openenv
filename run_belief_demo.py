"""Belief State Tracker — Live Demo

Simulates 8 steps of an agent investigating a contaminated supply chain.
Shows P(contaminated) rising for truly contaminated nodes while staying
low for safe nodes.  At step 6, the agent quarantines when P > 0.85.

Usage:
    python run_belief_demo.py              # saves frames to plots/
    python run_belief_demo.py --live       # live matplotlib animation
    python run_belief_demo.py --terminal   # terminal-only output

Designed to run in Colab, Jupyter, or a local terminal.
"""

from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from selfplay.belief_tracker import BeliefStateTracker


# ---------------------------------------------------------------------------
# Demo scenario: Lot_A and Lot_C are contaminated.
# Agent uses tool calls to gather evidence.
# ---------------------------------------------------------------------------

NODES = ["Lot_A", "Warehouse_B", "Lot_C", "Distributor_D", "Retailer_E", "Lot_F"]

HIDDEN_ARCS = [
    ("Lot_A", "Warehouse_B"),     # exists — contamination path
    ("Lot_A", "Lot_C"),           # exists — hidden relabel
    ("Warehouse_B", "Lot_F"),     # does NOT exist — false signal
    ("Distributor_D", "Retailer_E"),  # exists but irrelevant
]

# Each step: (tool_call_description, node_prob_updates, edge_prob_updates)
STEPS = [
    # Step 1: Agent inspects Distributor_D — finds suspicious report
    (
        "inspect_node(Distributor_D) -> partial contamination report",
        {"Distributor_D": 0.35, "Lot_A": 0.20, "Warehouse_B": 0.15},
        {("Lot_A", "Warehouse_B"): 0.55},
    ),

    # Step 2: Agent traces Lot_A — discovers relabel to Lot_C
    (
        "trace_lot(Lot_A) -> found repack event, Lot_C created",
        {"Lot_A": 0.55, "Lot_C": 0.40, "Distributor_D": 0.30},
        {("Lot_A", "Lot_C"): 0.72, ("Lot_A", "Warehouse_B"): 0.65},
    ),

    # Step 3: Agent inspects Warehouse_B — nothing significant
    (
        "inspect_node(Warehouse_B) -> clean inspection, no anomalies",
        {"Warehouse_B": 0.12, "Lot_A": 0.62},
        {("Warehouse_B", "Lot_F"): 0.20},
    ),

    # Step 4: Agent cross-references Lot_A and Lot_C
    (
        "cross_reference(Lot_A, Lot_C) -> shared origin confirmed",
        {"Lot_A": 0.78, "Lot_C": 0.70, "Retailer_E": 0.15},
        {("Lot_A", "Lot_C"): 0.91},
    ),

    # Step 5: Agent inspects Lot_C — finds contamination markers
    (
        "inspect_node(Lot_C) -> contamination markers detected",
        {"Lot_C": 0.82, "Lot_A": 0.85, "Distributor_D": 0.22},
        {("Lot_A", "Lot_C"): 0.95},
    ),

    # Step 6: P(Lot_A) crosses threshold — agent quarantines
    (
        "quarantine(Lot_A) -> P=0.88 > threshold, quarantine issued",
        {"Lot_A": 0.88},
        {},
    ),

    # Step 7: One more check on Lot_C to confirm
    (
        "request_lab_test(Lot_C) -> positive result",
        {"Lot_C": 0.93, "Lot_F": 0.08},
        {},
    ),

    # Step 8: Agent quarantines Lot_C and finalizes
    (
        "quarantine(Lot_C) -> P=0.93 > threshold, finalize()",
        {"Lot_C": 0.95},
        {},
    ),
]


def run_demo(mode: str = "save") -> None:
    """Run the belief tracker demo.

    Args:
        mode: "save" — save frames to plots/
              "live" — live matplotlib animation
              "terminal" — terminal-only output
    """
    tracker = BeliefStateTracker(
        nodes=NODES,
        hidden_arcs=HIDDEN_ARCS,
        quarantine_threshold=0.85,
    )

    print()
    print("=" * 62)
    print("  RecallTrace -- Belief State Tracker Demo")
    print("  Simulating 8 tool calls on a 6-node supply chain")
    print("=" * 62)

    os.makedirs("plots/belief_frames", exist_ok=True)

    for i, (action, node_probs, edge_probs) in enumerate(STEPS):
        step = i + 1

        # Update belief state
        tracker.update(node_probs, edge_probs)

        # Mark quarantine events
        if "quarantine(Lot_A)" in action:
            tracker.quarantine("Lot_A")
        if "quarantine(Lot_C)" in action:
            tracker.quarantine("Lot_C")

        # Print step header
        print(f"\n  Step {step}: {action}")

        if mode in ("terminal", "all"):
            tracker.render()

        if mode in ("save", "all"):
            frame_path = f"plots/belief_frames/step_{step:02d}.png"
            tracker.render_matplotlib(
                step=step,
                save_path=frame_path,
                action_text=action,
                live=False,
            )
            print(f"    -> Saved {frame_path}")

        if mode == "live":
            tracker.render_matplotlib(
                step=step,
                action_text=action,
                live=True,
            )
            time.sleep(0.8)

    # Save final composite frame
    if mode in ("save", "all"):
        final_path = "plots/belief_tracker_final.png"
        tracker.render_matplotlib(
            step=len(STEPS),
            save_path=final_path,
            action_text="finalize() -> Episode complete. 2 quarantined, 4 safe.",
            live=False,
        )
        print(f"\n  Final frame saved to {final_path}")

    # Print final state
    print("\n" + "=" * 62)
    print("  DEMO COMPLETE")
    print("=" * 62)

    state = tracker.get_state()
    print(f"\n  Final belief state at step {state['step']}:")
    print(f"    Quarantined: {list(state['quarantined'].keys())}")
    print(f"    Above threshold: {list(state['above_threshold'].keys())}")
    print(f"    Safe nodes confirmed: ", end="")
    safe = [n for n, p in state["node_probs"].items()
            if p < 0.3 and n not in state["quarantined"]]
    print(safe)

    if mode in ("save", "all"):
        print(f"\n  All frames saved to plots/belief_frames/")
        print(f"  Final composite: plots/belief_tracker_final.png")

    print()


if __name__ == "__main__":
    mode = "save"
    if "--live" in sys.argv:
        mode = "live"
    elif "--terminal" in sys.argv:
        mode = "terminal"
    elif "--all" in sys.argv:
        mode = "all"

    run_demo(mode)
