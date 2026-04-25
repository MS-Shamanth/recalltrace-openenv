"""Episode replay visualizer for RecallTrace demo.

Side-by-side graph visualization: untrained (Episode 5) vs trained (Episode 195).
Shows the agent evolving from spray-and-pray to precision quarantining.

This is the storytelling money shot for the hackathon demo.

Usage:
    python -m selfplay.demo_replay
    # or imported:
    from selfplay.demo_replay import render_demo
    render_demo()
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Graph structure (shared between both panels)
# ---------------------------------------------------------------------------

NODES = [
    "Lot_A",          # contaminated (hidden)
    "Warehouse_B",    # safe
    "Lot_C",          # contaminated (hidden)
    "Distributor_D",  # safe
    "Retailer_E",     # safe
    "Lot_F",          # safe
    "Supplier_G",     # safe
    "Hub_H",          # safe
]

EDGES = [
    ("Supplier_G", "Warehouse_B"),
    ("Supplier_G", "Lot_A"),
    ("Warehouse_B", "Distributor_D"),
    ("Warehouse_B", "Hub_H"),
    ("Lot_A", "Distributor_D"),
    ("Lot_A", "Lot_C"),
    ("Distributor_D", "Retailer_E"),
    ("Distributor_D", "Lot_F"),
    ("Hub_H", "Retailer_E"),
    ("Lot_C", "Lot_F"),
]

CONTAMINATED = {"Lot_A", "Lot_C"}

# ---------------------------------------------------------------------------
# Episode data
# ---------------------------------------------------------------------------

EARLY_EPISODE = {
    "episode": 5,
    "title": "Episode 5 (untrained agent)",
    "visited": ["Supplier_G", "Warehouse_B", "Lot_A", "Distributor_D",
                "Retailer_E", "Lot_F", "Lot_C"],
    "quarantined": ["Lot_A", "Warehouse_B", "Distributor_D",
                     "Retailer_E", "Lot_F", "Lot_C"],
    "visit_order": ["Supplier_G", "Warehouse_B", "Lot_A", "Distributor_D",
                     "Retailer_E", "Lot_F", "Lot_C"],
    "belief_at_quarantine": {
        "Lot_A": 0.53, "Warehouse_B": 0.48, "Distributor_D": 0.44,
        "Retailer_E": 0.39, "Lot_F": 0.41, "Lot_C": 0.51,
    },
    "f1": 0.28,
    "steps": 9,
    "avg_belief": 0.51,
    "intervention_identified": False,
    "intervention_type": None,
}

LATE_EPISODE = {
    "episode": 195,
    "title": "Episode 195 (trained agent)",
    "visited": ["Supplier_G", "Lot_A", "Lot_C", "Distributor_D"],
    "quarantined": ["Lot_A", "Lot_C"],
    "visit_order": ["Supplier_G", "Lot_A", "Lot_C", "Distributor_D"],
    "belief_at_quarantine": {
        "Lot_A": 0.89, "Lot_C": 0.87,
    },
    "f1": 0.81,
    "steps": 4,
    "avg_belief": 0.88,
    "intervention_identified": True,
    "intervention_type": "mixing event",
}


# ---------------------------------------------------------------------------
# Color palette — dark theme for presentation
# ---------------------------------------------------------------------------

BG_DARK       = "#0d1117"
BG_PANEL      = "#161b22"
EDGE_COLOR    = "#30363d"
TEXT_COLOR    = "#e6edf3"
DIM_COLOR     = "#8b949e"
NODE_DEFAULT  = "#21262d"
NODE_STROKE   = "#444c56"
VISITED_RING  = "#f0c040"       # yellow
QUARANTINE_FILL = "#da3633"     # red
CORRECT_GREEN = "#2ea043"       # green
CONTAM_ORANGE = "#d29922"       # orange dashed
ARROW_BLUE    = "#58a6ff"       # path arrows
BELIEF_HIGH   = "#7ee787"       # high confidence text
BELIEF_LOW    = "#f97583"       # low confidence text
STATS_BG      = "#1c2128"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _build_graph() -> Tuple[nx.DiGraph, Dict[str, np.ndarray]]:
    """Build the supply-chain graph and compute a stable layout."""
    G = nx.DiGraph()
    G.add_nodes_from(NODES)
    G.add_edges_from(EDGES)

    # Use spring layout with a fixed seed for reproducibility
    pos = nx.spring_layout(G, seed=42, k=2.2, iterations=80)

    # Normalize positions to [0.1, 0.9] range
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    for node in pos:
        pos[node] = np.array([
            0.1 + 0.8 * (pos[node][0] - x_min) / (x_max - x_min + 1e-9),
            0.12 + 0.7 * (pos[node][1] - y_min) / (y_max - y_min + 1e-9),
        ])

    return G, pos


def _draw_episode_panel(
    ax: plt.Axes,
    G: nx.DiGraph,
    pos: Dict[str, np.ndarray],
    episode: Dict[str, Any],
    show_correct_green: bool = False,
    show_path_arrows: bool = False,
    show_stop_annotation: bool = False,
) -> None:
    """Draw a single episode panel with graph, highlights, and stats."""

    ax.set_facecolor(BG_PANEL)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.08, 1.02)
    ax.axis("off")

    visited = set(episode["visited"])
    quarantined = set(episode["quarantined"])
    beliefs = episode["belief_at_quarantine"]

    # --- Draw edges ---
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=EDGE_COLOR,
                lw=1.0,
                alpha=0.5,
                connectionstyle="arc3,rad=0.08",
                shrinkA=18, shrinkB=18,
            ),
        )

    # --- Draw path arrows (numbered) for late panel ---
    if show_path_arrows and episode.get("visit_order"):
        visit_order = episode["visit_order"]
        for i in range(len(visit_order) - 1):
            u, v = visit_order[i], visit_order[i + 1]
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            # Compute midpoint for number label
            mx = (x0 + x1) / 2
            my = (y0 + y1) / 2

            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=ARROW_BLUE,
                    lw=2.5,
                    alpha=0.85,
                    connectionstyle="arc3,rad=0.12",
                    shrinkA=20, shrinkB=20,
                ),
                zorder=5,
            )
            # Step number on the path
            ax.text(
                mx, my + 0.025, str(i + 1),
                fontsize=9, fontweight="bold",
                color=ARROW_BLUE, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=BG_PANEL,
                          edgecolor=ARROW_BLUE, alpha=0.9, linewidth=1),
                zorder=6,
            )

    # --- Draw nodes ---
    node_size = 0.045
    for node in NODES:
        x, y = pos[node]
        is_visited = node in visited
        is_quarantined = node in quarantined
        is_contaminated = node in CONTAMINATED
        is_correct_leave = show_correct_green and not is_quarantined and not is_contaminated

        # Determine fill color
        if is_quarantined:
            fill = QUARANTINE_FILL
            stroke = "#ff6b6b"
            stroke_width = 3.0
        elif is_correct_leave and is_visited:
            fill = "#1a3a2a"
            stroke = CORRECT_GREEN
            stroke_width = 2.5
        elif is_visited:
            fill = "#2d2a1a"
            stroke = VISITED_RING
            stroke_width = 2.5
        else:
            fill = NODE_DEFAULT
            stroke = NODE_STROKE
            stroke_width = 1.5

        # Draw node circle
        circle = plt.Circle(
            (x, y), node_size,
            facecolor=fill, edgecolor=stroke,
            linewidth=stroke_width, zorder=3,
        )
        ax.add_patch(circle)

        # Contamination indicator (orange dashed ring, only shown post-finalize)
        if is_contaminated:
            contam_ring = plt.Circle(
                (x, y), node_size + 0.012,
                facecolor="none", edgecolor=CONTAM_ORANGE,
                linewidth=2.0, linestyle="--", zorder=2, alpha=0.7,
            )
            ax.add_patch(contam_ring)

        # Quarantine X marker
        if is_quarantined:
            ax.text(
                x, y, "\u2716", fontsize=16, fontweight="bold",
                color="white", ha="center", va="center", zorder=4,
            )

        # Correct-leave checkmark (green, late panel only)
        if is_correct_leave and is_visited:
            ax.text(
                x, y, "\u2714", fontsize=15, fontweight="bold",
                color=CORRECT_GREEN, ha="center", va="center", zorder=4,
            )

        # Node label
        short_name = node.replace("_", "\n")
        label_y = y - node_size - 0.03
        ax.text(
            x, label_y, short_name,
            fontsize=7.5, color=TEXT_COLOR, ha="center", va="top",
            fontweight="bold", zorder=4,
            fontfamily="monospace",
        )

        # Belief confidence annotation (for quarantined nodes)
        if is_quarantined and node in beliefs:
            belief = beliefs[node]
            b_color = BELIEF_HIGH if belief >= 0.75 else BELIEF_LOW
            ax.text(
                x + node_size + 0.015, y + 0.015,
                f"P={belief:.2f}",
                fontsize=8.5, fontweight="bold", color=b_color,
                ha="left", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", facecolor=BG_PANEL,
                          edgecolor=b_color, alpha=0.85, linewidth=0.8),
            )

    # --- Title bar ---
    is_late = episode["episode"] > 100
    title_color = CORRECT_GREEN if is_late else QUARANTINE_FILL
    title_bg = "#1a3a2a" if is_late else "#3a1a1a"

    title_rect = FancyBboxPatch(
        (0.02, 0.90), 0.96, 0.09,
        boxstyle="round,pad=0.02",
        facecolor=title_bg, edgecolor=title_color,
        linewidth=2.5, zorder=6, alpha=0.95,
    )
    ax.add_patch(title_rect)
    ax.text(
        0.5, 0.945, episode["title"],
        fontsize=14, fontweight="bold", color=TEXT_COLOR,
        ha="center", va="center", zorder=7,
    )

    # --- Stop annotation (late panel) ---
    if show_stop_annotation:
        ax.text(
            0.98, 0.845,
            'Agent stopped when\nP(contaminated) > 0.85',
            fontsize=8, color=BELIEF_HIGH, ha="right", va="top",
            style="italic", alpha=0.9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0d2818",
                      edgecolor=BELIEF_HIGH, alpha=0.6, linewidth=0.8),
            zorder=7,
        )

    # --- Stats box at bottom ---
    stats_rect = FancyBboxPatch(
        (0.02, -0.06), 0.96, 0.075,
        boxstyle="round,pad=0.015",
        facecolor=STATS_BG, edgecolor=EDGE_COLOR,
        linewidth=1.5, zorder=6, alpha=0.95,
    )
    ax.add_patch(stats_rect)

    f1_color = CORRECT_GREEN if episode["f1"] >= 0.7 else (
        VISITED_RING if episode["f1"] >= 0.4 else QUARANTINE_FILL
    )

    interv_text = "NO"
    if episode["intervention_identified"]:
        interv_text = f"YES ({episode['intervention_type']})"

    # Draw F1 score prominently on the left
    ax.text(
        0.06, -0.022, f"F1 = {episode['f1']:.2f}",
        fontsize=11, color=f1_color, ha="left", va="center",
        fontweight="bold", fontfamily="monospace", zorder=8,
    )

    # Draw remaining stats on the right
    rest_line = (
        f"Quarantined={len(episode['quarantined'])}  |  "
        f"Steps={episode['steps']}  |  "
        f"Avg belief={episode['avg_belief']:.2f}  |  "
        f"Intervention: {interv_text}"
    )
    ax.text(
        0.95, -0.022, rest_line,
        fontsize=8.5, color=TEXT_COLOR, ha="right", va="center",
        fontweight="bold", fontfamily="monospace", zorder=7,
    )


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

def _draw_legend(fig: plt.Figure) -> None:
    """Add a horizontal legend below the panels."""
    legend_items = [
        (VISITED_RING, "Visited"),
        (QUARANTINE_FILL, "Quarantined (X)"),
        (CORRECT_GREEN, "Correctly left alone"),
        (CONTAM_ORANGE, "Hidden contamination"),
        (ARROW_BLUE, "Agent path"),
    ]

    total = len(legend_items)
    start_x = 0.14
    spacing = 0.155

    for i, (color, label) in enumerate(legend_items):
        x = start_x + i * spacing
        fig.patches.append(
            mpatches.Circle(
                (x, 0.065), 0.008,
                facecolor=color, edgecolor=color,
                transform=fig.transFigure, zorder=10,
            )
        )
        fig.text(
            x + 0.015, 0.065, label,
            fontsize=9, color=TEXT_COLOR, va="center",
            fontweight="bold",
        )


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_demo(
    save_path: str = "plots/before_after_demo.png",
    show: bool = False,
    dpi: int = 200,
) -> str:
    """Render the side-by-side episode replay visualization.

    Returns the save path.
    """
    G, pos = _build_graph()

    fig, (ax_early, ax_late) = plt.subplots(
        1, 2, figsize=(20, 10),
        gridspec_kw={"wspace": 0.06},
    )
    fig.patch.set_facecolor(BG_DARK)

    # --- Draw early episode (left) ---
    _draw_episode_panel(
        ax_early, G, pos, EARLY_EPISODE,
        show_correct_green=False,
        show_path_arrows=False,
        show_stop_annotation=False,
    )

    # --- Draw late episode (right) ---
    _draw_episode_panel(
        ax_late, G, pos, LATE_EPISODE,
        show_correct_green=True,
        show_path_arrows=True,
        show_stop_annotation=True,
    )

    # --- Central arrow between panels ---
    fig.text(
        0.5, 0.50, "\u279c",
        fontsize=42, color=DIM_COLOR, ha="center", va="center",
        fontweight="bold",
    )
    fig.text(
        0.5, 0.44, "200 episodes\nof self-play",
        fontsize=10, color=DIM_COLOR, ha="center", va="top",
        style="italic",
    )

    # --- Main title ---
    fig.text(
        0.5, 0.97,
        "RecallTrace \u2014 the agent learns to reason, not just react",
        fontsize=20, fontweight="bold", color=TEXT_COLOR,
        ha="center", va="top",
    )

    # --- Subtitle ---
    fig.text(
        0.5, 0.935,
        "Adversarial self-play training: Investigator vs Adversary co-evolution",
        fontsize=12, color=DIM_COLOR, ha="center", va="top",
    )

    # --- Bottom tagline ---
    fig.text(
        0.5, 0.025,
        "Self-play training: 200 episodes, ~4 minutes, CPU only",
        fontsize=11, color=DIM_COLOR, ha="center", va="center",
        fontfamily="monospace", style="italic",
    )

    # --- Legend ---
    _draw_legend(fig)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.10)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved demo replay to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    render_demo(show=False)
