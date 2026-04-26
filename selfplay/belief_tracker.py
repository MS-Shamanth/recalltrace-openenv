"""Belief State Tracker for RecallTrace.

Tracks P(contaminated) per node and P(edge_exists) per hidden arc.
Updates after each agent tool call. Provides terminal and matplotlib
visualizations for live demo.

Usage:
    from selfplay.belief_tracker import BeliefStateTracker

    tracker = BeliefStateTracker(
        nodes=["Lot_A", "Warehouse_B", "Lot_C"],
        hidden_arcs=[("Lot_A", "Warehouse_B"), ("Warehouse_B", "Lot_C")],
    )
    tracker.update(
        node_probs={"Lot_A": 0.72, "Warehouse_B": 0.45, "Lot_C": 0.10},
        edge_probs={("Lot_A", "Warehouse_B"): 0.88},
    )
    tracker.render()                   # terminal version
    tracker.render_matplotlib(step=1)  # matplotlib version
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
# Use Agg backend when not in interactive mode (e.g. saving only)
# For live demo, caller should set the backend before importing this module
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _prob_to_color(p: float) -> str:
    """Map probability [0,1] to a hex color: gray(0) -> amber(0.5) -> red(1)."""
    if p < 0.5:
        # Gray to amber
        t = p / 0.5
        r = int(80 + t * (230 - 80))
        g = int(80 + t * (160 - 80))
        b = int(80 - t * 50)
        return f"#{r:02x}{g:02x}{b:02x}"
    else:
        # Amber to red
        t = (p - 0.5) / 0.5
        r = int(230 + t * (220 - 230))
        g = int(160 - t * 110)
        b = int(30 - t * 10)
        return f"#{r:02x}{g:02x}{b:02x}"


def _prob_to_terminal_color(p: float) -> str:
    """Return ANSI color code based on probability level."""
    if p >= 0.85:
        return "\033[91m"  # bright red — quarantine threshold
    elif p >= 0.5:
        return "\033[93m"  # yellow — suspicious
    elif p >= 0.3:
        return "\033[33m"  # dim yellow — weak signal
    else:
        return "\033[90m"  # gray — clean


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


# ---------------------------------------------------------------------------
# BeliefStateTracker
# ---------------------------------------------------------------------------

class BeliefStateTracker:
    """Tracks and visualizes belief state for RecallTrace episodes.

    Maintains P(contaminated) for each node and P(edge_exists) for each
    hidden arc. Updates incrementally after each agent tool call.

    Args:
        nodes: List of node names in the contamination propagation graph.
        hidden_arcs: List of (source, target) pairs for hidden edges.
        quarantine_threshold: P(contaminated) above which the trained
            agent should quarantine. Default 0.85.
    """

    def __init__(
        self,
        nodes: List[str],
        hidden_arcs: Optional[List[Tuple[str, str]]] = None,
        quarantine_threshold: float = 0.85,
    ):
        self.nodes = list(nodes)
        self.hidden_arcs = list(hidden_arcs or [])
        self.threshold = quarantine_threshold

        # Current belief state — start at uniform prior (0.1)
        self.node_probs: Dict[str, float] = {n: 0.10 for n in self.nodes}
        self.edge_probs: Dict[Tuple[str, str], float] = {
            arc: 0.50 for arc in self.hidden_arcs
        }

        # History for plotting belief evolution over time
        self.history: List[Dict[str, float]] = []
        self.step_count: int = 0

        # Track quarantine decisions
        self.quarantined: Dict[str, int] = {}  # node -> step quarantined

        # Matplotlib figure handle (reused for live updates)
        self._fig = None
        self._axes = None

    # ----- Core API -----

    def update(
        self,
        node_probs: Optional[Dict[str, float]] = None,
        edge_probs: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> None:
        """Update belief state with new probabilities from environment.

        Call this after each agent tool call. Only provided keys are
        updated; others remain at their previous value.

        Args:
            node_probs: {node_name: P(contaminated)} for updated nodes.
            edge_probs: {(src, tgt): P(edge_exists)} for updated arcs.
        """
        self.step_count += 1

        if node_probs:
            for node, prob in node_probs.items():
                self.node_probs[node] = max(0.0, min(1.0, prob))

        if edge_probs:
            for arc, prob in edge_probs.items():
                self.edge_probs[arc] = max(0.0, min(1.0, prob))

        # Save snapshot for history
        self.history.append(dict(self.node_probs))

    def quarantine(self, node: str) -> None:
        """Mark a node as quarantined at the current step."""
        self.quarantined[node] = self.step_count

    def get_state(self) -> dict:
        """Return the current belief state as a serializable dict.

        Returns:
            Dict with node_probs, edge_probs, step, quarantined, and
            any nodes above the quarantine threshold.
        """
        above_threshold = {
            n: p for n, p in self.node_probs.items()
            if p >= self.threshold
        }
        return {
            "step": self.step_count,
            "node_probs": dict(self.node_probs),
            "edge_probs": {f"{s}->{t}": p for (s, t), p in self.edge_probs.items()},
            "above_threshold": above_threshold,
            "quarantined": dict(self.quarantined),
        }

    def reset(self) -> None:
        """Reset all beliefs to priors for a new episode."""
        self.node_probs = {n: 0.10 for n in self.nodes}
        self.edge_probs = {arc: 0.50 for arc in self.hidden_arcs}
        self.history = []
        self.step_count = 0
        self.quarantined = {}

    # ----- Terminal rendering -----

    def render(self) -> None:
        """Print a clean terminal visualization of the current belief state.

        Shows a progress bar for each node's P(contaminated) and
        simple values for hidden arc probabilities.
        """
        bar_width = 30
        header = f"  Belief State - Step {self.step_count}"
        divider = "  " + "-" * 58

        lines = [
            "",
            divider,
            header,
            divider,
            "",
            f"  {'Node':<18s} {'P(contam)':>9s}  {'Bar':<{bar_width + 2}s}  Status",
            f"  {'----':<18s} {'---------':>9s}  {'---':<{bar_width + 2}s}  ------",
        ]

        for node in self.nodes:
            p = self.node_probs[node]
            filled = int(p * bar_width)
            bar = "#" * filled + "." * (bar_width - filled)
            color = _prob_to_terminal_color(p)

            # Status label
            if node in self.quarantined:
                status = f"\033[91mX QUARANTINED (step {self.quarantined[node]}){RESET}"
            elif p >= self.threshold:
                status = f"\033[91m! QUARANTINE NOW{RESET}"
            elif p >= 0.5:
                status = f"\033[93m? suspicious{RESET}"
            else:
                status = f"\033[90m- clean{RESET}"

            lines.append(
                f"  {node:<18s} {color}{p:>8.3f}{RESET}  "
                f"[{color}{bar}{RESET}]  {status}"
            )

        # Threshold indicator
        thresh_pos = int(self.threshold * bar_width) + 22
        lines.append(f"  {'':18s} {'':>9s}  {'':>{thresh_pos - 22}s}| {DIM}threshold={self.threshold}{RESET}")

        # Hidden arcs section (only if any exist)
        if self.edge_probs:
            lines.append("")
            lines.append(f"  Hidden Arcs:")
            for (src, tgt), p in self.edge_probs.items():
                color = "\033[92m" if p >= 0.7 else ("\033[93m" if p >= 0.4 else "\033[90m")
                confirmed = " (likely exists)" if p >= 0.7 else ""
                lines.append(f"    {src} -> {tgt}: {color}{p:.3f}{RESET}{confirmed}")

        lines.append(divider)
        lines.append("")

        print("\n".join(lines))

    # ----- Matplotlib rendering -----

    def render_matplotlib(
        self,
        step: Optional[int] = None,
        save_path: Optional[str] = None,
        action_text: Optional[str] = None,
        live: bool = True,
    ) -> None:
        """Render the belief state as a matplotlib horizontal bar chart.

        Designed for live demo — updates in place using plt.clf().

        Args:
            step: Step number to show in title. Defaults to self.step_count.
            save_path: If provided, save the figure to this path.
            action_text: Optional text describing the tool call just made.
            live: If True, use plt.pause() for animation. Set False for
                non-interactive (saving only).
        """
        if step is None:
            step = self.step_count

        # Create or reuse figure
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            self._fig, self._axes = plt.subplots(
                1, 2, figsize=(14, 5),
                gridspec_kw={"width_ratios": [3, 2], "wspace": 0.35},
            )
            self._fig.patch.set_facecolor("#0d1117")

        fig = self._fig
        ax_bars, ax_history = self._axes

        # ----- Left panel: horizontal bar chart -----
        ax_bars.clear()
        ax_bars.set_facecolor("#161b22")

        # Sort nodes by probability (highest at top)
        sorted_nodes = sorted(
            self.nodes,
            key=lambda n: self.node_probs[n],
        )
        probs = [self.node_probs[n] for n in sorted_nodes]
        y_pos = np.arange(len(sorted_nodes))

        # Color each bar based on probability
        colors = [_prob_to_color(p) for p in probs]

        bars = ax_bars.barh(
            y_pos, probs,
            height=0.6, color=colors,
            edgecolor="none", zorder=3,
        )

        # Background bars (full width)
        ax_bars.barh(
            y_pos, [1.0] * len(sorted_nodes),
            height=0.6, color="#21262d",
            edgecolor="none", zorder=1,
        )

        # Threshold line
        ax_bars.axvline(
            x=self.threshold, color="#f97583", linewidth=1.5,
            linestyle="--", zorder=4, alpha=0.8,
        )
        ax_bars.text(
            self.threshold + 0.02, len(sorted_nodes) - 0.3,
            f"quarantine\nthreshold",
            fontsize=8, color="#f97583", va="top",
            fontfamily="monospace", alpha=0.8,
        )

        # Labels
        ax_bars.set_yticks(y_pos)
        ax_bars.set_yticklabels(sorted_nodes, fontsize=10, fontfamily="monospace", color="#e6edf3")
        ax_bars.set_xlim(0, 1.05)
        ax_bars.set_xlabel("P(contaminated)", fontsize=10, color="#8b949e")

        # Probability values on bars
        for i, (node, p) in enumerate(zip(sorted_nodes, probs)):
            label_color = "#f97583" if p >= self.threshold else (
                "#fbbf24" if p >= 0.5 else "#8b949e"
            )
            # Add quarantine marker
            suffix = ""
            if node in self.quarantined:
                suffix = "  \u2716"
                label_color = "#f97583"

            ax_bars.text(
                p + 0.02, i, f"{p:.2f}{suffix}",
                va="center", fontsize=9, fontweight="bold",
                color=label_color, fontfamily="monospace",
            )

        # Title with step number
        title = f"Belief State \u2014 Step {step}"
        ax_bars.set_title(title, fontsize=14, fontweight="bold", color="#e6edf3", pad=12)

        # Action annotation
        if action_text:
            ax_bars.text(
                0.5, -0.12, f"\u25b6 {action_text}",
                transform=ax_bars.transAxes, fontsize=9,
                color="#58a6ff", ha="center", fontfamily="monospace",
                fontweight="bold",
            )

        # Style
        ax_bars.tick_params(colors="#8b949e", labelsize=9)
        ax_bars.spines["top"].set_visible(False)
        ax_bars.spines["right"].set_visible(False)
        ax_bars.spines["bottom"].set_color("#30363d")
        ax_bars.spines["left"].set_color("#30363d")

        # ----- Right panel: belief history sparklines -----
        ax_history.clear()
        ax_history.set_facecolor("#161b22")

        if len(self.history) > 1:
            steps_x = list(range(1, len(self.history) + 1))
            # Plot history for each node
            for node in self.nodes:
                node_hist = [h.get(node, 0) for h in self.history]
                p_current = node_hist[-1] if node_hist else 0
                color = _prob_to_color(p_current)
                alpha = 0.9 if p_current >= 0.3 else 0.35
                lw = 2.0 if p_current >= 0.5 else 1.0

                ax_history.plot(
                    steps_x, node_hist,
                    color=color, linewidth=lw, alpha=alpha,
                    marker="o", markersize=3, zorder=3,
                )

                # Label at the end of each line
                if p_current >= 0.25:
                    ax_history.text(
                        steps_x[-1] + 0.15, node_hist[-1],
                        node.split("_")[0],  # short name
                        fontsize=7.5, color=color, va="center",
                        fontfamily="monospace", fontweight="bold",
                        alpha=alpha,
                    )

            # Threshold line
            ax_history.axhline(
                y=self.threshold, color="#f97583", linewidth=1,
                linestyle="--", alpha=0.5, zorder=2,
            )

        ax_history.set_xlim(0.5, max(len(self.history) + 1.5, 3))
        ax_history.set_ylim(-0.02, 1.05)
        ax_history.set_xlabel("Tool Call Step", fontsize=10, color="#8b949e")
        ax_history.set_ylabel("P(contaminated)", fontsize=10, color="#8b949e")
        ax_history.set_title("Belief Evolution", fontsize=14, fontweight="bold", color="#e6edf3", pad=12)
        ax_history.tick_params(colors="#8b949e", labelsize=9)
        ax_history.spines["top"].set_visible(False)
        ax_history.spines["right"].set_visible(False)
        ax_history.spines["bottom"].set_color("#30363d")
        ax_history.spines["left"].set_color("#30363d")

        plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.15, wspace=0.35)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(
                save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )

        if live:
            plt.pause(0.05)
        else:
            plt.close(fig)
            self._fig = None
            self._axes = None

    def save_frame(self, save_path: str, step: Optional[int] = None) -> str:
        """Save the current belief state as a static image.

        Convenience wrapper around render_matplotlib for non-interactive use.
        Returns the save path.
        """
        self.render_matplotlib(step=step, save_path=save_path, live=False)
        return save_path
