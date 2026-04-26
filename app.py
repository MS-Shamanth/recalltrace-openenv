"""RecallTrace -- HuggingFace Gradio Application.

Premium UI with dark theme, animated elements, and live training.
"""

import gradio as gr
import sys
import os
import io
import time
from contextlib import redirect_stdout

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ── Custom CSS for premium dark theme ─────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* { font-family: 'Inter', sans-serif !important; }
code, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    background: #0d1117 !important;
}

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 16px !important;
    padding: 28px 32px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15) !important;
}

.main-header h1 {
    background: linear-gradient(135deg, #f97316, #ef4444, #ec4899) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin-bottom: 8px !important;
}

.main-header p {
    color: #8b949e !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
}

.stat-card {
    background: linear-gradient(135deg, #161b22, #1c2333) !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    padding: 20px !important;
    text-align: center !important;
}

.stat-card h3 {
    color: #8b949e !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin-bottom: 4px !important;
}

.stat-card .value {
    color: #e6edf3 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

.run-btn {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px 32px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(239, 68, 68, 0.3) !important;
}

.run-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(239, 68, 68, 0.5) !important;
}

.tab-nav button {
    color: #8b949e !important;
    border: none !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}

.tab-nav button.selected {
    color: #f97316 !important;
    border-bottom: 2px solid #f97316 !important;
}

.how-it-works {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    padding: 24px !important;
    margin-top: 16px !important;
}

.tool-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.85rem;
    color: #818cf8;
    font-family: 'JetBrains Mono', monospace;
}

footer { display: none !important; }
"""


def run_heuristic_training():
    """Run heuristic self-play training."""
    from selfplay.trainer import SelfPlayTrainer
    from selfplay.visualization import (
        show_training_curves, save_individual_plots,
        save_coevolution_plot, show_episode_comparison,
    )
    from selfplay.demo_replay import render_demo

    f = io.StringIO()
    with redirect_stdout(f):
        trainer = SelfPlayTrainer(num_nodes=10)
        stats = trainer.train(num_episodes=200)
        show_training_curves(stats, save_path="plots/selfplay_training.png")
        save_individual_plots(stats, plots_dir="plots")
        save_coevolution_plot(stats, save_path="plots/coevolution.png")
        early = stats[:30]
        late = stats[-30:]
        worst = min(early, key=lambda s: s["investigator_f1"])
        best = max(late, key=lambda s: s["investigator_f1"])
        show_episode_comparison(worst, best, save_path="plots/episode_comparison.png")
        render_demo(save_path="plots/before_after_demo.png")

    log = f.getvalue()
    e20 = stats[:20]
    l20 = stats[-20:]
    early_f1 = sum(s["investigator_f1"] for s in e20) / len(e20)
    late_f1 = sum(s["investigator_f1"] for s in l20) / len(l20)
    early_q = sum(s["num_quarantined"] for s in e20) / len(e20)
    late_q = sum(s["num_quarantined"] for s in l20) / len(l20)

    return (
        log,
        f"{early_f1:.3f} -> {late_f1:.3f}",
        f"{early_q:.1f} -> {late_q:.1f}",
        _img("plots/selfplay_training.png"),
        _img("plots/before_after_demo.png"),
        _img("plots/coevolution.png"),
        _img("plots/f1_curve.png"),
        _img("plots/belief_calibration.png"),
        _img("plots/episode_comparison.png"),
    )


def run_rl_training():
    """Run PyTorch RL training."""
    from selfplay.rl_trainer import RLSelfPlayTrainer
    from selfplay.visualization import (
        show_training_curves, save_individual_plots,
        save_coevolution_plot,
    )

    f = io.StringIO()
    with redirect_stdout(f):
        trainer = RLSelfPlayTrainer(num_nodes=10, lr=3e-4)
        stats = trainer.train(num_episodes=200)
        show_training_curves(stats, save_path="plots/rl_training.png")
        save_individual_plots(stats, plots_dir="plots/rl")
        save_coevolution_plot(stats, save_path="plots/rl_coevolution.png")

    log = f.getvalue()
    e20 = stats[:20]
    l20 = stats[-20:]
    early_f1 = sum(s["investigator_f1"] for s in e20) / len(e20)
    late_f1 = sum(s["investigator_f1"] for s in l20) / len(l20)
    early_q = sum(s["num_quarantined"] for s in e20) / len(e20)
    late_q = sum(s["num_quarantined"] for s in l20) / len(l20)
    avg_loss = sum(s.get("rl_metrics", {}).get("total_loss", 0) for s in l20) / len(l20)

    return (
        log,
        f"{early_f1:.3f} -> {late_f1:.3f}",
        f"{early_q:.1f} -> {late_q:.1f}",
        f"{avg_loss:.4f}",
        _img("plots/rl_training.png"),
        _img("plots/rl/f1_curve.png"),
        _img("plots/rl_coevolution.png"),
    )


def _img(path):
    return path if os.path.exists(path) else None


# ── Build the UI ─────────────────────────────────────────────────────
with gr.Blocks(
    title="RecallTrace: Causal Inference Benchmark",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.red,
        secondary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ),
) as demo:

    # ── Header ──
    gr.HTML("""
    <div class="main-header">
        <h1>RecallTrace</h1>
        <p style="font-size: 1.15rem; color: #c9d1d9; margin-bottom: 12px;">
            <strong>Causal Inference via Adversarial Self-Play</strong>
        </p>
        <p>
            A causal inference benchmark where an RL agent must identify which hidden intervention
            caused a contamination pattern in a partially observable graph, then quarantine precisely.
            Trained via adversarial self-play where the environment teaches itself to be harder.
        </p>
        <div style="display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap;">
            <span style="background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.3); border-radius: 20px; padding: 4px 14px; color: #f87171; font-size: 0.85rem;">
                Theme 3.1: World Modeling
            </span>
            <span style="background: rgba(249,115,22,0.15); border: 1px solid rgba(249,115,22,0.3); border-radius: 20px; padding: 4px 14px; color: #fb923c; font-size: 0.85rem;">
                Theme 4: Self-Play
            </span>
            <span style="background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.3); border-radius: 20px; padding: 4px 14px; color: #4ade80; font-size: 0.85rem;">
                7 Tools | 3 Intervention Types | CPU Only
            </span>
        </div>
    </div>
    """)

    # ── Training Tabs ──
    with gr.Tabs() as main_tabs:

        # ──────────── TAB 1: Heuristic Training ────────────
        with gr.TabItem("Heuristic Self-Play", id="heuristic"):
            gr.Markdown("### Adaptive Heuristic Agent (200 episodes, ~4s on CPU)")
            heuristic_btn = gr.Button(
                "Run Heuristic Training (200 episodes)",
                variant="primary", size="lg", elem_classes=["run-btn"],
            )

            with gr.Row():
                h_f1_display = gr.Textbox(label="F1 Score (Early -> Late)", interactive=False)
                h_q_display = gr.Textbox(label="Quarantined (Early -> Late)", interactive=False)

            with gr.Tabs():
                with gr.TabItem("Training Curves"):
                    h_training_plot = gr.Image(label="4-Panel Training Overview")
                with gr.TabItem("Before / After"):
                    h_before_after = gr.Image(label="Episode Behavior Comparison")
                with gr.TabItem("Co-Evolution"):
                    h_coevo = gr.Image(label="Both Agents Improving")
                with gr.TabItem("F1 Curve"):
                    h_f1_plot = gr.Image(label="F1 Score Over Episodes")
                with gr.TabItem("Belief Calibration"):
                    h_belief = gr.Image(label="P(contaminated) at Quarantine")
                with gr.TabItem("Episode Comparison"):
                    h_episode = gr.Image(label="Worst Early vs Best Late")
                with gr.TabItem("Training Log"):
                    h_log = gr.Textbox(label="Full Output", lines=20, max_lines=40)

            heuristic_btn.click(
                fn=run_heuristic_training,
                inputs=[],
                outputs=[h_log, h_f1_display, h_q_display,
                         h_training_plot, h_before_after, h_coevo,
                         h_f1_plot, h_belief, h_episode],
            )

        # ──────────── TAB 2: PyTorch RL Training ────────────
        with gr.TabItem("PyTorch RL Agent", id="rl"):
            gr.Markdown("### Neural Policy Network trained with REINFORCE (200 episodes)")
            rl_btn = gr.Button(
                "Train PyTorch RL Policy (200 episodes)",
                variant="primary", size="lg", elem_classes=["run-btn"],
            )

            with gr.Row():
                rl_f1_display = gr.Textbox(label="F1 Score (Early -> Late)", interactive=False)
                rl_q_display = gr.Textbox(label="Quarantined (Early -> Late)", interactive=False)
                rl_loss_display = gr.Textbox(label="Final Loss", interactive=False)

            with gr.Tabs():
                with gr.TabItem("RL Training Curves"):
                    rl_training_plot = gr.Image(label="4-Panel RL Training")
                with gr.TabItem("RL F1 Curve"):
                    rl_f1_plot = gr.Image(label="RL F1 Over Episodes")
                with gr.TabItem("RL Co-Evolution"):
                    rl_coevo = gr.Image(label="RL Agent vs Adversary")
                with gr.TabItem("Training Log"):
                    rl_log = gr.Textbox(label="Full Output", lines=20, max_lines=40)

            rl_btn.click(
                fn=run_rl_training,
                inputs=[],
                outputs=[rl_log, rl_f1_display, rl_q_display, rl_loss_display,
                         rl_training_plot, rl_f1_plot, rl_coevo],
            )

        # ──────────── TAB 3: Architecture ────────────
        with gr.TabItem("Architecture", id="arch"):
            gr.HTML("""
            <div style="padding: 24px;">
                <h2 style="color: #e6edf3; font-size: 1.5rem; margin-bottom: 20px;">
                    System Architecture
                </h2>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px;">
                    <div style="background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px;">
                        <h3 style="color: #f97316; font-size: 1.1rem; margin-bottom: 12px;">
                            Investigator (Agent 1)
                        </h3>
                        <p style="color: #8b949e; font-size: 0.9rem; line-height: 1.6;">
                            Uses 7 tools to investigate. Maintains belief state P(contaminated) per node.
                            Must identify the hidden intervention type before quarantining.
                        </p>
                        <div style="margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px;">
                            <span class="tool-badge">inspect_node</span>
                            <span class="tool-badge">trace_lot</span>
                            <span class="tool-badge">cross_reference</span>
                            <span class="tool-badge">request_lab_test</span>
                            <span class="tool-badge">quarantine</span>
                            <span class="tool-badge">notify</span>
                            <span class="tool-badge">finalize</span>
                        </div>
                    </div>

                    <div style="background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px;">
                        <h3 style="color: #ef4444; font-size: 1.1rem; margin-bottom: 12px;">
                            Adversary (Agent 2)
                        </h3>
                        <p style="color: #8b949e; font-size: 0.9rem; line-height: 1.6;">
                            Chooses which intervention to apply and where, maximizing investigator failure.
                            18-cell score table (type x region x density) adapts via EMA.
                        </p>
                        <div style="margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px;">
                            <span style="background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.3); border-radius: 6px; padding: 2px 8px; color: #f87171; font-size: 0.85rem;">lot_relabel</span>
                            <span style="background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.3); border-radius: 6px; padding: 2px 8px; color: #f87171; font-size: 0.85rem;">mixing_event</span>
                            <span style="background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.3); border-radius: 6px; padding: 2px 8px; color: #f87171; font-size: 0.85rem;">record_deletion</span>
                        </div>
                    </div>
                </div>

                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
                    <h3 style="color: #22c55e; font-size: 1.1rem; margin-bottom: 12px;">
                        Composable Reward Function (Ungameable)
                    </h3>
                    <table style="width: 100%; color: #c9d1d9; font-size: 0.9rem; border-collapse: collapse;">
                        <tr style="border-bottom: 1px solid #30363d;">
                            <td style="padding: 8px; color: #4ade80; font-weight: 600;">Recall</td>
                            <td style="padding: 8px;">+2.0 x (unsafe caught / total unsafe)</td>
                            <td style="padding: 8px; color: #8b949e;">Forces finding contamination</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #30363d;">
                            <td style="padding: 8px; color: #ef4444; font-weight: 600;">Precision</td>
                            <td style="padding: 8px;">-1.5 x (safe blocked / total safe)</td>
                            <td style="padding: 8px; color: #8b949e;">Prevents spray & pray</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #30363d;">
                            <td style="padding: 8px; color: #a78bfa; font-weight: 600;">Calibration</td>
                            <td style="padding: 8px;">+0.3 x (quarantined / total unsafe) if P > 0.8</td>
                            <td style="padding: 8px; color: #8b949e;">Rewards confident decisions</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; color: #f59e0b; font-weight: 600;">Efficiency</td>
                            <td style="padding: 8px;">-0.05 per step + speed bonus</td>
                            <td style="padding: 8px; color: #8b949e;">Encourages fast investigation</td>
                        </tr>
                    </table>
                </div>

                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px;">
                    <h3 style="color: #38bdf8; font-size: 1.1rem; margin-bottom: 12px;">
                        PyTorch RL Architecture
                    </h3>
                    <pre style="color: #c9d1d9; font-size: 0.85rem; line-height: 1.5; font-family: 'JetBrains Mono', monospace; background: #0d1117; padding: 16px; border-radius: 8px; overflow-x: auto;">
StateEncoder (112-dim)
  |-- Per-node features (12 nodes x 8 features)
  |    inventory, inspected, quarantined, evidence_strength, ...
  |-- Global features (16-dim)
       steps, budget, coverage, urgency, evidence_counts, ...

PolicyNetwork (MLP)
  |-- SharedBackbone: Linear(112,128) -> LN -> ReLU -> Linear(128,64) -> LN -> ReLU
  |-- ActionHead:     Linear(64, 7)   -> Categorical sampling
  |-- NodeHead:       Linear(64, 12)  -> Categorical sampling
  |-- ValueHead:      Linear(64, 1)   -> Baseline for variance reduction

Training: REINFORCE + learned baseline + entropy regularization
  |-- gamma=0.99, entropy_coef=0.02, lr=3e-4
  |-- Gradient clipping: max_norm=0.5
                    </pre>
                </div>
            </div>
            """)

    # ── Footer ──
    gr.HTML("""
    <div style="text-align: center; padding: 20px; color: #484f58; font-size: 0.85rem; margin-top: 16px;">
        RecallTrace - Meta PyTorch OpenEnv Hackathon | CPU Only | No External APIs
    </div>
    """)


if __name__ == "__main__":
    demo.launch()
