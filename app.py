import gradio as gr
import sys
import os

# Add the current directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from run_selfplay import run_demo

def run_simulation():
    # Capture the print output
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        run_demo()
    output_text = f.getvalue()
    
    # Return the text and the generated plots
    return (
        output_text,
        "plots/selfplay_training.png",
        "plots/before_after_demo.png"
    )

with gr.Blocks(title="RecallTrace: Causal Inference Demo") as demo:
    gr.Markdown("# 🚨 RecallTrace: Causal Inference via Adversarial Self-Play")
    gr.Markdown("An RL agent that doesn't just learn to detect contamination — it learns to infer the hidden causal intervention behind it. Trained via adversarial self-play.")
    
    with gr.Row():
        run_btn = gr.Button("🚀 Run Self-Play Training (200 episodes in ~1s)", variant="primary")
    
    with gr.Row():
        with gr.Column(scale=1):
            output_log = gr.Textbox(label="Training Log", lines=20)
        with gr.Column(scale=2):
            training_plot = gr.Image(label="Training Curves")
            before_after_plot = gr.Image(label="Before vs After Behavior")

    run_btn.click(
        fn=run_simulation,
        inputs=[],
        outputs=[output_log, training_plot, before_after_plot]
    )

if __name__ == "__main__":
    demo.launch()
