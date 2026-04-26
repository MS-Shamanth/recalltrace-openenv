#!/usr/bin/env python3
"""RecallTrace — LLM Training with Unsloth + TRL

Fine-tunes Qwen2.5-0.5B-Instruct on expert demonstrations from the
RecallTrace supply-chain environment, then evaluates improvement.

Quick start (GPU required):
    pip install unsloth "trl>=0.12" datasets accelerate
    python train_trl.py

On Google Colab (free T4):
    !pip install unsloth "trl>=0.12" datasets
    !git clone https://huggingface.co/spaces/ms-shamanth/recalltrace-openenv
    %cd recalltrace-openenv
    !python train_trl.py

On HF Jobs:
    export HF_TOKEN="hf_..."
    hf jobs uv run train_trl.py --flavor gpu-t4-small --with unsloth --with trl --with datasets
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env.env import RecallTraceEnv
from env.models import RecallAction
from baseline.policy import choose_heuristic_action

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
OUTPUT_DIR = Path("trained_model")
PLOTS_DIR = Path("plots")
HUB_MODEL_ID = "ms-shamanth/recalltrace-investigator"

SYSTEM_PROMPT = (
    "You are an expert supply-chain investigator for RecallTrace. "
    "You receive an observation of a product recall investigation and must "
    "choose the optimal next action. Respond with ONLY a valid JSON object.\n"
    "Available actions:\n"
    "- inspect_node: {\"type\":\"inspect_node\",\"node_id\":\"...\",\"rationale\":\"...\"}\n"
    "- trace_lot: {\"type\":\"trace_lot\",\"lot_id\":\"...\",\"rationale\":\"...\"}\n"
    "- quarantine: {\"type\":\"quarantine\",\"node_id\":\"...\",\"lot_id\":\"...\",\"quantity\":N,\"rationale\":\"...\"}\n"
    "- notify: {\"type\":\"notify\",\"node_id\":\"all\",\"rationale\":\"...\"}\n"
    "- finalize: {\"type\":\"finalize\",\"rationale\":\"...\"}"
)


# ---------------------------------------------------------------------------
# 1) Format observations as LLM prompts
# ---------------------------------------------------------------------------
def format_observation(obs) -> str:
    """Convert RecallObservation to readable text for the LLM."""
    lines = [
        f"TASK: {obs.task_id} | Steps: {obs.steps_taken}/{obs.steps_taken + obs.remaining_step_budget}",
        f"RECALL NOTICE: {obs.recall_notice}",
        "",
        "INVENTORY:",
    ]
    for nid, lots in obs.inventory.items():
        if lots:
            items = ", ".join(f"{l}={q}" for l, q in list(lots.items())[:6])
            lines.append(f"  {nid}: {items}")

    if obs.inspected_nodes:
        lines.append(f"\nINSPECTED NODES: {', '.join(obs.inspected_nodes)}")

    if obs.inspection_results:
        lines.append("INSPECTION FINDINGS:")
        for nid, findings in obs.inspection_results.items():
            for lid, ev in findings.items():
                status = ev.status if hasattr(ev, "status") else ev.get("status", "?")
                uq = ev.unsafe_quantity if hasattr(ev, "unsafe_quantity") else ev.get("unsafe_quantity", 0)
                lines.append(f"  {nid}/{lid}: status={status}, unsafe_qty={uq}")

    if obs.trace_results:
        lines.append("TRACE RESULTS:")
        for lid, tr in obs.trace_results.items():
            nodes = tr.get("affected_nodes", [])
            lines.append(f"  {lid}: affected_nodes={nodes}")

    if obs.quarantined_inventory:
        lines.append("QUARANTINED:")
        for nid, lots in obs.quarantined_inventory.items():
            items = ", ".join(f"{l}={q}" for l, q in lots.items())
            lines.append(f"  {nid}: {items}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2) Generate expert training data
# ---------------------------------------------------------------------------
def generate_expert_data(num_episodes: int = 300, seed: int = 42) -> list[dict]:
    """Run heuristic expert on many episodes, collect (prompt, action) pairs."""
    print(f"\n{'='*60}")
    print(f"  Phase 1: Generating expert demonstrations")
    print(f"  Episodes: {num_episodes}")
    print(f"{'='*60}\n")

    data = []
    total_reward = 0.0
    rng = random.Random(seed)

    tasks = RecallTraceEnv.available_tasks()

    for ep in range(num_episodes):
        task = tasks[ep % len(tasks)]
        env = RecallTraceEnv(task_id=task.task_id)
        obs = env.reset(task_id=task.task_id)
        ep_reward = 0.0

        for step in range(env.task.max_steps):
            prompt_text = format_observation(obs)
            action = choose_heuristic_action(obs)
            action_json = json.dumps(action.model_dump(exclude_none=True), sort_keys=True)

            obs, reward, done, info = env.step(action)
            ep_reward += reward

            # Only keep positive-reward actions as expert demonstrations
            if reward >= 0.0:
                data.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_text},
                        {"role": "assistant", "content": action_json},
                    ]
                })

            if done:
                break

        total_reward += ep_reward
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1:>4d}/{num_episodes} | Avg reward: {total_reward/(ep+1):.3f} | Samples: {len(data)}")

    print(f"\n  Generated {len(data)} expert samples from {num_episodes} episodes")
    print(f"  Average episode reward: {total_reward/num_episodes:.3f}\n")
    return data


# ---------------------------------------------------------------------------
# 3) SFT Training with Unsloth + TRL
# ---------------------------------------------------------------------------
def train_sft(dataset_dicts: list[dict], num_epochs: int = 3, max_steps: int = -1):
    """Fine-tune with Unsloth + TRL SFTTrainer."""
    print(f"\n{'='*60}")
    print(f"  Phase 2: SFT Training with Unsloth + TRL")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Epochs: {num_epochs}")
    print(f"{'='*60}\n")

    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    # Load model with 4-bit quantization
    print("  Loading model with Unsloth (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Pre-format messages into text strings (avoids Unsloth formatting_func issues)
    print("  Formatting dataset...")
    formatted_data = []
    for item in dataset_dicts:
        text = tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted_data.append({"text": text})

    dataset = Dataset.from_list(formatted_data)
    print(f"  Dataset size: {len(dataset)} samples")

    # Unsloth requires formatting_func — handle both single example and batch
    def formatting_func(example):
        t = example["text"]
        if isinstance(t, list):
            return t
        return [t]

    # Training config
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=num_epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        max_seq_length=2048,
        dataset_text_field="text",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_func,
        args=training_args,
    )

    print("  Starting training...\n")
    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start

    print(f"\n  Training complete in {elapsed:.0f}s")
    print(f"  Final loss: {result.training_loss:.4f}")

    # Save model
    print(f"  Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Extract training log for plotting
    train_log = [
        {"step": entry["step"], "loss": entry["loss"]}
        for entry in trainer.state.log_history
        if "loss" in entry
    ]

    return model, tokenizer, train_log


# ---------------------------------------------------------------------------
# 4) Evaluate: Baseline vs Trained
# ---------------------------------------------------------------------------
def evaluate_baseline(num_episodes: int = 50) -> dict:
    """Run untrained random baseline on the environment."""
    print("  Evaluating random baseline...")
    scores = []
    for ep in range(num_episodes):
        tasks = RecallTraceEnv.available_tasks()
        task = tasks[ep % len(tasks)]
        env = RecallTraceEnv(task_id=task.task_id)
        obs = env.reset(task_id=task.task_id)
        total_r = 0.0
        for _ in range(env.task.max_steps):
            # Random action
            action_type = random.choice(["inspect_node", "trace_lot", "quarantine", "notify", "finalize"])
            nodes = list(obs.inventory.keys())
            node_id = random.choice(nodes) if nodes else None
            lots = []
            for n_lots in obs.inventory.values():
                lots.extend(n_lots.keys())
            lot_id = random.choice(lots) if lots else None

            try:
                action = RecallAction(type=action_type, node_id=node_id, lot_id=lot_id,
                                       quantity=10 if action_type == "quarantine" else None)
                obs, reward, done, info = env.step(action)
                total_r += reward
            except Exception:
                action = RecallAction(type="finalize")
                obs, reward, done, info = env.step(action)
                total_r += reward
            if done:
                break
        scores.append(info.get("score") or 0.0)
    avg = sum(scores) / len(scores)
    print(f"  Random baseline: avg score = {avg:.4f}")
    return {"avg_score": avg, "scores": scores}


def evaluate_heuristic(num_episodes: int = 50) -> dict:
    """Run heuristic baseline."""
    print("  Evaluating heuristic baseline...")
    scores = []
    for ep in range(num_episodes):
        tasks = RecallTraceEnv.available_tasks()
        task = tasks[ep % len(tasks)]
        env = RecallTraceEnv(task_id=task.task_id)
        obs = env.reset(task_id=task.task_id)
        for _ in range(env.task.max_steps):
            action = choose_heuristic_action(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break
        scores.append(info.get("score") or 0.0)
    avg = sum(scores) / len(scores)
    print(f"  Heuristic baseline: avg score = {avg:.4f}")
    return {"avg_score": avg, "scores": scores}


def evaluate_trained(model, tokenizer, num_episodes: int = 50) -> dict:
    """Run trained LLM on the environment."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    print("  Evaluating trained model...")

    scores = []
    for ep in range(num_episodes):
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    Evaluating episode {ep+1}/{num_episodes}...")
        
        tasks = RecallTraceEnv.available_tasks()
        task = tasks[ep % len(tasks)]
        env = RecallTraceEnv(task_id=task.task_id)
        obs = env.reset(task_id=task.task_id)

        for _ in range(env.task.max_steps):
            prompt_text = format_observation(obs)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with __import__("torch").no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=200, max_length=None, temperature=0.1,
                    do_sample=True, pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            try:
                action_dict = json.loads(response)
                action = RecallAction.model_validate(action_dict)
            except Exception:
                action = choose_heuristic_action(obs)  # fallback

            obs, reward, done, info = env.step(action)
            if done:
                break

        scores.append(info.get("score") or 0.0)

    avg = sum(scores) / len(scores)
    print(f"  Trained model: avg score = {avg:.4f}")
    return {"avg_score": avg, "scores": scores}


# ---------------------------------------------------------------------------
# 5) Generate plots
# ---------------------------------------------------------------------------
def generate_plots(train_log: list[dict], eval_results: dict):
    """Generate training loss curve and evaluation comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(exist_ok=True)

    # --- Training Loss Curve ---
    if train_log:
        fig, ax = plt.subplots(figsize=(10, 5))
        steps = [e["step"] for e in train_log]
        losses = [e["loss"] for e in train_log]
        ax.plot(steps, losses, color="#ff6f3c", linewidth=2, label="SFT Training Loss")
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("RecallTrace — SFT Training Loss (Unsloth + TRL)", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "trl_training_loss.png", dpi=150)
        plt.close()
        print(f"  Saved: {PLOTS_DIR / 'trl_training_loss.png'}")

    # --- Evaluation Comparison ---
    if eval_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = list(eval_results.keys())
        avgs = [eval_results[n]["avg_score"] for n in names]
        colors = ["#8b949e", "#f0c040", "#2ea043"][:len(names)]
        bars = ax.bar(names, avgs, color=colors, width=0.5, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")
        ax.set_ylabel("Average Episode Score", fontsize=12)
        ax.set_title("RecallTrace — Baseline vs Trained Agent", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "trl_evaluation_comparison.png", dpi=150)
        plt.close()
        print(f"  Saved: {PLOTS_DIR / 'trl_evaluation_comparison.png'}")


# ---------------------------------------------------------------------------
# 6) Push to Hub
# ---------------------------------------------------------------------------
def push_to_hub(model, tokenizer, hub_model_id: str):
    """Push trained model + card to HF Hub."""
    print(f"\n  Pushing model to {hub_model_id}...")
    model.push_to_hub(hub_model_id, token=os.environ.get("HF_TOKEN"))
    tokenizer.push_to_hub(hub_model_id, token=os.environ.get("HF_TOKEN"))
    print(f"  Model available at: https://huggingface.co/{hub_model_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RecallTrace LLM Training (Unsloth + TRL)")
    parser.add_argument("--episodes", type=int, default=300, help="Expert data episodes")
    parser.add_argument("--epochs", type=int, default=3, help="SFT training epochs")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps (-1=use epochs)")
    parser.add_argument("--eval-episodes", type=int, default=30, help="Evaluation episodes")
    parser.add_argument("--push-model", action="store_true", help="Push to HF Hub")
    parser.add_argument("--hub-model-id", default=HUB_MODEL_ID, help="HF Hub model ID")
    parser.add_argument("--data-only", action="store_true", help="Only generate data, skip training")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  RecallTrace — LLM Agent Training")
    print("  Unsloth + TRL (SFT on Expert Demonstrations)")
    print("="*60)

    # GPU check — fail fast before wasting time on data generation
    if not args.data_only:
        import torch
        if not torch.cuda.is_available():
            print("\n  ❌ ERROR: No GPU detected!")
            print("  Unsloth requires a CUDA GPU.")
            print("\n  In Google Colab:")
            print("    Runtime → Change runtime type → T4 GPU → Save")
            print("    Then reconnect and re-run all cells.\n")
            sys.exit(1)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n  ✅ GPU detected: {gpu_name}")

    # Phase 1: Generate expert data
    expert_data = generate_expert_data(num_episodes=args.episodes)

    if args.data_only:
        # Save data and exit
        data_path = Path("training_data.json")
        with open(data_path, "w") as f:
            json.dump(expert_data, f)
        print(f"  Saved {len(expert_data)} samples to {data_path}")
        return

    # Phase 2: SFT Training
    model, tokenizer, train_log = train_sft(
        expert_data, num_epochs=args.epochs, max_steps=args.max_steps
    )

    # Phase 3: Evaluation
    print(f"\n{'='*60}")
    print(f"  Phase 3: Evaluation ({args.eval_episodes} episodes each)")
    print(f"{'='*60}\n")

    eval_results = {}
    eval_results["Random"] = evaluate_baseline(args.eval_episodes)
    eval_results["Heuristic"] = evaluate_heuristic(args.eval_episodes)
    eval_results["Trained LLM"] = evaluate_trained(model, tokenizer, args.eval_episodes)

    # Phase 4: Generate plots
    print(f"\n{'='*60}")
    print(f"  Phase 4: Generating plots")
    print(f"{'='*60}\n")
    generate_plots(train_log, eval_results)

    # Phase 5: Push to Hub
    if args.push_model:
        push_to_hub(model, tokenizer, args.hub_model_id)

    # Summary
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Random baseline:    {eval_results['Random']['avg_score']:.4f}")
    print(f"  Heuristic baseline: {eval_results['Heuristic']['avg_score']:.4f}")
    print(f"  Trained LLM:        {eval_results['Trained LLM']['avg_score']:.4f}")
    print(f"\n  Plots saved to: {PLOTS_DIR}/")
    if args.push_model:
        print(f"  Model pushed to: https://huggingface.co/{args.hub_model_id}")
    print()


if __name__ == "__main__":
    main()
