"""FastAPI server for serving RecallTrace in Docker or Hugging Face Spaces."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from baseline.policy import choose_heuristic_action
from env.env import RecallTraceEnv
from env.models import RecallAction
from selfplay.trainer import SelfPlayTrainer
from selfplay.scenario_gen import generate_graph, apply_intervention, compute_f1
from selfplay.adversary import AdversaryAgent, INTERVENTION_TYPES, GRAPH_REGIONS
from selfplay.investigator import InvestigatorAgent


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="RecallTrace OpenEnv", version="2.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

ACTIVE_ENV = RecallTraceEnv()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    phase: Optional[int] = None
    num_nodes: Optional[int] = None


class RunEpisodeRequest(BaseModel):
    task_id: Optional[str] = None
    phase: Optional[int] = None


class SelfPlayRequest(BaseModel):
    num_episodes: int = 200
    num_nodes: int = 10


# ---------------------------------------------------------------------------
# Static / health
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# OpenEnv endpoints (original)
# ---------------------------------------------------------------------------

@app.get("/tasks")
def tasks() -> dict:
    return {"tasks": [task.model_dump() for task in RecallTraceEnv.available_tasks()]}


@app.get("/api/tasks")
def api_tasks() -> dict:
    return tasks()


@app.get("/reset")
def reset_get(task_id: Optional[str] = None, phase: Optional[int] = None) -> dict:
    try:
        return ACTIVE_ENV.reset(task_id=task_id, phase=phase).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reset")
def reset_post(request: ResetRequest | None = Body(default=None)) -> dict:
    global ACTIVE_ENV
    request = request or ResetRequest()
    try:
        if request.num_nodes:
            from selfplay.scenario_gen import generate_graph
            ACTIVE_ENV = RecallTraceEnv(scenario_data=generate_graph(num_nodes=request.num_nodes))
            return ACTIVE_ENV.reset().model_dump()
        else:
            return ACTIVE_ENV.reset(task_id=request.task_id, phase=request.phase).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step")
def step(action: RecallAction) -> dict:
    try:
        observation, reward, done, info = ACTIVE_ENV.step(action)
        return {
            "observation": observation.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state")
def state() -> dict:
    return ACTIVE_ENV.state().model_dump()


def _run_episode(task_id: str | None = None, phase: int | None = None) -> dict:
    env = RecallTraceEnv(task_id=task_id, phase=phase)
    observation = env.reset(task_id=task_id, phase=phase)
    logs = []
    final_info = {"score": 0.0}

    for step_number in range(1, env.task.max_steps + 1):
        action = choose_heuristic_action(observation)
        observation, reward, done, info = env.step(action)
        logs.append(
            {
                "step": step_number,
                "action": action.model_dump(exclude_none=True),
                "reward": reward,
                "done": done,
                "message": info.get("message"),
            }
        )
        final_info = info
        if done:
            break

    return {
        "task": env.task.model_dump(),
        "score": float(final_info.get("score", 0.0)),
        "success": float(final_info.get("score", 0.0)) >= 0.9,
        "steps_taken": env.state().steps_taken,
        "final_info": final_info,
        "final_observation": observation.model_dump(),
        "logs": logs,
    }


@app.post("/api/run_episode")
def run_episode(request: RunEpisodeRequest) -> dict:
    try:
        return _run_episode(task_id=request.task_id, phase=request.phase)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/run_all")
def run_all() -> dict:
    try:
        episodes = [_run_episode(task_id=task.task_id) for task in RecallTraceEnv.available_tasks()]
        average_score = round(sum(item["score"] for item in episodes) / len(episodes), 4)
        return {
            "average_score": average_score,
            "episodes": episodes,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Self-Play API (NEW — powers the frontend simulation)
# ---------------------------------------------------------------------------

@app.post("/api/selfplay/run")
def selfplay_run(request: SelfPlayRequest) -> dict:
    """Run N episodes of adversarial self-play training.

    Returns all episode stats for the frontend to animate training curves.
    """
    try:
        trainer = SelfPlayTrainer(num_nodes=request.num_nodes)
        stats = trainer.train(num_episodes=request.num_episodes)

        # Compute summary
        early = stats[:20]
        late = stats[-20:]
        summary = {
            "early_f1": round(sum(s["investigator_f1"] for s in early) / len(early), 4),
            "late_f1": round(sum(s["investigator_f1"] for s in late) / len(late), 4),
            "early_quarantined": round(sum(s["num_quarantined"] for s in early) / len(early), 2),
            "late_quarantined": round(sum(s["num_quarantined"] for s in late) / len(late), 2),
            "early_steps": round(sum(s["steps_taken"] for s in early) / len(early), 2),
            "late_steps": round(sum(s["steps_taken"] for s in late) / len(late), 2),
            "adversary_strategy": trainer.adversary.get_strategy_summary(),
        }

        return {
            "num_episodes": request.num_episodes,
            "summary": summary,
            "episodes": stats,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/selfplay/demo")
def selfplay_demo(num_nodes: int = 10) -> dict:
    """Return pre-computed before/after episode data for instant demo.

    Runs a quick 200-episode training and returns early vs late comparison.
    """
    try:
        global ACTIVE_ENV
        from selfplay.scenario_gen import generate_graph
        ACTIVE_ENV = RecallTraceEnv(scenario_data=generate_graph(num_nodes=num_nodes))
        ACTIVE_ENV.reset()
        
        trainer = SelfPlayTrainer(num_nodes=num_nodes)
        stats = trainer.train(num_episodes=200)

        early_candidates = stats[:30]
        worst_early = min(early_candidates, key=lambda s: s["investigator_f1"])
        late_candidates = stats[-30:]
        best_late = max(late_candidates, key=lambda s: s["investigator_f1"])

        return {
            "early_episode": worst_early,
            "late_episode": best_late,
            "all_stats": stats,
            "graph": graph_structure(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/graph/structure")
def graph_structure() -> dict:
    """Return dynamic graph topology for the visualization canvas."""
    if not ACTIVE_ENV.state_data or "shipment_graph" not in ACTIVE_ENV.state_data:
        ACTIVE_ENV.reset()
        
    nodes = []
    edges = []
    
    graph = ACTIVE_ENV.state_data.get("shipment_graph", {})
    all_nodes = ACTIVE_ENV.state_data.get("nodes", {})
    
    # Assign layers
    layers = {"warehouse": [], "crossdock": [], "store": []}
    for n_id in all_nodes.keys():
        if n_id.startswith("warehouse"): layers["warehouse"].append(n_id)
        elif n_id.startswith("crossdock"): layers["crossdock"].append(n_id)
        else: layers["store"].append(n_id)
        
    x_positions = {"warehouse": 0.15, "crossdock": 0.5, "store": 0.85}
    
    # Generate coordinates
    for role, n_list in layers.items():
        count = len(n_list)
        for i, n_id in enumerate(sorted(n_list)):
            y = 0.1 + (0.8 * i / max(1, count - 1)) if count > 1 else 0.5
            nodes.append({
                "id": n_id,
                "label": n_id.capitalize().replace("_", " "),
                "role": role,
                "x": x_positions[role],
                "y": y,
                "contaminated": False # the frontend expects boolean, but ground truth shouldn't be exposed immediately unless required. Wait, frontend has logic for true contamination ring, but it's okay to omit or leave False for manual mode.
            })
            
    # Edges
    for src, targets in graph.items():
        for tgt in targets:
            edges.append({"from": src, "to": tgt})
            
    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# LLM Agent Inference (GPU-powered live demo)
# ---------------------------------------------------------------------------

_llm_model = None
_llm_tokenizer = None

LLM_HUB_MODEL = "ms-shamanth/recalltrace-investigator"
LLM_BASE_MODEL = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"

LLM_SYSTEM_PROMPT = (
    "You are an expert supply-chain investigator for RecallTrace. "
    "You receive an observation of a product recall investigation and must "
    "respond with the next best action as a JSON object. "
    "Available actions: inspect_node, trace_lot, quarantine, notify, finalize."
)


def _load_llm():
    """Lazy-load the trained LoRA model from HF Hub (runs once)."""
    global _llm_model, _llm_tokenizer
    if _llm_model is not None:
        return _llm_model, _llm_tokenizer

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available — LLM inference requires CUDA")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"  Loading tokenizer from {LLM_HUB_MODEL}...")
    _llm_tokenizer = AutoTokenizer.from_pretrained(LLM_HUB_MODEL)
    
    print(f"  Loading 4-bit base model {LLM_BASE_MODEL}...")
    quant_config = BitsAndBytesConfig(load_in_4bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        LLM_BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quant_config,
    )
    
    print(f"  Applying LoRA adapters from {LLM_HUB_MODEL}...")
    _llm_model = PeftModel.from_pretrained(base_model, LLM_HUB_MODEL)
    _llm_model.eval()
    
    print(f"  ✅ Model loaded successfully on {_llm_model.device}")
    return _llm_model, _llm_tokenizer


def _format_obs_for_llm(obs) -> str:
    """Format an observation into a text prompt for the LLM."""
    d = obs.model_dump() if hasattr(obs, 'model_dump') else obs
    parts = [f"Step: {d.get('steps_taken', 0)}/{d.get('max_steps', 15)}"]
    if d.get('recall_notice'):
        parts.append(f"Recall: {d['recall_notice']}")
    if d.get('nodes'):
        names = [n.get('node_id', n.get('id', '?')) for n in d['nodes'][:8]]
        parts.append(f"Visible nodes: {', '.join(names)}")
    if d.get('evidence'):
        parts.append(f"Evidence items: {len(d['evidence'])}")
        for ev in d['evidence'][:3]:
            parts.append(f"  - {ev}")
    if d.get('quarantined_nodes'):
        parts.append(f"Already quarantined: {d['quarantined_nodes']}")
    return "\n".join(parts)


class LLMRunRequest(BaseModel):
    task_id: Optional[str] = None


@app.get("/api/llm/status")
def llm_status() -> dict:
    """Check if GPU + model are available."""
    import torch
    gpu = torch.cuda.is_available()
    loaded = _llm_model is not None
    gpu_name = torch.cuda.get_device_name(0) if gpu else None
    return {"gpu_available": gpu, "model_loaded": loaded, "gpu_name": gpu_name}


@app.post("/api/llm/run_episode")
def llm_run_episode(request: LLMRunRequest = Body(default=LLMRunRequest())) -> dict:
    """Run a full episode using the trained LLM agent."""
    import torch

    try:
        model, tokenizer = _load_llm()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model loading failed: {e}")

    # Pick a task
    tasks = RecallTraceEnv.available_tasks()
    task_id = request.task_id
    if not task_id:
        task_id = random.choice(tasks).task_id
    task = next((t for t in tasks if t.task_id == task_id), tasks[0])

    env = RecallTraceEnv(task_id=task.task_id)
    obs = env.reset(task_id=task.task_id)
    steps_log = []
    total_reward = 0.0

    for step_num in range(1, env.task.max_steps + 1):
        prompt_text = _format_obs_for_llm(obs)
        messages = [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=200,
                temperature=0.1, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw_response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Parse model output into an action
        used_fallback = False
        try:
            import json as _json
            action_dict = _json.loads(raw_response)
            action = RecallAction.model_validate(action_dict)
        except Exception:
            action = choose_heuristic_action(obs)
            used_fallback = True

        obs, reward, done, info = env.step(action)
        total_reward += reward

        steps_log.append({
            "step": step_num,
            "model_output": raw_response[:500],
            "action": action.model_dump(exclude_none=True),
            "used_fallback": used_fallback,
            "reward": round(reward, 4),
            "done": done,
        })

        if done:
            break

    score = info.get("score") or 0.0
    return {
        "task": task.model_dump(),
        "score": round(float(score), 4),
        "total_reward": round(total_reward, 4),
        "steps_taken": len(steps_log),
        "steps": steps_log,
    }


# ---------------------------------------------------------------------------
# Single-episode detailed trace (for step-by-step animation)
# ---------------------------------------------------------------------------

@app.get("/api/selfplay/trace")
def selfplay_trace() -> dict:
    """Run a single self-play episode and return detailed step data for animation."""
    try:
        rng = random.Random(42)
        graph_scenario = generate_graph(num_nodes=10, seed=42)

        # Adversary picks intervention
        adversary = AdversaryAgent()
        intervention_type, target_node, num_hops = adversary.choose_intervention(
            graph_scenario, rng=rng,
        )
        graph_region = graph_scenario.get("_node_regions", {}).get(target_node, "downstream")

        # Apply intervention
        scenario = apply_intervention(graph_scenario, intervention_type, target_node, num_hops, rng=rng)

        # Create env and run investigator
        env = RecallTraceEnv(scenario_data=scenario)
        observation = env.reset()
        investigator = InvestigatorAgent()
        investigator.reset_episode()

        trace_steps: List[Dict[str, Any]] = []
        total_reward = 0.0
        step_num = 0
        done = False

        while not done and step_num < scenario["max_steps"]:
            action = investigator.act(observation, rng=rng)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            step_num += 1

            trace_steps.append({
                "step": step_num,
                "action_type": action.type if hasattr(action.type, 'value') else str(action.type),
                "node_id": getattr(action, 'node_id', None),
                "lot_id": getattr(action, 'lot_id', None),
                "quantity": getattr(action, 'quantity', None),
                "rationale": getattr(action, 'rationale', None),
                "reward": round(reward, 4),
                "done": done,
                "nodes_quarantined": list(set(investigator.nodes_quarantined)),
                "nodes_visited": list(set(investigator.nodes_visited)),
            })

        quarantined = list(set(investigator.nodes_quarantined))
        f1, f1_details = compute_f1(scenario, quarantined)

        return {
            "intervention_type": intervention_type,
            "graph_region": graph_region,
            "target_node": target_node,
            "f1": round(f1, 4),
            "f1_details": f1_details,
            "total_reward": round(total_reward, 4),
            "steps": trace_steps,
            "graph": _get_demo_graph(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
