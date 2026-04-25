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
    request = request or ResetRequest()
    try:
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
def selfplay_demo() -> dict:
    """Return pre-computed before/after episode data for instant demo.

    Runs a quick 200-episode training and returns early vs late comparison.
    """
    try:
        trainer = SelfPlayTrainer(num_nodes=10)
        stats = trainer.train(num_episodes=200)

        early_candidates = stats[:30]
        worst_early = min(early_candidates, key=lambda s: s["investigator_f1"])
        late_candidates = stats[-30:]
        best_late = max(late_candidates, key=lambda s: s["investigator_f1"])

        return {
            "early_episode": worst_early,
            "late_episode": best_late,
            "all_stats": stats,
            "graph": _get_demo_graph(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/graph/structure")
def graph_structure() -> dict:
    """Return graph topology for the visualization canvas."""
    return _get_demo_graph()


def _get_demo_graph() -> dict:
    """Build a sample graph for visualization."""
    nodes = [
        {"id": "Supplier_G", "label": "Supplier G", "role": "warehouse", "region": "source", "x": 0.15, "y": 0.5},
        {"id": "Lot_A", "label": "Lot A", "role": "warehouse", "region": "source", "x": 0.35, "y": 0.25, "contaminated": True},
        {"id": "Warehouse_B", "label": "Warehouse B", "role": "warehouse", "region": "source", "x": 0.35, "y": 0.75},
        {"id": "Lot_C", "label": "Lot C", "role": "crossdock", "region": "midstream", "x": 0.55, "y": 0.15, "contaminated": True},
        {"id": "Distributor_D", "label": "Distributor D", "role": "crossdock", "region": "midstream", "x": 0.55, "y": 0.55},
        {"id": "Hub_H", "label": "Hub H", "role": "crossdock", "region": "midstream", "x": 0.55, "y": 0.85},
        {"id": "Retailer_E", "label": "Retailer E", "role": "store", "region": "downstream", "x": 0.78, "y": 0.45},
        {"id": "Lot_F", "label": "Lot F", "role": "store", "region": "downstream", "x": 0.78, "y": 0.7},
    ]

    edges = [
        {"from": "Supplier_G", "to": "Warehouse_B"},
        {"from": "Supplier_G", "to": "Lot_A"},
        {"from": "Warehouse_B", "to": "Distributor_D"},
        {"from": "Warehouse_B", "to": "Hub_H"},
        {"from": "Lot_A", "to": "Distributor_D"},
        {"from": "Lot_A", "to": "Lot_C"},
        {"from": "Distributor_D", "to": "Retailer_E"},
        {"from": "Distributor_D", "to": "Lot_F"},
        {"from": "Hub_H", "to": "Retailer_E"},
        {"from": "Lot_C", "to": "Lot_F"},
    ]

    return {"nodes": nodes, "edges": edges}


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
