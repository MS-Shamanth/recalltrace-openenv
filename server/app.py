"""FastAPI server for serving RecallTrace in Docker or Hugging Face Spaces."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from baseline.policy import choose_heuristic_action
from env.env import RecallTraceEnv
from env.models import RecallAction


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="RecallTrace OpenEnv", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

ACTIVE_ENV = RecallTraceEnv()


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    phase: Optional[int] = None


class RunEpisodeRequest(BaseModel):
    task_id: Optional[str] = None
    phase: Optional[int] = None


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


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


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

