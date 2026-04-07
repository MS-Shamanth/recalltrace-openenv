"""FastAPI server for serving RecallTrace in Docker or Hugging Face Spaces."""

from __future__ import annotations

from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.env import RecallTraceEnv
from env.models import RecallAction


app = FastAPI(title="RecallTrace OpenEnv", version="1.0.0")
ACTIVE_ENV = RecallTraceEnv()


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    phase: Optional[int] = None


@app.get("/")
def root() -> dict:
    return {
        "name": "RecallTrace OpenEnv",
        "status": "ok",
        "tasks": [task.model_dump() for task in RecallTraceEnv.available_tasks()],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.get("/tasks")
def tasks() -> dict:
    return {"tasks": [task.model_dump() for task in RecallTraceEnv.available_tasks()]}


@app.get("/reset")
def reset_get(task_id: Optional[str] = None, phase: Optional[int] = None) -> dict:
    try:
        return ACTIVE_ENV.reset(task_id=task_id, phase=phase).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reset")
def reset_post(request: ResetRequest) -> dict:
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


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
