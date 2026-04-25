"""Submission-grade baseline inference runner for RecallTrace."""

from __future__ import annotations

import json
import os
from typing import Any, List

from openai import OpenAI

from env.env import RecallTraceEnv
from env.models import RecallAction
from grader.grader import grade_finalize_info
from baseline.policy import choose_heuristic_action, choose_llm_action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN", "")
BENCHMARK = "RecallTrace"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: RecallAction, reward: float, done: bool, error: str | None) -> None:
    payload = json.dumps(action.model_dump(exclude_none=True), sort_keys=True)
    error_text = error if error is not None else "null"
    print(f"[STEP] step={step} action={payload} reward={reward:.4f} done={str(done).lower()} error={error_text}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={json.dumps([round(r, 4) for r in rewards])}", flush=True)


def run_task(task_id: str, client: OpenAI | None) -> float:
    env = RecallTraceEnv(task_id=task_id)
    observation = env.reset()

    history: List[dict[str, Any]] = []
    rewards: List[float] = []
    steps_taken = 0
    final_info: dict[str, Any] = {"score": 0.0}

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME if client else "heuristic-baseline")

    for step in range(1, env.task.max_steps + 1):
        llm_action = choose_llm_action(client, MODEL_NAME, observation, history)
        action = llm_action or choose_heuristic_action(observation)

        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        steps_taken = step
        final_info = info
        log_step(step=step, action=action, reward=reward, done=done, error=info.get("error"))

        history.append(
            {
                "step": step,
                "action": action.model_dump(exclude_none=True),
                "reward": reward,
                "done": done,
                "message": info.get("message"),
            }
        )
        if done:
            break

    grade = grade_finalize_info(task_id, steps_taken, final_info)
    log_end(success=grade.success, steps=steps_taken, score=grade.score, rewards=rewards)
    return grade.score


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
    task_scores = [run_task(task.task_id, client) for task in RecallTraceEnv.available_tasks()]
    average_score = sum(task_scores) / len(task_scores)
    print(json.dumps({"benchmark": BENCHMARK, "average_score": round(average_score, 4), "task_scores": task_scores}), flush=True)


if __name__ == "__main__":
    main()
