"""Deterministic graders for RecallTrace tasks."""

from __future__ import annotations

from typing import Iterable, List

from env.env import RecallTraceEnv
from env.models import RecallAction, TaskGrade


def evaluate_action_plan(task_id: str, actions: Iterable[RecallAction | dict]) -> TaskGrade:
    """Run an action plan against a task and return a deterministic score."""
    env = RecallTraceEnv(task_id=task_id)
    env.reset()

    rewards: List[float] = []
    final_info = {"message": "Episode never finalized."}

    for action in actions:
        _, reward, done, info = env.step(action)
        rewards.append(reward)
        final_info = info
        if done:
            break

    if not env.done:
        _, reward, done, info = env.step(RecallAction(type="finalize"))
        rewards.append(reward)
        final_info = info
        assert done

    score = float(final_info.get("score", 0.0))
    state = env.state()
    return TaskGrade(
        task_id=task_id,
        score=score,
        success=score >= 0.9,
        steps_taken=state.steps_taken,
        max_steps=state.task.max_steps,
        reward_total=round(sum(rewards), 4),
        final_info=final_info,
    )


def grade_finalize_info(task_id: str, steps_taken: int, final_info: dict) -> TaskGrade:
    """Build a TaskGrade object from a finalized episode payload."""
    env = RecallTraceEnv(task_id=task_id)
    env.reset()
    return TaskGrade(
        task_id=task_id,
        score=float(final_info.get("score", 0.0)),
        success=float(final_info.get("score", 0.0)) >= 0.9,
        steps_taken=steps_taken,
        max_steps=env.task.max_steps,
        reward_total=float(final_info.get("score", 0.0)),
        final_info=final_info,
    )
