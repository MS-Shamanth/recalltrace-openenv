"""Typed models for the RecallTrace OpenEnv environment."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ActionType(str, Enum):
    INSPECT_NODE = "inspect_node"
    TRACE_LOT = "trace_lot"
    QUARANTINE = "quarantine"
    NOTIFY = "notify"
    FINALIZE = "finalize"


class RecallAction(BaseModel):
    """Action submitted by an agent."""

    model_config = ConfigDict(extra="forbid")

    type: ActionType
    node_id: Optional[str] = None
    lot_id: Optional[str] = None
    quantity: Optional[int] = Field(default=None, ge=1)
    rationale: Optional[str] = None


class RewardSignal(BaseModel):
    """Typed reward payload."""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=-1.0, le=1.0)
    reason: str
    components: Dict[str, float] = Field(default_factory=dict)


class InspectionEvidence(BaseModel):
    """Evidence revealed after inspecting a node."""

    model_config = ConfigDict(extra="allow")

    status: str
    unsafe_quantity: int = Field(ge=0)
    evidence: str
    safe_quantity: Optional[int] = Field(default=None, ge=0)


class TaskDefinition(BaseModel):
    """Static task descriptor."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    name: str
    difficulty: str
    objective: str
    max_steps: int = Field(ge=1)


class RecallObservation(BaseModel):
    """Observable state exposed to the agent."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    phase: int
    recall_notice: str
    available_actions: List[str]
    inventory: Dict[str, Dict[str, int]]
    discovered_shipments: Dict[str, List[str]]
    inspected_nodes: List[str]
    inspection_results: Dict[str, Dict[str, InspectionEvidence]]
    trace_results: Dict[str, Dict[str, Any]]
    notified_nodes: List[str]
    quarantined_inventory: Dict[str, Dict[str, int]]
    history: List[str]
    steps_taken: int = Field(ge=0)
    remaining_step_budget: int = Field(ge=0)


class StepInfo(BaseModel):
    """Structured info payload returned after each step."""

    model_config = ConfigDict(extra="allow")

    message: str
    action_type: str
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)


class EnvironmentState(BaseModel):
    """Full internal state for debugging and grading."""

    model_config = ConfigDict(extra="forbid")

    done: bool
    task: TaskDefinition
    steps_taken: int = Field(ge=0)
    state_data: Dict[str, Any]
    ground_truth: Dict[str, Any]


class TaskGrade(BaseModel):
    """Deterministic grader output."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    success: bool
    steps_taken: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    reward_total: float
    final_info: Dict[str, Any]
