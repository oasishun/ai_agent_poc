"""Pydantic models for Decision Trace system."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field
import uuid


class RetrievedKnowledge(BaseModel):
    source: str
    chunk: str
    relevance_score: float


class InputContext(BaseModel):
    user_request: str
    retrieved_knowledge: list[RetrievedKnowledge] = Field(default_factory=list)
    tool_results: dict[str, Any] = Field(default_factory=dict)


class RecommendationItem(BaseModel):
    rank: int
    schedule_id: str | None = None
    carrier: str | None = None
    reason: str


class DecisionOutput(BaseModel):
    recommendation_rank: list[RecommendationItem] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class UserFeedback(BaseModel):
    action: Literal["APPROVED", "REJECTED", "MODIFIED"]
    comment: str | None = None
    modified_output: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DecisionTrace(BaseModel):
    trace_id: str = Field(
        default_factory=lambda: f"dt-{uuid.uuid4()}"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_id: Literal["booking_agent", "tracking_agent", "orchestrator"]
    session_id: str
    decision_type: str  # e.g. carrier_selection, anomaly_detection, eta_update
    input_context: InputContext
    reasoning_steps: list[str] = Field(default_factory=list)
    decision_output: DecisionOutput = Field(default_factory=DecisionOutput)
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.8)
    tools_used: list[str] = Field(default_factory=list)
    parent_trace_id: str | None = None
    feedback: UserFeedback | None = None

    model_config = {"arbitrary_types_allowed": True}
