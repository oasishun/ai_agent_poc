"""Decision Trace CRUD: read/write traces.json."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.config import config
from src.decision.models import DecisionTrace, InputContext, DecisionOutput, UserFeedback


class TraceManager:
    """Manages persistence of DecisionTrace records."""

    def __init__(self, traces_file: Path | None = None):
        self._file = traces_file or config.traces_file
        self._file.parent.mkdir(parents=True, exist_ok=True)
        if not self._file.exists():
            self._file.write_text("[]", encoding="utf-8")

    def _load(self) -> list[dict]:
        return json.loads(self._file.read_text(encoding="utf-8"))

    def _save(self, traces: list[dict]) -> None:
        self._file.write_text(
            json.dumps(traces, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def create_trace(
        self,
        agent_id: str,
        session_id: str,
        decision_type: str,
        user_request: str,
        reasoning_steps: list[str],
        tools_used: list[str],
        decision_output: dict | None = None,
        confidence_score: float = 0.8,
        retrieved_knowledge: list[dict] | None = None,
        tool_results: dict | None = None,
        parent_trace_id: str | None = None,
    ) -> DecisionTrace:
        """Create and persist a new DecisionTrace."""
        from src.decision.models import RetrievedKnowledge, RecommendationItem

        knowledge = [
            RetrievedKnowledge(**k) if isinstance(k, dict) else k
            for k in (retrieved_knowledge or [])
        ]

        trace = DecisionTrace(
            agent_id=agent_id,
            session_id=session_id,
            decision_type=decision_type,
            input_context=InputContext(
                user_request=user_request,
                retrieved_knowledge=knowledge,
                tool_results=tool_results or {},
            ),
            reasoning_steps=reasoning_steps,
            decision_output=DecisionOutput(raw=decision_output or {}),
            confidence_score=confidence_score,
            tools_used=tools_used,
            parent_trace_id=parent_trace_id,
        )

        traces = self._load()
        traces.append(json.loads(trace.model_dump_json()))
        self._save(traces)
        return trace

    def get_trace(self, trace_id: str) -> dict | None:
        """Retrieve a trace by ID."""
        return next((t for t in self._load() if t["trace_id"] == trace_id), None)

    def list_traces(
        self,
        session_id: str | None = None,
        agent_id: str | None = None,
        decision_type: str | None = None,
    ) -> list[dict]:
        """List traces with optional filters."""
        traces = self._load()
        if session_id:
            traces = [t for t in traces if t.get("session_id") == session_id]
        if agent_id:
            traces = [t for t in traces if t.get("agent_id") == agent_id]
        if decision_type:
            traces = [t for t in traces if t.get("decision_type") == decision_type]
        return traces

    def update_feedback(
        self,
        trace_id: str,
        action: Literal["APPROVED", "REJECTED", "MODIFIED"],
        comment: str | None = None,
        modified_output: dict | None = None,
    ) -> bool:
        """Update feedback on an existing trace."""
        traces = self._load()
        for trace in traces:
            if trace["trace_id"] == trace_id:
                trace["feedback"] = {
                    "action": action,
                    "comment": comment,
                    "modified_output": modified_output,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self._save(traces)
                return True
        return False

    def export(self, output_path: Path | None = None) -> str:
        """Export all traces to a JSON file and return the path."""
        traces = self._load()
        if output_path is None:
            output_path = config.log_dir / "traces_export.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(traces, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return str(output_path)


# Module-level singleton
_trace_manager: TraceManager | None = None


def get_trace_manager() -> TraceManager:
    global _trace_manager
    if _trace_manager is None:
        _trace_manager = TraceManager()
    return _trace_manager
