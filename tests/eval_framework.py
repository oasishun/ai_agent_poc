"""Evaluation framework for agent scenario testing."""
from __future__ import annotations

import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

SCENARIOS_DIR = Path(__file__).parent / "scenarios"


@dataclass
class ScenarioResult:
    scenario_id: str
    name: str
    passed: bool
    tool_calls_used: int
    latency_seconds: float
    behaviors_verified: list[str] = field(default_factory=list)
    behaviors_missing: list[str] = field(default_factory=list)
    error: str | None = None
    agent_responses: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    total: int = 0
    passed: int = 0
    failed: int = 0
    results: list[ScenarioResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


def load_scenarios(file_path: Path) -> list[dict]:
    return json.loads(file_path.read_text(encoding="utf-8"))


def count_tool_calls_in_messages(messages: list) -> int:
    """Count tool call invocations in a message history."""
    count = 0
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            count += len(msg.tool_calls)
    return count


def check_behavior(behavior: str, messages: list, agent_output: str) -> bool:
    """Check if a behavior was exhibited (heuristic keyword matching)."""
    behavior_lower = behavior.lower()
    output_lower = agent_output.lower()

    # Tool-call checks
    tool_map = {
        "search_schedules": "search_schedules",
        "get_freight_rates": "get_freight_rates",
        "create_booking": "create_booking",
        "search_knowledge": "search_knowledge",
        "track_shipment": "track_shipment",
        "check_anomalies": "check_anomalies",
        "log_decision": "log_decision",
        "handoff_to_tracking": "handoff_to_tracking",
        "notify_stakeholder": "notify_stakeholder",
    }
    for keyword, tool in tool_map.items():
        if keyword in behavior_lower:
            for msg in messages:
                if hasattr(msg, "tool_calls"):
                    for tc in (msg.tool_calls or []):
                        if tc.get("name") == tool:
                            return True

    # Text-based checks
    keywords_map = {
        "부킹 확인": ["booking", "bk-", "confirmed"],
        "옵션 추천": ["hmm", "maersk", "option", "#1", "#2"],
        "reefer": ["reefer", "냉동"],
        "transit time": ["transit", "days", "일"],
        "cost": ["usd", "total", "rate"],
        "eta": ["eta", "도착", "arrival"],
        "지연": ["delay", "지연", "late"],
        "이상 감지": ["anomaly", "이상", "delay"],
    }
    for key, kws in keywords_map.items():
        if key in behavior_lower:
            if any(kw in output_lower for kw in kws):
                return True

    return False


def run_booking_scenario(scenario: dict) -> ScenarioResult:
    """Run a booking scenario and evaluate results."""
    from src.agents.orchestrator import get_orchestrator, OrchestratorState
    from langchain_core.messages import HumanMessage, AIMessage
    import uuid

    session_id = f"eval-{uuid.uuid4().hex[:8]}"
    orchestrator = get_orchestrator()
    conversation_history = []
    all_messages = []
    all_responses = []
    start_time = time.time()

    try:
        for user_msg in scenario["user_messages"]:
            state: OrchestratorState = {
                "messages": [HumanMessage(content=user_msg)],
                "current_agent": "booking",
                "shared_context": {},
                "handoff_payload": None,
                "session_id": session_id,
                "conversation_history": conversation_history,
            }
            result = orchestrator.invoke(state)
            conversation_history = result.get("conversation_history", conversation_history)
            msgs = result.get("messages", [])
            all_messages.extend(msgs)

            for msg in reversed(msgs):
                if isinstance(msg, AIMessage):
                    all_responses.append(msg.content)
                    break

        latency = time.time() - start_time
        combined_response = " ".join(all_responses)
        tool_calls = count_tool_calls_in_messages(all_messages)

        # Evaluate behaviors
        expected = scenario.get("expected_behaviors", [])
        verified = []
        missing = []
        for behavior in expected:
            if check_behavior(behavior, all_messages, combined_response):
                verified.append(behavior)
            else:
                missing.append(behavior)

        max_calls = scenario.get("evaluation_criteria", {}).get("max_tool_calls", 999)
        max_latency = scenario.get("evaluation_criteria", {}).get("max_latency_seconds", 999)

        passed = (
            len(missing) == 0
            and tool_calls <= max_calls
            and latency <= max_latency
        )

        return ScenarioResult(
            scenario_id=scenario["scenario_id"],
            name=scenario["name"],
            passed=passed,
            tool_calls_used=tool_calls,
            latency_seconds=round(latency, 2),
            behaviors_verified=verified,
            behaviors_missing=missing,
            agent_responses=all_responses,
        )

    except Exception as e:
        return ScenarioResult(
            scenario_id=scenario["scenario_id"],
            name=scenario["name"],
            passed=False,
            tool_calls_used=0,
            latency_seconds=round(time.time() - start_time, 2),
            error=str(e),
        )


def run_all_evaluations() -> EvalReport:
    """Run all scenario evaluations and return a report."""
    report = EvalReport()

    for scenario_file in SCENARIOS_DIR.glob("*.json"):
        scenarios = load_scenarios(scenario_file)
        for scenario in scenarios:
            print(f"\n  Running: {scenario['scenario_id']} — {scenario['name']}")
            result = run_booking_scenario(scenario)
            report.results.append(result)
            report.total += 1
            if result.passed:
                report.passed += 1
                print(f"  ✓ PASSED ({result.latency_seconds:.1f}s, {result.tool_calls_used} tool calls)")
            else:
                report.failed += 1
                if result.error:
                    print(f"  ✗ FAILED (error: {result.error})")
                else:
                    print(f"  ✗ FAILED — Missing behaviors: {result.behaviors_missing}")

    return report


def print_report(report: EvalReport) -> None:
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"Total: {report.total} | Passed: {report.passed} | Failed: {report.failed}")
    print(f"Pass Rate: {report.pass_rate:.0%}")
    print(f"{'='*60}")
    for r in report.results:
        status = "✓" if r.passed else "✗"
        print(f"{status} {r.scenario_id}: {r.name}")
        if not r.passed:
            if r.error:
                print(f"  Error: {r.error}")
            else:
                print(f"  Missing: {r.behaviors_missing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation scenarios")
    parser.add_argument("--output", help="Output JSON file for results")
    args = parser.parse_args()

    print("Running evaluation scenarios...")
    report = run_all_evaluations()
    print_report(report)

    if args.output:
        out = {
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "pass_rate": report.pass_rate,
            "results": [
                {
                    "scenario_id": r.scenario_id,
                    "name": r.name,
                    "passed": r.passed,
                    "tool_calls_used": r.tool_calls_used,
                    "latency_seconds": r.latency_seconds,
                    "behaviors_missing": r.behaviors_missing,
                    "error": r.error,
                }
                for r in report.results
            ],
        }
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nReport saved to: {args.output}")
