"""Multi-Agent Orchestrator — routes between Booking and Tracking agents."""
from __future__ import annotations

import json
from typing import Annotated, Literal, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.config import config
from src.agents.booking_agent import run_booking_agent
from src.agents.tracking_agent import run_tracking_agent

ROUTER_PROMPT = """You are a logistics AI orchestrator. Analyze the user's message in the context of the recent conversation and determine which agent should handle it.

Respond with ONLY one of:
- "booking" — for scheduling, rates, freight quotes, creating bookings, carrier comparisons
- "tracking" — for tracking shipments, checking status, anomalies, ETA updates, milestones
- "general" — for general questions or explicit help requests
- "continue" — if the user is answering a question, making a selection (e.g., "1번으로 해줘", "옵션2"), or continuing the ongoing context.

Examples:
- "부산에서 LA로 부킹해줘" → booking
- "컨테이너 어디있어?" → tracking
- "요율 얼마야?" → booking
- "도움말" → general
- "1번 옵션으로 진행해줘" → continue
- "응 그렇게 해" → continue
"""


class OrchestratorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: Literal["orchestrator", "booking", "tracking"]
    shared_context: dict
    handoff_payload: dict | None
    session_id: str
    conversation_history: list[BaseMessage]


def _get_router_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.openai_model_fallback,  # Use cheaper model for routing
        openai_api_key=config.openai_api_key,
        temperature=0,
    )


def route_intent(state: OrchestratorState) -> dict[str, Any]:
    """Analyze user intent and determine target agent."""
    last_user_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg:
        return {"current_agent": "general"}

    history = state.get("conversation_history", [])
    history_text = ""
    if history:
        history_text = "Recent Conversation:\n"
        for msg in history[-4:]:
            role = "User" if isinstance(msg, HumanMessage) else "Agent"
            history_text += f"{role}: {msg.content}\n"

    llm = _get_router_llm()
    response = llm.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=f"{history_text}\nUser message: {last_user_msg}"),
    ])
    intent = response.content.strip().lower()
    
    if "booking" in intent:
        agent = "booking"
    elif "tracking" in intent:
        agent = "tracking"
    elif "continue" in intent:
        agent = state.get("current_agent", "general")
    else:
        agent = "general"

    return {"current_agent": agent}


def booking_node(state: OrchestratorState) -> dict[str, Any]:
    """Execute Booking Agent and handle handoff signals."""
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    result = run_booking_agent(
        user_message=last_user_msg,
        session_id=state["session_id"],
        history=state["conversation_history"],
    )

    # Extract agent response
    agent_messages = result.get("messages", [])
    last_ai = None
    for msg in reversed(agent_messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            last_ai = msg
            break

    # Check for handoff in tool call results
    handoff_payload = None
    for msg in agent_messages:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict) and data.get("handoff"):
                    handoff_payload = data
                    shared = dict(state["shared_context"])
                    shared.update({
                        "booking_id": data.get("booking_id"),
                        "shipment_ref": data.get("shipment_ref"),
                    })
                    state["shared_context"] = shared
            except (json.JSONDecodeError, TypeError):
                pass

    updates: dict[str, Any] = {}
    if last_ai:
        updates["messages"] = [last_ai]
        updates["conversation_history"] = list(state["conversation_history"]) + [
            HumanMessage(content=last_user_msg),
            last_ai,
        ]
    if handoff_payload:
        updates["handoff_payload"] = handoff_payload
        updates["current_agent"] = "tracking"

    return updates


def tracking_node(state: OrchestratorState) -> dict[str, Any]:
    """Execute Tracking Agent with booking context if available."""
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    booking_context = state.get("shared_context") or None

    result = run_tracking_agent(
        user_message=last_user_msg,
        session_id=state["session_id"],
        history=state["conversation_history"],
        booking_context=booking_context,
    )

    agent_messages = result.get("messages", [])
    last_ai = None
    for msg in reversed(agent_messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            last_ai = msg
            break

    updates: dict[str, Any] = {}
    if last_ai:
        updates["messages"] = [last_ai]
        updates["conversation_history"] = list(state["conversation_history"]) + [
            HumanMessage(content=last_user_msg),
            last_ai,
        ]
    return updates


def general_node(state: OrchestratorState) -> dict[str, Any]:
    """Handle general/help requests."""
    help_text = """안녕하세요! Logistics AI Agent입니다.

다음과 같은 업무를 도와드릴 수 있습니다:

🚢 **Booking Agent** (부킹 관련):
- 항구 간 스케줄 조회
- 운임 비교 및 추천
- 부킹 생성 및 확인

📦 **Track & Trace Agent** (추적 관련):
- B/L 또는 컨테이너 번호로 화물 추적
- 이상 감지 및 ETA 업데이트
- 이동 이력 조회

**슬래시 커맨드**: /help, /switch, /trace [id], /traces, /knowledge, /log, /export, /exit

무엇을 도와드릴까요?"""
    return {"messages": [AIMessage(content=help_text)]}


def route_after_router(state: OrchestratorState) -> Literal["booking", "tracking", "general", "__end__"]:
    agent = state.get("current_agent", "orchestrator")
    if agent == "booking":
        return "booking"
    elif agent == "tracking":
        return "tracking"
    else:
        return "general"


def build_orchestrator_graph() -> Any:
    """Build and compile the Orchestrator StateGraph."""
    graph = StateGraph(OrchestratorState)

    graph.add_node("router", route_intent)
    graph.add_node("booking", booking_node)
    graph.add_node("tracking", tracking_node)
    graph.add_node("general", general_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"booking": "booking", "tracking": "tracking", "general": "general"},
    )
    graph.add_edge("booking", END)
    graph.add_edge("tracking", END)
    graph.add_edge("general", END)

    return graph.compile()


_orchestrator_graph = None


def get_orchestrator() -> Any:
    global _orchestrator_graph
    if _orchestrator_graph is None:
        _orchestrator_graph = build_orchestrator_graph()
    return _orchestrator_graph
