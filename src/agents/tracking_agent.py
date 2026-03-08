"""Track & Trace Agent — LangGraph StateGraph for shipment tracking."""
from __future__ import annotations

from typing import Annotated, Literal, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.config import config
from src.cli import logger as log_module
from src.tools.tracking_tools import track_shipment, get_milestones, check_anomalies, notify_stakeholder
from src.tools.knowledge_tools import search_knowledge, log_decision

TRACKING_TOOLS = [
    track_shipment,
    get_milestones,
    check_anomalies,
    notify_stakeholder,
    search_knowledge,
    log_decision,
]

TRACKING_SYSTEM_PROMPT = """You are an expert Track & Trace Agent for international logistics shipments.

Your capabilities:
- Track shipments by B/L number, booking ID, or container number
- Retrieve complete milestone event history
- Detect anomalies: ETA delays, route deviations, rollovers
- Notify stakeholders about critical events
- Search knowledge base for port congestion and regulations
- Log anomaly detection decisions for transparency

Guidelines:
1. When tracking, always provide: current status, location, original ETA, current ETA
2. Always run check_anomalies to detect any issues
3. If anomalies detected, notify stakeholders using notify_stakeholder tool
4. Log anomaly detection decisions using log_decision tool
5. Search knowledge base for context on port congestion when delays are detected
6. Explain root causes clearly in plain language
7. For significant delays (>24h), suggest mitigation actions

Be proactive about flagging ETA changes and explain what caused them.
Always express times in local timezone context when possible.
"""


class TrackingState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_request: str
    tracking_reference: str | None
    tracking_data: dict | None
    milestones: list[dict]
    anomalies: list[dict]
    notifications_sent: list[str]
    decision_traces: list[str]
    current_step: Literal["init", "track", "check_anomalies", "notify", "done"]
    session_id: str
    booking_context: dict | None  # Passed from Booking Agent on handoff


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.openai_model,
        openai_api_key=config.openai_api_key,
        temperature=0.1,
    ).bind_tools(TRACKING_TOOLS)


def agent_node(state: TrackingState) -> dict[str, Any]:
    """Main agent node: LLM decides which tools to call."""
    llm = _get_llm()

    system = TRACKING_SYSTEM_PROMPT
    # Inject booking context if available
    if state.get("booking_context"):
        ctx = state["booking_context"]
        system += f"\n\nBooking context from Booking Agent:\n{ctx}"

    messages = [SystemMessage(content=system)] + state["messages"]
    
    log_module.log("INFO", "tracking", "llm", "Agent is analyzing tracking data and context...")
    response = llm.invoke(messages)
    
    if response.content:
        log_module.log("DEBUG", "tracking", "llm", f"Agent reasoning/response: {response.content}")
        
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            log_module.log("INFO", "tracking", "tool", f"Agent decided to execute: {tc['name']}")
            
    return {"messages": [response]}


def should_continue(state: TrackingState) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


def build_tracking_graph() -> Any:
    """Build and compile the Tracking Agent StateGraph."""
    tool_node = ToolNode(TRACKING_TOOLS)

    graph = StateGraph(TrackingState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile()


_tracking_graph = None


def get_tracking_graph():
    global _tracking_graph
    if _tracking_graph is None:
        _tracking_graph = build_tracking_graph()
    return _tracking_graph


def run_tracking_agent(
    user_message: str,
    session_id: str,
    history: list[BaseMessage] | None = None,
    booking_context: dict | None = None,
) -> dict:
    """Run the tracking agent for a single user turn."""
    graph = get_tracking_graph()

    initial_state: TrackingState = {
        "messages": (history or []) + [HumanMessage(content=user_message)],
        "user_request": user_message,
        "tracking_reference": None,
        "tracking_data": None,
        "milestones": [],
        "anomalies": [],
        "notifications_sent": [],
        "decision_traces": [],
        "current_step": "init",
        "session_id": session_id,
        "booking_context": booking_context,
    }

    result = graph.invoke(initial_state, {"recursion_limit": config.agent_max_iterations})
    return result
