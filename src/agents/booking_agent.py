"""Booking Agent — LangGraph StateGraph for international booking workflow."""
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
from src.tools.booking_tools import search_schedules, get_freight_rates, create_booking, handoff_to_tracking
from src.tools.knowledge_tools import search_knowledge, log_decision

BOOKING_TOOLS = [
    search_schedules,
    get_freight_rates,
    create_booking,
    handoff_to_tracking,
    search_knowledge,
    log_decision,
]

BOOKING_SYSTEM_PROMPT = """You are an expert International Logistics Booking Agent specializing in ocean freight.

Your capabilities:
- Search vessel schedules and compare options
- Retrieve and compare freight rates (base rate + surcharges)
- Recommend optimal carrier/route based on cost, transit time, and reliability
- Create bookings and provide confirmation details
- Search knowledge base for carrier tips, regulations, and best practices
- Log your decision reasoning for transparency

Guidelines:
1. Always search schedules AND freight rates before making recommendations
2. Always search knowledge base for relevant tips (tribal knowledge)
3. Present at least 2 options with clear pros/cons when possible
4. Mention important cut-off dates and special requirements
5. Log your carrier selection decision using log_decision tool
6. After booking confirmation, ask if user wants tracking setup
7. Use handoff_to_tracking when user confirms they want tracking

When you need to hand off to tracking after a booking, use handoff_to_tracking tool.
Always be specific about costs in USD and transit times in days.
"""


class BookingState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_request: str
    cargo_info: dict | None
    schedules: list[dict]
    rates: list[dict]
    knowledge_context: list[dict]
    recommendations: list[dict]
    selected_option: dict | None
    booking_result: dict | None
    decision_traces: list[str]
    current_step: Literal["init", "search", "analyze", "recommend", "confirm", "handoff", "done"]
    session_id: str
    handoff_requested: bool


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.openai_model,
        openai_api_key=config.openai_api_key,
        temperature=0.1,
    ).bind_tools(BOOKING_TOOLS)


def agent_node(state: BookingState) -> dict[str, Any]:
    """Main agent node: LLM decides which tools to call."""
    llm = _get_llm()
    messages = [SystemMessage(content=BOOKING_SYSTEM_PROMPT)] + state["messages"]
    
    log_module.log("INFO", "booking", "llm", "Agent is thinking and analyzing context...")
    response = llm.invoke(messages)
    
    if response.content:
        log_module.log("DEBUG", "booking", "llm", f"Agent reasoning/response: {response.content}")
        
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            log_module.log("INFO", "booking", "tool", f"Agent decided to execute: {tc['name']}")
            
    return {"messages": [response]}


def should_continue(state: BookingState) -> Literal["tools", "end"]:
    """Route to tools if last message has tool calls, else end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        # Check for handoff
        for tc in last.tool_calls:
            if tc["name"] == "handoff_to_tracking":
                state["handoff_requested"] = True
                return "tools"
        return "tools"
    return "end"


def build_booking_graph() -> Any:
    """Build and compile the Booking Agent StateGraph."""
    tool_node = ToolNode(BOOKING_TOOLS)

    graph = StateGraph(BookingState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# Module-level compiled graph (lazy)
_booking_graph = None


def get_booking_graph():
    global _booking_graph
    if _booking_graph is None:
        _booking_graph = build_booking_graph()
    return _booking_graph


def run_booking_agent(user_message: str, session_id: str, history: list[BaseMessage] | None = None) -> dict:
    """Run the booking agent for a single user turn."""
    graph = get_booking_graph()

    initial_state: BookingState = {
        "messages": (history or []) + [HumanMessage(content=user_message)],
        "user_request": user_message,
        "cargo_info": None,
        "schedules": [],
        "rates": [],
        "knowledge_context": [],
        "recommendations": [],
        "selected_option": None,
        "booking_result": None,
        "decision_traces": [],
        "current_step": "init",
        "session_id": session_id,
        "handoff_requested": False,
    }

    result = graph.invoke(initial_state, {"recursion_limit": config.agent_max_iterations})
    return result
