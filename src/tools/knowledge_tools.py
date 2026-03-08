"""LangChain tools for knowledge search and decision logging."""
from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool


@tool
def search_knowledge(query: str, knowledge_type: str = "", top_k: int = 5) -> str:
    """Search the knowledge base using semantic similarity (RAG).

    Args:
        query: Natural language search query
        knowledge_type: Filter by type — one of: structured, unstructured, tribal (empty = all)
        top_k: Maximum number of results to return (default: 5)

    Returns:
        JSON string with relevant knowledge chunks and relevance scores
    """
    from src.knowledge.rag import get_rag_pipeline

    rag = get_rag_pipeline()
    ktype = knowledge_type.lower() if knowledge_type in ("structured", "unstructured", "tribal") else None
    results = rag.similarity_search(query, knowledge_type=ktype, k=top_k)  # type: ignore[arg-type]
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def log_decision(
    agent_id: str,
    session_id: str,
    decision_type: str,
    user_request: str,
    reasoning: str,
    output: str,
    confidence: float = 0.8,
    tools_used: str = "",
) -> str:
    """Log an agent decision to the Decision Trace system.

    Args:
        agent_id: Agent identifier (booking_agent or tracking_agent)
        session_id: Current session ID
        decision_type: Type of decision (e.g., carrier_selection, anomaly_detection)
        user_request: The original user request that triggered this decision
        reasoning: JSON array string of reasoning steps
        output: JSON string of the decision output
        confidence: Confidence score 0.0-1.0 (default: 0.8)
        tools_used: Comma-separated list of tools used

    Returns:
        JSON string with the created trace_id
    """
    from src.decision.trace import get_trace_manager

    try:
        steps = json.loads(reasoning) if reasoning.startswith("[") else [reasoning]
    except json.JSONDecodeError:
        steps = [reasoning]

    try:
        output_dict = json.loads(output) if output else {}
    except json.JSONDecodeError:
        output_dict = {"raw": output}

    tool_list = [t.strip() for t in tools_used.split(",") if t.strip()]

    trace = get_trace_manager().create_trace(
        agent_id=agent_id,
        session_id=session_id,
        decision_type=decision_type,
        user_request=user_request,
        reasoning_steps=steps,
        tools_used=tool_list,
        decision_output=output_dict,
        confidence_score=confidence,
    )

    return json.dumps({"trace_id": trace.trace_id, "status": "logged"}, ensure_ascii=False)
