"""Rich-based output formatter for agent responses."""
from __future__ import annotations

import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich.rule import Rule
from rich import box

console = Console()

AGENT_COLORS = {
    "booking": "blue",
    "tracking": "green",
    "orchestrator": "magenta",
    "system": "dim white",
}


def render_header(session_id: str, active_agent: str = "booking") -> None:
    color = AGENT_COLORS.get(active_agent, "white")
    agent_label = {
        "booking": "🚢 Booking Agent",
        "tracking": "📦 Track & Trace Agent",
        "orchestrator": "🤖 Orchestrator",
    }.get(active_agent, active_agent)

    console.print(
        Panel(
            f"  [bold]Logistics AI Agent PoC v1.0[/bold]          [dim]Session: {session_id}[/dim]\n"
            f"  Active Agent: [{color}]{agent_label}[/{color}]",
            border_style=color,
        )
    )


def render_agent_response(content: str, agent: str = "booking") -> None:
    color = AGENT_COLORS.get(agent, "white")
    label = {
        "booking": "🚢 Booking Agent",
        "tracking": "📦 Track & Trace Agent",
    }.get(agent, f"🤖 {agent.title()} Agent")

    console.print(
        Panel(
            content,
            title=f"[{color}]{label}[/{color}]",
            border_style=color,
            padding=(1, 2),
        )
    )


def render_tool_call(tool_name: str, args: dict) -> None:
    args_str = json.dumps(args, ensure_ascii=False)
    console.print(
        Panel(
            f"[bold yellow]→ {tool_name}[/bold yellow]({args_str})",
            title="[yellow]Tool Call[/yellow]",
            border_style="yellow",
            padding=(0, 1),
        )
    )


def render_decision_trace(trace: dict) -> None:
    """Render a DecisionTrace using Rich Tree."""
    confidence = trace.get("confidence_score", 0)
    color = "green" if confidence >= 0.8 else "yellow" if confidence >= 0.7 else "red"

    tree = Tree(
        f"[bold]Decision Trace[/bold] [{color}]{trace.get('trace_id', '')}[/{color}] "
        f"| {trace.get('decision_type', '')} "
        f"| Confidence: [{color}]{confidence:.0%}[/{color}]"
    )

    # Input
    ctx_node = tree.add("[cyan]Input Context[/cyan]")
    ctx = trace.get("input_context", {})
    ctx_node.add(f"Request: {ctx.get('user_request', '')[:80]}")
    knowledge = ctx.get("retrieved_knowledge", [])
    if knowledge:
        k_node = ctx_node.add(f"Knowledge ({len(knowledge)} items)")
        for k in knowledge[:3]:
            k_node.add(f"[dim]{k.get('source', '')} (score: {k.get('relevance_score', 0):.2f})[/dim]")

    # Reasoning
    steps_node = tree.add("[green]Reasoning Steps[/green]")
    for step in trace.get("reasoning_steps", []):
        steps_node.add(f"[dim]{step}[/dim]")

    # Tools
    tools = trace.get("tools_used", [])
    if tools:
        tree.add(f"[yellow]Tools Used:[/yellow] {', '.join(tools)}")

    console.print(tree)


def render_traces_table(traces: list[dict]) -> None:
    """Render a summary table of decision traces."""
    table = Table(title="Decision Traces", box=box.ROUNDED)
    table.add_column("Trace ID", style="cyan", no_wrap=True, max_width=20)
    table.add_column("Agent", style="blue")
    table.add_column("Type", style="green")
    table.add_column("Confidence", justify="right")
    table.add_column("Feedback")
    table.add_column("Timestamp", style="dim")

    for t in traces:
        conf = t.get("confidence_score", 0)
        color = "green" if conf >= 0.8 else "yellow" if conf >= 0.7 else "red"
        feedback = t.get("feedback")
        fb_str = feedback["action"] if feedback else "—"
        table.add_row(
            t.get("trace_id", "")[-12:],
            t.get("agent_id", ""),
            t.get("decision_type", ""),
            f"[{color}]{conf:.0%}[/{color}]",
            fb_str,
            str(t.get("timestamp", ""))[:19],
        )
    console.print(table)


def render_knowledge_summary(stats: dict[str, int]) -> None:
    table = Table(title="Knowledge Store", box=box.SIMPLE)
    table.add_column("Collection", style="cyan")
    table.add_column("Chunks", justify="right", style="green")
    for k, v in stats.items():
        table.add_row(k, str(v))
    console.print(table)


def render_log_rule() -> None:
    console.print(Rule("Agent Log", style="dim"))


def render_separator() -> None:
    console.print(Rule(style="dim"))
