"""Main CLI loop for the Logistics AI Agent PoC."""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage

from rich.console import Console
from rich.prompt import Prompt

from src.config import config
from src.cli import logger as log_module
from src.cli.renderer import (
    render_header,
    render_agent_response,
    render_decision_trace,
    render_traces_table,
    render_knowledge_summary,
    render_separator,
    console,
)
from src.agents.orchestrator import get_orchestrator, OrchestratorState
from src.decision.trace import get_trace_manager

HELP_TEXT = """
[bold]Logistics AI Agent PoC — Help[/bold]

[cyan]Slash Commands:[/cyan]
  /help                Show this help message
  /switch [booking|tracking]  Manually switch active agent
  /trace [id]          Show a specific Decision Trace
  /traces              List all Decision Traces for this session
  /knowledge           Show Knowledge Store stats
  /log [DEBUG|INFO|WARN]  Change log level
  /export              Export session traces to file
  /exit                Exit the session

[cyan]Example queries:[/cyan]
  "부산에서 LA로 40HC 1대 부킹해줘. 4월 초 출항 희망."
  "HDMU1234567 컨테이너 지금 어디있어?"
  "BK-2026-001 부킹 화물 추적해줘. 이상 있으면 알려줘."
"""


class LogisticsAgentCLI:
    def __init__(self):
        self.session_id = f"sess-{uuid.uuid4().hex[:8]}"
        self.active_agent: Literal["booking", "tracking"] = "booking"
        self.conversation_history: list = []
        self.shared_context: dict = {}
        self.trace_manager = get_trace_manager()
        self.orchestrator = get_orchestrator()

        # Setup logging
        log_module.setup_session_logging(self.session_id)

    def run(self) -> None:
        render_header(self.session_id, self.active_agent)
        console.print(f"[dim]Session: {self.session_id} | Type /help for commands[/dim]\n")

        while True:
            try:
                user_input = Prompt.ask("[bold green]You[/bold green]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                if not self._handle_command(user_input):
                    break
                continue

            self._handle_chat(user_input)

    def _handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns False to exit."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if command == "/exit":
            console.print("[dim]Session ended. Goodbye![/dim]")
            return False

        elif command == "/help":
            console.print(HELP_TEXT)

        elif command == "/switch":
            if arg in ("booking", "tracking"):
                self.active_agent = arg  # type: ignore
                render_header(self.session_id, self.active_agent)
                console.print(f"[green]Switched to {self.active_agent} agent.[/green]")
            else:
                console.print("[yellow]Usage: /switch [booking|tracking][/yellow]")

        elif command == "/trace":
            if not arg:
                console.print("[yellow]Usage: /trace [trace_id][/yellow]")
            else:
                trace = self.trace_manager.get_trace(arg)
                if trace:
                    render_decision_trace(trace)
                else:
                    console.print(f"[red]Trace not found: {arg}[/red]")

        elif command == "/traces":
            traces = self.trace_manager.list_traces(session_id=self.session_id)
            if traces:
                render_traces_table(traces)
            else:
                console.print("[dim]No traces for this session yet.[/dim]")

        elif command == "/knowledge":
            try:
                from src.knowledge.rag import get_rag_pipeline
                rag = get_rag_pipeline()
                stats = {}
                for ktype in ("structured", "unstructured", "tribal"):
                    col = rag._stores.get(ktype)  # type: ignore
                    count = col._collection.count() if col else 0
                    stats[ktype] = count
                render_knowledge_summary(stats)
            except Exception as e:
                console.print(f"[yellow]Knowledge store not initialized: {e}[/yellow]")
                console.print("[dim]Run: python -c \"from src.knowledge.rag import RAGPipeline; RAGPipeline().ingest_all()\"[/dim]")

        elif command == "/log":
            level = arg.upper() if arg else "INFO"
            log_module.set_log_level(level)
            console.print(f"[green]Log level set to {level}[/green]")

        elif command == "/export":
            path = self.trace_manager.export()
            console.print(f"[green]Traces exported to: {path}[/green]")

        else:
            console.print(f"[yellow]Unknown command: {command}. Type /help for help.[/yellow]")

        return True

    def _handle_chat(self, user_input: str) -> None:
        """Process a user message through the orchestrator."""
        log_module.log("INFO", "system", "chat", f"User input: {user_input[:80]}")
        render_separator()

        try:
            state: OrchestratorState = {
                "messages": [HumanMessage(content=user_input)],
                "current_agent": self.active_agent,
                "shared_context": self.shared_context,
                "handoff_payload": None,
                "session_id": self.session_id,
                "conversation_history": self.conversation_history,
            }

            result = self.orchestrator.invoke(state, {"recursion_limit": config.agent_max_iterations})

            # Update active agent based on routing result
            routed_agent = result.get("current_agent", self.active_agent)
            if routed_agent in ("booking", "tracking"):
                self.active_agent = routed_agent

            # Update shared context and history
            if result.get("shared_context"):
                self.shared_context.update(result["shared_context"])
            if result.get("conversation_history"):
                self.conversation_history = result["conversation_history"]

            # Handle handoff
            if result.get("handoff_payload"):
                hp = result["handoff_payload"]
                self.active_agent = "tracking"
                log_module.log("INFO", "orchestrator", "handoff",
                               f"Handoff → tracking | booking_id={hp.get('booking_id')}")

            # Extract and render response
            last_ai = None
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage):
                    last_ai = msg
                    break

            render_separator()
            if last_ai:
                render_agent_response(last_ai.content, self.active_agent)
            else:
                console.print("[dim]No response from agent.[/dim]")

        except Exception as e:
            log_module.log("ERROR", "system", "chat", f"Error: {e}")
            console.print(f"[red]Error processing request: {e}[/red]")

        render_separator()


def main() -> None:
    cli = LogisticsAgentCLI()
    cli.run()


if __name__ == "__main__":
    main()
