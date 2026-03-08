"""Structured logging using Rich console."""
from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

from src.config import config

# Shared Rich console (stderr for logs, stdout for agent output)
console = Console(stderr=True, highlight=False)
agent_console = Console(highlight=True)

_LEVEL_COLORS = {
    "DEBUG": "dim white",
    "INFO": "cyan",
    "WARN": "yellow",
    "WARNING": "yellow",
    "ERROR": "red bold",
}

_current_log_level = config.log_level.upper()
_session_log_file: Path | None = None


def setup_session_logging(session_id: str) -> None:
    """Set up file logging for a session."""
    global _session_log_file
    log_dir = config.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    _session_log_file = log_dir / f"session_{session_id}.log"

    # Configure stdlib logging
    logging.basicConfig(
        level=getattr(logging, _current_log_level, logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(_session_log_file, encoding="utf-8"),
        ],
    )


def set_log_level(level: str) -> None:
    global _current_log_level
    _current_log_level = level.upper()
    logging.getLogger().setLevel(getattr(logging, _current_log_level, logging.INFO))


def log(
    level: str,
    agent: str,
    component: str,
    message: str,
) -> None:
    """Emit a structured log line."""
    level_upper = level.upper()
    if level_upper == "DEBUG" and _current_log_level != "DEBUG":
        return

    color = _LEVEL_COLORS.get(level_upper, "white")
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{level_upper}] [{agent}] [{component}] {message}"

    console.print(Text(line, style=color))

    # Also write to file
    logger = logging.getLogger(f"{agent}.{component}")
    log_fn = getattr(logger, level.lower(), logger.info)
    log_fn(f"[{agent}] [{component}] {message}")
