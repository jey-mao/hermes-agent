#!/usr/bin/env python3
"""
AgentCommand — LangGraph Command protocol for Hermes external control.

Provides a structured way for external callers (Gateway, TUI, ACP adapter)
to inject directives into a running AIAgent:
  - INTERRUPT : pause agent loop, preserve state for resume
  - RESUME    : continue from last interrupt point
  - GOTO      : jump to a named phase/stage
  - UPDATE    : inject state modifications (e.g. messages, context)
  - STOP      : terminate cleanly

Usage:
    from agent_command import AgentCommand, CommandType, CommandBus

    # External caller (Gateway/TUI) sends a command:
    bus = get_command_bus(session_id)
    bus.enqueue(AgentCommand(type=CommandType.INTERRUPT, reason="user_requested"))

    # Inside run_agent.py, the main loop reads and processes:
    for cmd in self._command_bus.dequeue_all():
        if cmd.type == CommandType.INTERRUPT:
            self._interrupt_requested = True
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional, Dict


class CommandType(Enum):
    """All supported external control directives."""
    INTERRUPT = "interrupt"   # Pause the agent loop, preserve state
    RESUME    = "resume"      # Continue from last interrupt point
    GOTO      = "goto"        # Jump to a named phase/stage
    UPDATE    = "update"      # Inject state modifications
    STOP      = "stop"       # Terminate cleanly
    MESSAGE   = "message"    # Step 5: inject a message dict via inbox queue


@dataclass
class AgentCommand:
    """A single command injectable from external callers.

    Attributes:
        type:      Which directive to execute.
        goto:      Named phase/stage to jump to (GOTO only).
        update:    State patches to merge (UPDATE only). e.g. {"messages": [...]} injects messages.
        resume_at: Turn ID to resume from (RESUME only).
        reason:    Human-readable reason for the command.
        sender:    Who sent this command ("gateway", "tui", "acp", "external").
    """
    type: CommandType
    goto: Optional[str] = None
    update: Optional[Dict[str, Any]] = None
    resume_at: Optional[int] = None
    reason: str = ""
    sender: str = "external"

    def to_json(self) -> str:
        _d = asdict(self)
        _d["type"] = _d["type"].value  # CommandType enum → str for JSON
        return json.dumps(_d, ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str | Dict[str, Any]) -> AgentCommand:
        if isinstance(data, str):
            data = json.loads(data)
        data = dict(data)
        data["type"] = CommandType(data.get("type", "interrupt"))  # str → CommandType
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CommandBus:
    """Thread-safe command queue.

    External callers write commands here; the agent loop reads and processes
    them on each iteration. Mimics LangGraph's Command injection model
    without requiring LangGraph itself.

    Thread-safe: all operations are guarded by an RLock.
    """

    def __init__(self) -> None:
        self._queue: list[AgentCommand] = []
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)

    def enqueue(self, cmd: AgentCommand) -> None:
        """Add a command. Wakes any thread waiting on wait_for_command."""
        with self._lock:
            self._queue.append(cmd)
            self._not_empty.notify()

    def dequeue_all(self) -> list[AgentCommand]:
        """Atomically drain and return all pending commands."""
        with self._lock:
            cmds = list(self._queue)
            self._queue.clear()
            return cmds

    def peek(self) -> list[AgentCommand]:
        """Return a snapshot of pending commands without consuming them."""
        with self._lock:
            return list(self._queue)

    def wait_for_command(self, timeout: float = 0.1) -> list[AgentCommand]:
        """Block up to *timeout* seconds, then return all pending commands."""
        with self._not_empty:
            self._not_empty.wait_for(lambda: bool(self._queue), timeout=timeout)
            return self.dequeue_all()

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    def size(self) -> int:
        """Return the number of pending commands (non-blocking snapshot)."""
        with self._lock:
            return len(self._queue)

    def clear(self) -> None:
        """Drop all pending commands."""
        with self._lock:
            self._queue.clear()


# ---------------------------------------------------------------------------
# Global command bus registry (keyed by session_id)
# ---------------------------------------------------------------------------

_command_buses: Dict[str, CommandBus] = {}
_command_bus_lock = threading.RLock()


def get_command_bus(session_id: str) -> CommandBus:
    """Return (creating if needed) the CommandBus for *session_id*."""
    with _command_bus_lock:
        if session_id not in _command_buses:
            _command_buses[session_id] = CommandBus()
        return _command_buses[session_id]


def drop_command_bus(session_id: str) -> None:
    """Remove the CommandBus for *session_id*. Call on session teardown."""
    with _command_bus_lock:
        _command_buses.pop(session_id, None)


def has_pending_commands(session_id: str) -> bool:
    """Return True if there are unprocessed commands for *session_id*."""
    with _command_bus_lock:
        bus = _command_buses.get(session_id)
        return bool(bus and not bus.is_empty())
