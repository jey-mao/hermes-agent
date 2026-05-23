#!/usr/bin/env python3
"""
AgentRegistry — AutoGen-style multi-agent registry for Hermes.

Provides stable `HermesAgentId` addresses for AIAgent instances, enabling
external callers (Gateway, TUI, ACP adapter) to send messages or commands
to a specific agent without holding a direct Python object reference.

AutoGen comparison:
  AutoGen            →  Hermes
  AgentId            →  HermesAgentId  (frozen dataclass)
  Runtime            →  AgentRegistry  (singleton)
  runtime.send()     →  AgentRegistry.send_message()
  runtime.register() →  AgentRegistry.register()

Usage:
    from agent_registry import AgentRegistry, HermesAgentId

    # Register an agent
    aid = AgentRegistry().register(agent=my_agent, name="supervisor", session_id="sess_123")

    # Later, resolve and send a message
    agent = AgentRegistry().resolve(aid)
    if agent:
        agent._inbox.put({"role": "user", "content": "hello"})

    # Or via command injection
    from agent_command import AgentCommand, CommandType
    AgentRegistry().send_command(aid, AgentCommand(type=CommandType.INTERRUPT, reason="pause"))
"""

from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, List

from agent_command import AgentCommand, CommandBus, get_command_bus


@dataclass(frozen=True)
class HermesAgentId:
    """Stable address for a Hermes agent instance.

    Comparable to AutoGen's AgentId. Use as dict key, hash target, or string
    representation for logging/APIs.

    Format: "name@session_id#instance_id"
    Example: "supervisor@sess_abc123#7f3a9c2d"
    """
    name: str                       # Logical role: "supervisor", "browser_agent", "default"
    session_id: str                 # Session isolation boundary
    instance_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __str__(self) -> str:
        return f"{self.name}@{self.session_id}#{self.instance_id}"

    @property
    def short_id(self) -> str:
        """Short display form: name@session_id (ignores instance_id)."""
        return f"{self.name}@{self.session_id}"


class AgentRegistry:
    """Thread-safe global registry of AIAgent instances.

    Acts as the AutoGen Runtime for Hermes. External callers resolve
    HermesAgentId → AIAgent instance through here.

    Singleton: use ``AgentRegistry()`` directly (creates on first call).
    """

    _instance: Optional[AgentRegistry] = None
    _lock = threading.RLock()

    def __new__(cls) -> AgentRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._agents: Dict[HermesAgentId, Any] = {}
                    cls._instance._inbox_queues: Dict[HermesAgentId, queue.Queue] = {}
                    cls._instance._agent_locks: Dict[HermesAgentId, threading.RLock] = {}
        return cls._instance

    def register(
        self,
        agent: Any,
        name: str,
        session_id: str,
    ) -> HermesAgentId:
        """Register an AIAgent and return its HermesAgentId.

        If an agent with the same (name, session_id) is already registered,
        a new instance_id is generated (allowing multiple agents with the
        same logical name in the same session).

        Also attaches a per-agent inbox Queue and CommandBus.
        """
        agent_id = HermesAgentId(name=name, session_id=session_id)
        with self._lock:
            self._agents[agent_id] = agent
            self._inbox_queues[agent_id] = queue.Queue()
            self._agent_locks[agent_id] = threading.RLock()

        # Attach routing metadata to the agent itself
        agent._agent_id = agent_id          # type: ignore[attr-defined]
        agent._inbox = self._inbox_queues[agent_id]  # type: ignore[attr-defined]

        # Attach per-session command bus
        agent._command_bus = get_command_bus(session_id)  # type: ignore[attr-defined]

        return agent_id

    def resolve(self, agent_id: HermesAgentId) -> Optional[Any]:
        """Look up an agent by exact HermesAgentId. Returns None if not found."""
        with self._lock:
            return self._agents.get(agent_id)

    def resolve_by_name(self, name: str, session_id: str) -> Optional[Any]:
        """Find any agent matching name+session_id (ignores instance_id).

        If multiple agents match, returns the first registered.
        """
        with self._lock:
            for aid in self._agents:
                if aid.name == name and aid.session_id == session_id:
                    return self._agents[aid]
        return None

    def resolve_all(self, name: str, session_id: str) -> List[Any]:
        """Find all agents matching name+session_id."""
        with self._lock:
            return [
                self._agents[aid]
                for aid in self._agents
                if aid.name == name and aid.session_id == session_id
            ]

    def list_agents(self) -> List[HermesAgentId]:
        """Return all registered agent IDs."""
        with self._lock:
            return list(self._agents.keys())

    def list_agents_by_session(self, session_id: str) -> List[HermesAgentId]:
        """Return all registered agent IDs for a given session."""
        with self._lock:
            return [
                aid for aid in self._agents
                if aid.session_id == session_id
            ]

    def unregister(self, agent_id: HermesAgentId) -> bool:
        """Remove an agent from the registry. Returns True if it was present."""
        with self._lock:
            if agent_id not in self._agents:
                return False
            del self._agents[agent_id]
            self._inbox_queues.pop(agent_id, None)
            self._agent_locks.pop(agent_id, None)
        # Clean up agent routing metadata
        agent = None
        with self._lock:
            if agent_id in self._agents:  # re-check after lock
                agent = self._agents[agent_id]
        if agent is not None:
            agent._agent_id = None       # type: ignore[attr-defined]
            agent._inbox = queue.Queue()  # type: ignore[attr-defined]
        return True

    # ------------------------------------------------------------------------
    # Command injection
    # ------------------------------------------------------------------------

    def send_command(
        self,
        target: HermesAgentId,
        command: AgentCommand,
    ) -> bool:
        """Inject a command into a specific agent's command bus.

        Returns True if the agent was found (command was enqueued),
        False if the agent is not registered.
        """
        agent = self.resolve(target)
        if not agent:
            return False
        bus = getattr(agent, "_command_bus", None)
        if bus:
            bus.enqueue(command)
        return True

    def broadcast(
        self,
        command: AgentCommand,
        predicate: Optional[Callable[[HermesAgentId], bool]] = None,
    ) -> int:
        """Send a command to all agents matching *predicate*.

        If predicate is None, sends to all registered agents.

        Returns the number of agents that received the command.
        """
        with self._lock:
            targets = [
                aid for aid in self._agents
                if predicate is None or predicate(aid)
            ]
        count = 0
        for target in targets:
            if self.send_command(target, command):
                count += 1
        return count

    # ------------------------------------------------------------------------
    # Message routing (for Step 5 inbox-based routing)
    # ------------------------------------------------------------------------

    def send_message(
        self,
        target: HermesAgentId,
        message: Dict[str, Any],
        block: bool = False,
        timeout: Optional[float] = None,
    ) -> bool:
        """Send a message to an agent's inbox.

        Args:
            target:   HermesAgentId to route to.
            message:  Message dict (e.g. {"role": "user", "content": "..."}).
            block:    Whether to block if the inbox is full.
            timeout:  Max seconds to block (ignored if block=False).

        Returns:
            True if the message was enqueued, False if the agent was not found
            or the queue was full (non-blocking mode).
        """
        inbox = None
        with self._lock:
            inbox = self._inbox_queues.get(target)
        if inbox is None:
            return False
        try:
            inbox.put(message, block=block, timeout=timeout)
            return True
        except queue.Full:
            return False

    def get_inbox_size(self, target: HermesAgentId) -> int:
        """Return the number of messages waiting in an agent's inbox."""
        with self._lock:
            inbox = self._inbox_queues.get(target)
        if inbox is None:
            return 0
        return inbox.qsize()

    def drain_inbox(self, target: HermesAgentId) -> List[Dict[str, Any]]:
        """Atomically drain all messages from an agent's inbox."""
        with self._lock:
            inbox = self._inbox_queues.get(target)
        if inbox is None:
            return []
        messages = []
        while True:
            try:
                messages.append(inbox.get_nowait())
            except queue.Empty:
                break
        return messages
