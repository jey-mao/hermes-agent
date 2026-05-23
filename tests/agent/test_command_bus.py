#!/usr/bin/env python3
"""
test_command_bus.py
===================
Verifies the CommandBus and AgentCommand classes in agent_command.py.

Coverage:
  - AgentCommand: to_json / from_json round-trip
  - CommandBus: enqueue / dequeue_all (FIFO)
  - CommandBus: peek (non-destructive)
  - CommandBus: wait_for_command (blocking with timeout)
  - CommandBus: is_empty / size
  - CommandBus: clear
  - Global registry: get_command_bus / drop_command_bus per session
  - Global registry: has_pending_commands

Run: pytest tests/agent/test_command_bus.py -v
"""

from __future__ import annotations

import json
import threading
import time
import pytest
from agent_command import (
    AgentCommand,
    CommandType,
    CommandBus,
    get_command_bus,
    drop_command_bus,
    has_pending_commands,
)


# ---------------------------------------------------------------------------
# AgentCommand round-trip
# ---------------------------------------------------------------------------

class TestAgentCommand:
    def test_to_json_and_back_all_fields(self):
        """All fields survive the JSON round-trip."""
        original = AgentCommand(
            type=CommandType.GOTO,
            goto="my_phase",
            update={"messages": [{"role": "user", "content": "hi"}]},
            resume_at=42,
            reason="test reason",
            sender="test_sender",
        )
        json_str = original.to_json()
        restored = AgentCommand.from_json(json_str)

        assert restored.type == CommandType.GOTO
        assert restored.goto == "my_phase"
        assert restored.update == {"messages": [{"role": "user", "content": "hi"}]}
        assert restored.resume_at == 42
        assert restored.reason == "test reason"
        assert restored.sender == "test_sender"

    def test_to_json_and_back_minimal(self):
        """Only required 'type' field — others use defaults."""
        cmd = AgentCommand(type=CommandType.INTERRUPT)
        json_str = cmd.to_json()
        restored = AgentCommand.from_json(json_str)

        assert restored.type == CommandType.INTERRUPT
        assert restored.goto is None
        assert restored.update is None
        assert restored.resume_at is None
        assert restored.reason == ""
        assert restored.sender == "external"

    def test_from_json_dict(self):
        """from_json accepts a dict, not just a string."""
        data = {"type": "stop", "reason": "user quit"}
        cmd = AgentCommand.from_json(data)

        assert cmd.type == CommandType.STOP
        assert cmd.reason == "user quit"

    def test_message_command_type(self):
        """CommandType.MESSAGE exists and serialises correctly."""
        cmd = AgentCommand(type=CommandType.MESSAGE)
        json_str = cmd.to_json()
        restored = AgentCommand.from_json(json_str)

        assert restored.type == CommandType.MESSAGE


# ---------------------------------------------------------------------------
# CommandBus FIFO
# ---------------------------------------------------------------------------

class TestCommandBusFIFO:
    def test_enqueue_dequeue_returns_items(self):
        """Items dequeued in FIFO order."""
        bus = CommandBus()
        bus.enqueue(AgentCommand(type=CommandType.INTERRUPT, reason="first"))
        bus.enqueue(AgentCommand(type=CommandType.RESUME, reason="second"))

        dequeued = bus.dequeue_all()

        assert len(dequeued) == 2
        assert dequeued[0].reason == "first"
        assert dequeued[1].reason == "second"

    def test_dequeue_all_drains_queue(self):
        """Calling dequeue_all clears the queue."""
        bus = CommandBus()
        bus.enqueue(AgentCommand(type=CommandType.STOP))
        first = bus.dequeue_all()
        second = bus.dequeue_all()

        assert len(first) == 1
        assert len(second) == 0

    def test_enqueue_from_multiple_threads(self):
        """Thread-safe: enqueue from many threads, all items recovered."""
        bus = CommandBus()
        errors = []

        def worker(idx):
            try:
                for j in range(20):
                    bus.enqueue(AgentCommand(type=CommandType.MESSAGE, reason=f"{idx}-{j}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        all_items = bus.dequeue_all()
        assert len(all_items) == 100  # 5 threads × 20 items


# ---------------------------------------------------------------------------
# CommandBus peek / is_empty / size
# ---------------------------------------------------------------------------

class TestCommandBusObservers:
    def test_peek_does_not_consume(self):
        """peek returns items but leaves them in the queue."""
        bus = CommandBus()
        bus.enqueue(AgentCommand(type=CommandType.GOTO, goto="phase1"))
        bus.enqueue(AgentCommand(type=CommandType.GOTO, goto="phase2"))

        snapshot1 = bus.peek()
        snapshot2 = bus.peek()

        assert len(snapshot1) == 2
        assert len(snapshot2) == 2
        # Queue still has them
        assert bus.size() == 2

    def test_is_empty_after_clear(self):
        """is_empty returns True after clear()."""
        bus = CommandBus()
        bus.enqueue(AgentCommand(type=CommandType.STOP))
        assert bus.is_empty() is False
        bus.clear()
        assert bus.is_empty() is True

    def test_size_reflects_enqueued_count(self):
        """size() matches number of enqueued items."""
        bus = CommandBus()
        assert bus.size() == 0
        bus.enqueue(AgentCommand(type=CommandType.INTERRUPT))
        assert bus.size() == 1
        bus.enqueue(AgentCommand(type=CommandType.INTERRUPT))
        assert bus.size() == 2
        bus.dequeue_all()
        assert bus.size() == 0


# ---------------------------------------------------------------------------
# CommandBus wait_for_command
# ---------------------------------------------------------------------------

class TestCommandBusWait:
    def test_wait_returns_immediately_when_items_present(self):
        """wait_for_command with items already present returns without blocking."""
        bus = CommandBus()
        bus.enqueue(AgentCommand(type=CommandType.RESUME))

        start = time.time()
        result = bus.wait_for_command(timeout=2.0)
        elapsed = time.time() - start

        assert len(result) == 1
        assert elapsed < 0.5  # should not have waited

    def test_wait_times_out_when_empty(self):
        """wait_for_command returns empty list after timeout."""
        bus = CommandBus()

        start = time.time()
        result = bus.wait_for_command(timeout=0.1)
        elapsed = time.time() - start

        assert len(result) == 0
        assert elapsed >= 0.09  # must have waited at least the timeout

    def test_wait_wakes_on_enqueue(self):
        """wait_for_command returns when a new item is enqueued."""
        bus = CommandBus()
        result_holder = [None]

        def waiter():
            result_holder[0] = bus.wait_for_command(timeout=2.0)

        t = threading.Thread(target=waiter)
        t.start()

        # Enqueue after a short delay
        time.sleep(0.05)
        bus.enqueue(AgentCommand(type=CommandType.INTERRUPT, reason="wakeup"))

        t.join(timeout=1.0)
        assert result_holder[0] is not None
        assert len(result_holder[0]) == 1
        assert result_holder[0][0].reason == "wakeup"


# ---------------------------------------------------------------------------
# Global registry (per-session)
# ---------------------------------------------------------------------------

class TestGlobalRegistry:
    def test_get_command_bus_creates_per_session(self):
        """get_command_bus returns a unique bus per session_id."""
        bus_a = get_command_bus("session_A")
        bus_b = get_command_bus("session_B")
        bus_a2 = get_command_bus("session_A")

        assert bus_a is bus_a2       # same session → same bus
        assert bus_a is not bus_b   # different session → different bus

    def test_drop_command_bus_removes_it(self):
        """drop_command_bus removes the bus; next call creates a new one."""
        sid = "session_drop_test"
        bus1 = get_command_bus(sid)
        drop_command_bus(sid)
        bus2 = get_command_bus(sid)

        assert bus1 is not bus2  # new instance after drop

    def test_has_pending_commands(self):
        """has_pending_commands returns True when items are enqueued."""
        sid = "session_pending_test"
        bus = get_command_bus(sid)

        assert has_pending_commands(sid) is False
        bus.enqueue(AgentCommand(type=CommandType.STOP))
        assert has_pending_commands(sid) is True
        bus.dequeue_all()
        assert has_pending_commands(sid) is False

    def test_different_sessions_isolated(self):
        """Items in one session's bus don't affect another session's check."""
        sid1, sid2 = "session_iso_1", "session_iso_2"
        get_command_bus(sid1).enqueue(AgentCommand(type=CommandType.STOP))

        assert has_pending_commands(sid1) is True
        assert has_pending_commands(sid2) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])