#!/usr/bin/env python3
"""
Channel — LangGraph-style multi-agent coordination for Hermes.

Provides three channel types for agent communication:

  TopicChannel  : Pub/sub broadcast. All subscribers receive every event.
                  Use for: task dispatched, agent finished, event notifications.

  LastValueChannel : Shared KV. New subscribers receive the current value
                    immediately. Senders overwrite. Use for: task result,
                    shared state, latest artifact.

  ElephantQueueChannel : FIFO queue. Each subscriber gets its own cursor.
                         Use for: work queue, task queue, sequential processing.

Usage:
    from agent.channels import get_channel, TopicChannel, LastValueChannel

    # Get (or create) a channel for a session
    ch = get_channel("my_task_result", channel_type="lastvalue", session_id="sess_123")

    # Publish / Set
    ch.set({"status": "done", "result": [...]})          # LastValueChannel
    ch.publish({"type": "task_dispatched", "payload": {...}})  # TopicChannel

    # Subscribe
    ch.subscribe(callback=lambda event: print(event))     # Both types

    # Receive (blocking)
    event = ch.receive(timeout=30)   # TopicChannel / ElephantQueueChannel
    value = ch.get()                # LastValueChannel

    # Check
    ch.has_value()      # LastValueChannel: True if value exists
    ch.size()           # TopicChannel: number of subscribers
    ch.is_empty()       # ElephantQueueChannel: True if no events queued
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Channel Type Enum
# ---------------------------------------------------------------------------

class ChannelType(Enum):
    TOPIC = "topic"              # Pub/sub broadcast
    LASTVALUE = "lastvalue"      # Shared KV (latest value)
    ELEPHANT_QUEUE = "elephant_queue"  # FIFO per-subscriber queue


# ---------------------------------------------------------------------------
# Channel Metadata
# ---------------------------------------------------------------------------

@dataclass
class ChannelMetadata:
    """Immutable metadata for a channel."""
    name: str
    channel_type: ChannelType
    session_id: str
    created_at: float = field(default_factory=time.time)
    created_by: str = "hermes"

    @property
    def address(self) -> str:
        """Full channel address: session::name"""
        return f"{self.session_id}::{self.name}"


# ---------------------------------------------------------------------------
# Subscriber
# ---------------------------------------------------------------------------

@dataclass
class Subscriber:
    """A callback subscription."""
    subscriber_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    callback: Optional[Callable] = None
    # For ElephantQueueChannel: each subscriber gets their own Queue
    queue: Optional[queue.Queue] = None
    created_at: float = field(default_factory=time.time)
    # For TopicChannel: next read position in _events (0-indexed).
    # Enables replay of missed events to late subscribers via subscribe(include_last=True).
    _position: int = 0


# ---------------------------------------------------------------------------
# Base Channel
# ---------------------------------------------------------------------------

class Channel:
    """Abstract base for all channel types."""

    def __init__(
        self,
        name: str,
        channel_type: ChannelType,
        session_id: str,
        max_size: int = 1000,
    ):
        self.meta = ChannelMetadata(
            name=name,
            channel_type=channel_type,
            session_id=session_id,
        )
        self._max_size = max_size
        self._lock = threading.RLock()
        self._subscribers: Dict[str, Subscriber] = {}  # subscriber_id → Subscriber
        self._not_empty = threading.Condition(self._lock)

    # ---- Properties ----

    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def channel_type(self) -> ChannelType:
        return self.meta.channel_type

    @property
    def session_id(self) -> str:
        return self.meta.session_id

    @property
    def address(self) -> str:
        return self.meta.address

    # ---- Subscriber management ----

    def subscribe(
        self,
        callback: Optional[Callable] = None,
    ) -> str:
        """Subscribe to this channel. Returns subscriber_id."""
        with self._lock:
            sub = Subscriber(callback=callback)
            self._subscribers[sub.subscriber_id] = sub
            self._not_empty.notify_all()
            return sub.subscriber_id

    def unsubscribe(self, subscriber_id: str) -> bool:
        """Remove a subscriber. Returns True if found."""
        with self._lock:
            if subscriber_id in self._subscribers:
                del self._subscribers[subscriber_id]
                return True
            return False

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def get_subscribers(self) -> List[str]:
        """Return list of subscriber_ids."""
        with self._lock:
            return list(self._subscribers.keys())

    # ---- Abstract methods to override ----

    def publish(self, event: Any) -> int:
        """Publish an event. Returns number of subscribers notified."""
        raise NotImplementedError

    def receive(self, timeout: float = 0.1) -> Optional[Any]:
        """Block and receive next event. Returns None on timeout."""
        raise NotImplementedError

    def get(self) -> Optional[Any]:
        """Get current value (LastValueChannel only)."""
        raise NotImplementedError

    def set(self, value: Any) -> None:
        """Set current value (LastValueChannel only)."""
        raise NotImplementedError

    def has_value(self) -> bool:
        """Check if channel has a value (LastValueChannel only)."""
        raise NotImplementedError

    def is_empty(self) -> bool:
        """Check if channel is empty (TopicChannel: no events, ElephantQueue: no queued)."""
        raise NotImplementedError

    def size(self) -> int:
        """Number of pending items (queue depth) or 1 if LastValue."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all pending events/values."""
        raise NotImplementedError

    # ---- Internals ----

    def _notify_subscribers(self, event: Any) -> int:
        """Call each subscriber's callback. Returns count of successful calls."""
        notified = 0
        for sub in list(self._subscribers.values()):
            if sub.callback is not None:
                try:
                    sub.callback(event)
                    notified += 1
                except Exception:
                    pass  # Don't let one subscriber crash others
        return notified

    def _prune_stale_subscribers(self) -> None:
        """Remove subscribers with dead (None) callbacks."""
        stale = [sid for sid, sub in self._subscribers.items() if sub.callback is None]
        for sid in stale:
            del self._subscribers[sid]


# ---------------------------------------------------------------------------
# TopicChannel — Pub/Sub Broadcast
# ---------------------------------------------------------------------------

class TopicChannel(Channel):
    """Pub/sub broadcast channel.

    All subscribers receive every published event.
    Each subscriber has their own internal queue (max_size deep).

    Use for: task_dispatched, agent_finished, system_events, notifications.
    """

    def __init__(
        self,
        name: str,
        session_id: str,
        max_size: int = 1000,
    ):
        super().__init__(
            name=name,
            channel_type=ChannelType.TOPIC,
            session_id=session_id,
            max_size=max_size,
        )
        self._events: queue.Queue = queue.Queue(maxsize=max_size)
        self._last_event: Optional[Any] = None  # For new subscribers

    def publish(self, event: Any) -> int:
        """Publish an event to all subscribers. Returns number notified."""
        with self._lock:
            # Store last event for late subscribers
            self._last_event = event

            # Enqueue in shared _events ONLY if there are no active subscribers.
            # This is the "catch-up window" for late subscribers: when _events is non-empty,
            # receive() reads from it first. Once subscribers exist, new events go
            # directly into their queues and _events drains naturally.
            if not self._subscribers:
                try:
                    self._events.put_nowait(event)
                except queue.Full:
                    try:
                        self._events.get_nowait()
                        self._events.put_nowait(event)
                    except queue.Empty:
                        pass

            # Enqueue for all subscriber queues (created on demand)
            for sub in list(self._subscribers.values()):
                if sub.queue is None:
                    sub.queue = queue.Queue(maxsize=self._max_size)
                try:
                    sub.queue.put_nowait(event)
                except queue.Full:
                    # Drop oldest to make room
                    try:
                        sub.queue.get_nowait()
                        sub.queue.put_nowait(event)
                    except queue.Empty:
                        pass
            self._not_empty.notify_all()

            # Call sync callbacks immediately
            notified = self._notify_subscribers(event)
            return notified

    def receive(self, timeout: float = 0.1) -> Optional[Any]:
        """Block and receive the next event.

        Priority order:
          1. Historical events in _events (replayed via subscribe(include_last=True))
          2. New events in subscriber queues (arrived after subscribe)

        This ordering is essential: when a Channel Observer calls subscribe(include_last=True),
        all historical events are restored to _events. receive() should consume those first.
        After _events is drained, receive() switches to subscriber queues for new events.

        For Channel Observer use: each Hermes turn calls subscribe() with a unique
        consumer_id, then drains via receive(). The subscriber is unsubscribed after draining.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None

            with self._lock:
                # Priority 1: historical events in _events
                if not self._events.empty():
                    try:
                        return self._events.get_nowait()
                    except queue.Empty:
                        pass

                # Priority 2: new events in subscriber queues
                for sid, sub in self._subscribers.items():
                    if sub.queue is not None and not sub.queue.empty():
                        try:
                            return sub.queue.get_nowait()
                        except queue.Empty:
                            pass

                # Wait a short interval before retrying
                wait_time = min(0.05, remaining)
            time.sleep(wait_time)

        return None

    def subscribe(
        self,
        callback: Optional[Callable] = None,
        include_last: bool = False,
        consumer_id: Optional[str] = None,
    ) -> str:
        """Subscribe. If include_last=True, replay all historical events to the subscriber.

        This is the mechanism that enables "late subscriber" catching up:
        when a new consumer (e.g. Hermes Channel Observer) subscribes after
        events have already been published, include_last=True replays every
        historical event so the subscriber can consume them via receive().

        The replay strategy:
          - Historical events are put into the shared _events queue.
          - The subscriber's queue only receives NEW events going forward.
          - receive() checks _events first (for historical), then subscriber queue
            (for new events that arrive while subscribed).

        Args:
            callback: Sync callback invoked on each event (optional).
            include_last: If True, replay all historical _events via receive().
                Also calls the callback for each historical event if provided.
            consumer_id: Optional explicit consumer_id (must be unique). If None,
                a random id is generated.
        """
        with self._lock:
            sub = Subscriber(callback=callback)
            if consumer_id:
                sub.subscriber_id = consumer_id
            self._subscribers[sub.subscriber_id] = sub

            if include_last:
                # Drain all historical events from _events into a replay list.
                # These will be restored to _events so receive() can consume them
                # (receive() waits on _events, not subscriber queue).
                replay_events = []
                while True:
                    try:
                        replay_events.append(self._events.get_nowait())
                    except queue.Empty:
                        break

                # Restore historical events to _events so receive() sees them.
                # (This also sets up the _events condition for wait_for.)
                for ev in replay_events:
                    try:
                        self._events.put_nowait(ev)
                    except queue.Full:
                        # Should not happen if historical count < max_size
                        try:
                            self._events.get_nowait()
                            self._events.put_nowait(ev)
                        except queue.Empty:
                            pass

                # Set subscriber's position to the number of historical events
                # so the next publish() correctly increments from there.
                sub._position = len(replay_events)

                # Call callbacks for each historical event (if provided)
                if callback is not None:
                    for ev in replay_events:
                        try:
                            callback(ev)
                        except Exception:
                            pass
                    # Also call for the last event
                    if self._last_event is not None:
                        try:
                            callback(self._last_event)
                        except Exception:
                            pass

            self._not_empty.notify_all()
            return sub.subscriber_id

    def get(self) -> Optional[Any]:
        return self._last_event

    def set(self, value: Any) -> None:
        self.publish(value)

    def has_value(self) -> bool:
        return self._last_event is not None

    def is_empty(self) -> bool:
        return self._last_event is None

    def size(self) -> int:
        """Number of subscribers (not events — topic uses broadcast)."""
        with self._lock:
            return len(self._subscribers)

    def clear(self) -> None:
        with self._lock:
            self._last_event = None
            while not self._events.empty():
                try:
                    self._events.get_nowait()
                except queue.Empty:
                    break


# ---------------------------------------------------------------------------
# LastValueChannel — Shared KV (Latest Value)
# ---------------------------------------------------------------------------

class LastValueChannel(Channel):
    """Shared key-value store that always reflects the latest value.

    New subscribers immediately receive the current value.
    Senders overwrite. Use for: task_result, shared_artifact,
    latest_checkpoint, current_state.
    """

    def __init__(
        self,
        name: str,
        session_id: str,
        ttl_seconds: Optional[float] = None,
    ):
        super().__init__(
            name=name,
            channel_type=ChannelType.LASTVALUE,
            session_id=session_id,
        )
        self._value: Optional[Any] = None
        self._value_timestamp: Optional[float] = None
        self._ttl_seconds = ttl_seconds
        self._version: int = 0

    def publish(self, event: Any) -> int:
        """Publish (set) a new value. Alias for set()."""
        return self.set(event)

    def set(self, value: Any) -> int:
        """Set the current value. Returns number of subscribers notified."""
        with self._lock:
            self._value = value
            self._value_timestamp = time.time()
            self._version += 1
            self._not_empty.notify_all()
            notified = self._notify_subscribers(value)
            return notified

    def get(self) -> Optional[Any]:
        """Return current value, or None if expired (TTL) or unset."""
        with self._lock:
            if self._value is None:
                return None
            if self._ttl_seconds and self._value_timestamp:
                if time.time() - self._value_timestamp > self._ttl_seconds:
                    self._value = None
                    self._value_timestamp = None
                    return None
            return self._value

    def receive(self, timeout: float = 0.1) -> Optional[Any]:
        """Block until value changes. Returns latest value on change or timeout."""
        with self._not_empty:
            start = time.time()
            initial_version = self._version

            while self._version == initial_version:
                remaining = timeout - (time.time() - start)
                if remaining <= 0:
                    return self.get()
                self._not_empty.wait(timeout=remaining)

            return self.get()

    def has_value(self) -> bool:
        return self.get() is not None

    def is_empty(self) -> bool:
        return not self.has_value()

    def size(self) -> int:
        return 1 if self.has_value() else 0

    def clear(self) -> None:
        with self._lock:
            self._value = None
            self._value_timestamp = None
            self._version += 1

    @property
    def ttl_seconds(self) -> Optional[float]:
        return self._ttl_seconds

    @property
    def age_seconds(self) -> Optional[float]:
        if self._value_timestamp is None:
            return None
        return time.time() - self._value_timestamp

    @property
    def version(self) -> int:
        return self._version


# ---------------------------------------------------------------------------
# ElephantQueueChannel — FIFO Per-Subscriber Queue
# ---------------------------------------------------------------------------

class ElephantQueueChannel(Channel):
    """FIFO queue where each subscriber gets their own cursor.

    Items are added to the queue; each subscriber processes at their own pace.
    Use for: work queue, task queue, sequential processing, load balancing.
    """

    def __init__(
        self,
        name: str,
        session_id: str,
        max_size: int = 1000,
    ):
        super().__init__(
            name=name,
            channel_type=ChannelType.ELEPHANT_QUEUE,
            session_id=session_id,
            max_size=max_size,
        )
        self._queue: queue.Queue = queue.Queue(maxsize=max_size)

    def publish(self, event: Any) -> int:
        """Add an event to the queue. Returns number of waiting subscribers (queued = True means delivered)."""
        with self._lock:
            try:
                self._queue.put_nowait(event)
                self._not_empty.notify_all()
                # For elephant queue, publishing just enqueues.
                # Each subscriber reads at their own pace.
                # We report success as having room (queued successfully).
                return 1 if not self._queue.full() else 0
            except queue.Full:
                return 0

    def set(self, value: Any) -> None:
        """Elephant queue doesn't support set — use publish() for FIFO."""
        self.publish(value)

    def enqueue(self, item: Any) -> bool:
        """Alias for publish(). Returns True if enqueued, False if queue full."""
        try:
            self._queue.put_nowait(item)
            with self._lock:
                self._not_empty.notify_all()
            return True
        except queue.Full:
            return False

    def receive(self, timeout: float = 0.1) -> Optional[Any]:
        """Block and receive the next item from the shared queue.

        NOTE: This is a SHARED queue — all subscribers compete for the same items.
        For per-subscriber queues, use subscribe() to get a subscriber_id,
        then call get_subscriber_queue(subscriber_id).get(timeout).
        """
        with self._not_empty:
            result = self._not_empty.wait_for(
                lambda: not self._queue.empty(),
                timeout=timeout,
            )
            if not result:
                return None
            try:
                return self._queue.get_nowait()
            except queue.Empty:
                return None

    def get_subscriber_queue(self, subscriber_id: str) -> queue.Queue:
        """Get the per-subscriber queue for a subscriber. Creates on demand.

        Each subscriber processes items at their own pace — no competition.
        """
        with self._lock:
            sub = self._subscribers.get(subscriber_id)
            if sub is None:
                return queue.Queue(maxsize=self._max_size)
            if sub.queue is None:
                sub.queue = queue.Queue(maxsize=self._max_size)
            return sub.queue

    def subscribe(
        self,
        callback: Optional[Callable] = None,
        own_queue: bool = True,
    ) -> str:
        """Subscribe. If own_queue=True, subscriber gets their own FIFO queue."""
        with self._lock:
            sub = Subscriber(callback=callback)
            if own_queue:
                sub.queue = queue.Queue(maxsize=self._max_size)
            self._subscribers[sub.subscriber_id] = sub
            self._not_empty.notify_all()
            return sub.subscriber_id

    def get(self) -> Optional[Any]:
        """Peek at next item without consuming (queue.front())."""
        with self._lock:
            if self._queue.empty():
                return None
            return self._queue.queue[0]  # peek without consuming

    def has_value(self) -> bool:
        return not self._queue.empty()

    def is_empty(self) -> bool:
        return self._queue.empty()

    def size(self) -> int:
        """Current queue depth."""
        return self._queue.qsize()

    def clear(self) -> None:
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break


# ---------------------------------------------------------------------------
# Channel Registry (keyed by session_id::channel_name)
# ---------------------------------------------------------------------------

_channels: Dict[str, Channel] = {}
_channel_lock = threading.RLock()


def get_channel(
    name: str,
    channel_type: str | ChannelType = "topic",
    session_id: str = "default",
    **kwargs,
) -> Channel:
    """Get or create a channel by address (session_id::name).

    channel_type: "topic" | "lastvalue" | "elephant_queue" (case-insensitive)
    """
    addr = f"{session_id}::{name}"
    with _channel_lock:
        if addr not in _channels:
            if isinstance(channel_type, str):
                ct = ChannelType(channel_type.lower().replace("elephant-queue", "elephant_queue"))
            else:
                ct = channel_type

            if ct == ChannelType.TOPIC:
                _channels[addr] = TopicChannel(name=name, session_id=session_id, **kwargs)
            elif ct == ChannelType.LASTVALUE:
                _channels[addr] = LastValueChannel(name=name, session_id=session_id, **kwargs)
            elif ct == ChannelType.ELEPHANT_QUEUE:
                _channels[addr] = ElephantQueueChannel(name=name, session_id=session_id, **kwargs)
            else:
                raise ValueError(f"Unknown channel type: {ct}")

        return _channels[addr]


def drop_channel(name: str, session_id: str = "default") -> bool:
    """Remove a channel. Returns True if found and removed."""
    addr = f"{session_id}::{name}"
    with _channel_lock:
        if addr in _channels:
            del _channels[addr]
            return True
        return False


def list_channels(session_id: Optional[str] = None) -> List[ChannelMetadata]:
    """List all channels, optionally filtered by session_id."""
    with _channel_lock:
        result = []
        for addr, ch in _channels.items():
            if session_id is None or ch.session_id == session_id:
                result.append(ch.meta)
        return result


def get_channel_addresses(session_id: Optional[str] = None) -> List[str]:
    """List all channel addresses (session_id::name)."""
    with _channel_lock:
        if session_id is None:
            return list(_channels.keys())
        return [a for a, ch in _channels.items() if ch.session_id == session_id]


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def topic(name: str, session_id: str = "default", **kwargs) -> TopicChannel:
    """Create or get a TopicChannel."""
    return get_channel(name, channel_type=ChannelType.TOPIC, session_id=session_id, **kwargs)


def lastvalue(name: str, session_id: str = "default", **kwargs) -> LastValueChannel:
    """Create or get a LastValueChannel."""
    return get_channel(name, channel_type=ChannelType.LASTVALUE, session_id=session_id, **kwargs)


def elephant_queue(name: str, session_id: str = "default", **kwargs) -> ElephantQueueChannel:
    """Create or get an ElephantQueueChannel."""
    return get_channel(name, channel_type=ChannelType.ELEPHANT_QUEUE, session_id=session_id, **kwargs)


# ---------------------------------------------------------------------------
# Integration with AgentRegistry
# ---------------------------------------------------------------------------

def publish_to_agent(
    agent_name: str,
    channel_name: str,
    event: Any,
    session_id: str = "default",
    channel_type: str = "topic",
) -> int:
    """Publish an event to a channel associated with an agent."""
    ch = get_channel(f"{agent_name}::{channel_name}", channel_type=channel_type, session_id=session_id)
    return ch.publish(event)


# ---------------------------------------------------------------------------
# Self-test (run with: python agent/channels.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import threading as t

    print("=" * 60)
    print("Channel self-test")
    print("=" * 60)

    # --- TopicChannel ---
    print("\n[TopicChannel] Broadcast test")
    ch = topic("test_topic", session_id="test_sess")
    received = []

    def callback(event):
        received.append(event)

    sid = ch.subscribe(callback=callback, include_last=False)
    print(f"  Subscribed: {sid}, count={ch.subscriber_count}")

    count = ch.publish({"type": "event", "data": "hello"})
    print(f"  Published → {count} notified, received={received}")

    ch.publish({"type": "event2"})
    print(f"  Second publish: received={received}")

    ch.unsubscribe(sid)
    print(f"  Unsubscribed: count={ch.subscriber_count}")

    # --- LastValueChannel ---
    print("\n[LastValueChannel] KV test")
    lv = lastvalue("test_lv", session_id="test_sess")
    print(f"  Initial: has_value={lv.has_value()}, value={lv.get()}")

    notified = lv.set({"result": "data123"})
    print(f"  Set: notified={notified}, value={lv.get()}")

    sid2 = lv.subscribe(callback=lambda e: print(f"  [callback] new value: {e}"))
    print(f"  Subscribed (immediate delivery): count={lv.subscriber_count}")

    lv.set({"result": "updated"})
    print(f"  Update: value={lv.get()}, version={lv.version}")

    print(f"  Age: {lv.age_seconds:.3f}s, TTL: {lv.ttl_seconds}")

    lv.clear()
    print(f"  Cleared: has_value={lv.has_value()}")

    lv.unsubscribe(sid2)

    # --- ElephantQueueChannel ---
    print("\n[ElephantQueueChannel] FIFO test")
    eq = elephant_queue("test_eq", session_id="test_sess")
    print(f"  Initial: size={eq.size()}, empty={eq.is_empty()}")

    eq.enqueue("task_1")
    eq.enqueue("task_2")
    eq.enqueue("task_3")
    print(f"  Enqueued 3: size={eq.size()}, peek={eq.get()}")

    item = eq.receive(timeout=1)
    print(f"  Received: {item}, size={eq.size()}")

    sid3 = eq.subscribe(own_queue=True)
    q = eq.get_subscriber_queue(sid3)
    print(f"  Subscriber own queue: {q}")

    # --- Cross-session isolation ---
    print("\n[Session isolation]")
    ch_a = get_channel("shared", channel_type="topic", session_id="sess_A")
    ch_b = get_channel("shared", channel_type="topic", session_id="sess_B")
    ch_a.set({"session": "A"})
    ch_b.set({"session": "B"})
    print(f"  sess_A::shared = {ch_a.get()}")
    print(f"  sess_B::shared = {ch_b.get()}")
    print(f"  Different channels: {ch_a is not ch_b}")

    # --- Channel registry ---
    print("\n[Registry]")
    print(f"  Channels: {get_channel_addresses()}")
    drop_channel("shared", session_id="sess_A")
    drop_channel("shared", session_id="sess_B")
    print(f"  After drop: {get_channel_addresses()}")

    print("\n" + "=" * 60)
    print("self-test complete")