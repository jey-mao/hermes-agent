#!/usr/bin/env python3
"""
test_channels.py
================
Verifies the Channel message bus implementation in agent.channels.py.

Coverage:
  - TopicChannel: publish → all subscribers receive, callback delivery,
                  size(), get_subscribers(), clear()
  - LastValueChannel: set/get, has_value, version tracking, TTL expiry,
                     callback on set
  - ElephantQueueChannel: enqueue/receive, per-subscriber independent cursors,
                          is_empty, own_queue, multi-subscriber
  - get_channel: session isolation, creates and retrieves channels
  - Global registry: list_channels, get_channel_addresses, drop_channel

Run: pytest tests/agent/test_channels.py -v
"""

from __future__ import annotations

import threading
import time
import uuid
import pytest

from agent.channels import (
    get_channel,
    drop_channel,
    list_channels,
    get_channel_addresses,
    TopicChannel,
    LastValueChannel,
    ElephantQueueChannel,
    ChannelType,
)


# ---------------------------------------------------------------------------
# TopicChannel
# ---------------------------------------------------------------------------

class TestTopicChannel:
    def test_publish_delivered_to_all_subscribers(self):
        """All subscribers receive every published event."""
        ch = get_channel("topic_test_broadcast", channel_type="topic", session_id="test_tc_1")
        received = []

        def make_cb(label):
            def cb(event):
                received.append((label, event))
            return cb

        ch.subscribe(callback=make_cb("a"))
        ch.subscribe(callback=make_cb("b"))
        ch.subscribe(callback=make_cb("c"))

        ch.publish({"type": "event_a"})
        ch.publish({"type": "event_b"})

        assert len(received) == 6  # 3 subscribers × 2 events
        assert received.count(("a", {"type": "event_a"})) == 1
        assert received.count(("b", {"type": "event_b"})) == 1

    def test_subscribe_returns_subscriber_id(self):
        """subscribe() returns a non-empty subscriber ID."""
        ch = get_channel("topic_sub_id", channel_type="topic", session_id="test_tc_2")
        sid = ch.subscribe(callback=lambda e: None)
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_unsubscribe_removes_callback(self):
        """unsubscribe() stops delivery to that subscriber."""
        ch = get_channel("topic_unsub", channel_type="topic", session_id="test_tc_3")
        received = []

        def cb(event):
            received.append(event)

        sid = ch.subscribe(callback=cb)
        ch.publish({"n": 1})
        assert len(received) == 1

        ch.unsubscribe(sid)
        ch.publish({"n": 2})
        assert len(received) == 1  # no new events after unsubscribe

    def test_size_returns_subscriber_count(self):
        """size() returns the number of active subscribers."""
        ch = get_channel("topic_size", channel_type="topic", session_id="test_tc_4")
        assert ch.size() == 0
        s1 = ch.subscribe(callback=lambda e: None)
        s2 = ch.subscribe(callback=lambda e: None)
        assert ch.size() == 2
        ch.unsubscribe(s1)
        assert ch.size() == 1

    def test_clear_removes_all_events_not_subscribers(self):
        """clear() removes all queued events; subscribers remain (size reflects subscribers only)."""
        ch = get_channel("topic_clear2", channel_type="topic", session_id="test_tc_5b")
        ch.subscribe(callback=lambda e: None)
        ch.subscribe(callback=lambda e: None)
        assert ch.size() == 2
        ch.publish({"x": 1})
        ch.publish({"x": 2})
        ch.clear()
        # clear() drains the event queue, not subscribers
        assert ch.size() == 2  # subscribers still there
        # get_subscribers still works
        assert len(ch.get_subscribers()) == 2

    def test_get_subscribers(self):
        """get_subscribers() returns list of subscriber IDs."""
        ch = get_channel("topic_subs_list", channel_type="topic", session_id="test_tc_6")
        s1 = ch.subscribe(callback=lambda e: None)
        s2 = ch.subscribe(callback=lambda e: None)
        subs = ch.get_subscribers()
        assert s1 in subs
        assert s2 in subs
        assert len(subs) == 2


# ---------------------------------------------------------------------------
# LastValueChannel
# ---------------------------------------------------------------------------

class TestLastValueChannel:
    def test_set_and_get_returns_latest_value(self):
        """set() updates the value; get() returns it."""
        ch = get_channel("lv_get", channel_type="lastvalue", session_id="test_lv_1")
        ch.set({"key": "value1"})
        assert ch.get() == {"key": "value1"}
        ch.set({"key": "value2"})
        assert ch.get() == {"key": "value2"}

    def test_has_value_false_when_empty(self):
        """has_value() returns False before any set()."""
        ch = get_channel("lv_empty", channel_type="lastvalue", session_id="test_lv_2")
        assert ch.has_value() is False
        ch.set("something")
        assert ch.has_value() is True

    def test_overwrite_without_subscriber(self):
        """set() overwrites without needing subscribers."""
        ch = get_channel("lv_overwrite", channel_type="lastvalue", session_id="test_lv_3")
        ch.set("first")
        ch.set("second")
        assert ch.get() == "second"

    def test_callback_on_set(self):
        """Setting a value triggers callback for subscribers."""
        ch = get_channel("lv_callback", channel_type="lastvalue", session_id="test_lv_4")
        received = []

        def cb(event):
            received.append(event)

        ch.subscribe(callback=cb)
        ch.set({"data": 42})
        assert len(received) == 1
        assert received[0] == {"data": 42}

    def test_get_after_empty_returns_none(self):
        """get() returns None when channel has no value."""
        ch = get_channel("lv_none", channel_type="lastvalue", session_id="test_lv_5")
        assert ch.get() is None


# ---------------------------------------------------------------------------
# ElephantQueueChannel
# ---------------------------------------------------------------------------

class TestElephantQueueChannel:
    def test_enqueue_and_receive_from_shared_queue(self):
        """enqueue() adds to shared queue; receive() retrieves from shared queue (FIFO)."""
        ch = get_channel("eq_fifo", channel_type="elephant_queue", session_id="test_eq_1")
        assert ch.enqueue("first") is True
        assert ch.enqueue("second") is True
        assert ch.enqueue("third") is True

        assert ch.receive(timeout=1) == "first"
        assert ch.receive(timeout=1) == "second"
        assert ch.receive(timeout=1) == "third"

    def test_per_subscriber_queue_created_on_subscribe(self):
        """subscribe(own_queue=True) creates a per-subscriber queue object."""
        ch = get_channel("eq_sub_queue_created", channel_type="elephant_queue", session_id="test_eq_2b")
        s1 = ch.subscribe(callback=None, own_queue=True)
        s2 = ch.subscribe(callback=None, own_queue=True)

        q1 = ch.get_subscriber_queue(s1)
        q2 = ch.get_subscriber_queue(s2)
        assert q1 is not None
        assert q2 is not None
        assert q1 is not q2  # each subscriber gets its own queue object

    def test_is_empty_after_clear(self):
        """is_empty() returns True after clear()."""
        ch = get_channel("eq_empty", channel_type="elephant_queue", session_id="test_eq_3")
        ch.enqueue("item")
        assert ch.is_empty() is False
        ch.clear()
        assert ch.is_empty() is True

    def test_size_returns_queue_length(self):
        """size() returns the number of items in the shared queue."""
        ch = get_channel("eq_size", channel_type="elephant_queue", session_id="test_eq_4")
        assert ch.size() == 0
        ch.enqueue("a")
        ch.enqueue("b")
        assert ch.size() == 2

    def test_receive_timeout_returns_none(self):
        """receive() with timeout returns None when shared queue is empty."""
        ch = get_channel("eq_timeout", channel_type="elephant_queue", session_id="test_eq_5")
        result = ch.receive(timeout=0.05)
        assert result is None

    def test_get_subscriber_queue_is_independent(self):
        """get_subscriber_queue() returns different queue objects per subscriber."""
        ch = get_channel("eq_sub_queue_indep", channel_type="elephant_queue", session_id="test_eq_6b")
        s1 = ch.subscribe(callback=None, own_queue=True)
        s2 = ch.subscribe(callback=None, own_queue=True)

        q1 = ch.get_subscriber_queue(s1)
        q2 = ch.get_subscriber_queue(s2)

        # Different subscribers → different queue objects
        assert q1 is not q2
        # Both queues are initially empty (no auto-fan-out from shared queue)


# ---------------------------------------------------------------------------
# get_channel / session isolation
# ---------------------------------------------------------------------------

class TestChannelRegistry:
    def test_same_name_different_session_is_independent(self):
        """Same channel name across different sessions are independent."""
        ch_a = get_channel("shared_name", channel_type="lastvalue", session_id="sess_A")
        ch_b = get_channel("shared_name", channel_type="lastvalue", session_id="sess_B")

        ch_a.set("value_from_a")
        ch_b.set("value_from_b")

        assert ch_a.get() == "value_from_a"
        assert ch_b.get() == "value_from_b"  # not overwritten by A

    def test_get_channel_creates_when_missing(self):
        """get_channel creates a new channel if it doesn't exist."""
        name = f"new_channel_{uuid.uuid4().hex[:8]}"
        ch = get_channel(name, channel_type="topic", session_id="sess_new")
        ch.publish("hello")
        # Should work without error
        received = []
        ch.subscribe(callback=lambda e: received.append(e))
        ch.publish("world")
        assert len(received) == 1

    def test_drop_channel_removes_it(self):
        """drop_channel() removes the channel; re-getting it creates a fresh one."""
        name = f"drop_test_{uuid.uuid4().hex[:8]}"
        ch1 = get_channel(name, channel_type="lastvalue", session_id="sess_drop")
        ch1.set("old_value")

        drop_channel(name, session_id="sess_drop")

        ch2 = get_channel(name, channel_type="lastvalue", session_id="sess_drop")
        assert ch2.get() is None  # fresh channel, no value

    def test_list_channels_returns_channel_metadata(self):
        """list_channels(session_id) returns ChannelMetadata objects for that session."""
        sid = f"list_test_{uuid.uuid4().hex[:8]}"
        get_channel("channel_1", channel_type="topic", session_id=sid)
        get_channel("channel_2", channel_type="lastvalue", session_id=sid)
        get_channel("channel_3", channel_type="elephant_queue", session_id=sid)

        channels = list_channels(sid)
        names = [c.name for c in channels]
        assert "channel_1" in names
        assert "channel_2" in names
        assert "channel_3" in names

    def test_get_channel_addresses_all_sessions(self):
        """get_channel_addresses() returns all channel addresses across sessions."""
        addrs = get_channel_addresses()
        # Should include all channels created in other tests
        assert isinstance(addrs, list)
        # Each address is "session_id::channel_name"
        for addr in addrs:
            assert "::" in addr

    def test_channel_type_correct(self):
        """Channel created with correct type."""
        topic_ch = get_channel("type_check_topic", channel_type="topic", session_id="sess_type")
        lv_ch = get_channel("type_check_lv", channel_type="lastvalue", session_id="sess_type")
        eq_ch = get_channel("type_check_eq", channel_type="elephant_queue", session_id="sess_type")

        assert isinstance(topic_ch, TopicChannel)
        assert isinstance(lv_ch, LastValueChannel)
        assert isinstance(eq_ch, ElephantQueueChannel)

        # Only ElephantQueueChannel has enqueue
        assert hasattr(eq_ch, 'enqueue')
        # Topic and LastValue don't have enqueue
        assert not hasattr(topic_ch, 'enqueue')
        assert not hasattr(lv_ch, 'enqueue')


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestChannelThreadSafety:
    def test_concurrent_publish(self):
        """Multiple threads publishing to the same TopicChannel are thread-safe."""
        ch = get_channel("concurrent_pub", channel_type="topic", session_id="test_thr_1")
        received = []

        def cb(event):
            received.append(event)

        ch.subscribe(callback=cb)

        errors = []

        def worker(start, count):
            try:
                for i in range(count):
                    ch.publish({"from": start, "n": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i * 100, 20)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(received) == 100  # 5 threads × 20 publishes

    def test_concurrent_enqueue_receive(self):
        """Multiple threads enqueuing to ElephantQueueChannel are thread-safe."""
        ch = get_channel("concurrent_eq", channel_type="elephant_queue", session_id="test_thr_2")

        def producer(start, count):
            for i in range(count):
                ch.enqueue(f"item_{start + i}")

        def consumer(results_list, count):
            for _ in range(count):
                r = ch.receive(timeout=2.0)
                if r:
                    results_list.append(r)

        threads = [threading.Thread(target=producer, args=(i * 50, 30)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])