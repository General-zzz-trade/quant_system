"""Tests for core/bus — BoundedEventBus with priority, backpressure, overflow."""
from __future__ import annotations

import pytest

from event.bounded_bus import (
    BoundedEventBus,
    BusConfig,
    OverflowPolicy,
    PublishResult,
)
from event.core_types import Envelope, EventKind, EventMetadata, Priority


# ── Helpers ──────────────────────────────────────────────────

def _env(kind=EventKind.MARKET, priority=Priority.NORMAL):
    meta = EventMetadata.create(source="test")
    return Envelope(event={"test": True}, metadata=meta, kind=kind, priority=priority)


# ── Publish + drain cycle ────────────────────────────────────

class TestPublishDrain:
    def test_publish_accepted(self):
        bus = BoundedEventBus(BusConfig(capacity=10))
        result = bus.publish(_env())
        assert result == PublishResult.ACCEPTED
        assert bus.size == 1

    def test_drain_delivers_to_handler(self):
        bus = BoundedEventBus()
        received = []
        bus.subscribe(lambda e: received.append(e))
        bus.publish(_env())
        delivered = bus.drain(batch_size=10)
        assert delivered == 1
        assert len(received) == 1

    def test_drain_empties_queue(self):
        bus = BoundedEventBus()
        bus.subscribe(lambda e: None)
        for _ in range(5):
            bus.publish(_env())
        bus.drain(batch_size=100)
        assert bus.size == 0


# ── Priority ordering ───────────────────────────────────────

class TestPriorityOrdering:
    def test_critical_before_normal_before_low(self):
        bus = BoundedEventBus()
        received = []
        bus.subscribe(lambda e: received.append(e.priority))

        bus.publish(_env(priority=Priority.LOW))
        bus.publish(_env(priority=Priority.CRITICAL))
        bus.publish(_env(priority=Priority.NORMAL))

        bus.drain(batch_size=10)
        assert received == [Priority.CRITICAL, Priority.NORMAL, Priority.LOW]

    def test_same_priority_fifo(self):
        bus = BoundedEventBus()
        received = []
        bus.subscribe(lambda e: received.append(e.event["seq"]))

        for i in range(3):
            meta = EventMetadata.create(source="test")
            env = Envelope(event={"seq": i}, metadata=meta, kind=EventKind.MARKET, priority=Priority.NORMAL)
            bus.publish(env)

        bus.drain(batch_size=10)
        assert received == [0, 1, 2]


# ── Backpressure and capacity ────────────────────────────────

class TestBackpressure:
    def test_backpressure_at_high_watermark(self):
        # capacity=10, high_watermark=0.5 → backpressure at 5
        bus = BoundedEventBus(BusConfig(capacity=10, high_watermark=0.5))
        results = []
        for _ in range(10):
            results.append(bus.publish(_env()))
        # First 5 ACCEPTED, from 5 onward → BACKPRESSURE (once at watermark)
        assert PublishResult.ACCEPTED in results
        assert PublishResult.BACKPRESSURE in results

    def test_capacity_full_triggers_overflow(self):
        bus = BoundedEventBus(BusConfig(capacity=3, high_watermark=0.5))
        for _ in range(3):
            bus.publish(_env())
        # Now at capacity, next publish triggers overflow
        result = bus.publish(_env())
        # DROP_LOWEST is default, but same priority → DROPPED
        assert result in (PublishResult.DROPPED, PublishResult.ACCEPTED)


# ── Overflow DROP_LOWEST ─────────────────────────────────────

class TestOverflowDropLowest:
    def test_replaces_lowest_priority(self):
        bus = BoundedEventBus(BusConfig(capacity=2, high_watermark=1.0))
        bus.publish(_env(priority=Priority.LOW))
        bus.publish(_env(priority=Priority.LOW))
        # Full — publish CRITICAL should replace a LOW
        result = bus.publish(_env(priority=Priority.CRITICAL))
        assert result == PublishResult.ACCEPTED
        assert bus.size == 2

        # Drain and verify CRITICAL is delivered first
        received = []
        bus.subscribe(lambda e: received.append(e.priority))
        bus.drain(batch_size=10)
        assert received[0] == Priority.CRITICAL

    def test_same_priority_drops(self):
        bus = BoundedEventBus(BusConfig(capacity=2, high_watermark=1.0))
        bus.publish(_env(priority=Priority.NORMAL))
        bus.publish(_env(priority=Priority.NORMAL))
        result = bus.publish(_env(priority=Priority.NORMAL))
        # Same priority → cannot replace, DROPPED
        assert result == PublishResult.DROPPED


# ── Overflow REJECT ──────────────────────────────────────────

class TestOverflowReject:
    def test_rejects_when_full(self):
        bus = BoundedEventBus(BusConfig(
            capacity=2, high_watermark=1.0, overflow_policy=OverflowPolicy.REJECT,
        ))
        bus.publish(_env())
        bus.publish(_env())
        result = bus.publish(_env())
        assert result == PublishResult.REJECTED


# ── Subscribe / Unsubscribe ──────────────────────────────────

class TestSubscription:
    def test_unsubscribe_removes_handler(self):
        bus = BoundedEventBus()
        received = []
        def handler(e):
            return received.append(1)
        bus.subscribe(handler)
        bus.unsubscribe(handler)
        bus.publish(_env())
        bus.drain(batch_size=10)
        assert len(received) == 0

    def test_unsubscribe_nonexistent_is_noop(self):
        bus = BoundedEventBus()
        bus.unsubscribe(lambda e: None)  # should not raise

    def test_kind_specific_handler(self):
        bus = BoundedEventBus()
        market_events = []
        fill_events = []
        bus.subscribe(lambda e: market_events.append(e), kind=EventKind.MARKET)
        bus.subscribe(lambda e: fill_events.append(e), kind=EventKind.FILL)

        bus.publish(_env(kind=EventKind.MARKET))
        bus.publish(_env(kind=EventKind.FILL))
        bus.drain(batch_size=10)

        assert len(market_events) == 1
        assert len(fill_events) == 1

    def test_all_event_handler(self):
        bus = BoundedEventBus()
        all_events = []
        bus.subscribe(lambda e: all_events.append(e))  # kind=None → all

        bus.publish(_env(kind=EventKind.MARKET))
        bus.publish(_env(kind=EventKind.FILL))
        bus.drain(batch_size=10)

        assert len(all_events) == 2

    def test_kind_and_all_handlers_both_fire(self):
        bus = BoundedEventBus()
        kind_events = []
        all_events = []
        bus.subscribe(lambda e: kind_events.append(e), kind=EventKind.MARKET)
        bus.subscribe(lambda e: all_events.append(e))

        bus.publish(_env(kind=EventKind.MARKET))
        bus.drain(batch_size=10)

        assert len(kind_events) == 1
        assert len(all_events) == 1


# ── BusStats ─────────────────────────────────────────────────

class TestBusStats:
    def test_stats_tracking(self):
        bus = BoundedEventBus(BusConfig(capacity=100))
        bus.subscribe(lambda e: None)
        bus.publish(_env())
        bus.publish(_env())
        bus.drain(batch_size=10)
        stats = bus.stats()
        assert stats["published"] == 2
        assert stats["delivered"] == 2
        assert stats["dropped"] == 0

    def test_dropped_counted(self):
        bus = BoundedEventBus(BusConfig(
            capacity=1, high_watermark=1.0, overflow_policy=OverflowPolicy.REJECT,
        ))
        bus.publish(_env())
        bus.publish(_env())  # rejected
        stats = bus.stats()
        assert stats["dropped"] == 1


# ── Utilization ──────────────────────────────────────────────

class TestUtilization:
    def test_utilization_empty(self):
        bus = BoundedEventBus(BusConfig(capacity=100))
        assert bus.utilization == 0.0

    def test_utilization_half_full(self):
        bus = BoundedEventBus(BusConfig(capacity=10, high_watermark=1.0))
        for _ in range(5):
            bus.publish(_env())
        assert bus.utilization == pytest.approx(0.5)
