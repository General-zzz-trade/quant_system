"""Tests for core.bus — BoundedEventBus with backpressure and priority."""
from __future__ import annotations

from core.bus import BoundedEventBus, BusConfig, OverflowPolicy, PublishResult
from core.types import Envelope, EventKind, EventMetadata, Priority


def _make_envelope(
    kind: EventKind = EventKind.MARKET,
    priority: Priority = Priority.NORMAL,
) -> Envelope:
    meta = EventMetadata.create(source="test")
    return Envelope(event={"test": True}, metadata=meta, kind=kind, priority=priority)


class TestBoundedEventBus:
    def test_publish_and_drain(self) -> None:
        bus = BoundedEventBus()
        env = _make_envelope()
        result = bus.publish(env)
        assert result == PublishResult.ACCEPTED
        assert bus.size == 1

        received: list[Envelope] = []
        bus.subscribe(received.append)
        delivered = bus.drain(batch_size=10)
        assert delivered == 1
        assert len(received) == 1
        assert received[0].event_id == env.event_id

    def test_subscribe_by_kind(self) -> None:
        bus = BoundedEventBus()
        market_events: list[Envelope] = []
        order_events: list[Envelope] = []

        bus.subscribe(market_events.append, kind=EventKind.MARKET)
        bus.subscribe(order_events.append, kind=EventKind.ORDER)

        bus.publish(_make_envelope(kind=EventKind.MARKET))
        bus.publish(_make_envelope(kind=EventKind.ORDER))
        bus.publish(_make_envelope(kind=EventKind.MARKET))

        bus.drain(batch_size=10)
        assert len(market_events) == 2
        assert len(order_events) == 1

    def test_unsubscribe(self) -> None:
        bus = BoundedEventBus()
        events: list[Envelope] = []
        bus.subscribe(events.append)
        bus.publish(_make_envelope())
        bus.drain()
        assert len(events) == 1

        bus.unsubscribe(events.append)
        bus.publish(_make_envelope())
        bus.drain()
        assert len(events) == 1  # no new events

    def test_priority_ordering(self) -> None:
        bus = BoundedEventBus()
        received: list[Envelope] = []
        bus.subscribe(received.append)

        # Publish low first, then critical
        bus.publish(_make_envelope(priority=Priority.LOW))
        bus.publish(_make_envelope(priority=Priority.CRITICAL))
        bus.publish(_make_envelope(priority=Priority.NORMAL))

        bus.drain(batch_size=10)
        priorities = [e.priority for e in received]
        assert priorities == [Priority.CRITICAL, Priority.NORMAL, Priority.LOW]

    def test_backpressure_signal(self) -> None:
        cfg = BusConfig(capacity=10, high_watermark=0.5)
        bus = BoundedEventBus(cfg)

        # Fill to 50% → should start signaling backpressure
        for _ in range(5):
            bus.publish(_make_envelope())

        result = bus.publish(_make_envelope())
        assert result == PublishResult.BACKPRESSURE

    def test_capacity_reject_policy(self) -> None:
        cfg = BusConfig(capacity=3, overflow_policy=OverflowPolicy.REJECT)
        bus = BoundedEventBus(cfg)

        for _ in range(3):
            bus.publish(_make_envelope())

        result = bus.publish(_make_envelope())
        assert result in (PublishResult.REJECTED, PublishResult.BACKPRESSURE)

    def test_capacity_drop_lowest_policy(self) -> None:
        cfg = BusConfig(capacity=3, high_watermark=1.0, overflow_policy=OverflowPolicy.DROP_LOWEST)
        bus = BoundedEventBus(cfg)

        # Fill with LOW priority
        for _ in range(3):
            bus.publish(_make_envelope(priority=Priority.LOW))

        # Publish CRITICAL — should drop a LOW event
        result = bus.publish(_make_envelope(priority=Priority.CRITICAL))
        assert result == PublishResult.ACCEPTED
        assert bus.size == 3  # still at capacity

    def test_stats_tracking(self) -> None:
        bus = BoundedEventBus()
        bus.subscribe(lambda e: None)

        bus.publish(_make_envelope())
        bus.publish(_make_envelope())
        bus.drain()

        stats = bus.stats()
        assert stats["published"] == 2
        assert stats["delivered"] == 2
        assert stats["dropped"] == 0

    def test_utilization(self) -> None:
        cfg = BusConfig(capacity=100)
        bus = BoundedEventBus(cfg)
        assert bus.utilization == 0.0

        for _ in range(50):
            bus.publish(_make_envelope())
        assert abs(bus.utilization - 0.5) < 0.01

    def test_empty_drain(self) -> None:
        bus = BoundedEventBus()
        delivered = bus.drain()
        assert delivered == 0

    def test_subscribe_all(self) -> None:
        bus = BoundedEventBus()
        all_events: list[Envelope] = []
        bus.subscribe(all_events.append)  # kind=None → subscribe all

        bus.publish(_make_envelope(kind=EventKind.MARKET))
        bus.publish(_make_envelope(kind=EventKind.ORDER))
        bus.publish(_make_envelope(kind=EventKind.FILL))
        bus.drain()

        assert len(all_events) == 3
