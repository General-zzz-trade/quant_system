"""Property-based tests for BoundedEventBus."""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from core.bus import BoundedEventBus, BusConfig, OverflowPolicy, PublishResult
from core.types import Envelope, EventKind, EventMetadata, Priority

from tests_unit.properties.strategies import envelopes


def _make_bus(capacity: int, policy: OverflowPolicy = OverflowPolicy.DROP_LOWEST) -> BoundedEventBus:
    return BoundedEventBus(BusConfig(
        capacity=capacity,
        high_watermark=1.0,  # No backpressure signal before full
        overflow_policy=policy,
    ))


@given(envs=st.lists(envelopes(), min_size=1, max_size=100))
@settings(max_examples=200)
def test_no_lost_events_under_capacity(envs):
    """When publish count <= capacity, all events are delivered."""
    bus = _make_bus(capacity=max(len(envs), 10))
    received = []
    bus.subscribe(lambda e: received.append(e))

    for env in envs:
        result = bus.publish(env)
        assert result == PublishResult.ACCEPTED

    bus.drain(batch_size=len(envs) + 10)
    assert len(received) == len(envs)


@given(envs=st.lists(envelopes(), min_size=2, max_size=50))
@settings(max_examples=200)
def test_drain_delivers_in_priority_order(envs):
    """Drained events respect priority ordering (lower value = higher priority)."""
    bus = _make_bus(capacity=100)
    received = []
    bus.subscribe(lambda e: received.append(e))

    for env in envs:
        bus.publish(env)

    bus.drain(batch_size=len(envs) + 10)

    for i in range(len(received) - 1):
        assert received[i].priority.value <= received[i + 1].priority.value


@given(envs=st.lists(envelopes(), min_size=1, max_size=50))
@settings(max_examples=200)
def test_bus_size_matches_published(envs):
    """Bus size equals number of published events before drain."""
    bus = _make_bus(capacity=max(len(envs), 10))
    for env in envs:
        bus.publish(env)
    assert bus.size == len(envs)


@given(envs=st.lists(envelopes(), min_size=1, max_size=50))
@settings(max_examples=200)
def test_bus_empty_after_full_drain(envs):
    """Bus is empty after draining all events."""
    bus = _make_bus(capacity=max(len(envs), 10))
    for env in envs:
        bus.publish(env)
    bus.drain(batch_size=len(envs) + 10)
    assert bus.size == 0


@given(
    n_events=st.integers(min_value=5, max_value=30),
    capacity=st.integers(min_value=2, max_value=4),
)
@settings(max_examples=100)
def test_reject_policy_never_exceeds_capacity(n_events, capacity):
    """With REJECT policy, bus never holds more than capacity events."""
    bus = _make_bus(capacity=capacity, policy=OverflowPolicy.REJECT)
    env = Envelope(
        event={"test": True},
        metadata=EventMetadata.create(source="test"),
        kind=EventKind.MARKET,
        priority=Priority.NORMAL,
    )
    for _ in range(n_events):
        bus.publish(env)
        assert bus.size <= capacity


@given(envs=st.lists(envelopes(), min_size=1, max_size=50))
@settings(max_examples=200)
def test_stats_published_count_consistent(envs):
    """stats.published matches number of successfully published events."""
    bus = _make_bus(capacity=max(len(envs), 10))
    accepted = 0
    for env in envs:
        result = bus.publish(env)
        if result in (PublishResult.ACCEPTED, PublishResult.BACKPRESSURE):
            accepted += 1
    stats = bus.stats()
    assert stats["published"] == accepted
