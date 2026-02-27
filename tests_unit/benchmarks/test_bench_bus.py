"""Performance benchmark: BoundedEventBus publish/drain latency."""
from __future__ import annotations

import time

import pytest

from core.bus import BoundedEventBus, BusConfig
from core.types import Envelope, EventKind, EventMetadata, Priority


def _make_envelope(i: int) -> Envelope:
    priorities = [Priority.LOW, Priority.NORMAL, Priority.HIGH, Priority.CRITICAL]
    return Envelope(
        event={"i": i},
        metadata=EventMetadata.create(source="bench"),
        kind=EventKind.MARKET,
        priority=priorities[i % len(priorities)],
    )


@pytest.mark.benchmark
def test_bus_publish_drain_throughput():
    """Bus should publish >10K events/sec and drain >5K events/sec."""
    n_events = 10_000
    bus = BoundedEventBus(BusConfig(capacity=n_events * 2, high_watermark=0.95))
    received = []
    bus.subscribe(lambda e: received.append(e))

    envelopes = [_make_envelope(i) for i in range(n_events)]

    # Publish phase
    start = time.perf_counter()
    for env in envelopes:
        bus.publish(env)
    publish_elapsed = time.perf_counter() - start

    # Drain phase
    start = time.perf_counter()
    bus.drain(batch_size=n_events + 100)
    drain_elapsed = time.perf_counter() - start

    pub_rate = n_events / publish_elapsed
    drain_rate = n_events / drain_elapsed

    print(f"\nBus publish: {pub_rate:,.0f} events/sec ({publish_elapsed:.3f}s)")
    print(f"Bus drain:   {drain_rate:,.0f} events/sec ({drain_elapsed:.3f}s)")
    print(f"Total delivered: {len(received)}")

    assert pub_rate > 10_000, f"Publish too slow: {pub_rate:.0f}/s"
    assert drain_rate > 5_000, f"Drain too slow: {drain_rate:.0f}/s"
    assert len(received) == n_events
