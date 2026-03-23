"""Bounded event bus with backpressure and priority ordering.

Replaces the unbounded ``event/bus.py`` with a capacity-limited design:

  * **Bounded ring buffer** — prevents OOM under high-frequency feeds.
  * **Priority ordering** — CRITICAL > HIGH > NORMAL > LOW.
  * **Backpressure signal** — callers know when to slow down.
  * **Overflow policy** — configurable: DROP_LOWEST or REJECT.

The bus is thread-safe.  Producers call ``publish()``, consumers call
``drain()`` in a loop.

Migrated from core/bus.py.
"""
from __future__ import annotations

import heapq
import threading
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, List, Optional, Tuple

from event.core_types import Envelope, EventKind


# -- Publish result -------------------------------------------

class PublishResult(Enum):
    ACCEPTED = auto()
    BACKPRESSURE = auto()
    DROPPED = auto()
    REJECTED = auto()


# -- Configuration --------------------------------------------

class OverflowPolicy(Enum):
    DROP_LOWEST = auto()   # drop the lowest-priority event
    REJECT = auto()        # reject the new event


@dataclass(frozen=True, slots=True)
class BusConfig:
    """Bounded bus tuning knobs."""
    capacity: int = 10_000
    high_watermark: float = 0.8       # fraction — triggers backpressure signal
    overflow_policy: OverflowPolicy = OverflowPolicy.DROP_LOWEST


# -- Stats ----------------------------------------------------

@dataclass
class BusStats:
    published: int = 0
    delivered: int = 0
    dropped: int = 0
    backpressure_signals: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "published": self.published,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "backpressure_signals": self.backpressure_signals,
        }


# -- Bounded Event Bus ----------------------------------------

# Priority queue entry: (priority_value, sequence, envelope)
_QueueEntry = Tuple[int, int, Envelope[Any]]


class BoundedEventBus:
    """Thread-safe, bounded, priority-aware event bus.

    Usage::

        bus = BoundedEventBus(BusConfig(capacity=5000))
        bus.subscribe(handler, kind=EventKind.MARKET)
        result = bus.publish(envelope)
        delivered = bus.drain(batch_size=64)
    """

    def __init__(self, config: Optional[BusConfig] = None) -> None:
        self._cfg = config or BusConfig()
        self._lock = threading.Lock()
        self._seq: int = 0
        self._heap: List[_QueueEntry] = []
        self._stats = BusStats()

        # Subscriptions: kind -> [handler]  (None key = subscribe-all)
        self._handlers_by_kind: dict[Optional[EventKind], List[Callable[[Envelope[Any]], None]]] = defaultdict(list)

    # -- Subscribe --------------------------------------------

    def subscribe(
        self,
        handler: Callable[[Envelope[Any]], None],
        *,
        kind: Optional[EventKind] = None,
    ) -> None:
        """Register *handler* for events of *kind* (``None`` = all events)."""
        with self._lock:
            self._handlers_by_kind[kind].append(handler)

    def unsubscribe(
        self,
        handler: Callable[[Envelope[Any]], None],
        *,
        kind: Optional[EventKind] = None,
    ) -> None:
        """Remove *handler* from *kind* subscription list."""
        with self._lock:
            handlers = self._handlers_by_kind.get(kind, [])
            try:
                handlers.remove(handler)
            except ValueError:
                pass

    # -- Publish ------------------------------------------------

    def publish(self, envelope: Envelope[Any]) -> PublishResult:
        """Enqueue an event.  Returns the outcome."""
        with self._lock:
            size = len(self._heap)

            # Backpressure check
            if size >= int(self._cfg.capacity * self._cfg.high_watermark):
                self._stats.backpressure_signals += 1
                if size >= self._cfg.capacity:
                    return self._handle_overflow(envelope)

            self._seq += 1
            entry: _QueueEntry = (envelope.priority.value, self._seq, envelope)
            heapq.heappush(self._heap, entry)
            self._stats.published += 1

            if size >= int(self._cfg.capacity * self._cfg.high_watermark):
                return PublishResult.BACKPRESSURE

            return PublishResult.ACCEPTED

    def _handle_overflow(self, envelope: Envelope[Any]) -> PublishResult:
        """Called under lock when at capacity."""
        if self._cfg.overflow_policy == OverflowPolicy.REJECT:
            self._stats.dropped += 1
            return PublishResult.REJECTED

        # DROP_LOWEST: if new event is higher priority than worst, swap
        if self._heap:
            worst_priority = self._heap[-1][0]  # approximate — heap isn't fully sorted
            if envelope.priority.value < worst_priority:
                # Find and remove the actual lowest-priority entry
                # (heapq is a min-heap, so highest numeric priority = lowest importance)
                worst_idx = max(range(len(self._heap)), key=lambda i: self._heap[i][0])
                self._heap[worst_idx] = self._heap[-1]
                self._heap.pop()
                heapq.heapify(self._heap)

                self._seq += 1
                entry: _QueueEntry = (envelope.priority.value, self._seq, envelope)
                heapq.heappush(self._heap, entry)
                self._stats.dropped += 1  # dropped the old one
                self._stats.published += 1
                return PublishResult.ACCEPTED

        self._stats.dropped += 1
        return PublishResult.DROPPED

    # -- Drain (consumer side) ----------------------------------

    def drain(self, batch_size: int = 64) -> int:
        """Pop up to *batch_size* events and deliver them to handlers.

        Returns the number of events delivered.
        """
        # Pop events under lock
        with self._lock:
            batch: list[Envelope[Any]] = []
            for _ in range(min(batch_size, len(self._heap))):
                _, _, env = heapq.heappop(self._heap)
                batch.append(env)
            # Snapshot handlers to avoid holding lock during dispatch
            handlers_snapshot = {
                k: list(v) for k, v in self._handlers_by_kind.items()
            }

        # Dispatch outside lock
        delivered = 0
        for env in batch:
            # Handlers subscribed to this specific kind
            for h in handlers_snapshot.get(env.kind, []):
                h(env)
                delivered += 1
            # Handlers subscribed to all events
            for h in handlers_snapshot.get(None, []):
                h(env)
                delivered += 1

        with self._lock:
            self._stats.delivered += delivered
        return delivered

    # -- Observability ------------------------------------------

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._heap)

    @property
    def utilization(self) -> float:
        with self._lock:
            return len(self._heap) / max(1, self._cfg.capacity)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return self._stats.snapshot()
