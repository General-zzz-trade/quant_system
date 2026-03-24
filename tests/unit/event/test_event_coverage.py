"""
Tests for low/zero-coverage event modules:
  - event/event.py       (0%)
  - event/tick_types.py  (0%)
  - event/metrics.py     (28%)
Note: event/validators.py and event/dispatcher.py deleted (dead code).
"""
from __future__ import annotations

import dataclasses
import sys
import threading
import time
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Patch _quant_hotpath before any event.* imports that depend on it
# ---------------------------------------------------------------------------
try:
    import _quant_hotpath  # noqa: F401 — real .so
except ImportError:
    _mock_hotpath = MagicMock()
    _mock_hotpath.rust_event_id.side_effect = lambda: "test-eid-001"
    _mock_hotpath.rust_now_ns.return_value = 1_700_000_000_000_000_000
    sys.modules["_quant_hotpath"] = _mock_hotpath

# Also patch the depth_processor import used by tick_types, but only if the real
# module still cannot be imported after the _quant_hotpath shim above.
try:
    import execution.adapters.binance.depth_processor as _depth_processor  # noqa: F401
    _FakeOrderBookLevel = _depth_processor.OrderBookLevel
    _FakeOrderBookSnapshot = _depth_processor.OrderBookSnapshot
except Exception:
    _mock_dp = types.ModuleType("execution.adapters.binance.depth_processor")

    @dataclass(frozen=True, slots=True)
    class _FakeOrderBookLevel:
        price: Decimal
        qty: Decimal

    @dataclass(frozen=True, slots=True)
    class _FakeOrderBookSnapshot:
        symbol: str
        bids: tuple
        asks: tuple
        ts_ms: int
        last_update_id: int = 0

    _mock_dp.OrderBookLevel = _FakeOrderBookLevel
    _mock_dp.OrderBookSnapshot = _FakeOrderBookSnapshot
    sys.modules["execution.adapters.binance.depth_processor"] = _mock_dp

# Now safe to import
from event.errors import EventValidationError, EventDispatchError  # noqa: E402
from event.lifecycle import EventLifecycle, LifecycleState          # noqa: E402
from event.types import EventType                                   # noqa: E402
from event.header import EventHeader                                # noqa: E402


# ---------------------------------------------------------------------------
# Helpers / fixtures shared across test classes
# ---------------------------------------------------------------------------

# Lightweight fake header for tests that do NOT go through _get_header()
# (i.e. tests of event.event, event.metrics, event.dispatcher internals
#  that only need header.event_id)
@dataclass(frozen=True)
class _SimpleFakeHeader:
    event_id: str


@dataclass(frozen=True)
class _SimpleFakeEvent:
    header: _SimpleFakeHeader


def _simple_evt(eid: str = "evt-001") -> _SimpleFakeEvent:
    return _SimpleFakeEvent(header=_SimpleFakeHeader(event_id=eid))


# ---------------------------------------------------------------------------
# ValidatorHeader: a real EventHeader-compatible object for validators tests.
# event.validators._get_header checks isinstance(header, EventHeader).
# We use the real EventHeader dataclass (imported above) with extra attrs
# monkey-patched on for validator-specific fields.
# ---------------------------------------------------------------------------

_TS_NS = 1_700_000_000_000_000_000
_DEFAULT_DT = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _make_real_header(
    eid: str = "evt-001",
    event_time: Any = _DEFAULT_DT,
    source: str = "test",
    meta: Any = None,
    mode: Any = None,
    trace_id: Any = None,
    span_id: Any = None,
    parent_span_id: Any = None,
    trace_depth: Any = None,
) -> EventHeader:
    """
    Build a real EventHeader and monkey-patch extra fields used by validators.
    EventHeader is frozen=True, so we set attrs via object.__setattr__ bypass.
    """
    h = EventHeader(
        event_id=eid,
        event_type=EventType.MARKET,
        version=1,
        ts_ns=_TS_NS,
        source=source,
    )
    # Monkey-patch the validator-specific attributes
    for attr, val in [
        ("event_time", event_time),
        ("meta", meta),
        ("mode", mode),
        ("trace_id", trace_id),
        ("span_id", span_id),
        ("parent_span_id", parent_span_id),
        ("trace_depth", trace_depth),
    ]:
        object.__setattr__(h, attr, val)
    return h


@dataclass(frozen=True)
class _ValidatorEvent:
    """Event wrapping a real EventHeader, for validators tests."""
    header: EventHeader
    symbol: str = "ETHUSDT"
    timeframe: str = "1h"


def _evt(
    eid: str = "evt-001",
    dt: datetime | None = None,
    symbol: str = "ETHUSDT",
    timeframe: str = "1h",
    source: str = "test",
    meta: Any = None,
    mode: Any = None,
    trace_id: Any = None,
    span_id: Any = None,
    parent_span_id: Any = None,
    trace_depth: Any = None,
) -> _ValidatorEvent:
    event_time = dt or _DEFAULT_DT
    h = _make_real_header(
        eid=eid,
        event_time=event_time,
        source=source,
        meta=meta,
        mode=mode,
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        trace_depth=trace_depth,
    )
    return _ValidatorEvent(header=h, symbol=symbol, timeframe=timeframe)


# ---------------------------------------------------------------------------
# event/event.py — DELETED (dead code)
# event/tick_types.py — DELETED (dead code)
# event/validators.py — DELETED (dead code), all validator tests removed
# ---------------------------------------------------------------------------


class TestCounter:
    """Tests for Counter primitive."""

    def test_initial_value_is_zero(self) -> None:
        from event.metrics import Counter

        c = Counter()
        assert c.value() == 0

    def test_inc_default(self) -> None:
        from event.metrics import Counter

        c = Counter()
        c.inc()
        assert c.value() == 1

    def test_inc_by_n(self) -> None:
        from event.metrics import Counter

        c = Counter()
        c.inc(5)
        assert c.value() == 5

    def test_multiple_incs(self) -> None:
        from event.metrics import Counter

        c = Counter()
        c.inc(3)
        c.inc(2)
        assert c.value() == 5

    def test_thread_safe_increments(self) -> None:
        from event.metrics import Counter

        c = Counter()
        threads = [threading.Thread(target=lambda: c.inc()) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert c.value() == 100


class TestLatencyStat:
    """Tests for LatencyStat primitive."""

    def test_snapshot_when_empty(self) -> None:
        from event.metrics import LatencyStat

        ls = LatencyStat()
        snap = ls.snapshot()
        assert snap["count"] == 0
        assert snap["avg_ms"] == 0.0
        assert snap["max_ms"] == 0.0

    def test_single_observation(self) -> None:
        from event.metrics import LatencyStat

        ls = LatencyStat()
        ls.observe(10.0)
        snap = ls.snapshot()
        assert snap["count"] == 1
        assert snap["avg_ms"] == 10.0
        assert snap["max_ms"] == 10.0

    def test_multiple_observations(self) -> None:
        from event.metrics import LatencyStat

        ls = LatencyStat()
        ls.observe(10.0)
        ls.observe(20.0)
        snap = ls.snapshot()
        assert snap["count"] == 2
        assert snap["avg_ms"] == 15.0
        assert snap["max_ms"] == 20.0

    def test_max_tracks_correctly(self) -> None:
        from event.metrics import LatencyStat

        ls = LatencyStat()
        for v in [5.0, 100.0, 3.0, 50.0]:
            ls.observe(v)
        snap = ls.snapshot()
        assert snap["max_ms"] == 100.0


class TestMetricsSnapshot:
    """Tests for MetricsSnapshot dataclass."""

    def test_creation(self) -> None:
        from event.metrics import MetricsSnapshot

        snap = MetricsSnapshot(
            counters={"created": 10, "dispatched": 5},
            latencies={"emit_to_dispatch": {"count": 5, "avg_ms": 1.0, "max_ms": 2.0}},
        )
        assert snap.counters["created"] == 10
        assert snap.latencies["emit_to_dispatch"]["avg_ms"] == 1.0


class TestEventMetrics:
    """Tests for EventMetrics — all lifecycle state transitions."""

    def _make_event(self, event_id: str) -> Any:
        @dataclass(frozen=True)
        class _H:
            event_id: str

        @dataclass(frozen=True)
        class _E:
            header: _H

        return _E(header=_H(event_id=event_id))

    def _make_no_header_event(self) -> Any:
        """Event without header — falls back to id()."""
        return object()

    def test_on_lifecycle_created(self) -> None:
        from event.metrics import EventMetrics

        m = EventMetrics()
        e = self._make_event("eid-001")
        m.on_lifecycle(e, LifecycleState.CREATED)
        snap = m.snapshot()
        assert snap.counters["created"] == 1

    def test_on_lifecycle_dispatching(self) -> None:
        from event.metrics import EventMetrics

        m = EventMetrics()
        e = self._make_event("eid-002")
        m.on_lifecycle(e, LifecycleState.CREATED)
        m.on_lifecycle(e, LifecycleState.DISPATCH_START)
        snap = m.snapshot()
        assert snap.counters["dispatching"] == 1

    def test_emit_to_dispatch_latency_observed(self) -> None:
        from event.metrics import EventMetrics

        m = EventMetrics()
        e = self._make_event("eid-003")
        m.on_lifecycle(e, LifecycleState.CREATED)
        m.on_lifecycle(e, LifecycleState.DISPATCH_START)
        snap = m.snapshot()
        assert snap.latencies["emit_to_dispatch"]["count"] == 1

    def test_dispatching_without_prior_created_no_latency(self) -> None:
        """DISPATCHING without CREATED → no emit_to_dispatch observation."""
        from event.metrics import EventMetrics

        m = EventMetrics()
        e = self._make_event("eid-004")
        m.on_lifecycle(e, LifecycleState.DISPATCH_START)
        snap = m.snapshot()
        assert snap.latencies["emit_to_dispatch"]["count"] == 0

    def test_on_lifecycle_dispatched(self) -> None:
        from event.metrics import EventMetrics

        m = EventMetrics()
        e = self._make_event("eid-005")
        m.on_lifecycle(e, LifecycleState.CREATED)
        m.on_lifecycle(e, LifecycleState.DISPATCH_START)
        m.on_lifecycle(e, LifecycleState.DISPATCHED)
        snap = m.snapshot()
        assert snap.counters["dispatched"] == 1
        assert snap.latencies["dispatch_to_done"]["count"] == 1

    def test_dispatched_without_dispatching_no_latency(self) -> None:
        from event.metrics import EventMetrics

        m = EventMetrics()
        e = self._make_event("eid-006")
        m.on_lifecycle(e, LifecycleState.DISPATCHED)
        snap = m.snapshot()
        assert snap.counters["dispatched"] == 1
        assert snap.latencies["dispatch_to_done"]["count"] == 0

    def test_on_lifecycle_retry(self) -> None:
        from event.metrics import EventMetrics

        m = EventMetrics()
        e = self._make_event("eid-007")
        m.on_lifecycle(e, LifecycleState.CREATED)
        m.on_lifecycle(e, LifecycleState.RETRY)
        snap = m.snapshot()
        assert snap.counters["retry"] == 1

    def test_on_lifecycle_failed(self) -> None:
        from event.metrics import EventMetrics

        m = EventMetrics()
        e = self._make_event("eid-008")
        m.on_lifecycle(e, LifecycleState.CREATED)
        m.on_lifecycle(e, LifecycleState.FAILED)
        snap = m.snapshot()
        assert snap.counters["failed"] == 1

    def test_on_lifecycle_dropped(self) -> None:
        from event.metrics import EventMetrics

        m = EventMetrics()
        e = self._make_event("eid-009")
        m.on_lifecycle(e, LifecycleState.CREATED)
        m.on_lifecycle(e, LifecycleState.DROPPED)
        snap = m.snapshot()
        assert snap.counters["dropped"] == 1

    def test_cleanup_on_dispatched(self) -> None:
        """After DISPATCHED, internal timestamps should be cleaned up."""
        from event.metrics import EventMetrics

        m = EventMetrics()
        e = self._make_event("eid-010")
        m.on_lifecycle(e, LifecycleState.CREATED)
        m.on_lifecycle(e, LifecycleState.DISPATCH_START)
        m.on_lifecycle(e, LifecycleState.DISPATCHED)
        # Internal dicts should be empty
        assert "eid-010" not in m._emit_ts
        assert "eid-010" not in m._dispatch_ts

    def test_cleanup_on_retry(self) -> None:
        from event.metrics import EventMetrics

        m = EventMetrics()
        e = self._make_event("eid-011")
        m.on_lifecycle(e, LifecycleState.CREATED)
        m.on_lifecycle(e, LifecycleState.RETRY)
        assert "eid-011" not in m._emit_ts

    def test_cleanup_on_failed(self) -> None:
        from event.metrics import EventMetrics

        m = EventMetrics()
        e = self._make_event("eid-012")
        m.on_lifecycle(e, LifecycleState.CREATED)
        m.on_lifecycle(e, LifecycleState.FAILED)
        assert "eid-012" not in m._emit_ts

    def test_cleanup_on_dropped(self) -> None:
        from event.metrics import EventMetrics

        m = EventMetrics()
        e = self._make_event("eid-013")
        m.on_lifecycle(e, LifecycleState.CREATED)
        m.on_lifecycle(e, LifecycleState.DROPPED)
        assert "eid-013" not in m._emit_ts

    def test_event_key_fallback_to_id(self) -> None:
        """Event without header falls back to str(id(event))."""
        from event.metrics import EventMetrics

        m = EventMetrics()
        e = self._make_no_header_event()
        # Should not raise, just use object id as key
        m.on_lifecycle(e, LifecycleState.CREATED)
        snap = m.snapshot()
        assert snap.counters["created"] == 1

    def test_snapshot_all_counters_present(self) -> None:
        from event.metrics import EventMetrics

        m = EventMetrics()
        snap = m.snapshot()
        for key in ("created", "dispatching", "dispatched", "retry", "failed", "dropped"):
            assert key in snap.counters

    def test_snapshot_all_latencies_present(self) -> None:
        from event.metrics import EventMetrics

        m = EventMetrics()
        snap = m.snapshot()
        assert "emit_to_dispatch" in snap.latencies
        assert "dispatch_to_done" in snap.latencies

    def test_unrecognized_lifecycle_state_is_ignored(self) -> None:
        """Passing a state not handled by on_lifecycle should be a no-op."""
        from event.metrics import EventMetrics

        m = EventMetrics()
        e = self._make_event("eid-014")
        # ENQUEUED / DISPATCH_START / HANDLED are not handled by EventMetrics
        m.on_lifecycle(e, LifecycleState.ENQUEUED)
        snap = m.snapshot()
        # Nothing should increment
        assert all(v == 0 for v in snap.counters.values())

