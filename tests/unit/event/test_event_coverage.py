"""
Tests for low/zero-coverage event modules:
  - event/validators.py  (0%)
  - event/dispatcher.py  (0%)
  - event/event.py       (0%)
  - event/tick_types.py  (0%)
  - event/metrics.py     (28%)
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
# event/event.py
# ---------------------------------------------------------------------------

class TestEventClass:
    """Tests for event.event.Event dataclass."""

    def test_event_construction(self) -> None:
        from event.event import Event

        h = _make_real_header(eid="hdr-1")
        e = Event(header=h, payload={"key": "value"})
        assert e.header is h
        assert e.payload == {"key": "value"}

    def test_event_is_frozen(self) -> None:
        from event.event import Event

        h = _make_real_header(eid="hdr-2")
        e = Event(header=h, payload={})
        with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
            e.payload = {"new": "data"}  # type: ignore[misc]

    def test_event_equality(self) -> None:
        from event.event import Event

        h = _make_real_header(eid="hdr-3")
        e1 = Event(header=h, payload={"x": 1})
        e2 = Event(header=h, payload={"x": 1})
        assert e1 == e2

    def test_event_payload_type(self) -> None:
        from event.event import Event

        h = _make_real_header(eid="hdr-4")
        payload: Dict[str, Any] = {"price": 1234.5, "qty": 0.5}
        e = Event(header=h, payload=payload)
        assert isinstance(e.payload, dict)


# ---------------------------------------------------------------------------
# event/tick_types.py
# ---------------------------------------------------------------------------

class TestTickTypes:
    """Tests for TradeTickEvent and DepthUpdateEvent."""

    def test_trade_tick_event_creation(self) -> None:
        from event.tick_types import TradeTickEvent

        ev = TradeTickEvent(
            symbol="BTCUSDT",
            price=Decimal("50000.0"),
            qty=Decimal("0.01"),
            side="buy",
            trade_id=123456,
            ts_ms=1_700_000_000_000,
        )
        assert ev.symbol == "BTCUSDT"
        assert ev.price == Decimal("50000.0")
        assert ev.qty == Decimal("0.01")
        assert ev.side == "buy"
        assert ev.trade_id == 123456
        assert ev.ts_ms == 1_700_000_000_000
        assert isinstance(ev.received_at, float)

    def test_trade_tick_event_sell_side(self) -> None:
        from event.tick_types import TradeTickEvent

        ev = TradeTickEvent(
            symbol="ETHUSDT",
            price=Decimal("2000.0"),
            qty=Decimal("1.0"),
            side="sell",
            trade_id=999,
            ts_ms=1_700_000_001_000,
        )
        assert ev.side == "sell"

    def test_trade_tick_event_is_frozen(self) -> None:
        from event.tick_types import TradeTickEvent

        ev = TradeTickEvent(
            symbol="BTCUSDT",
            price=Decimal("50000.0"),
            qty=Decimal("0.01"),
            side="buy",
            trade_id=1,
            ts_ms=1_000,
        )
        with pytest.raises((AttributeError, TypeError)):
            ev.symbol = "OTHER"  # type: ignore[misc]

    def test_trade_tick_received_at_default(self) -> None:
        from event.tick_types import TradeTickEvent

        before = time.monotonic()
        ev = TradeTickEvent(
            symbol="BTCUSDT",
            price=Decimal("1"),
            qty=Decimal("1"),
            side="buy",
            trade_id=1,
            ts_ms=1,
        )
        after = time.monotonic()
        assert before <= ev.received_at <= after + 0.1

    def test_depth_update_event_creation(self) -> None:
        from event.tick_types import DepthUpdateEvent

        snap = _FakeOrderBookSnapshot(
            symbol="BTCUSDT",
            bids=(),
            asks=(),
            ts_ms=1_700_000_000_000,
            last_update_id=0,
        )
        ev = DepthUpdateEvent(snapshot=snap)
        assert ev.snapshot is snap
        assert isinstance(ev.received_at, float)

    def test_depth_update_event_is_frozen(self) -> None:
        from event.tick_types import DepthUpdateEvent

        snap = _FakeOrderBookSnapshot(symbol="X", bids=(), asks=(), ts_ms=0, last_update_id=0)
        ev = DepthUpdateEvent(snapshot=snap)
        with pytest.raises((AttributeError, TypeError)):
            ev.snapshot = snap  # type: ignore[misc]

    def test_depth_update_received_at_default(self) -> None:
        from event.tick_types import DepthUpdateEvent

        snap = _FakeOrderBookSnapshot(symbol="X", bids=(), asks=(), ts_ms=0, last_update_id=0)
        before = time.monotonic()
        ev = DepthUpdateEvent(snapshot=snap)
        after = time.monotonic()
        assert before <= ev.received_at <= after + 0.1


# ---------------------------------------------------------------------------
# event/validators.py
# ---------------------------------------------------------------------------

class TestGetHeader:
    """Tests for _get_header helper."""

    def test_returns_header_when_valid(self) -> None:
        from event.validators import _get_header

        e = _evt()  # uses real EventHeader
        result = _get_header(e)
        assert isinstance(result, EventHeader)

    def test_raises_when_no_header_attr(self) -> None:
        from event.validators import _get_header

        class _NoHeader:
            pass

        with pytest.raises(EventValidationError, match="EventHeader"):
            _get_header(_NoHeader())

    def test_raises_when_header_wrong_type(self) -> None:
        from event.validators import _get_header

        class _BadEvent:
            header = "not-a-header"

        with pytest.raises(EventValidationError):
            _get_header(_BadEvent())


class TestGetMeta:
    """Tests for _get_meta helper."""

    def test_returns_meta_when_mapping(self) -> None:
        from event.validators import _get_meta

        h = _make_real_header(meta={"k": "v"})
        result = _get_meta(h)
        assert result == {"k": "v"}

    def test_returns_empty_dict_when_no_meta(self) -> None:
        from event.validators import _get_meta

        h = _make_real_header(meta=None)
        result = _get_meta(h)
        assert result == {}

    def test_returns_empty_dict_when_meta_not_mapping(self) -> None:
        from event.validators import _get_meta

        h = _make_real_header(meta="bad-type")
        result = _get_meta(h)
        assert result == {}


class TestGetEventTime:
    """Tests for _get_event_time helper."""

    def test_returns_datetime_when_valid(self) -> None:
        from event.validators import _get_event_time

        dt = datetime(2024, 6, 1, tzinfo=timezone.utc)
        h = _make_real_header(event_time=dt)
        result = _get_event_time(h)
        assert result == dt

    def test_raises_when_event_time_missing(self) -> None:
        from event.validators import _get_event_time

        h = _make_real_header()
        # Remove the event_time attr completely
        object.__setattr__(h, "event_time", None)
        with pytest.raises(EventValidationError, match="event_time"):
            _get_event_time(h)

    def test_raises_when_event_time_not_datetime(self) -> None:
        from event.validators import _get_event_time

        h = _make_real_header()
        object.__setattr__(h, "event_time", "not-a-datetime")
        with pytest.raises(EventValidationError):
            _get_event_time(h)

    def test_raises_when_event_time_naive(self) -> None:
        from event.validators import _get_event_time

        naive_dt = datetime(2024, 1, 1)  # no tzinfo
        h = _make_real_header(event_time=naive_dt)
        with pytest.raises(EventValidationError, match="tz-aware"):
            _get_event_time(h)


class TestDefaultStreamKey:
    """Tests for _default_stream_key helper."""

    def test_returns_tuple_of_four_strings(self) -> None:
        from event.validators import _default_stream_key

        e = _evt(eid="e1", symbol="ETHUSDT", timeframe="1h", source="bybit")
        key = _default_stream_key(e)
        assert isinstance(key, tuple)
        assert len(key) == 4

    def test_key_contains_symbol_and_timeframe(self) -> None:
        from event.validators import _default_stream_key

        e = _evt(symbol="BTCUSDT", timeframe="15m", source="binance")
        key = _default_stream_key(e)
        assert "BTCUSDT" in key
        assert "15m" in key
        assert "binance" in key

    def test_event_without_symbol(self) -> None:
        from event.validators import _default_stream_key

        @dataclass(frozen=True)
        class _NoSymbolEvent:
            header: EventHeader

        e = _NoSymbolEvent(header=_make_real_header())
        key = _default_stream_key(e)
        assert key[1] == ""

    def test_event_without_timeframe(self) -> None:
        from event.validators import _default_stream_key

        @dataclass(frozen=True)
        class _NoTfEvent:
            header: EventHeader
            symbol: str = "ETHUSDT"

        e = _NoTfEvent(header=_make_real_header())
        key = _default_stream_key(e)
        assert key[2] == ""


class TestRequiredHeaderFieldsValidator:
    """Tests for RequiredHeaderFieldsValidator."""

    def test_passes_valid_event(self) -> None:
        from event.validators import RequiredHeaderFieldsValidator

        v = RequiredHeaderFieldsValidator()
        e = _evt()
        v.validate(e)  # should not raise

    def test_raises_when_missing_header(self) -> None:
        from event.validators import RequiredHeaderFieldsValidator

        v = RequiredHeaderFieldsValidator()

        class _Bare:
            pass

        with pytest.raises(EventValidationError):
            v.validate(_Bare())

    def test_raises_when_event_id_wrong_type(self) -> None:
        from event.validators import RequiredHeaderFieldsValidator

        v = RequiredHeaderFieldsValidator()
        h = _make_real_header()
        object.__setattr__(h, "event_id", 12345)  # not a str
        e = _ValidatorEvent(header=h)
        with pytest.raises(EventValidationError, match="event_id"):
            v.validate(e)

    def test_passes_when_event_id_is_none(self) -> None:
        """event_id=None should pass (not required by this validator)."""
        from event.validators import RequiredHeaderFieldsValidator

        v = RequiredHeaderFieldsValidator()
        h = _make_real_header()
        object.__setattr__(h, "event_id", None)
        e = _ValidatorEvent(header=h)
        v.validate(e)  # should not raise


class TestLifecycleTerminalValidator:
    """Tests for LifecycleTerminalValidator."""

    def _lifecycle_with_state(
        self, state: LifecycleState, event: Any
    ) -> EventLifecycle:
        lc = MagicMock(spec=EventLifecycle)
        lc.state_of.return_value = state
        return lc

    def test_passes_when_block_terminal_false(self) -> None:
        from event.validators import LifecycleTerminalValidator

        lc = MagicMock(spec=EventLifecycle)
        v = LifecycleTerminalValidator(lifecycle=lc, block_terminal=False)
        e = _evt()
        v.validate(e)
        lc.state_of.assert_not_called()

    def test_passes_when_state_is_none(self) -> None:
        from event.validators import LifecycleTerminalValidator

        lc = self._lifecycle_with_state(None, None)  # type: ignore[arg-type]
        v = LifecycleTerminalValidator(lifecycle=lc)
        e = _evt()
        v.validate(e)  # not terminal → should pass

    def test_passes_when_state_is_enqueued(self) -> None:
        from event.validators import LifecycleTerminalValidator

        lc = self._lifecycle_with_state(LifecycleState.ENQUEUED, None)
        v = LifecycleTerminalValidator(lifecycle=lc)
        e = _evt()
        v.validate(e)  # not terminal → should pass

    def test_raises_when_dispatched(self) -> None:
        from event.validators import LifecycleTerminalValidator

        lc = self._lifecycle_with_state(LifecycleState.DISPATCHED, None)
        v = LifecycleTerminalValidator(lifecycle=lc)
        e = _evt()
        with pytest.raises(EventValidationError, match="终态"):
            v.validate(e)

    def test_raises_when_failed(self) -> None:
        from event.validators import LifecycleTerminalValidator

        lc = self._lifecycle_with_state(LifecycleState.FAILED, None)
        v = LifecycleTerminalValidator(lifecycle=lc)
        with pytest.raises(EventValidationError, match="终态"):
            v.validate(_evt())

    def test_raises_when_dropped(self) -> None:
        from event.validators import LifecycleTerminalValidator

        lc = self._lifecycle_with_state(LifecycleState.DROPPED, None)
        v = LifecycleTerminalValidator(lifecycle=lc)
        with pytest.raises(EventValidationError, match="终态"):
            v.validate(_evt())


class TestStreamMonotonicTimeValidator:
    """Tests for StreamMonotonicTimeValidator."""

    def _make(self, allow_equal: bool = False, thread_safe: bool = True):
        from event.validators import StreamMonotonicTimeValidator
        return StreamMonotonicTimeValidator(
            allow_equal=allow_equal, thread_safe=thread_safe
        )

    def test_first_event_always_passes(self) -> None:
        v = self._make()
        e = _evt(dt=datetime(2024, 1, 1, tzinfo=timezone.utc))
        v.validate(e)  # no prior state

    def test_subsequent_increasing_time_passes(self) -> None:
        v = self._make()
        e1 = _evt("e1", dt=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc))
        e2 = _evt("e1", dt=datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc))
        v.validate(e1)
        v.validate(e2)

    def test_raises_on_time_going_back(self) -> None:
        v = self._make()
        e1 = _evt("e1", dt=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc))
        e2 = _evt("e1", dt=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc))
        v.validate(e1)
        with pytest.raises(EventValidationError, match="时间"):
            v.validate(e2)

    def test_raises_on_equal_time_when_not_allow_equal(self) -> None:
        v = self._make(allow_equal=False)
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        e1 = _evt("e1", dt=dt)
        e2 = _evt("e1", dt=dt)
        v.validate(e1)
        with pytest.raises(EventValidationError, match="非递增"):
            v.validate(e2)

    def test_passes_on_equal_time_when_allow_equal(self) -> None:
        v = self._make(allow_equal=True)
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        e1 = _evt("e1", dt=dt)
        e2 = _evt("e1", dt=dt)
        v.validate(e1)
        v.validate(e2)  # same time, allow_equal=True → should pass

    def test_raises_on_backward_time_when_allow_equal(self) -> None:
        v = self._make(allow_equal=True)
        e1 = _evt("e1", dt=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc))
        e2 = _evt("e1", dt=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc))
        v.validate(e1)
        with pytest.raises(EventValidationError, match="倒流"):
            v.validate(e2)

    def test_different_streams_are_independent(self) -> None:
        """Two different symbols should not interfere with each other."""
        v = self._make()
        e_eth1 = _evt("e1", dt=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc), symbol="ETHUSDT")
        e_btc1 = _evt("e2", dt=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc), symbol="BTCUSDT")
        v.validate(e_eth1)
        v.validate(e_btc1)  # earlier time but different stream key → should pass

    def test_without_thread_safe(self) -> None:
        v = self._make(thread_safe=False)
        e1 = _evt("e1", dt=datetime(2024, 1, 1, tzinfo=timezone.utc))
        e2 = _evt("e1", dt=datetime(2024, 1, 2, tzinfo=timezone.utc))
        v.validate(e1)
        v.validate(e2)  # should pass


class TestReplayModeValidator:
    """Tests for ReplayModeValidator."""

    def test_passes_when_neither_mode_set(self) -> None:
        from event.validators import ReplayModeValidator

        v = ReplayModeValidator(replay_only=False, live_only=False)
        v.validate(_evt())

    def test_passes_when_no_mode_on_event(self) -> None:
        from event.validators import ReplayModeValidator

        v = ReplayModeValidator(replay_only=True, live_only=False)
        e = _evt(mode=None)
        v.validate(e)  # no mode → default pass

    def test_passes_replay_only_with_replay_mode(self) -> None:
        from event.validators import ReplayModeValidator

        v = ReplayModeValidator(replay_only=True)
        for mode in ("replay", "backtest", "paper_replay"):
            e = _evt(mode=mode)
            v.validate(e)

    def test_raises_replay_only_with_live_mode(self) -> None:
        from event.validators import ReplayModeValidator

        v = ReplayModeValidator(replay_only=True)
        e = _evt(mode="live")
        with pytest.raises(EventValidationError, match="回放模式"):
            v.validate(e)

    def test_passes_live_only_with_live_mode(self) -> None:
        from event.validators import ReplayModeValidator

        v = ReplayModeValidator(live_only=True)
        for mode in ("live", "paper_live", "production"):
            e = _evt(mode=mode)
            v.validate(e)

    def test_raises_live_only_with_replay_mode(self) -> None:
        from event.validators import ReplayModeValidator

        v = ReplayModeValidator(live_only=True)
        e = _evt(mode="backtest")
        with pytest.raises(EventValidationError, match="实盘模式"):
            v.validate(e)

    def test_mode_from_meta_when_header_mode_is_none(self) -> None:
        from event.validators import ReplayModeValidator

        v = ReplayModeValidator(replay_only=True)
        e = _evt(meta={"mode": "live"})
        with pytest.raises(EventValidationError):
            v.validate(e)

    def test_mode_case_insensitive(self) -> None:
        from event.validators import ReplayModeValidator

        v = ReplayModeValidator(replay_only=True)
        e = _evt(mode="REPLAY")
        v.validate(e)


class TestTraceValidator:
    """Tests for TraceValidator."""

    def test_passes_when_no_trace_info(self) -> None:
        from event.validators import TraceValidator

        v = TraceValidator()
        e = _evt()  # no trace fields
        v.validate(e)

    def test_passes_with_valid_trace(self) -> None:
        from event.validators import TraceValidator

        v = TraceValidator()
        e = _evt(trace_id="tid-001", span_id="sid-001", trace_depth=0)
        v.validate(e)

    def test_raises_when_trace_id_empty(self) -> None:
        from event.validators import TraceValidator

        v = TraceValidator()
        e = _evt(span_id="sid-001")  # span_id exists but trace_id is absent
        with pytest.raises(EventValidationError, match="trace_id"):
            v.validate(e)

    def test_raises_when_trace_id_not_str(self) -> None:
        from event.validators import TraceValidator

        v = TraceValidator()
        e = _evt(trace_id=12345)  # not a string
        with pytest.raises(EventValidationError, match="trace_id"):
            v.validate(e)

    def test_raises_when_span_id_invalid_in_meta(self) -> None:
        """span_id="" in meta (not header attr) triggers validation."""
        from event.validators import TraceValidator

        v = TraceValidator()
        # Put span_id in meta so it is not subject to the `or` short-circuit
        e = _evt(trace_id="tid-001", meta={"span_id": ""})
        with pytest.raises(EventValidationError, match="span_id"):
            v.validate(e)

    def test_raises_when_parent_span_id_invalid_in_meta(self) -> None:
        from event.validators import TraceValidator

        v = TraceValidator()
        e = _evt(trace_id="tid-001", meta={"parent_span_id": ""})
        with pytest.raises(EventValidationError, match="parent_span_id"):
            v.validate(e)

    def test_passes_with_parent_span_id(self) -> None:
        from event.validators import TraceValidator

        v = TraceValidator()
        e = _evt(trace_id="tid-001", parent_span_id="psid-001")
        v.validate(e)

    def test_raises_when_trace_depth_not_int(self) -> None:
        from event.validators import TraceValidator

        v = TraceValidator()
        e = _evt(trace_id="tid-001", trace_depth="not-an-int")
        # "not-an-int" → int("not-an-int") → ValueError → EventValidationError
        with pytest.raises(EventValidationError, match="int"):
            v.validate(e)

    def test_raises_when_trace_depth_negative(self) -> None:
        from event.validators import TraceValidator

        v = TraceValidator()
        e = _evt(trace_id="tid-001", trace_depth=-1)
        with pytest.raises(EventValidationError, match="负数"):
            v.validate(e)

    def test_passes_when_trace_depth_zero(self) -> None:
        from event.validators import TraceValidator

        v = TraceValidator()
        e = _evt(trace_id="tid-001", trace_depth=0)
        v.validate(e)

    def test_trace_id_from_meta(self) -> None:
        from event.validators import TraceValidator

        v = TraceValidator()
        # span_id in meta triggers trace_id requirement
        e = _evt(meta={"span_id": "s1"})
        with pytest.raises(EventValidationError, match="trace_id"):
            v.validate(e)


class TestCompositeValidator:
    """Tests for CompositeValidator."""

    def test_passes_when_all_validators_pass(self) -> None:
        from event.validators import CompositeValidator, BaseValidator

        class _PassV(BaseValidator):
            def validate(self, event: Any) -> None:
                pass

        v = CompositeValidator(validators=(_PassV(), _PassV()))
        v.validate(_evt())

    def test_stops_at_first_failure(self) -> None:
        from event.validators import CompositeValidator, BaseValidator

        calls: list[str] = []

        class _PassV(BaseValidator):
            def validate(self, event: Any) -> None:
                calls.append("pass")

        class _FailV(BaseValidator):
            def validate(self, event: Any) -> None:
                raise EventValidationError("fail")

        class _ShouldNotRun(BaseValidator):
            def validate(self, event: Any) -> None:
                calls.append("should-not-run")

        v = CompositeValidator(validators=(_PassV(), _FailV(), _ShouldNotRun()))
        with pytest.raises(EventValidationError):
            v.validate(_evt())
        assert "should-not-run" not in calls

    def test_empty_validators_passes(self) -> None:
        from event.validators import CompositeValidator

        v = CompositeValidator(validators=())
        v.validate(_evt())  # no validators → always passes


class TestBuildDefaultValidators:
    """Tests for build_default_validators factory."""

    def test_builds_without_lifecycle(self) -> None:
        from event.validators import build_default_validators

        ev = build_default_validators()
        e = _evt()
        ev.validate(e)  # should not raise

    def test_builds_with_lifecycle(self) -> None:
        from event.validators import build_default_validators

        lc = MagicMock(spec=EventLifecycle)
        lc.state_of.return_value = LifecycleState.ENQUEUED
        ev = build_default_validators(lifecycle=lc)
        e = _evt()
        ev.validate(e)

    def test_time_monotonic_disabled(self) -> None:
        from event.validators import build_default_validators

        ev = build_default_validators(enable_time_monotonic=False)
        # Same timestamp twice should pass (no monotonic check)
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ev.validate(_evt("e1", dt=dt))
        ev.validate(_evt("e1", dt=dt))

    def test_time_allow_equal(self) -> None:
        from event.validators import build_default_validators

        ev = build_default_validators(time_allow_equal=True)
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ev.validate(_evt("e1", dt=dt))
        ev.validate(_evt("e1", dt=dt))  # equal time ok


class TestEventValidators:
    """Tests for EventValidators facade."""

    def test_delegates_to_inner_validator(self) -> None:
        from event.validators import EventValidators

        inner = MagicMock()
        ev = EventValidators(validator=inner)
        e = _evt()
        ev.validate(e)
        inner.validate.assert_called_once_with(e)

    def test_propagates_validation_error(self) -> None:
        from event.validators import EventValidators

        inner = MagicMock()
        inner.validate.side_effect = EventValidationError("fail")
        ev = EventValidators(validator=inner)
        with pytest.raises(EventValidationError):
            ev.validate(_evt())


# ---------------------------------------------------------------------------
# event/metrics.py
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


# ---------------------------------------------------------------------------
# event/dispatcher.py
# ---------------------------------------------------------------------------

class TestEventDispatcher:
    """Tests for EventDispatcher.

    EventDispatcher calls mark_dispatching / mark_dispatched / mark_failed /
    mark_dropped on its lifecycle object — these are duck-typed methods not on
    EventLifecycle directly, so we use a plain MagicMock (no spec).
    """

    def _make_lifecycle(self) -> MagicMock:
        lc = MagicMock()  # no spec — dispatcher uses duck-typed lifecycle methods
        return lc

    def _make_dispatcher(
        self,
        handlers=None,
        lifecycle=None,
        queue_size=100,
        retry_on_exception=False,
    ):
        from event.dispatcher import EventDispatcher

        lc = lifecycle or self._make_lifecycle()
        h = handlers or []
        return EventDispatcher(
            handlers=h,
            lifecycle=lc,
            queue_size=queue_size,
            retry_on_exception=retry_on_exception,
        )

    def _make_event(self, eid: str = "ev-001") -> MagicMock:
        e = MagicMock()
        e.header.event_id = eid
        return e

    # --- start/stop ---

    def test_start_creates_thread(self) -> None:
        d = self._make_dispatcher()
        try:
            d.start()
            assert d._thread is not None
            assert d._thread.is_alive()
        finally:
            with patch("infra.threading_utils.safe_join_thread"):
                d.stop()

    def test_double_start_is_idempotent(self) -> None:
        d = self._make_dispatcher()
        try:
            d.start()
            thread1 = d._thread
            d.start()  # second call should be no-op
            assert d._thread is thread1
        finally:
            with patch("infra.threading_utils.safe_join_thread"):
                d.stop()

    def test_stop_when_not_started_is_noop(self) -> None:
        d = self._make_dispatcher()
        d.stop()  # should not raise

    def test_stop_sets_started_false(self) -> None:
        d = self._make_dispatcher()
        d.start()
        with patch("infra.threading_utils.safe_join_thread"):
            d.stop()
        assert not d._started

    # --- submit ---

    def test_submit_raises_when_not_started(self) -> None:
        d = self._make_dispatcher()
        e = self._make_event()
        with pytest.raises(EventDispatchError, match="尚未启动"):
            d.submit(e)

    def test_submit_raises_and_marks_dropped_on_full_queue(self) -> None:
        lc = self._make_lifecycle()
        d = self._make_dispatcher(lifecycle=lc, queue_size=1)
        d._started = True
        e1 = self._make_event("e1")
        e2 = self._make_event("e2")
        d._queue.put(e1, block=False)  # fill the queue manually
        with pytest.raises(EventDispatchError, match="queue full"):
            d.submit(e2)
        lc.mark_dropped.assert_called_once_with(e2)

    def test_submit_adds_to_queue(self) -> None:
        d = self._make_dispatcher()
        d._started = True
        e = self._make_event()
        d.submit(e)
        assert d._queue.qsize() == 1

    # --- _dispatch_one ---

    def test_dispatch_one_calls_handlers(self) -> None:
        lc = self._make_lifecycle()
        results: list[str] = []
        handler = lambda ev: results.append("called")  # noqa: E731
        d = self._make_dispatcher(handlers=[handler], lifecycle=lc)
        e = self._make_event()
        d._dispatch_one(e)
        assert results == ["called"]
        lc.mark_dispatched.assert_called_once_with(e)

    def test_dispatch_one_marks_dispatching_first(self) -> None:
        lc = self._make_lifecycle()
        order: list[str] = []
        lc.mark_dispatching.side_effect = lambda ev: order.append("dispatching")
        lc.mark_dispatched.side_effect = lambda ev: order.append("dispatched")
        handler = lambda ev: order.append("handler")  # noqa: E731
        d = self._make_dispatcher(handlers=[handler], lifecycle=lc)
        e = self._make_event()
        d._dispatch_one(e)
        assert order == ["dispatching", "handler", "dispatched"]

    def test_dispatch_one_marks_failed_on_exception(self) -> None:
        lc = self._make_lifecycle()

        def _fail(ev: Any) -> None:
            raise RuntimeError("handler error")

        d = self._make_dispatcher(handlers=[_fail], lifecycle=lc)
        e = self._make_event()
        d._dispatch_one(e)
        lc.mark_failed.assert_called_once_with(e)
        lc.mark_dispatched.assert_not_called()

    def test_dispatch_one_retry_requeues_event(self) -> None:
        lc = self._make_lifecycle()
        call_count = 0

        def _fail(ev: Any) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        d = self._make_dispatcher(
            handlers=[_fail], lifecycle=lc, retry_on_exception=True
        )
        d._started = True
        e = self._make_event()
        d._dispatch_one(e)
        # Should have been re-queued
        assert d._queue.qsize() == 1
        lc.mark_failed.assert_called_once()

    def test_dispatch_one_retry_drops_when_queue_full(self) -> None:
        lc = self._make_lifecycle()

        def _fail(ev: Any) -> None:
            raise RuntimeError("fail")

        d = self._make_dispatcher(
            handlers=[_fail], lifecycle=lc, queue_size=1, retry_on_exception=True
        )
        # Pre-fill the queue so put(block=False) raises queue.Full
        filler = self._make_event("filler")
        d._queue.put(filler, block=False)
        e = self._make_event("retry-target")
        d._dispatch_one(e)
        # Should be dropped (queue full during retry)
        lc.mark_dropped.assert_called_once_with(e)

    def test_multiple_handlers_called_in_order(self) -> None:
        lc = self._make_lifecycle()
        order: list[int] = []
        d = self._make_dispatcher(
            handlers=[
                lambda ev: order.append(1),
                lambda ev: order.append(2),
                lambda ev: order.append(3),
            ],
            lifecycle=lc,
        )
        e = self._make_event()
        d._dispatch_one(e)
        assert order == [1, 2, 3]

    def test_handler_exception_stops_subsequent_handlers(self) -> None:
        """After a handler raises, remaining handlers are NOT called."""
        lc = self._make_lifecycle()
        called: list[str] = []

        def _fail(ev: Any) -> None:
            raise RuntimeError("fail")

        d = self._make_dispatcher(
            handlers=[
                lambda ev: called.append("first"),
                _fail,
                lambda ev: called.append("third"),
            ],
            lifecycle=lc,
        )
        e = self._make_event()
        d._dispatch_one(e)
        assert "third" not in called

    def test_end_to_end_dispatch_via_thread(self) -> None:
        """Integration: start dispatcher, submit event, verify handler called."""
        lc = self._make_lifecycle()
        received: list[Any] = []
        barrier = threading.Event()

        def _handler(ev: Any) -> None:
            received.append(ev)
            barrier.set()

        d = self._make_dispatcher(handlers=[_handler], lifecycle=lc, queue_size=10)
        try:
            d.start()
            e = self._make_event()
            d.submit(e)
            barrier.wait(timeout=2.0)
            assert len(received) == 1
            assert received[0] is e
        finally:
            with patch("infra.threading_utils.safe_join_thread"):
                d.stop()
