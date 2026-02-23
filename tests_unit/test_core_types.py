"""Tests for core.types — Envelope, EventMetadata, TraceContext, Symbol."""
from __future__ import annotations

from datetime import datetime, timezone

from core.types import (
    Envelope,
    EventKind,
    EventMetadata,
    Priority,
    Route,
    Symbol,
    TraceContext,
    VenueSymbol,
)


# ── TraceContext ──────────────────────────────────────────


class TestTraceContext:
    def test_new_root_generates_unique_ids(self) -> None:
        t1 = TraceContext.new_root()
        t2 = TraceContext.new_root()
        assert t1.trace_id != t2.trace_id
        assert t1.span_id != t2.span_id
        assert t1.parent_span_id is None

    def test_child_span_preserves_trace_id(self) -> None:
        root = TraceContext.new_root()
        child = root.child_span()
        assert child.trace_id == root.trace_id
        assert child.parent_span_id == root.span_id
        assert child.span_id != root.span_id

    def test_grandchild_chain(self) -> None:
        root = TraceContext.new_root()
        child = root.child_span()
        grandchild = child.child_span()
        assert grandchild.trace_id == root.trace_id
        assert grandchild.parent_span_id == child.span_id

    def test_frozen(self) -> None:
        tc = TraceContext.new_root()
        try:
            tc.trace_id = "hack"  # type: ignore[misc]
            assert False, "Should raise"
        except AttributeError:
            pass


# ── EventMetadata ─────────────────────────────────────────


class TestEventMetadata:
    def test_create_defaults(self) -> None:
        m = EventMetadata.create(source="test")
        assert m.source == "test"
        assert m.event_id
        assert m.trace.trace_id
        assert isinstance(m.timestamp, datetime)

    def test_create_with_trace(self) -> None:
        trace = TraceContext.new_root()
        m = EventMetadata.create(source="live", trace=trace)
        assert m.trace.trace_id == trace.trace_id

    def test_create_with_causation(self) -> None:
        m = EventMetadata.create(source="decision", causation_id="abc123")
        assert m.causation_id == "abc123"

    def test_unique_event_ids(self) -> None:
        ids = {EventMetadata.create(source="t").event_id for _ in range(100)}
        assert len(ids) == 100


# ── Envelope ──────────────────────────────────────────────


class TestEnvelope:
    def test_envelope_wraps_event(self) -> None:
        payload = {"symbol": "BTCUSDT", "close": 50000}
        meta = EventMetadata.create(source="test")
        env = Envelope(event=payload, metadata=meta, kind=EventKind.MARKET)
        assert env.event == payload
        assert env.kind == EventKind.MARKET
        assert env.priority == Priority.NORMAL

    def test_envelope_priority_override(self) -> None:
        meta = EventMetadata.create(source="risk")
        env = Envelope(event="kill", metadata=meta, kind=EventKind.RISK, priority=Priority.CRITICAL)
        assert env.priority == Priority.CRITICAL

    def test_envelope_properties(self) -> None:
        meta = EventMetadata.create(source="test")
        env = Envelope(event=42, metadata=meta, kind=EventKind.CONTROL)
        assert env.event_id == meta.event_id
        assert env.trace_id == meta.trace.trace_id
        assert env.timestamp == meta.timestamp

    def test_envelope_frozen(self) -> None:
        meta = EventMetadata.create(source="test")
        env = Envelope(event=1, metadata=meta, kind=EventKind.MARKET)
        try:
            env.event = 2  # type: ignore[misc]
            assert False, "Should raise"
        except AttributeError:
            pass


# ── Symbol ────────────────────────────────────────────────


class TestSymbol:
    def test_parse_btcusdt(self) -> None:
        s = Symbol.parse("BTCUSDT")
        assert s.base == "BTC"
        assert s.quote == "USDT"
        assert s.canonical == "BTCUSDT"

    def test_parse_with_slash(self) -> None:
        s = Symbol.parse("ETH/USDT")
        assert s.base == "ETH"
        assert s.quote == "USDT"

    def test_parse_with_dash(self) -> None:
        s = Symbol.parse("SOL-USDT")
        assert s.base == "SOL"
        assert s.quote == "USDT"

    def test_parse_btc_quote(self) -> None:
        s = Symbol.parse("ETHBTC")
        assert s.base == "ETH"
        assert s.quote == "BTC"

    def test_parse_invalid_raises(self) -> None:
        try:
            Symbol.parse("X")
            assert False, "Should raise"
        except ValueError:
            pass

    def test_str(self) -> None:
        s = Symbol(base="BTC", quote="USDT")
        assert str(s) == "BTCUSDT"


class TestVenueSymbol:
    def test_str(self) -> None:
        vs = VenueSymbol(venue="binance", symbol=Symbol(base="BTC", quote="USDT"), raw="BTCUSDT")
        assert str(vs) == "binance:BTCUSDT"
