"""Tests for core/bootstrap, core/types, core/clock, core/interceptors."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.bootstrap import SystemContext, bootstrap, bootstrap_test
from core.clock import SimulatedClock
from core.interceptors import (
    EventKindGate,
    InterceptAction,
    InterceptorChain,
    InterceptResult,
    PassthroughInterceptor,
)
from core.types import (
    Envelope,
    EventKind,
    EventMetadata,
    Priority,
    Symbol,
    TraceContext,
)


# ── Helpers ──────────────────────────────────────────────────

def _env(kind=EventKind.MARKET, priority=Priority.NORMAL):
    meta = EventMetadata.create(source="test")
    return Envelope(event={}, metadata=meta, kind=kind, priority=priority)


# ── bootstrap_test ───────────────────────────────────────────

class TestBootstrapTest:
    def test_returns_system_context(self):
        ctx = bootstrap_test()
        assert isinstance(ctx, SystemContext)
        assert ctx.config is not None
        assert ctx.bus is not None
        assert ctx.chain is not None
        assert ctx.saga_manager is not None
        assert ctx.effects is not None

    def test_custom_defaults_merged(self):
        ctx = bootstrap_test(defaults={"bus.capacity": 500})
        assert ctx.config.get("bus.capacity", int) == 500


class TestBootstrap:
    def test_with_defaults(self):
        ctx = bootstrap(defaults={"bus.capacity": 2000})
        assert ctx.config.get("bus.capacity", int) == 2000

    def test_default_config_applied(self):
        ctx = bootstrap()
        # Check a known default from _DEFAULT_CONFIG
        assert ctx.config.get("saga.max_completed", int) == 10_000


# ── TraceContext ─────────────────────────────────────────────

class TestTraceContext:
    def test_new_root(self):
        tc = TraceContext.new_root()
        assert len(tc.trace_id) == 32  # uuid4 hex
        assert len(tc.span_id) == 16
        assert tc.parent_span_id is None

    def test_child_span(self):
        root = TraceContext.new_root()
        child = root.child_span()
        assert child.trace_id == root.trace_id
        assert child.parent_span_id == root.span_id
        assert child.span_id != root.span_id

    def test_child_inherits_baggage(self):
        root = TraceContext(trace_id="t", span_id="s", baggage=(("key", "val"),))
        child = root.child_span()
        assert child.baggage == (("key", "val"),)

    def test_unique_roots(self):
        r1 = TraceContext.new_root()
        r2 = TraceContext.new_root()
        assert r1.trace_id != r2.trace_id


# ── EventMetadata ────────────────────────────────────────────

class TestEventMetadata:
    def test_create_factory(self):
        meta = EventMetadata.create(source="binance_ws")
        assert meta.source == "binance_ws"
        assert len(meta.event_id) == 32
        assert meta.timestamp.tzinfo is not None

    def test_unique_event_ids(self):
        m1 = EventMetadata.create(source="a")
        m2 = EventMetadata.create(source="a")
        assert m1.event_id != m2.event_id

    def test_custom_trace(self):
        tc = TraceContext.new_root()
        meta = EventMetadata.create(source="x", trace=tc)
        assert meta.trace is tc

    def test_causation_id(self):
        m1 = EventMetadata.create(source="a")
        m2 = EventMetadata.create(source="b", causation_id=m1.event_id)
        assert m2.causation_id == m1.event_id


# ── Envelope ─────────────────────────────────────────────────

class TestEnvelope:
    def test_wrap_event(self):
        meta = EventMetadata.create(source="test")
        env = Envelope(event={"price": 100}, metadata=meta, kind=EventKind.MARKET)
        assert env.event == {"price": 100}
        assert env.kind == EventKind.MARKET
        assert env.priority == Priority.NORMAL

    def test_properties(self):
        meta = EventMetadata.create(source="test")
        env = Envelope(event={}, metadata=meta, kind=EventKind.FILL, priority=Priority.HIGH)
        assert env.event_id == meta.event_id
        assert env.trace_id == meta.trace.trace_id
        assert env.timestamp == meta.timestamp


# ── Symbol ───────────────────────────────────────────────────

class TestSymbol:
    def test_parse_slash_format(self):
        s = Symbol.parse("BTC/USDT")
        assert s.base == "BTC"
        assert s.quote == "USDT"

    def test_parse_dash_format(self):
        s = Symbol.parse("ETH-USDT")
        assert s.base == "ETH"
        assert s.quote == "USDT"

    def test_parse_concatenated(self):
        s = Symbol.parse("BTCUSDT")
        assert s.base == "BTC"
        assert s.quote == "USDT"

    def test_canonical_property(self):
        s = Symbol(base="BTC", quote="USDT")
        assert s.canonical == "BTCUSDT"
        assert str(s) == "BTCUSDT"

    def test_parse_invalid_raises(self):
        with pytest.raises(ValueError):
            Symbol.parse("X")

    def test_case_insensitive(self):
        s = Symbol.parse("btc/usdt")
        assert s.base == "BTC"
        assert s.quote == "USDT"


# ── SimulatedClock ───────────────────────────────────────────

class TestSimulatedClock:
    def test_default_time(self):
        clock = SimulatedClock()
        assert clock.now() == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_advance(self):
        clock = SimulatedClock()
        clock.advance(timedelta(hours=1))
        expected = datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc)
        assert clock.now() == expected
        assert clock.monotonic() == pytest.approx(3600.0)

    def test_set(self):
        clock = SimulatedClock()
        target = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
        clock.set(target)
        assert clock.now() == target

    def test_set_naive_adds_utc(self):
        clock = SimulatedClock()
        naive = datetime(2025, 1, 1, 0, 0)
        clock.set(naive)
        assert clock.now().tzinfo is not None

    def test_sleep_advances_simulated_time(self):
        clock = SimulatedClock()
        clock.sleep(10.0)
        # Default real_sleep=False → advances simulated time
        assert clock.monotonic() == pytest.approx(10.0)


# ── InterceptorChain ─────────────────────────────────────────

class TestInterceptorChain:
    def test_empty_chain_continues(self):
        chain = InterceptorChain()
        result = chain.run_before(_env(), state={})
        assert result.action == InterceptAction.CONTINUE

    def test_passthrough_continues(self):
        chain = InterceptorChain([PassthroughInterceptor()])
        result = chain.run_before(_env(), state={})
        assert result.action == InterceptAction.CONTINUE

    def test_reject_stops_chain(self):
        class RejectAll:
            @property
            def name(self):
                return "reject_all"
            def before_reduce(self, envelope, state):
                return InterceptResult.reject(self.name, "no")
            def after_reduce(self, envelope, old_state, new_state):
                return InterceptResult.ok(self.name)

        reached = []
        class Tracker:
            @property
            def name(self):
                return "tracker"
            def before_reduce(self, envelope, state):
                reached.append(True)
                return InterceptResult.ok(self.name)
            def after_reduce(self, envelope, old_state, new_state):
                return InterceptResult.ok(self.name)

        chain = InterceptorChain([RejectAll(), Tracker()])
        result = chain.run_before(_env(), state={})
        assert result.action == InterceptAction.REJECT
        assert result.interceptor == "reject_all"
        assert len(reached) == 0  # second interceptor never ran

    def test_chain_ordering(self):
        order = []
        class Numbered:
            def __init__(self, n):
                self._n = n
            @property
            def name(self):
                return f"ic_{self._n}"
            def before_reduce(self, envelope, state):
                order.append(self._n)
                return InterceptResult.ok(self.name)
            def after_reduce(self, envelope, old_state, new_state):
                return InterceptResult.ok(self.name)

        chain = InterceptorChain([Numbered(1), Numbered(2), Numbered(3)])
        chain.run_before(_env(), state={})
        assert order == [1, 2, 3]

    def test_run_after_kill_stops_chain(self):
        class KillAll:
            @property
            def name(self):
                return "kill"
            def before_reduce(self, envelope, state):
                return InterceptResult.ok(self.name)
            def after_reduce(self, envelope, old_state, new_state):
                return InterceptResult.kill(self.name, "emergency")

        chain = InterceptorChain([KillAll(), PassthroughInterceptor()])
        result = chain.run_after(_env(), old_state={}, new_state={})
        assert result.action == InterceptAction.KILL


# ── EventKindGate ────────────────────────────────────────────

class TestEventKindGate:
    def test_blocks_specified_kinds(self):
        gate = EventKindGate(blocked_kinds=frozenset({EventKind.ORDER}))
        env = _env(kind=EventKind.ORDER)
        result = gate.before_reduce(env, state={})
        assert result.action == InterceptAction.REJECT

    def test_allows_non_blocked_kinds(self):
        gate = EventKindGate(blocked_kinds=frozenset({EventKind.ORDER}))
        env = _env(kind=EventKind.MARKET)
        result = gate.before_reduce(env, state={})
        assert result.action == InterceptAction.CONTINUE

    def test_after_reduce_always_ok(self):
        gate = EventKindGate(blocked_kinds=frozenset({EventKind.ORDER}))
        env = _env(kind=EventKind.ORDER)
        result = gate.after_reduce(env, old_state={}, new_state={})
        assert result.action == InterceptAction.CONTINUE

    def test_custom_reason(self):
        gate = EventKindGate(blocked_kinds=frozenset({EventKind.ORDER}), reason="kill-switch active")
        env = _env(kind=EventKind.ORDER)
        result = gate.before_reduce(env, state={})
        assert "kill-switch" in result.reason
