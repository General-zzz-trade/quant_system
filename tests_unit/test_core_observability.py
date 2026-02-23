"""Tests for core.observability — TracingInterceptor, LoggingInterceptor, MetricsInterceptor."""
from __future__ import annotations

import time
from typing import Any

from core.effects import InMemoryMetrics
from core.interceptors import InterceptAction, InterceptResult, InterceptorChain
from core.observability import (
    LoggingInterceptor,
    MetricsInterceptor,
    SpanRecord,
    TracingInterceptor,
)
from core.types import Envelope, EventKind, EventMetadata, TraceContext


# ── Helpers ──────────────────────────────────────────────

def _make_envelope(kind: EventKind = EventKind.MARKET, source: str = "test") -> Envelope:
    meta = EventMetadata.create(source=source)
    return Envelope(event={"data": 1}, metadata=meta, kind=kind)


# ── TracingInterceptor Tests ─────────────────────────────


class TestTracingInterceptor:
    def test_span_recorded_after_reduce(self) -> None:
        t = TracingInterceptor(max_spans=100)
        env = _make_envelope()

        r1 = t.before_reduce(env, state=None)
        assert r1.action == InterceptAction.CONTINUE

        r2 = t.after_reduce(env, old_state=None, new_state=None)
        assert r2.action == InterceptAction.CONTINUE

        spans = t.spans
        assert len(spans) == 1
        assert spans[0].event_id == env.event_id
        assert spans[0].trace_id == env.trace_id
        assert spans[0].event_kind == "MARKET"
        assert spans[0].duration_ms >= 0.0

    def test_multiple_spans(self) -> None:
        t = TracingInterceptor(max_spans=100)

        for _ in range(3):
            env = _make_envelope()
            t.before_reduce(env, state=None)
            t.after_reduce(env, old_state=None, new_state=None)

        assert len(t.spans) == 3

    def test_max_spans_eviction(self) -> None:
        t = TracingInterceptor(max_spans=5)

        for _ in range(10):
            env = _make_envelope()
            t.before_reduce(env, state=None)
            t.after_reduce(env, old_state=None, new_state=None)

        assert len(t.spans) == 5

    def test_span_has_trace_context_fields(self) -> None:
        t = TracingInterceptor(max_spans=100)
        trace = TraceContext.new_root()
        meta = EventMetadata.create(source="test", trace=trace)
        env = Envelope(event="x", metadata=meta, kind=EventKind.CONTROL)

        t.before_reduce(env, state=None)
        t.after_reduce(env, old_state=None, new_state=None)

        span = t.spans[0]
        assert span.trace_id == trace.trace_id
        assert span.span_id == trace.span_id
        assert span.parent_span_id == trace.parent_span_id

    def test_name(self) -> None:
        t = TracingInterceptor()
        assert t.name == "tracing"


# ── LoggingInterceptor Tests ─────────────────────────────


class TestLoggingInterceptor:
    def test_record_non_continue(self) -> None:
        li = LoggingInterceptor()
        env = _make_envelope()
        result = InterceptResult.reject("risk", "too big")
        li.record("before_reduce", env, result)

        entries = li.entries
        assert len(entries) == 1
        assert entries[0]["action"] == "REJECT"
        assert entries[0]["interceptor"] == "risk"

    def test_skip_continue_by_default(self) -> None:
        li = LoggingInterceptor()
        env = _make_envelope()
        result = InterceptResult.ok("passthrough")
        li.record("before_reduce", env, result)

        assert len(li.entries) == 0

    def test_log_continue_enabled(self) -> None:
        li = LoggingInterceptor(log_continue=True)
        env = _make_envelope()
        result = InterceptResult.ok("passthrough")
        li.record("before_reduce", env, result)

        assert len(li.entries) == 1

    def test_external_log_fn_called(self) -> None:
        logged = []

        def log_fn(level: str, message: str, **kw: Any) -> None:
            logged.append((level, message))

        li = LoggingInterceptor(log_fn=log_fn)
        env = _make_envelope()
        result = InterceptResult.kill("emergency", "system halt")
        li.record("before_reduce", env, result)

        assert len(logged) == 1
        assert logged[0][0] == "warning"
        assert "emergency" in logged[0][1]

    def test_max_entries_eviction(self) -> None:
        li = LoggingInterceptor(max_entries=3)

        for i in range(10):
            env = _make_envelope()
            result = InterceptResult.reject("test", f"reason {i}")
            li.record("before_reduce", env, result)

        assert len(li.entries) == 3

    def test_before_and_after_reduce_pass_through(self) -> None:
        li = LoggingInterceptor()
        env = _make_envelope()
        r1 = li.before_reduce(env, state=None)
        r2 = li.after_reduce(env, old_state=None, new_state=None)
        assert r1.action == InterceptAction.CONTINUE
        assert r2.action == InterceptAction.CONTINUE

    def test_name(self) -> None:
        li = LoggingInterceptor()
        assert li.name == "logging"


# ── MetricsInterceptor Tests ─────────────────────────────


class TestMetricsInterceptor:
    def test_counts_events(self) -> None:
        m = InMemoryMetrics()
        mi = MetricsInterceptor(metrics=m)
        env = _make_envelope(kind=EventKind.MARKET)

        mi.before_reduce(env, state=None)
        mi.after_reduce(env, old_state=None, new_state=None)

        snap = m.snapshot()
        # Check counters were incremented
        assert any("pipeline.events_in" in k for k in snap)
        assert any("pipeline.events_out" in k for k in snap)

    def test_records_latency(self) -> None:
        m = InMemoryMetrics()
        mi = MetricsInterceptor(metrics=m)
        env = _make_envelope()

        mi.before_reduce(env, state=None)
        mi.after_reduce(env, old_state=None, new_state=None)

        snap = m.snapshot()
        assert any("pipeline.reduce_ms" in k for k in snap)

    def test_name(self) -> None:
        mi = MetricsInterceptor(metrics=InMemoryMetrics())
        assert mi.name == "metrics"

    def test_chain_integration(self) -> None:
        """MetricsInterceptor works in a chain without interfering."""
        m = InMemoryMetrics()
        mi = MetricsInterceptor(metrics=m)
        chain = InterceptorChain([mi])

        env = _make_envelope()
        result = chain.run_before(env, state=None)
        assert result.action == InterceptAction.CONTINUE

        result = chain.run_after(env, old_state=None, new_state=None)
        assert result.action == InterceptAction.CONTINUE
