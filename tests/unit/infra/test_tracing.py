"""Tests for lightweight distributed tracing."""
from __future__ import annotations

import time

import pytest

from infra.tracing.otel import Span, SpanContext, Tracer


class TestSpan:
    def test_duration_ms(self):
        span = Span(
            trace_id="t1",
            span_id="s1",
            start_ts=1.0,
            end_ts=1.05,
        )
        assert span.duration_ms == pytest.approx(50.0)

    def test_frozen(self):
        span = Span(trace_id="t1", span_id="s1")
        with pytest.raises(AttributeError):
            span.trace_id = "t2"  # type: ignore[misc]

    def test_default_tags_empty(self):
        span = Span(trace_id="t1", span_id="s1")
        assert span.tags == {}


class TestTracer:
    def test_start_span_returns_context(self):
        tracer = Tracer("test-service")
        ctx = tracer.start_span("my-op")
        assert isinstance(ctx, SpanContext)

    def test_span_context_records_duration(self):
        tracer = Tracer("test-service")
        with tracer.start_span("compute") as span:
            assert span.trace_id
            assert span.span_id
            assert span.operation == "compute"
            time.sleep(0.01)

        spans = tracer.get_spans()
        assert len(spans) == 1
        assert spans[0].duration_ms > 0
        assert spans[0].end_ts > spans[0].start_ts

    def test_parent_id_propagated(self):
        tracer = Tracer("test-service")
        with tracer.start_span("parent") as parent_span:
            pass

        with tracer.start_span("child", parent_id=parent_span.span_id) as child:
            pass

        spans = tracer.get_spans()
        assert len(spans) == 2
        child_span = spans[1]
        assert child_span.parent_id == parent_span.span_id

    def test_trace_id_shared_when_provided(self):
        tracer = Tracer("test-service")
        tid = "shared-trace-id"
        with tracer.start_span("a", trace_id=tid):
            pass
        with tracer.start_span("b", trace_id=tid):
            pass

        spans = tracer.get_spans()
        assert all(s.trace_id == tid for s in spans)

    def test_tags_recorded(self):
        tracer = Tracer("test-service")
        with tracer.start_span("op", tags={"env": "test", "symbol": "BTC"}):
            pass

        spans = tracer.get_spans()
        assert spans[0].tags["env"] == "test"
        assert spans[0].tags["symbol"] == "BTC"

    def test_error_tag_on_exception(self):
        tracer = Tracer("test-service")
        with pytest.raises(ValueError):
            with tracer.start_span("failing") as span:
                raise ValueError("boom")

        spans = tracer.get_spans()
        assert len(spans) == 1
        assert "error" in spans[0].tags
        assert "boom" in spans[0].tags["error"]

    def test_get_spans_limit(self):
        tracer = Tracer("test-service")
        for i in range(10):
            with tracer.start_span(f"op-{i}"):
                pass

        assert len(tracer.get_spans(limit=3)) == 3
        assert len(tracer.get_spans(limit=100)) == 10

    def test_clear_spans(self):
        tracer = Tracer("test-service")
        with tracer.start_span("op"):
            pass
        assert len(tracer.get_spans()) == 1
        tracer.clear()
        assert len(tracer.get_spans()) == 0

    def test_fallback_without_otel(self):
        tracer = Tracer("test-service")
        # OTel likely not installed in test env
        # Either way, the tracer should work
        with tracer.start_span("fallback-op") as span:
            assert span.trace_id
            assert span.operation == "fallback-op"

        assert len(tracer.get_spans()) == 1
