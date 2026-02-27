"""Unit tests for TracingInterceptor → OTel Tracer bridge."""
from __future__ import annotations

from types import SimpleNamespace
from datetime import datetime
from unittest.mock import MagicMock, patch

from core.observability import TracingInterceptor, SpanRecord
from infra.tracing.otel import Tracer


def _make_envelope(event_id: str = "evt-1", kind: str = "MARKET"):
    return SimpleNamespace(
        event_id=event_id,
        trace_id="trace-abc",
        kind=SimpleNamespace(name=kind),
        metadata=SimpleNamespace(
            source="test",
            trace=SimpleNamespace(span_id="span-1", parent_span_id=None),
        ),
    )


class TestTracingOtelBridge:

    def test_span_exported_to_tracer(self):
        tracer = Tracer(service_name="test")
        ti = TracingInterceptor(max_spans=100, tracer=tracer)

        envelope = _make_envelope()
        ti.before_reduce(envelope, state=None)
        ti.after_reduce(envelope, old_state=None, new_state=None)

        # SpanRecord should be in internal buffer
        assert len(ti.spans) == 1
        assert ti.spans[0].event_kind == "MARKET"

        # Tracer should have received the exported span
        otel_spans = tracer.get_spans(limit=10)
        assert len(otel_spans) == 1
        assert otel_spans[0].operation == "pipeline.MARKET"
        assert otel_spans[0].tags["event_id"] == "evt-1"

    def test_no_tracer_still_works(self):
        ti = TracingInterceptor(max_spans=100)  # no tracer

        envelope = _make_envelope()
        ti.before_reduce(envelope, state=None)
        result = ti.after_reduce(envelope, old_state=None, new_state=None)

        assert len(ti.spans) == 1
        assert result.action.name == "CONTINUE"

    def test_tracer_exception_does_not_break_pipeline(self):
        broken_tracer = MagicMock()
        broken_tracer.start_span.side_effect = RuntimeError("otel broken")

        ti = TracingInterceptor(max_spans=100, tracer=broken_tracer)

        envelope = _make_envelope()
        ti.before_reduce(envelope, state=None)
        result = ti.after_reduce(envelope, old_state=None, new_state=None)

        # Pipeline should continue even with broken tracer
        assert result.action.name == "CONTINUE"
        assert len(ti.spans) == 1

    def test_multiple_events_exported(self):
        tracer = Tracer(service_name="test")
        ti = TracingInterceptor(max_spans=100, tracer=tracer)

        for i in range(5):
            env = _make_envelope(event_id=f"evt-{i}", kind="MARKET")
            ti.before_reduce(env, state=None)
            ti.after_reduce(env, old_state=None, new_state=None)

        assert len(ti.spans) == 5
        assert len(tracer.get_spans(limit=100)) == 5

    def test_span_tags_include_source(self):
        tracer = Tracer(service_name="test")
        ti = TracingInterceptor(max_spans=100, tracer=tracer)

        envelope = _make_envelope()
        ti.before_reduce(envelope, state=None)
        ti.after_reduce(envelope, old_state=None, new_state=None)

        span = tracer.get_spans(limit=1)[0]
        assert span.tags["source"] == "test"
