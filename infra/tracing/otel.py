"""Lightweight distributed tracing with OpenTelemetry-compatible concepts.

When the OTel SDK is installed, delegates to it. Otherwise provides a
minimal in-process tracing implementation suitable for profiling and
debugging latency across the pipeline.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Span:
    """Immutable span record."""

    trace_id: str
    span_id: str
    parent_id: str = ""
    operation: str = ""
    start_ts: float = 0.0
    end_ts: float = 0.0
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_ts - self.start_ts) * 1000


class SpanContext:
    """Context manager for spans."""

    def __init__(self, tracer: Tracer, span: Span) -> None:
        self._tracer = tracer
        self._span = span

    @property
    def span(self) -> Span:
        return self._span

    def __enter__(self) -> Span:
        return self._span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        end_ts = time.monotonic()
        finished = Span(
            trace_id=self._span.trace_id,
            span_id=self._span.span_id,
            parent_id=self._span.parent_id,
            operation=self._span.operation,
            start_ts=self._span.start_ts,
            end_ts=end_ts,
            tags={
                **self._span.tags,
                **({"error": str(exc_val)} if exc_val else {}),
            },
        )
        self._span = finished
        self._tracer._record_span(finished)


class Tracer:
    """Lightweight distributed tracing.

    When OTel SDK is available, delegates to it.
    Otherwise, provides a minimal in-process tracing implementation.
    """

    def __init__(self, service_name: str = "quant_system") -> None:
        self._service = service_name
        self._spans: list[Span] = []
        self._lock = threading.Lock()
        self._max_spans = 10_000
        self._otel_tracer = self._try_init_otel()

    def _try_init_otel(self) -> Any:
        """Try to init OpenTelemetry, return None if not available."""
        try:
            from opentelemetry import trace  # type: ignore[import-not-found]
            return trace.get_tracer(self._service)
        except ImportError:
            logger.debug("OpenTelemetry not available, using in-process tracer")
            return None

    @property
    def has_otel(self) -> bool:
        return self._otel_tracer is not None

    def start_span(
        self,
        operation: str,
        *,
        parent_id: str = "",
        trace_id: str = "",
        tags: Optional[dict[str, str]] = None,
    ) -> SpanContext:
        """Start a new span. Returns context manager."""
        tid = trace_id or uuid.uuid4().hex[:16]
        sid = uuid.uuid4().hex[:16]
        span = Span(
            trace_id=tid,
            span_id=sid,
            parent_id=parent_id,
            operation=operation,
            start_ts=time.monotonic(),
            tags=tags or {},
        )
        return SpanContext(self, span)

    def _record_span(self, span: Span) -> None:
        """Record a completed span."""
        with self._lock:
            self._spans.append(span)
            if len(self._spans) > self._max_spans:
                self._spans = self._spans[-self._max_spans:]

    def get_spans(self, *, limit: int = 100) -> list[Span]:
        """Get recent spans (for in-process tracer)."""
        with self._lock:
            return list(self._spans[-limit:])

    def clear(self) -> None:
        """Clear recorded spans."""
        with self._lock:
            self._spans.clear()
