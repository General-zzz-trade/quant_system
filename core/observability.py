"""Structured observability interceptors — tracing and logging.

These interceptors inject into the pipeline via the InterceptorChain.
They are non-blocking and never reject events — their purpose is
recording, not gate-keeping.

Design:
    - ``TracingInterceptor``: ensures every envelope carries a TraceContext
      and records span timing for before/after reduce phases.
    - ``LoggingInterceptor``: structured-logs intercept decisions and state
      transitions for audit and debugging.
    - ``MetricsInterceptor``: records pipeline throughput and latency to
      the Effects metrics sink.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from core.interceptors import InterceptAction, InterceptResult
from core.types import Envelope


# ── Tracing Interceptor ─────────────────────────────────

class TracingInterceptor:
    """Ensures trace context propagation through the pipeline.

    Every event that enters the pipeline gets a trace span recorded.
    If the event already has a TraceContext (via its metadata), a child
    span is created.  Otherwise a new root trace is started.

    Span timings are collected in ``spans`` for export or inspection.

    Parameters
    ----------
    max_spans : int
        Maximum retained spans (ring-buffer style).  Oldest are evicted.
    """

    def __init__(
        self,
        *,
        max_spans: int = 10_000,
        tracer: Optional[Any] = None,
        active_ttl_seconds: float = 60.0,
    ) -> None:
        self._max_spans = max_spans
        self._spans: List[SpanRecord] = []
        self._active: Dict[str, float] = {}  # event_id → start_mono
        self._tracer = tracer  # infra.tracing.otel.Tracer (optional)
        self._active_ttl = active_ttl_seconds

    @property
    def name(self) -> str:
        return "tracing"

    @property
    def spans(self) -> List[SpanRecord]:
        return list(self._spans)

    def cleanup_stale(self) -> int:
        """Remove _active entries older than ``active_ttl_seconds``.

        Returns the number of entries removed.  Safe to call at any time;
        intended for periodic maintenance or triggered from ``before_reduce``
        when the dict grows large.
        """
        cutoff = time.monotonic() - self._active_ttl
        stale = [eid for eid, t in self._active.items() if t < cutoff]
        for eid in stale:
            del self._active[eid]
        return len(stale)

    def before_reduce(self, envelope: Envelope[Any], state: Any) -> InterceptResult:
        event_id = envelope.event_id
        # Auto-evict stale entries to prevent unbounded growth when
        # after_reduce is never called (e.g. dropped / short-circuited events).
        if len(self._active) > 100:
            self.cleanup_stale()
        self._active[event_id] = time.monotonic()
        return InterceptResult.ok(self.name)

    def after_reduce(
        self,
        envelope: Envelope[Any],
        old_state: Any,
        new_state: Any,
    ) -> InterceptResult:
        event_id = envelope.event_id
        start = self._active.pop(event_id, None)
        duration_ms = (time.monotonic() - start) * 1000.0 if start is not None else 0.0

        span = SpanRecord(
            trace_id=envelope.trace_id,
            span_id=envelope.metadata.trace.span_id,
            parent_span_id=envelope.metadata.trace.parent_span_id,
            event_id=event_id,
            event_kind=envelope.kind.name,
            source=envelope.metadata.source,
            duration_ms=duration_ms,
        )

        self._spans.append(span)
        if len(self._spans) > self._max_spans:
            self._spans = self._spans[-self._max_spans:]

        if self._tracer is not None:
            self._export_to_tracer(span)

        return InterceptResult.ok(self.name)

    def _export_to_tracer(self, span: SpanRecord) -> None:
        if self._tracer is None:
            return
        try:
            with self._tracer.start_span(
                f"pipeline.{span.event_kind}",
                trace_id=span.trace_id,
                parent_id=span.parent_span_id or "",
                tags={"event_id": span.event_id, "source": span.source},
            ):
                pass  # SpanContext __exit__ records it
        except Exception:
            pass  # tracing must never break pipeline


@dataclass(frozen=True, slots=True)
class SpanRecord:
    """A recorded pipeline span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    event_id: str
    event_kind: str
    source: str
    duration_ms: float


# ── Logging Interceptor ──────────────────────────────────

class LoggingInterceptor:
    """Structured-logs every pipeline intercept for audit.

    Logs are appended to an internal buffer and optionally forwarded
    to an external log function (e.g., Effects.log).

    Parameters
    ----------
    log_fn : callable, optional
        External log sink.  Signature: ``(level: str, message: str, **kw) -> None``
    max_entries : int
        Maximum log entries retained.
    log_continue : bool
        If False (default), only non-CONTINUE results are logged.
        If True, every intercept is logged (verbose).
    """

    def __init__(
        self,
        *,
        log_fn: Optional[Callable[..., None]] = None,
        max_entries: int = 5_000,
        log_continue: bool = False,
    ) -> None:
        self._log_fn = log_fn
        self._max_entries = max_entries
        self._log_continue = log_continue
        self._entries: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "logging"

    @property
    def entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def before_reduce(self, envelope: Envelope[Any], state: Any) -> InterceptResult:
        # Logging interceptor never blocks — it's observability-only
        return InterceptResult.ok(self.name)

    def after_reduce(
        self,
        envelope: Envelope[Any],
        old_state: Any,
        new_state: Any,
    ) -> InterceptResult:
        return InterceptResult.ok(self.name)

    def record(self, phase: str, envelope: Envelope[Any], result: InterceptResult) -> None:
        """Record an intercept decision from the chain.

        This is meant to be called by the pipeline runner after the
        interceptor chain produces a result, not by the interceptor itself.
        """
        if not self._log_continue and result.action == InterceptAction.CONTINUE:
            return

        entry = {
            "phase": phase,
            "event_id": envelope.event_id,
            "event_kind": envelope.kind.name,
            "interceptor": result.interceptor,
            "action": result.action.name,
            "reason": result.reason,
        }

        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        if self._log_fn is not None:
            level = "info" if result.action == InterceptAction.CONTINUE else "warning"
            self._log_fn(
                level,
                f"pipeline.{phase}: {result.interceptor} → {result.action.name}",
                **entry,
            )


# ── Metrics Interceptor ──────────────────────────────────

class MetricsInterceptor:
    """Records pipeline throughput and latency to a metrics sink.

    Parameters
    ----------
    metrics : MetricsEffect-like object
        Must have ``counter(name, value, **tags)`` and
        ``histogram(name, value, **tags)`` methods.
    """

    def __init__(self, metrics: Any) -> None:
        self._metrics = metrics
        self._active: Dict[str, float] = {}

    @property
    def name(self) -> str:
        return "metrics"

    def before_reduce(self, envelope: Envelope[Any], state: Any) -> InterceptResult:
        self._active[envelope.event_id] = time.monotonic()
        self._metrics.counter("pipeline.events_in", 1, kind=envelope.kind.name)
        return InterceptResult.ok(self.name)

    def after_reduce(
        self,
        envelope: Envelope[Any],
        old_state: Any,
        new_state: Any,
    ) -> InterceptResult:
        start = self._active.pop(envelope.event_id, None)
        if start is not None:
            dt_ms = (time.monotonic() - start) * 1000.0
            self._metrics.histogram("pipeline.reduce_ms", dt_ms, kind=envelope.kind.name)
        self._metrics.counter("pipeline.events_out", 1, kind=envelope.kind.name)
        return InterceptResult.ok(self.name)
