# engine/tracing.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import threading
import time
import uuid


def _gen_id() -> str:
    return uuid.uuid4().hex


@dataclass(slots=True)
class Span:
    span_id: str
    name: str
    start_ns: int
    parent_id: Optional[str] = None
    end_ns: Optional[int] = None
    attrs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def finish(self) -> None:
        self.end_ns = time.perf_counter_ns()


@dataclass(slots=True)
class Trace:
    trace_id: str
    spans: Dict[str, Span] = field(default_factory=dict)

    def add_span(self, span: Span) -> None:
        self.spans[span.span_id] = span


class Tracer:
    """
    冻结版 v1.0 Tracer：
    - 只负责关联（trace_id / span_id / parent）
    - 不做采样、不做导出
    """
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._traces: Dict[str, Trace] = {}
        self._local = threading.local()

    # ---- context ----
    def current_trace_id(self) -> Optional[str]:
        return getattr(self._local, "trace_id", None)

    def current_span_id(self) -> Optional[str]:
        return getattr(self._local, "span_id", None)

    # ---- lifecycle ----
    def start_trace(self, name: str, *, attrs: Optional[Dict[str, Any]] = None) -> str:
        with self._lock:
            trace_id = _gen_id()
            trace = Trace(trace_id=trace_id)
            self._traces[trace_id] = trace

        span = Span(
            span_id=_gen_id(),
            name=name,
            start_ns=time.perf_counter_ns(),
            parent_id=None,
            attrs=attrs or {},
        )
        trace.add_span(span)

        self._local.trace_id = trace_id
        self._local.span_id = span.span_id
        return trace_id

    def start_span(self, name: str, *, attrs: Optional[Dict[str, Any]] = None) -> Span:
        trace_id = self.current_trace_id()
        if trace_id is None:
            # 自动补一个 trace（v1.0 容错）
            self.start_trace(name="auto")

        trace_id = self.current_trace_id()
        assert trace_id is not None
        parent = self.current_span_id()

        span = Span(
            span_id=_gen_id(),
            name=name,
            start_ns=time.perf_counter_ns(),
            parent_id=parent,
            attrs=attrs or {},
        )

        with self._lock:
            self._traces[trace_id].add_span(span)

        self._local.span_id = span.span_id
        return span

    def end_span(self, span: Span, *, error: Optional[str] = None) -> None:
        span.finish()
        if error:
            span.error = error
        # 回退到父 span
        self._local.span_id = span.parent_id

    def end_trace(self) -> None:
        self._local.trace_id = None
        self._local.span_id = None

    # ---- read-only snapshot ----
    def snapshot(self, trace_id: str) -> Optional[Trace]:
        with self._lock:
            return self._traces.get(trace_id)
