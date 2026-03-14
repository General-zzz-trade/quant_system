# execution/observability/tracing.py
"""Execution tracing — track operation lifecycle through the system."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True, slots=True)
class Span:
    """追踪跨度 — 单个操作的生命周期记录。"""
    span_id: str
    trace_id: str
    operation: str
    start_ts: float
    end_ts: float
    status: str = "ok"          # "ok" / "error"
    tags: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        return (self.end_ts - self.start_ts) * 1000


class SpanBuilder:
    """Span 构建器 — 自动计时。"""

    def __init__(self, operation: str, trace_id: Optional[str] = None) -> None:
        self._span_id = uuid.uuid4().hex[:12]
        self._trace_id = trace_id or uuid.uuid4().hex[:16]
        self._operation = operation
        self._start = time.monotonic()
        self._tags: Dict[str, str] = {}
        self._error: Optional[str] = None
        self._status = "ok"

    def tag(self, key: str, value: str) -> SpanBuilder:
        self._tags[key] = value
        return self

    def error(self, msg: str) -> SpanBuilder:
        self._error = msg
        self._status = "error"
        return self

    def finish(self) -> Span:
        return Span(
            span_id=self._span_id,
            trace_id=self._trace_id,
            operation=self._operation,
            start_ts=self._start,
            end_ts=time.monotonic(),
            status=self._status,
            tags=dict(self._tags),
            error=self._error,
        )


class Tracer:
    """简单追踪器 — 收集所有 span。"""

    def __init__(self, max_spans: int = 10000) -> None:
        self._spans: List[Span] = []
        self._max = max_spans

    def start_span(
        self, operation: str, *, trace_id: Optional[str] = None,
    ) -> SpanBuilder:
        return SpanBuilder(operation, trace_id)

    def record(self, span: Span) -> None:
        self._spans.append(span)
        if len(self._spans) > self._max:
            self._spans = self._spans[-self._max:]

    def query(
        self,
        *,
        operation: Optional[str] = None,
        trace_id: Optional[str] = None,
        limit: int = 100,
    ) -> Sequence[Span]:
        results: list[Span] = []
        for s in reversed(self._spans):
            if operation and s.operation != operation:
                continue
            if trace_id and s.trace_id != trace_id:
                continue
            results.append(s)
            if len(results) >= limit:
                break
        return results
