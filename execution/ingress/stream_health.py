# execution/ingress/stream_health.py
"""Stream health monitoring for execution data feeds."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class StreamStatus(str, Enum):
    """数据流健康状态。"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"        # 延迟高但仍有数据
    STALE = "stale"              # 长时间无数据
    DISCONNECTED = "disconnected"


@dataclass(frozen=True, slots=True)
class StreamHealthSnapshot:
    """某个数据流的健康快照。"""
    stream_id: str
    status: StreamStatus
    last_message_age_sec: float
    message_count: int
    error_count: int
    latency_ms: float


class StreamHealthMonitor:
    """
    数据流健康监控器。

    追踪每个流（WS/REST）的最新消息时间、消息计数、错误计数。
    """

    def __init__(
        self,
        *,
        stale_threshold_sec: float = 30.0,
        degraded_latency_ms: float = 1000.0,
    ) -> None:
        self._stale_threshold = stale_threshold_sec
        self._degraded_latency = degraded_latency_ms
        self._streams: Dict[str, _StreamState] = {}

    def record_message(self, stream_id: str, *, latency_ms: float = 0.0) -> None:
        """记录收到一条消息。"""
        state = self._get_or_create(stream_id)
        state.last_message_time = time.monotonic()
        state.message_count += 1
        state.latest_latency_ms = latency_ms

    def record_error(self, stream_id: str) -> None:
        """记录一次错误。"""
        state = self._get_or_create(stream_id)
        state.error_count += 1

    def record_disconnect(self, stream_id: str) -> None:
        """记录断线。"""
        state = self._get_or_create(stream_id)
        state.disconnected = True

    def record_reconnect(self, stream_id: str) -> None:
        """记录重连。"""
        state = self._get_or_create(stream_id)
        state.disconnected = False

    def check(self, stream_id: str) -> StreamHealthSnapshot:
        """检查某个流的健康状态。"""
        state = self._streams.get(stream_id)
        if state is None:
            return StreamHealthSnapshot(
                stream_id=stream_id,
                status=StreamStatus.DISCONNECTED,
                last_message_age_sec=0.0,
                message_count=0,
                error_count=0,
                latency_ms=0.0,
            )

        now = time.monotonic()
        age = now - state.last_message_time if state.last_message_time > 0 else float("inf")

        if state.disconnected:
            status = StreamStatus.DISCONNECTED
        elif age > self._stale_threshold:
            status = StreamStatus.STALE
        elif state.latest_latency_ms > self._degraded_latency:
            status = StreamStatus.DEGRADED
        else:
            status = StreamStatus.HEALTHY

        return StreamHealthSnapshot(
            stream_id=stream_id,
            status=status,
            last_message_age_sec=age if age != float("inf") else 0.0,
            message_count=state.message_count,
            error_count=state.error_count,
            latency_ms=state.latest_latency_ms,
        )

    def check_all(self) -> list[StreamHealthSnapshot]:
        return [self.check(sid) for sid in self._streams]

    def _get_or_create(self, stream_id: str) -> _StreamState:
        if stream_id not in self._streams:
            self._streams[stream_id] = _StreamState()
        return self._streams[stream_id]


class _StreamState:
    __slots__ = ("last_message_time", "message_count", "error_count",
                 "latest_latency_ms", "disconnected")

    def __init__(self) -> None:
        self.last_message_time: float = 0.0
        self.message_count: int = 0
        self.error_count: int = 0
        self.latest_latency_ms: float = 0.0
        self.disconnected: bool = False
