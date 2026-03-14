# event/metrics.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict

from event.lifecycle import LifecycleState


# ============================================================
# 基础原语（冻结）
# ============================================================

class Counter:
    """
    线程安全计数器（单调递增）
    """
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: int = 0

    def inc(self, n: int = 1) -> None:
        with self._lock:
            self._value += n

    def value(self) -> int:
        with self._lock:
            return self._value


class LatencyStat:
    """
    延迟统计（轻量）
    - 提供 count / avg_ms / max_ms
    - 不提供分位数（分位由外部监控系统完成）
    """
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count: int = 0
        self._sum_ms: float = 0.0
        self._max_ms: float = 0.0

    def observe(self, ms: float) -> None:
        with self._lock:
            self._count += 1
            self._sum_ms += ms
            if ms > self._max_ms:
                self._max_ms = ms

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            avg = (self._sum_ms / self._count) if self._count else 0.0
            return {
                "count": self._count,
                "avg_ms": round(avg, 3),
                "max_ms": round(self._max_ms, 3),
            }


# ============================================================
# EventMetrics（只监听 Lifecycle）
# ============================================================

@dataclass(frozen=True)
class MetricsSnapshot:
    """
    可序列化快照（稳定协议）
    """
    counters: Dict[str, int]
    latencies: Dict[str, Dict[str, float]]


class EventMetrics:
    """
    EventMetrics —— 生命周期驱动的被动指标收集器（机构级）

    冻结约定：
    - Metrics 只由 Lifecycle 状态变化驱动
    - Runtime / Dispatcher / Handler 不直接调用 Metrics
    - 终态自动清理内部时间戳，避免泄漏
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # counters
        self._created = Counter()
        self._dispatching = Counter()
        self._dispatched = Counter()
        self._retry = Counter()
        self._failed = Counter()
        self._dropped = Counter()

        # latency stats
        self._emit_to_dispatch = LatencyStat()
        self._dispatch_to_done = LatencyStat()

        # internal timestamps（event_id -> monotonic seconds）
        self._emit_ts: Dict[str, float] = {}
        self._dispatch_ts: Dict[str, float] = {}

    # --------------------------------------------------------
    # 内部工具
    # --------------------------------------------------------

    def _event_key(self, event: Any) -> str:
        try:
            return event.header.event_id
        except Exception:
            return str(id(event))

    def _now(self) -> float:
        return time.monotonic()

    def _cleanup(self, key: str) -> None:
        # 终态清理，防止时间戳泄漏
        self._emit_ts.pop(key, None)
        self._dispatch_ts.pop(key, None)

    # --------------------------------------------------------
    # Lifecycle hooks（唯一入口）
    # --------------------------------------------------------

    def on_lifecycle(self, event: Any, state: LifecycleState) -> None:
        """
        由 Lifecycle 在状态转移时调用
        """
        key = self._event_key(event)
        now = self._now()

        with self._lock:
            if state == LifecycleState.CREATED:
                self._created.inc()
                self._emit_ts[key] = now

            elif state == LifecycleState.DISPATCHING:
                self._dispatching.inc()
                self._dispatch_ts[key] = now
                # emit -> dispatch 延迟
                t0 = self._emit_ts.get(key)
                if t0 is not None:
                    self._emit_to_dispatch.observe((now - t0) * 1000.0)

            elif state == LifecycleState.DISPATCHED:
                self._dispatched.inc()
                # dispatch -> done 延迟
                t1 = self._dispatch_ts.get(key)
                if t1 is not None:
                    self._dispatch_to_done.observe((now - t1) * 1000.0)
                self._cleanup(key)

            elif state == LifecycleState.RETRY:
                self._retry.inc()
                # retry 视为本轮终态
                self._cleanup(key)

            elif state == LifecycleState.FAILED:
                self._failed.inc()
                self._cleanup(key)

            elif state == LifecycleState.DROPPED:
                self._dropped.inc()
                self._cleanup(key)

    # --------------------------------------------------------
    # Snapshot
    # --------------------------------------------------------

    def snapshot(self) -> MetricsSnapshot:
        """
        生成可序列化快照
        """
        return MetricsSnapshot(
            counters={
                "created": self._created.value(),
                "dispatching": self._dispatching.value(),
                "dispatched": self._dispatched.value(),
                "retry": self._retry.value(),
                "failed": self._failed.value(),
                "dropped": self._dropped.value(),
            },
            latencies={
                "emit_to_dispatch": self._emit_to_dispatch.snapshot(),
                "dispatch_to_done": self._dispatch_to_done.snapshot(),
            },
        )
