# context/views/monitoring_view.py
"""Monitoring view — what monitoring/dashboards can see."""
from __future__ import annotations

from typing import Any, Mapping

from context.context import Context


class MonitoringView:
    """
    监控视图 — 监控仪表盘可以看到的上下文。

    暴露：全量只读信息用于展示。
    """

    def __init__(self, context: Context) -> None:
        self._context = context

    def clock(self) -> Mapping[str, Any]:
        return self._context.clock_snapshot()

    def event_info(self) -> Mapping[str, Any]:
        return self._context.last_event_info()

    def full_snapshot(self) -> Mapping[str, Any]:
        snap = self._context.snapshot()
        return {
            "context_id": snap.context_id,
            "snapshot_id": snap.snapshot_id,
            "ts": snap.ts,
            "bar_index": snap.bar_index,
            "created_at_ms": snap.created_at_ms,
        }

    def refresh(self) -> None:
        pass
