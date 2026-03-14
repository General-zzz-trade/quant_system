# execution/bridge/venue_router.py
"""Route commands to the correct venue client based on venue name."""
from __future__ import annotations

from typing import Any, Dict, Optional

from execution.bridge.execution_bridge import ExecutionBridge


class VenueRouter:
    """
    交易所路由器 — 根据 venue 名称路由到对应的 ExecutionBridge。

    支持多交易所并行执行。
    """

    def __init__(self) -> None:
        self._bridges: Dict[str, ExecutionBridge] = {}

    def register(self, venue: str, bridge: ExecutionBridge) -> None:
        """注册一个交易所的执行桥。"""
        self._bridges[venue.lower()] = bridge

    def get_bridge(self, venue: str) -> Optional[ExecutionBridge]:
        """获取指定交易所的执行桥。"""
        return self._bridges.get(venue.lower())

    def submit(self, cmd: Any) -> Any:
        """提交订单 — 自动路由到对应交易所。"""
        venue = str(getattr(cmd, "venue", "")).lower()
        bridge = self._bridges.get(venue)
        if bridge is None:
            raise KeyError(f"no bridge registered for venue {venue!r}")
        return bridge.submit(cmd)

    def cancel(self, cmd: Any) -> Any:
        """撤单 — 自动路由到对应交易所。"""
        venue = str(getattr(cmd, "venue", "")).lower()
        bridge = self._bridges.get(venue)
        if bridge is None:
            raise KeyError(f"no bridge registered for venue {venue!r}")
        return bridge.cancel(cmd)

    @property
    def venues(self) -> list[str]:
        return list(self._bridges.keys())
