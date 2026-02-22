# context/views/risk_view.py
"""Risk view — what risk modules can see."""
from __future__ import annotations

from typing import Any, Mapping, Optional

from context.context import Context


class RiskView:
    """
    风控视图 — 风控模块可以看到的上下文。

    暴露：账户状态、仓位信息、风险指标。
    """

    def __init__(self, context: Context) -> None:
        self._context = context

    def clock(self) -> Mapping[str, Any]:
        return self._context.clock_snapshot()

    def context_info(self) -> Mapping[str, Any]:
        return self._context.last_event_info()

    def refresh(self) -> None:
        pass
