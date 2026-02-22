# context/views/execution_view.py
"""Execution view — what execution layer can see."""
from __future__ import annotations

from typing import Any, Mapping, Optional

from context.context import Context


class ExecutionView:
    """
    执行视图 — 执行层可以看到的上下文。

    暴露：市场数据（用于滑点计算）、时钟信息。
    """

    def __init__(self, context: Context) -> None:
        self._context = context

    def clock(self) -> Mapping[str, Any]:
        return self._context.clock_snapshot()

    def get_market_price(self, symbol: str, venue: str) -> Optional[Any]:
        snap = self._context.get_market(symbol=symbol, venue=venue)
        if snap is None:
            return None
        return snap.last_price

    def refresh(self) -> None:
        pass
