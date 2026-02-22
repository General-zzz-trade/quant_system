# context/views/strategy_view.py
"""Strategy view — what strategy modules can see."""
from __future__ import annotations

from typing import Any, Mapping, Optional

from context.context import Context, ContextSnapshot
from context.market.market_state import MarketSnapshot


class StrategyView:
    """
    策略视图 — 策略模块可以看到的上下文。

    只暴露：市场数据、时钟信息、只读账户快照。
    不暴露：内部状态、可变引用。
    """

    def __init__(self, context: Context) -> None:
        self._context = context

    def get_market(self, symbol: str, venue: str) -> Optional[MarketSnapshot]:
        return self._context.get_market(symbol=symbol, venue=venue)

    def require_market(self, symbol: str, venue: str) -> MarketSnapshot:
        return self._context.require_market(symbol=symbol, venue=venue)

    def clock(self) -> Mapping[str, Any]:
        return self._context.clock_snapshot()

    def snapshot(self) -> ContextSnapshot:
        return self._context.snapshot()

    def refresh(self) -> None:
        pass  # strategy view is always fresh (reads from context directly)
