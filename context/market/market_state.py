# quant_system/context/market/market_state.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from event.types import MarketEvent


# ============================================================
# 不可变快照（对外只读）
# ============================================================

@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """
    单一 symbol + venue 的市场快照（不可变）

    顶级机构约束：
    - 必须是值语义（value object）
    - 不包含可变结构
    - 可以安全在策略 / 风控 / 回测间共享
    """
    symbol: str
    venue: str

    ts: Any
    bar_index: int

    last_price: Optional[float]
    last_size: Optional[float]

    # OHLCV（bar 级别，可选）
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    volume: Optional[float]


# ============================================================
# 可变状态（Context 内部使用）
# ============================================================

class MarketState:
    """
    MarketState：市场事实的可变容器（Context 内部）

    设计原则：
    - 只保存“发生了什么”
    - 不保存“应该如何交易”
    - 所有更新必须来自 reducer
    """

    def __init__(self) -> None:
        # key = (symbol, venue)
        self._states: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # ------------------------------------------------------------
    # Reducer 唯一写入口
    # ------------------------------------------------------------

    def update_from_market_event(
        self,
        event: MarketEvent,
        *,
        ts: Any,
        bar_index: int,
    ) -> None:
        """
        从 MarketEvent 更新 MarketState（只能被 reducer 调用）

        约束：
        - ts / bar_index 必须来自 Context.clock
        - 不在此方法中推进时间
        """
        symbol = event.symbol
        venue = event.venue
        key = (symbol, venue)

        state = self._states.get(key)
        if state is None:
            # 初始化最小状态
            state = {
                "symbol": symbol,
                "venue": venue,
                "ts": ts,
                "bar_index": bar_index,
                "last_price": None,
                "last_size": None,
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "volume": None,
            }
            self._states[key] = state

        # --------------------------------------------------------
        # 价格 / 成交类事件（tick / trade）
        # --------------------------------------------------------
        if getattr(event, "price", None) is not None:
            price = float(event.price)
            state["last_price"] = price

            # bar 逻辑（如果当前事件属于 bar）
            if getattr(event, "kind", None) is not None:
                kind_value = getattr(event.kind, "value", event.kind)
                if kind_value == "bar":
                    # bar 初始化
                    if state["open"] is None or bar_index != state["bar_index"]:
                        state["open"] = price
                        state["high"] = price
                        state["low"] = price
                        state["close"] = price
                        state["volume"] = 0.0
                    else:
                        state["high"] = max(state["high"], price)
                        state["low"] = min(state["low"], price)
                        state["close"] = price

        if getattr(event, "size", None) is not None:
            size = float(event.size)
            state["last_size"] = size
            if state["volume"] is not None:
                state["volume"] += size

        # --------------------------------------------------------
        # 最后统一更新时间戳
        # --------------------------------------------------------
        state["ts"] = ts
        state["bar_index"] = bar_index

    # ------------------------------------------------------------
    # 对外只读接口（返回不可变快照）
    # ------------------------------------------------------------

    def get_snapshot(
        self,
        *,
        symbol: str,
        venue: str,
    ) -> Optional[MarketSnapshot]:
        state = self._states.get((symbol, venue))
        if state is None:
            return None
        return self._to_snapshot(state)

    def require_snapshot(
        self,
        *,
        symbol: str,
        venue: str,
    ) -> MarketSnapshot:
        snap = self.get_snapshot(symbol=symbol, venue=venue)
        if snap is None:
            raise KeyError(f"MarketSnapshot 不存在: symbol={symbol}, venue={venue}")
        return snap

    # ------------------------------------------------------------
    # 内部：构造不可变快照
    # ------------------------------------------------------------

    def _to_snapshot(self, state: Dict[str, Any]) -> MarketSnapshot:
        """
        将内部可变 state 转换为不可变 MarketSnapshot
        """
        return MarketSnapshot(
            symbol=state["symbol"],
            venue=state["venue"],
            ts=state["ts"],
            bar_index=state["bar_index"],
            last_price=state["last_price"],
            last_size=state["last_size"],
            open=state["open"],
            high=state["high"],
            low=state["low"],
            close=state["close"],
            volume=state["volume"],
        )
