"""MovingAverageCrossModule and snapshot helpers for backtest_runner.

Extracted from backtest_runner.py to keep it under 500 lines.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from event.header import EventHeader
from event.types import EventType, IntentEvent, OrderEvent
from _quant_hotpath import RustPositionState as PositionState
from runner.backtest.adapter import _make_id


class MovingAverageCrossModule:
    def __init__(self, *, symbol: str, window: int, order_qty: Decimal, origin: str = "ma_cross") -> None:
        self.symbol = symbol.upper()
        self.window = int(window)
        self.order_qty = Decimal(str(order_qty))
        self.origin = origin
        self._closes: List[Decimal] = []

    def decide(self, snapshot: Any) -> Iterable[Any]:
        market, positions, event_id = _snapshot_views(snapshot)
        close = getattr(market, "close", None) or getattr(market, "last_price", None)
        if close is None:
            return ()

        close_d = Decimal(str(close))
        self._closes.append(close_d)
        if len(self._closes) > self.window:
            self._closes.pop(0)
        if len(self._closes) < self.window:
            return ()

        ma = sum(self._closes) / Decimal(str(self.window))

        pos = positions.get(self.symbol) or PositionState.empty(self.symbol)
        qty = getattr(pos, "qty", Decimal("0"))

        want_long = close_d > ma

        events: List[Any] = []
        if qty == 0 and want_long:
            events.extend(self._open_long(event_id=event_id))
        elif qty > 0 and (not want_long):
            events.extend(self._close_long(qty=qty, event_id=event_id))

        return events

    def _open_long(self, *, event_id: Optional[str]) -> Sequence[Any]:
        intent_id = _make_id("intent")
        order_id = _make_id("order")

        intent_h = EventHeader.new_root(
            event_type=EventType.INTENT,
            version=1,
            source=f"decision:{self.origin}",
            correlation_id=str(event_id) if event_id else None,
        )
        order_h = EventHeader.from_parent(
            parent=intent_h,
            event_type=EventType.ORDER,
            version=1,
            source=f"decision:{self.origin}",
        )

        return (
            IntentEvent(
                header=intent_h,
                intent_id=intent_id,
                symbol=self.symbol,
                side="buy",
                target_qty=self.order_qty,
                reason_code="ma_cross_long",
                origin=self.origin,
            ),
            OrderEvent(
                header=order_h,
                order_id=order_id,
                intent_id=intent_id,
                symbol=self.symbol,
                side="buy",
                qty=self.order_qty,
                price=None,
            ),
        )

    def _close_long(self, *, qty: Decimal, event_id: Optional[str]) -> Sequence[Any]:
        intent_id = _make_id("intent")
        order_id = _make_id("order")

        intent_h = EventHeader.new_root(
            event_type=EventType.INTENT,
            version=1,
            source=f"decision:{self.origin}",
            correlation_id=str(event_id) if event_id else None,
        )
        order_h = EventHeader.from_parent(
            parent=intent_h,
            event_type=EventType.ORDER,
            version=1,
            source=f"decision:{self.origin}",
        )

        q = abs(qty)
        return (
            IntentEvent(
                header=intent_h,
                intent_id=intent_id,
                symbol=self.symbol,
                side="sell",
                target_qty=q,
                reason_code="ma_cross_exit",
                origin=self.origin,
            ),
            OrderEvent(
                header=order_h,
                order_id=order_id,
                intent_id=intent_id,
                symbol=self.symbol,
                side="sell",
                qty=q,
                price=None,
            ),
        )


def _snapshot_views(snapshot: Any) -> Tuple[Any, Mapping[str, Any], Optional[str]]:
    if hasattr(snapshot, "market") and hasattr(snapshot, "positions"):
        market = getattr(snapshot, "market")
        positions = getattr(snapshot, "positions")
        event_id = getattr(snapshot, "event_id", None)
        return market, positions, event_id

    if isinstance(snapshot, dict):
        market = snapshot.get("market")
        if market is None:
            markets = snapshot.get("markets") or {}
            market = next(iter(markets.values()), None) if markets else None
        positions = snapshot.get("positions") or {}
        event_id = snapshot.get("event_id")
        if market is None:
            raise RuntimeError("snapshot missing market/markets")
        return market, positions, event_id

    raise RuntimeError(f"unsupported snapshot type: {type(snapshot).__name__}")
