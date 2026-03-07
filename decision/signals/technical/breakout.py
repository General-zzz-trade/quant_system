from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from decision.market_access import get_decimal_attr
from decision.types import SignalResult


@dataclass(frozen=True, slots=True)
class BreakoutSignal:
    """Uses current close vs high/low in snapshot as a micro-breakout proxy."""
    name: str = "breakout"

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        m = getattr(snapshot, "market", None)
        if m is None:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))
        c = get_decimal_attr(m, "close", "last_price")
        h = get_decimal_attr(m, "high")
        l = get_decimal_attr(m, "low")
        if c is None or h is None or l is None:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))
        if h > 0 and c >= h:
            return SignalResult(symbol=symbol, side="buy", score=Decimal("1"), confidence=Decimal("1"), meta={"c": str(c), "h": str(h)})
        if l > 0 and c <= l:
            return SignalResult(symbol=symbol, side="sell", score=Decimal("-1"), confidence=Decimal("1"), meta={"c": str(c), "l": str(l)})
        return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0.2"), meta={"c": str(c), "h": str(h), "l": str(l)})
