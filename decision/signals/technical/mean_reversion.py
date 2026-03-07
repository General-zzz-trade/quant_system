from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from decision.market_access import get_decimal_attr
from decision.types import SignalResult


@dataclass(frozen=True, slots=True)
class MeanReversionSignal:
    """Very small proxy: if close < open => buy (revert), if close > open => sell."""
    name: str = "mean_reversion"

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        m = getattr(snapshot, "market", None)
        if m is None:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))
        o = get_decimal_attr(m, "open")
        c = get_decimal_attr(m, "close", "last_price")
        if o is None or c is None:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))
        if c < o:
            return SignalResult(symbol=symbol, side="buy", score=(o - c) / max(o, Decimal("1")), confidence=Decimal("0.6"), meta={"o": str(o), "c": str(c)})
        if c > o:
            return SignalResult(symbol=symbol, side="sell", score=-(c - o) / max(o, Decimal("1")), confidence=Decimal("0.6"), meta={"o": str(o), "c": str(c)})
        return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0.2"), meta={"o": str(o), "c": str(c)})
