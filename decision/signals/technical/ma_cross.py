from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping

from decision.types import SignalResult


@dataclass(frozen=True, slots=True)
class MACrossSignal:
    """MA cross signal.

    Requires features (e.g. fast_ma, slow_ma). If missing, returns neutral.
    """
    fast_key: str = "ma_fast"
    slow_key: str = "ma_slow"
    name: str = "ma_cross"

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        feats = getattr(snapshot, "features", None)
        if not isinstance(feats, Mapping):
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))
        f = feats.get(self.fast_key)
        s = feats.get(self.slow_key)
        try:
            f = Decimal(str(f))
            s = Decimal(str(s))
        except Exception:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))
        if f > s:
            return SignalResult(symbol=symbol, side="buy", score=Decimal("1"), confidence=Decimal("1"), meta={"fast": str(f), "slow": str(s)})
        if f < s:
            return SignalResult(symbol=symbol, side="sell", score=Decimal("-1"), confidence=Decimal("1"), meta={"fast": str(f), "slow": str(s)})
        return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("1"), meta={"fast": str(f), "slow": str(s)})
