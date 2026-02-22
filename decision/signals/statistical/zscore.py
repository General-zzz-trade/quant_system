from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping

from decision.types import SignalResult


@dataclass(frozen=True, slots=True)
class ZScoreSignal:
    key: str = "zscore"
    threshold: Decimal = Decimal("1.0")
    name: str = "zscore"

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        feats = getattr(snapshot, "features", None)
        if not isinstance(feats, Mapping):
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))
        z = feats.get(self.key)
        try:
            z = Decimal(str(z))
        except Exception:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))
        if z >= self.threshold:
            return SignalResult(symbol=symbol, side="sell", score=-(z / self.threshold), confidence=Decimal("0.8"), meta={"z": str(z)})
        if z <= -self.threshold:
            return SignalResult(symbol=symbol, side="buy", score=(-z / self.threshold), confidence=Decimal("0.8"), meta={"z": str(z)})
        return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0.2"), meta={"z": str(z)})
