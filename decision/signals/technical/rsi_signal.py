from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Optional
from decision.types import SignalResult


@dataclass(frozen=True, slots=True)
class RSISignal:
    rsi_key: str = "rsi"
    overbought: Decimal = Decimal("70")
    oversold: Decimal = Decimal("30")
    name: str = "rsi_signal"

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        feats = getattr(snapshot, "features", None)
        if not isinstance(feats, Mapping):
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        raw_rsi = feats.get(self.rsi_key)
        if raw_rsi is None:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        try:
            rsi = Decimal(str(raw_rsi))
        except (InvalidOperation, Exception):
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        if rsi > self.overbought:
            divisor = Decimal("100") - self.overbought
            score = -(rsi - self.overbought) / divisor if divisor != Decimal("0") else Decimal("-1")
            return SignalResult(
                symbol=symbol,
                side="sell",
                score=score,
                confidence=Decimal("0.8"),
                meta={"rsi": str(rsi)},
            )

        if rsi < self.oversold:
            divisor = self.oversold
            score = (self.oversold - rsi) / divisor if divisor != Decimal("0") else Decimal("1")
            return SignalResult(
                symbol=symbol,
                side="buy",
                score=score,
                confidence=Decimal("0.8"),
                meta={"rsi": str(rsi)},
            )

        return SignalResult(
            symbol=symbol,
            side="flat",
            score=Decimal("0"),
            confidence=Decimal("0.5"),
            meta={"rsi": str(rsi)},
        )
