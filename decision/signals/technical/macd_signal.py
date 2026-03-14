from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping
from decision.types import SignalResult


@dataclass(frozen=True, slots=True)
class MACDSignal:
    macd_key: str = "macd"
    signal_key: str = "macd_signal"
    hist_key: str = "macd_hist"
    name: str = "macd_signal"

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        feats = getattr(snapshot, "features", None)
        if not isinstance(feats, Mapping):
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        raw_macd = feats.get(self.macd_key)
        raw_signal = feats.get(self.signal_key)
        raw_hist = feats.get(self.hist_key)

        if any(v is None for v in (raw_macd, raw_signal, raw_hist)):
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        try:
            macd = Decimal(str(raw_macd))
            signal = Decimal(str(raw_signal))
            histogram = Decimal(str(raw_hist))
        except (InvalidOperation, Exception):
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        meta = {
            "macd": str(macd),
            "signal": str(signal),
            "histogram": str(histogram),
        }

        if histogram > Decimal("0"):
            score = min(Decimal("1"), Decimal(str(abs(histogram))))
            return SignalResult(
                symbol=symbol,
                side="buy",
                score=score,
                confidence=Decimal("0.7"),
                meta=meta,
            )

        if histogram < Decimal("0"):
            score = -min(Decimal("1"), Decimal(str(abs(histogram))))
            return SignalResult(
                symbol=symbol,
                side="sell",
                score=score,
                confidence=Decimal("0.7"),
                meta=meta,
            )

        return SignalResult(
            symbol=symbol,
            side="flat",
            score=Decimal("0"),
            confidence=Decimal("0.5"),
            meta=meta,
        )
