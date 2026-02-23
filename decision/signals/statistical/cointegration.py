"""Cointegration signal — spread-based mean reversion for pair trading.

Reads a pre-computed spread z-score from snapshot features. When the spread
deviates beyond threshold, generates buy/sell signals expecting reversion.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping

from decision.types import SignalResult


@dataclass(frozen=True, slots=True)
class CointegrationSignal:
    """Spread z-score signal for cointegrated pairs.

    Reads the pre-computed spread z-score from snapshot features.
    If not available, returns flat (neutral).
    """
    name: str = "cointegration"
    spread_key: str = "spread_zscore"
    threshold: Decimal = Decimal("2.0")
    confidence_at_threshold: Decimal = Decimal("0.7")

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        feats = getattr(snapshot, "features", None)
        if not isinstance(feats, Mapping):
            return self._flat(symbol)

        raw = feats.get(self.spread_key)
        if raw is None:
            return self._flat(symbol)

        try:
            z = Decimal(str(raw))
        except (ValueError, TypeError):
            return self._flat(symbol)

        meta = {"spread_z": str(z), "threshold": str(self.threshold)}

        # Spread too high -> expect reversion down -> sell
        if z >= self.threshold:
            score = -(z / self.threshold)
            conf = min(Decimal("1"), self.confidence_at_threshold * abs(z) / self.threshold)
            return SignalResult(symbol=symbol, side="sell", score=score, confidence=conf, meta=meta)

        # Spread too low -> expect reversion up -> buy
        if z <= -self.threshold:
            score = -z / self.threshold
            conf = min(Decimal("1"), self.confidence_at_threshold * abs(z) / self.threshold)
            return SignalResult(symbol=symbol, side="buy", score=score, confidence=conf, meta=meta)

        return SignalResult(
            symbol=symbol, side="flat", score=Decimal("0"),
            confidence=Decimal("0.1"), meta=meta,
        )

    def _flat(self, symbol: str) -> SignalResult:
        return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))
