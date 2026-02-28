"""Volatility factor signal: mean-reversion on realized volatility."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import sqrt
from typing import Any, List

from decision.types import SignalResult


@dataclass(frozen=True, slots=True)
class VolatilitySignal:
    name: str = "volatility"
    lookback: int = 20

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        closes = _get_closes(snapshot, symbol)
        if len(closes) < self.lookback + 1:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        recent = closes[-self.lookback - 1:]
        rets = [(recent[i] / recent[i - 1]) - 1.0 for i in range(1, len(recent)) if recent[i - 1] != 0]
        if len(rets) < 4:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        # Rolling vol: compare recent half to full window
        half = len(rets) // 2
        recent_rets = rets[half:]
        full_rets = rets

        def _vol(rs: list) -> float:
            m = sum(rs) / len(rs)
            v = sum((r - m) ** 2 for r in rs) / len(rs)
            return sqrt(v) if v > 0 else 0.0

        recent_vol = _vol(recent_rets)
        full_vol = _vol(full_rets)

        if full_vol < 1e-12:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        # Z-score: (recent_vol - full_vol) / full_vol
        # Negative z = vol contracting (buy), positive z = vol expanding (sell)
        z = (recent_vol - full_vol) / full_vol
        score = Decimal(str(round(-z, 6)))  # invert: expanding vol → sell
        conf = Decimal(str(round(min(abs(z), 1.0), 4)))
        side = "buy" if score > 0 else ("sell" if score < 0 else "flat")
        return SignalResult(symbol=symbol, side=side, score=score, confidence=conf)


def _get_closes(snapshot: Any, symbol: str) -> List[float]:
    bars = getattr(snapshot, "bars", None)
    if bars is None:
        bars = getattr(snapshot, "get_bars", lambda s: [])(symbol)
    if isinstance(bars, dict):
        bars = bars.get(symbol, [])
    return [float(getattr(b, "close", b.get("close", 0) if isinstance(b, dict) else 0)) for b in bars]
