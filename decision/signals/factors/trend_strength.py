"""Trend strength signal: ADX-based trend following."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, List

from decision.types import SignalResult
from features.types import Bar


@dataclass(frozen=True, slots=True)
class TrendStrengthSignal:
    name: str = "trend_strength"
    adx_window: int = 14
    adx_threshold: float = 25.0
    lookback: int = 20

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        bars = _get_bars(snapshot, symbol)
        if len(bars) < self.lookback + self.adx_window * 2:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        from features.technical.adx import adx
        adx_series = adx(bars, window=self.adx_window)

        # Get latest valid ADX
        latest_adx = None
        for v in reversed(adx_series):
            if v is not None:
                latest_adx = v
                break
        if latest_adx is None:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        if latest_adx < self.adx_threshold:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"),
                              confidence=Decimal(str(round(latest_adx / 100.0, 4))))

        # Strong trend: follow price direction
        closes = [b.close for b in bars[-self.lookback:]]
        if len(closes) < 2 or closes[0] == 0:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        direction = (closes[-1] - closes[0]) / abs(closes[0])
        strength = latest_adx / 100.0  # normalize to 0-1
        score = Decimal(str(round(direction * strength, 6)))
        conf = Decimal(str(round(min(strength, 1.0), 4)))
        side = "buy" if score > 0 else ("sell" if score < 0 else "flat")
        return SignalResult(symbol=symbol, side=side, score=score, confidence=conf)


def _get_bars(snapshot: Any, symbol: str) -> List[Bar]:
    bars = getattr(snapshot, "bars", None)
    if bars is None:
        bars = getattr(snapshot, "get_bars", lambda s: [])(symbol)
    if isinstance(bars, dict):
        bars = bars.get(symbol, [])
    result = []
    for b in bars:
        if isinstance(b, Bar):
            result.append(b)
        elif isinstance(b, dict):
            from datetime import datetime
            result.append(Bar(
                ts=b.get("ts", datetime.min),
                open=float(b.get("open", 0)),
                high=float(b.get("high", 0)),
                low=float(b.get("low", 0)),
                close=float(b.get("close", 0)),
                volume=float(b.get("volume", 0)),
            ))
        else:
            result.append(Bar(
                ts=getattr(b, "ts", None),
                open=float(getattr(b, "open", 0)),
                high=float(getattr(b, "high", 0)),
                low=float(getattr(b, "low", 0)),
                close=float(getattr(b, "close", 0)),
                volume=float(getattr(b, "volume", 0)),
            ))
    return result
