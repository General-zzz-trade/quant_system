"""Volume-price divergence signal: detect bearish/bullish divergences."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, List

from decision.types import SignalResult


@dataclass(frozen=True, slots=True)
class VolumePriceDivergenceSignal:
    name: str = "volume_price_div"
    lookback: int = 10

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        closes, volumes = _get_close_volume(snapshot, symbol)
        if len(closes) < self.lookback + 1 or len(volumes) < self.lookback + 1:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        recent_c = closes[-self.lookback - 1:]
        recent_v = volumes[-self.lookback:]

        # Price change over lookback
        if recent_c[0] == 0:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))
        price_change = (recent_c[-1] - recent_c[0]) / abs(recent_c[0])

        # Volume trend: simple linear slope
        mean_v = sum(recent_v) / len(recent_v)
        if mean_v == 0:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        n = len(recent_v)
        mean_i = (n - 1) / 2.0
        num = sum((i - mean_i) * (v - mean_v) for i, v in enumerate(recent_v))
        den = sum((i - mean_i) ** 2 for i in range(n))
        vol_slope = (num / den) / mean_v if den > 0 else 0.0  # normalized slope

        if abs(price_change) < 1e-6:
            score = Decimal("0")
            return SignalResult(symbol=symbol, side="flat", score=score, confidence=Decimal("0"))

        # Divergence: volume slope should confirm price direction
        # price↑ vol↓ → bearish (sell), price↓ vol↑ → bullish (buy)
        # score = vol_slope if price going up, -vol_slope if price going down
        # This way: price↑ + vol↓ → vol_slope<0 → negative score → sell
        #           price↓ + vol↑ → -vol_slope<0... no.
        # Simplest: multiply by sign of price change
        direction = 1.0 if price_change > 0 else -1.0
        divergence_score = vol_slope * direction  # negative = divergence
        score = Decimal(str(round(divergence_score * 100, 6)))
        conf = Decimal(str(round(min(abs(divergence_score) * 10, 1.0), 4)))
        side = "buy" if score > 0 else ("sell" if score < 0 else "flat")
        return SignalResult(symbol=symbol, side=side, score=score, confidence=conf)


def _get_close_volume(snapshot: Any, symbol: str) -> tuple[List[float], List[float]]:
    bars = getattr(snapshot, "bars", None)
    if bars is None:
        bars = getattr(snapshot, "get_bars", lambda s: [])(symbol)
    if isinstance(bars, dict):
        bars = bars.get(symbol, [])
    closes = []
    volumes = []
    for b in bars:
        if isinstance(b, dict):
            closes.append(float(b.get("close", 0)))
            volumes.append(float(b.get("volume", 0)))
        else:
            closes.append(float(getattr(b, "close", 0)))
            volumes.append(float(getattr(b, "volume", 0)))
    return closes, volumes
