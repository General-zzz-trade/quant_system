"""Liquidity factor signal: volume z-score as liquidity premium indicator."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import sqrt
from typing import Any, List

from decision.types import SignalResult


@dataclass(frozen=True, slots=True)
class LiquiditySignal:
    name: str = "liquidity"
    lookback: int = 20

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        volumes = _get_volumes(snapshot, symbol)
        if len(volumes) < self.lookback:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        recent = volumes[-self.lookback:]
        mean_vol = sum(recent) / len(recent)
        var = sum((v - mean_vol) ** 2 for v in recent) / len(recent)
        std = sqrt(var) if var > 0 else 1e-10

        current = recent[-1]
        z = (current - mean_vol) / std if std > 1e-10 else 0.0

        # High volume (positive z) → liquidity available → slight buy bias (liquidity premium)
        # Low volume (negative z) → illiquidity → slight sell bias (risk)
        score = Decimal(str(round(z, 6)))
        conf = Decimal(str(round(min(abs(z) / 3.0, 1.0), 4)))
        side = "buy" if score > 0 else ("sell" if score < 0 else "flat")
        return SignalResult(symbol=symbol, side=side, score=score, confidence=conf)


def _get_volumes(snapshot: Any, symbol: str) -> List[float]:
    bars = getattr(snapshot, "bars", None)
    if bars is None:
        bars = getattr(snapshot, "get_bars", lambda s: [])(symbol)
    if isinstance(bars, dict):
        bars = bars.get(symbol, [])
    return [float(getattr(b, "volume", b.get("volume", 0) if isinstance(b, dict) else 0)) for b in bars]
