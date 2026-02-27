"""Momentum factor signal: cumulative returns over lookback → z-score."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import sqrt
from typing import Any, List, Optional

from decision.types import SignalResult


@dataclass(frozen=True, slots=True)
class MomentumSignal:
    name: str = "momentum"
    lookback: int = 20

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        closes = _get_closes(snapshot, symbol)
        if len(closes) < self.lookback + 1:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        recent = closes[-self.lookback - 1:]
        rets = [(recent[i] / recent[i - 1]) - 1.0 for i in range(1, len(recent)) if recent[i - 1] != 0]
        if len(rets) < 2:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        cum_ret = 1.0
        for r in rets:
            cum_ret *= (1.0 + r)
        cum_ret -= 1.0

        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        std = sqrt(var) if var > 0 else 1e-10
        z = cum_ret / std if std > 1e-10 else 0.0

        score = Decimal(str(round(z, 6)))
        conf = Decimal(str(round(min(abs(z) / 3.0, 1.0), 4)))
        side = "buy" if score > 0 else ("sell" if score < 0 else "flat")
        return SignalResult(symbol=symbol, side=side, score=score, confidence=conf)


def _get_closes(snapshot: Any, symbol: str) -> List[float]:
    bars = getattr(snapshot, "bars", None)
    if bars is None:
        bars = getattr(snapshot, "get_bars", lambda s: [])(symbol)
    if isinstance(bars, dict):
        bars = bars.get(symbol, [])
    return [float(getattr(b, "close", b.get("close", 0) if isinstance(b, dict) else 0)) for b in bars]
