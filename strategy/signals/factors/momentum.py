"""Momentum factor signal: cumulative returns over lookback -> z-score."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, List

from _quant_hotpath import rust_momentum_score
from decision.types import SignalResult


@dataclass(frozen=True, slots=True)
class MomentumSignal:
    name: str = "momentum"
    lookback: int = 20

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        closes = _get_closes(snapshot, symbol)
        side, score, conf = rust_momentum_score(closes, self.lookback)
        return SignalResult(
            symbol=symbol,
            side=side,
            score=Decimal(str(score)),
            confidence=Decimal(str(conf)),
        )


def _get_closes(snapshot: Any, symbol: str) -> List[float]:
    bars = getattr(snapshot, "bars", None)
    if bars is None:
        bars = getattr(snapshot, "get_bars", lambda s: [])(symbol)
    if isinstance(bars, dict):
        bars = bars.get(symbol, [])
    return [float(getattr(b, "close", b.get("close", 0) if isinstance(b, dict) else 0)) for b in bars]
