"""Liquidity factor signal: volume z-score as liquidity premium indicator."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, List

from _quant_hotpath import rust_liquidity_score
from decision.types import SignalResult


@dataclass(frozen=True, slots=True)
class LiquiditySignal:
    name: str = "liquidity"
    lookback: int = 20

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        volumes = _get_volumes(snapshot, symbol)
        side, score, conf = rust_liquidity_score(volumes, self.lookback)
        return SignalResult(
            symbol=symbol,
            side=side,
            score=Decimal(str(score)),
            confidence=Decimal(str(conf)),
        )


def _get_volumes(snapshot: Any, symbol: str) -> List[float]:
    bars = getattr(snapshot, "bars", None)
    if bars is None:
        bars = getattr(snapshot, "get_bars", lambda s: [])(symbol)
    if isinstance(bars, dict):
        bars = bars.get(symbol, [])
    return [float(getattr(b, "volume", b.get("volume", 0) if isinstance(b, dict) else 0)) for b in bars]
