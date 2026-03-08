"""Volume-price divergence signal: detect bearish/bullish divergences."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, List

from _quant_hotpath import rust_volume_price_div_score
from decision.types import SignalResult


@dataclass(frozen=True, slots=True)
class VolumePriceDivergenceSignal:
    name: str = "volume_price_div"
    lookback: int = 10

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        closes, volumes = _get_close_volume(snapshot, symbol)
        side, score, conf = rust_volume_price_div_score(closes, volumes, self.lookback)
        return SignalResult(
            symbol=symbol,
            side=side,
            score=Decimal(str(score)),
            confidence=Decimal(str(conf)),
        )


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
