"""Carry factor signal: trade against funding rate direction."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from decision.types import SignalResult


@dataclass(frozen=True, slots=True)
class CarrySignal:
    name: str = "carry"

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        rate = _get_funding_rate(snapshot, symbol)
        if rate is None:
            return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))

        # Negative funding → longs pay less / receive → go long
        # Positive funding → shorts receive → go short
        score = Decimal(str(round(-rate * 10000, 6)))  # scale for readability
        conf = Decimal(str(round(min(abs(rate) * 10000, 1.0), 4)))
        side = "buy" if score > 0 else ("sell" if score < 0 else "flat")
        return SignalResult(symbol=symbol, side=side, score=score, confidence=conf)


def _get_funding_rate(snapshot: Any, symbol: str) -> float | None:
    fr = getattr(snapshot, "funding_rate", None)
    if fr is not None:
        if isinstance(fr, dict):
            val = fr.get(symbol)
            return float(val) if val is not None else None
        return float(fr)
    get_fr = getattr(snapshot, "get_funding_rate", None)
    if get_fr is not None:
        val = get_fr(symbol)
        return float(val) if val is not None else None
    return None
