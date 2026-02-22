# portfolio/risk_model/liquidity/spread.py
"""Bid-ask spread estimation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class SpreadEstimate:
    """买卖价差估计。"""
    symbol: str
    avg_spread_bps: float
    median_spread_bps: float
    max_spread_bps: float
    stability: float  # coefficient of variation


class SpreadModel:
    """买卖价差模型。"""
    name: str = "spread"

    def __init__(self, lookback: int = 30) -> None:
        self.lookback = lookback

    def estimate(self, symbol: str, spreads: Sequence[float]) -> SpreadEstimate:
        recent = list(spreads)[-self.lookback:]
        if not recent:
            return SpreadEstimate(symbol, 0.0, 0.0, 0.0, 0.0)

        avg = sum(recent) / len(recent)
        sorted_s = sorted(recent)
        median = sorted_s[len(sorted_s) // 2]
        max_s = max(recent)
        std = math.sqrt(sum((s - avg) ** 2 for s in recent) / max(len(recent) - 1, 1))
        cv = std / avg if avg > 0 else 0.0

        return SpreadEstimate(
            symbol=symbol,
            avg_spread_bps=avg,
            median_spread_bps=median,
            max_spread_bps=max_s,
            stability=1.0 / (1.0 + cv),
        )
