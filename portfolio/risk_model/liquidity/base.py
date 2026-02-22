# portfolio/risk_model/liquidity/base.py
"""Liquidity model protocol."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True, slots=True)
class LiquidityEstimate:
    """流动性估计。"""
    symbol: str
    bid_ask_spread_bps: float
    daily_volume_usd: float
    market_impact_bps: float
    liquidity_score: float  # 0-1, 1 = most liquid


class LiquidityModel(Protocol):
    """流动性模型协议。"""
    name: str

    def estimate(
        self,
        symbol: str,
        volumes: Sequence[float],
        spreads: Sequence[float] | None = None,
    ) -> LiquidityEstimate:
        """估计流动性指标。"""
        ...
