# portfolio/risk_model/liquidity/volume.py
"""Volume-based liquidity analysis."""
from __future__ import annotations

import math
from typing import Sequence

from portfolio.risk_model.liquidity.base import LiquidityEstimate


class VolumeProfile:
    """基于成交量的流动性分析。"""
    name: str = "volume"

    def __init__(self, lookback: int = 30) -> None:
        self.lookback = lookback

    def estimate(
        self,
        symbol: str,
        volumes: Sequence[float],
        spreads: Sequence[float] | None = None,
    ) -> LiquidityEstimate:
        recent = list(volumes)[-self.lookback:]
        if not recent:
            return LiquidityEstimate(
                symbol=symbol,
                bid_ask_spread_bps=0.0,
                daily_volume_usd=0.0,
                market_impact_bps=0.0,
                liquidity_score=0.0,
            )

        avg_vol = sum(recent) / len(recent)
        vol_std = math.sqrt(
            sum((v - avg_vol) ** 2 for v in recent) / max(len(recent) - 1, 1)
        )

        # 流动性评分: 基于日均成交量
        if avg_vol > 1_000_000:
            score = min(1.0, avg_vol / 100_000_000)
        else:
            score = max(0.0, avg_vol / 1_000_000)

        spread_bps = 0.0
        if spreads:
            recent_spreads = list(spreads)[-self.lookback:]
            spread_bps = sum(recent_spreads) / len(recent_spreads) if recent_spreads else 0.0

        return LiquidityEstimate(
            symbol=symbol,
            bid_ask_spread_bps=spread_bps,
            daily_volume_usd=avg_vol,
            market_impact_bps=0.0,
            liquidity_score=score,
        )
