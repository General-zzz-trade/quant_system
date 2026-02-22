# portfolio/risk_model/liquidity/market_impact.py
"""Market impact model (simplified Almgren-Chriss)."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImpactEstimate:
    """市场冲击估计。"""
    symbol: str
    permanent_impact_bps: float
    temporary_impact_bps: float
    total_impact_bps: float
    optimal_participation_rate: float


class MarketImpactModel:
    """简化 Almgren-Chriss 市场冲击模型。"""
    name: str = "market_impact"

    def __init__(
        self,
        permanent_coeff: float = 0.1,
        temporary_coeff: float = 0.5,
        volatility_scale: float = 1.0,
    ) -> None:
        self.permanent_coeff = permanent_coeff
        self.temporary_coeff = temporary_coeff
        self.volatility_scale = volatility_scale

    def estimate(
        self,
        symbol: str,
        order_size_usd: float,
        daily_volume_usd: float,
        volatility: float = 0.02,
    ) -> ImpactEstimate:
        if daily_volume_usd <= 0:
            return ImpactEstimate(symbol, 0.0, 0.0, 0.0, 0.0)

        participation = order_size_usd / daily_volume_usd
        sigma = volatility * self.volatility_scale

        # 永久冲击 ∝ σ · √participation
        perm = self.permanent_coeff * sigma * math.sqrt(participation) * 10000
        # 临时冲击 ∝ σ · participation
        temp = self.temporary_coeff * sigma * participation * 10000

        # 最优参与率 (简化)
        optimal_pr = min(0.1, math.sqrt(0.01 / max(sigma, 1e-6)))

        return ImpactEstimate(
            symbol=symbol,
            permanent_impact_bps=perm,
            temporary_impact_bps=temp,
            total_impact_bps=perm + temp,
            optimal_participation_rate=optimal_pr,
        )
