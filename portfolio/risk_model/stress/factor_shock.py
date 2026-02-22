# portfolio/risk_model/stress/factor_shock.py
"""Factor-based stress testing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from portfolio.risk_model.stress.scenarios import StressResult


@dataclass(frozen=True, slots=True)
class FactorShock:
    """因子冲击定义。"""
    factor_name: str
    shock_magnitude: float  # 标准差倍数


class FactorShockTest:
    """因子冲击压力测试。"""
    name: str = "factor_shock"

    def run(
        self,
        weights: Mapping[str, float],
        exposures: Mapping[str, Mapping[str, float]],
        shocks: Sequence[FactorShock],
    ) -> StressResult:
        """通过因子暴露传导冲击到组合。"""
        asset_losses: dict[str, float] = {}
        port_loss = 0.0

        for symbol, w in weights.items():
            betas = exposures.get(symbol, {})
            asset_shock = sum(
                betas.get(s.factor_name, 0.0) * s.shock_magnitude
                for s in shocks
            )
            loss = w * asset_shock
            asset_losses[symbol] = loss
            port_loss += loss

        shock_names = "+".join(s.factor_name for s in shocks)
        return StressResult(
            scenario_name=f"factor_shock_{shock_names}",
            portfolio_loss=port_loss,
            asset_losses=asset_losses,
            max_drawdown=abs(port_loss),
        )
