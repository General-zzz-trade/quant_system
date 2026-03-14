# portfolio/risk_model/stress/historical.py
"""Historical stress testing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from portfolio.risk_model.stress.scenarios import StressScenario


@dataclass(frozen=True, slots=True)
class StressResult:
    """压力测试结果。"""
    scenario_name: str
    portfolio_loss: float
    asset_losses: Mapping[str, float]
    max_drawdown: float


class HistoricalStress:
    """历史情景压力测试。"""
    name: str = "historical_stress"

    def run(
        self,
        weights: Mapping[str, float],
        scenario: StressScenario,
    ) -> StressResult:
        """基于场景冲击计算组合损失。"""
        asset_losses: dict[str, float] = {}
        port_loss = 0.0
        for symbol, w in weights.items():
            shock = scenario.shocks.get(symbol, 0.0)
            loss = w * shock
            asset_losses[symbol] = loss
            port_loss += loss

        return StressResult(
            scenario_name=scenario.name,
            portfolio_loss=port_loss,
            asset_losses=asset_losses,
            max_drawdown=abs(port_loss),
        )

    def run_from_returns(
        self,
        weights: Mapping[str, float],
        returns: Mapping[str, Sequence[float]],
        worst_n: int = 5,
    ) -> list[StressResult]:
        """从历史收益率中找到最差的 N 天。"""
        symbols = list(weights.keys())
        n_obs = min(len(returns.get(s, ())) for s in symbols) if symbols else 0
        if n_obs == 0:
            return []

        # 计算历史每日组合收益率
        daily = []
        for t in range(n_obs):
            ret = sum(weights[s] * returns[s][t] for s in symbols)
            daily.append((t, ret))

        # 排序找最差
        daily.sort(key=lambda x: x[1])
        results = []
        for t, port_ret in daily[:worst_n]:
            asset_losses = {s: weights[s] * returns[s][t] for s in symbols}
            results.append(StressResult(
                scenario_name=f"historical_day_{t}",
                portfolio_loss=port_ret,
                asset_losses=asset_losses,
                max_drawdown=abs(port_ret),
            ))
        return results
