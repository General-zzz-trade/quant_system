# context/portfolio/portfolio_state.py
"""Portfolio state — derived from account + positions + market."""
from __future__ import annotations

from decimal import Decimal
from typing import Dict, Mapping, Optional

from context.portfolio.portfolio_snapshot import ContextPortfolioSnapshot


class ContextPortfolioState:
    """
    组合状态 — 从账户和仓位汇总计算。

    不直接存储数据，而是根据输入计算。
    """

    def compute(
        self,
        *,
        equity: Decimal,
        positions: Mapping[str, Decimal],       # symbol -> signed qty
        prices: Mapping[str, Decimal],           # symbol -> mark price
    ) -> ContextPortfolioSnapshot:
        long_exp = Decimal("0")
        short_exp = Decimal("0")
        for symbol, qty in positions.items():
            price = prices.get(symbol, Decimal("0"))
            notional = abs(qty) * price
            if qty > 0:
                long_exp += notional
            elif qty < 0:
                short_exp += notional

        gross = long_exp + short_exp
        net = long_exp - short_exp
        leverage = gross / equity if equity > 0 else Decimal("0")

        return ContextPortfolioSnapshot(
            total_equity=equity,
            gross_exposure=gross,
            net_exposure=net,
            leverage=leverage,
            position_count=len([q for q in positions.values() if q != 0]),
            long_exposure=long_exp,
            short_exposure=short_exp,
            cash=equity - gross,
        )
