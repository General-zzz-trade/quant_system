# attribution/report.py
"""Attribution report — consolidated P&L and cost attribution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from attribution.pnl import PnLBreakdown, compute_pnl
from attribution.cost import CostBreakdown, compute_cost_attribution


@dataclass(frozen=True)
class AttributionReport:
    """归因报告。"""
    pnl: PnLBreakdown
    cost: CostBreakdown
    net_return_pct: float
    period_label: str = ""


def build_report(
    fills: Sequence[Mapping[str, object]],
    initial_equity: float,
    current_prices: Mapping[str, float] | None = None,
    reference_prices: Mapping[str, float] | None = None,
    period_label: str = "",
) -> AttributionReport:
    """从成交记录构建归因报告。"""
    pnl = compute_pnl(fills, current_prices)
    cost = compute_cost_attribution(fills, reference_prices)
    net_return = pnl.total_pnl / initial_equity if initial_equity > 0 else 0.0
    return AttributionReport(
        pnl=pnl,
        cost=cost,
        net_return_pct=net_return * 100,
        period_label=period_label,
    )
