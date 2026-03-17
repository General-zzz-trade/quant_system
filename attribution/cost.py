# attribution/cost.py
"""Trading cost attribution — fees, slippage, market impact."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from _quant_hotpath import rust_compute_cost_attribution


@dataclass(frozen=True)
class CostBreakdown:
    """交易成本分解。"""
    total_cost_bps: float
    fee_bps: float
    slippage_bps: float
    market_impact_bps: float


def estimate_slippage(
    fills: Sequence[Mapping[str, object]],
    reference_prices: Mapping[str, float],
) -> float:
    """估计滑点 (bps)。"""
    r = rust_compute_cost_attribution(list(fills), dict(reference_prices))
    return float(r["slippage_bps"])


def compute_cost_attribution(
    fills: Sequence[Mapping[str, object]],
    reference_prices: Mapping[str, float] | None = None,
) -> CostBreakdown:
    """计算完整交易成本归因。"""
    r = rust_compute_cost_attribution(
        list(fills), dict(reference_prices) if reference_prices else None
    )
    return CostBreakdown(
        total_cost_bps=r["total_cost_bps"],
        fee_bps=r["fee_bps"],
        slippage_bps=r["slippage_bps"],
        market_impact_bps=r["market_impact_bps"],
    )
