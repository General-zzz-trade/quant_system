# attribution/cost.py
"""Trading cost attribution — fees, slippage, market impact."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Mapping


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
    total_slip = 0.0
    total_notional = 0.0
    for fill in fills:
        symbol = str(fill.get("symbol", ""))
        price = float(fill.get("price", 0))
        qty = float(fill.get("qty", 0))
        ref = reference_prices.get(symbol, price)
        if ref > 0:
            slip = abs(price - ref) / ref
            notional = qty * price
            total_slip += slip * notional
            total_notional += notional
    return (total_slip / total_notional * 10000) if total_notional > 0 else 0.0


def compute_cost_attribution(
    fills: Sequence[Mapping[str, object]],
    reference_prices: Mapping[str, float] | None = None,
) -> CostBreakdown:
    """计算完整交易成本归因。"""
    total_fees = 0.0
    total_notional = 0.0
    for fill in fills:
        total_fees += float(fill.get("fee", 0))
        total_notional += float(fill.get("qty", 0)) * float(fill.get("price", 0))

    fee_bps = (total_fees / total_notional * 10000) if total_notional > 0 else 0.0
    slip_bps = estimate_slippage(fills, reference_prices or {}) if reference_prices else 0.0

    return CostBreakdown(
        total_cost_bps=fee_bps + slip_bps,
        fee_bps=fee_bps,
        slippage_bps=slip_bps,
        market_impact_bps=0.0,
    )
