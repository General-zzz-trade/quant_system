# attribution/pnl.py
"""P&L attribution — breakdown of returns by source."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from _quant_hotpath import rust_compute_pnl


@dataclass(frozen=True)
class PnLBreakdown:
    """P&L 归因分解。"""
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    fee_cost: float
    funding_pnl: float = 0.0
    by_symbol: Mapping[str, float] = field(default_factory=dict)


def compute_pnl(
    fills: Sequence[Mapping[str, object]],
    current_prices: Mapping[str, float] | None = None,
) -> PnLBreakdown:
    """从成交记录计算 P&L 归因。"""
    r = rust_compute_pnl(list(fills), dict(current_prices) if current_prices else None)
    return PnLBreakdown(
        total_pnl=r["total_pnl"],
        realized_pnl=r["realized_pnl"],
        unrealized_pnl=r["unrealized_pnl"],
        fee_cost=r["fee_cost"],
        funding_pnl=r["funding_pnl"],
        by_symbol=r["by_symbol"],
    )
