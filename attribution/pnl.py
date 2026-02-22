# attribution/pnl.py
"""P&L attribution — breakdown of returns by source."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


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
    realized = 0.0
    fees = 0.0
    by_symbol: dict[str, float] = {}
    positions: dict[str, tuple[float, float]] = {}  # symbol → (qty, avg_price)

    for fill in fills:
        symbol = str(fill.get("symbol", ""))
        qty = float(fill.get("qty", 0))
        price = float(fill.get("price", 0))
        fee = float(fill.get("fee", 0))
        side = str(fill.get("side", "buy"))

        signed_qty = qty if side == "buy" else -qty
        fees += fee

        cur_qty, cur_avg = positions.get(symbol, (0.0, 0.0))
        new_qty = cur_qty + signed_qty

        # 平仓部分产生已实现 P&L
        if cur_qty != 0 and (cur_qty > 0) != (signed_qty > 0):
            closed_qty = min(abs(signed_qty), abs(cur_qty))
            pnl = closed_qty * (price - cur_avg) * (1 if cur_qty > 0 else -1)
            realized += pnl
            by_symbol[symbol] = by_symbol.get(symbol, 0.0) + pnl

        # 更新均价
        if abs(new_qty) > abs(cur_qty):
            total_cost = cur_avg * abs(cur_qty) + price * abs(signed_qty)
            new_avg = total_cost / abs(new_qty) if new_qty != 0 else 0.0
        else:
            new_avg = cur_avg
        positions[symbol] = (new_qty, new_avg)

    # 未实现 P&L
    unrealized = 0.0
    if current_prices:
        for symbol, (qty, avg) in positions.items():
            if qty != 0 and symbol in current_prices:
                unrealized += qty * (current_prices[symbol] - avg)

    return PnLBreakdown(
        total_pnl=realized + unrealized - fees,
        realized_pnl=realized,
        unrealized_pnl=unrealized,
        fee_cost=fees,
        by_symbol=by_symbol,
    )
