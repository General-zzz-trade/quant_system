"""Signal-level P&L attribution.

Traces every fill back to the originating signal (IntentEvent.origin)
via the intent→order→fill chain, then computes per-signal P&L.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class SignalPnL:
    """P&L breakdown for a single signal origin."""

    origin: str
    realized_pnl: float
    unrealized_pnl: float
    fee_cost: float
    trade_count: int
    win_rate: float


@dataclass(frozen=True)
class SignalAttributionReport:
    """Full signal attribution report."""

    by_signal: Mapping[str, SignalPnL]
    total_pnl: float
    unattributed_pnl: float


def attribute_by_signal(
    intents: Sequence[Mapping[str, object]],
    orders: Sequence[Mapping[str, object]],
    fills: Sequence[Mapping[str, object]],
    current_prices: Optional[Mapping[str, float]] = None,
) -> SignalAttributionReport:
    """Attribute fills to signal origins via intent→order→fill chain.

    Parameters
    ----------
    intents : sequence of dicts with keys intent_id, origin, symbol, side
    orders  : sequence of dicts with keys order_id, intent_id, symbol, side
    fills   : sequence of dicts with keys fill_id, order_id, symbol, side, qty, price, fee (optional)
    current_prices : optional symbol→price map for unrealized P&L

    Returns
    -------
    SignalAttributionReport with per-signal P&L breakdown.
    """
    # Build lookup: intent_id → origin
    intent_origin: Dict[str, str] = {}
    intent_side: Dict[str, str] = {}
    for intent in intents:
        iid = str(intent.get("intent_id", ""))
        intent_origin[iid] = str(intent.get("origin", ""))
        intent_side[iid] = str(intent.get("side", ""))

    # Build lookup: order_id → intent_id
    order_intent: Dict[str, str] = {}
    for order in orders:
        oid = str(order.get("order_id", ""))
        order_intent[oid] = str(order.get("intent_id", ""))

    # Group fills by origin
    origin_fills: Dict[str, List[Mapping[str, object]]] = {}
    unattributed_fills: List[Mapping[str, object]] = []

    for fill in fills:
        order_id = str(fill.get("order_id", ""))
        intent_id = order_intent.get(order_id, "")
        origin = intent_origin.get(intent_id, "") if intent_id else ""

        if origin:
            origin_fills.setdefault(origin, []).append(fill)
        else:
            unattributed_fills.append(fill)

    # Compute P&L per origin
    by_signal: Dict[str, SignalPnL] = {}
    total_pnl = 0.0

    for origin, o_fills in origin_fills.items():
        pnl = _compute_origin_pnl(origin, o_fills, current_prices)
        by_signal[origin] = pnl
        total_pnl += pnl.realized_pnl + pnl.unrealized_pnl - pnl.fee_cost

    # Unattributed
    unattr_pnl = _compute_origin_pnl("__unattributed__", unattributed_fills, current_prices) if unattributed_fills else None
    unattributed_total = 0.0
    if unattr_pnl is not None:
        unattributed_total = unattr_pnl.realized_pnl + unattr_pnl.unrealized_pnl - unattr_pnl.fee_cost
        total_pnl += unattributed_total

    return SignalAttributionReport(
        by_signal=by_signal,
        total_pnl=total_pnl,
        unattributed_pnl=unattributed_total,
    )


def _compute_origin_pnl(
    origin: str,
    fills: Sequence[Mapping[str, object]],
    current_prices: Optional[Mapping[str, float]] = None,
) -> SignalPnL:
    """Compute P&L for a group of fills sharing the same origin."""
    realized = 0.0
    fees = 0.0
    positions: Dict[str, tuple[float, float]] = {}  # symbol → (qty, avg_price)
    wins = 0
    trades = 0

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

        # Closing portion → realized P&L
        if cur_qty != 0 and (cur_qty > 0) != (signed_qty > 0):
            closed_qty = min(abs(signed_qty), abs(cur_qty))
            pnl = closed_qty * (price - cur_avg) * (1 if cur_qty > 0 else -1)
            realized += pnl
            trades += 1
            if pnl > 0:
                wins += 1

        # Update average price
        if abs(new_qty) > abs(cur_qty):
            total_cost = cur_avg * abs(cur_qty) + price * abs(signed_qty)
            new_avg = total_cost / abs(new_qty) if new_qty != 0 else 0.0
        else:
            new_avg = cur_avg
        positions[symbol] = (new_qty, new_avg)

    # Unrealized P&L
    unrealized = 0.0
    if current_prices:
        for symbol, (qty, avg) in positions.items():
            if qty != 0 and symbol in current_prices:
                unrealized += qty * (current_prices[symbol] - avg)

    win_rate = wins / trades if trades > 0 else 0.0
    trade_count = len(fills)

    return SignalPnL(
        origin=origin,
        realized_pnl=realized,
        unrealized_pnl=unrealized,
        fee_cost=fees,
        trade_count=trade_count,
        win_rate=win_rate,
    )
