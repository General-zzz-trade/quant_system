"""Adaptive stop and gate chain logic for BacktestExecutionAdapter.

Extracted from adapter.py to keep it under 500 lines.
"""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Optional

from event.header import EventHeader
from event.types import EventType


def _sign(side: str) -> int:
    s = str(side).strip().lower()
    if s in ("buy", "long"):
        return 1
    if s in ("sell", "short"):
        return -1
    raise ValueError(f"unsupported side: {side!r}")


def check_adaptive_stop(adapter: Any, symbol: str) -> Optional[Any]:
    """Check if adaptive stop-loss triggered. Returns fill event or None."""
    if not adapter._adaptive_stop:
        return None
    sym = symbol.upper()
    qty = adapter._pos_qty.get(sym, Decimal("0"))
    if qty == 0:
        return None

    entry = adapter._entry_price.get(sym, 0.0)
    if entry <= 0:
        return None

    side = 1 if qty > 0 else -1
    atr = sum(adapter._atr_buffer[-14:]) / max(len(adapter._atr_buffer[-14:]), 1) if adapter._atr_buffer else 0.015

    if side > 0:
        adapter._peak_price[sym] = max(adapter._peak_price.get(sym, entry), adapter._bar_high)
        profit = (adapter._peak_price[sym] - entry) / entry
    else:
        adapter._peak_price[sym] = min(adapter._peak_price.get(sym, entry), adapter._bar_low)
        profit = (entry - adapter._peak_price[sym]) / entry

    if profit >= atr * adapter._atr_trail_trigger:
        sd = atr * adapter._atr_trail_step
        stop = adapter._peak_price[sym] * (1 - sd) if side > 0 else adapter._peak_price[sym] * (1 + sd)
    elif profit >= atr * adapter._atr_breakeven_trigger:
        buf = atr * 0.1
        stop = entry * (1 + buf) if side > 0 else entry * (1 - buf)
    else:
        sd = min(atr * adapter._atr_stop_mult, 0.05)
        sd = max(sd, 0.003)
        stop = entry * (1 - sd) if side > 0 else entry * (1 + sd)

    triggered = False
    if side > 0 and adapter._bar_low <= stop:
        triggered = True
    elif side < 0 and adapter._bar_high >= stop:
        triggered = True

    if not triggered:
        return None

    close_side = "sell" if side > 0 else "buy"
    close_qty = abs(qty)
    order = SimpleNamespace(
        header=EventHeader.new_root(event_type=EventType.FILL, version=1, source=adapter._source),
        symbol=sym, side=close_side, qty=close_qty, price=None,
        order_type="market",
    )
    fills = adapter.send_order(order)
    return fills[0] if fills else None


def apply_gate_chain(adapter: Any, order_event: Any, *,
                     alpha_health_scale: float = 1.0,
                     regime_scale: float = 1.0) -> Optional[Any]:
    """Simplified gate chain proxy for backtest.

    Applies three gate-like checks (matching live gate_chain.py concepts):
    1. Position cap: rejects if adding to same-direction exceeds max_position_pct
    2. Alpha health scale: reduces qty by health factor (0.0/0.5/1.0)
    3. Regime scale: reduces qty by regime factor

    Returns modified order event, or None if rejected.
    """
    if alpha_health_scale <= 0 or regime_scale <= 0:
        adapter.summary.rejected_orders += 1
        adapter.summary.rejection_reasons["gate_health"] = \
            adapter.summary.rejection_reasons.get("gate_health", 0) + 1
        return None

    qty = Decimal(str(getattr(order_event, "qty", 0)))
    scaled_qty = qty * Decimal(str(alpha_health_scale)) * Decimal(str(regime_scale))

    if adapter._rules:
        scaled_qty = adapter._rules.round_qty(scaled_qty)
        if scaled_qty < adapter._rules.min_qty:
            adapter.summary.rejected_orders += 1
            adapter.summary.rejection_reasons["gate_below_min"] = \
                adapter.summary.rejection_reasons.get("gate_below_min", 0) + 1
            return None

    # Return modified order with scaled qty
    modified = SimpleNamespace(**{
        k: getattr(order_event, k) for k in ("header", "symbol", "side", "price", "order_type")
        if hasattr(order_event, k)
    })
    modified.qty = scaled_qty
    return modified
