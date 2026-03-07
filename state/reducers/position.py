# DEPRECATED: Superseded by Rust PositionReducer (rust_pipeline_apply). Retained for parity tests.
from __future__ import annotations

from decimal import Decimal
from typing import Any

from state.errors import ReducerError
from state.position import PositionState
from state.reducers.base import ReducerResult
from state._util import get_event_type, get_event_ts, get_symbol, signed_qty, to_decimal


class PositionReducer:
    """Project fill events into PositionState (qty/avg_price)."""

    def reduce(self, state: PositionState, event: Any) -> ReducerResult[PositionState]:
        et = get_event_type(event)
        if et not in ("fill", "trade_fill", "execution_fill"):
            return ReducerResult(state=state, changed=False)

        sym = get_symbol(event, state.symbol)
        if sym != state.symbol:
            return ReducerResult(state=state, changed=False)

        ts = get_event_ts(event)

        qty_raw = getattr(event, "qty", None)
        if qty_raw is None:
            qty_raw = getattr(event, "quantity", None)
        if qty_raw is None:
            raise ReducerError("fill event missing qty/quantity")

        side = getattr(event, "side", None)
        if side is None:
            raise ReducerError("fill event missing side")

        qty = signed_qty(qty_raw, side)
        if qty == 0:
            raise ReducerError("fill qty cannot be 0")

        price_raw = getattr(event, "price", None)
        if price_raw is None:
            raise ReducerError("fill event missing price")
        price = to_decimal(price_raw)
        if price <= 0:
            raise ReducerError("fill price must be > 0")

        prev_qty = state.qty
        new_qty = prev_qty + qty

        # 1) flat
        if new_qty == 0:
            new_state = state.with_update(qty=Decimal("0"), avg_price=None, last_price=price, ts=ts)
            return ReducerResult(state=new_state, changed=True, note="position_flat")

        # 2) opening or adding (same direction as position)
        if prev_qty == 0 or (prev_qty > 0) == (qty > 0):
            if prev_qty == 0 or state.avg_price is None:
                new_avg = price
            else:
                # weighted avg for adds only
                new_avg = (abs(prev_qty) * state.avg_price + abs(qty) * price) / abs(new_qty)
            new_state = state.with_update(qty=new_qty, avg_price=new_avg, last_price=price, ts=ts)
            return ReducerResult(state=new_state, changed=True, note="position_add")

        # 3) reducing (opposite direction fill)
        # - if not crossing zero: avg_price must remain unchanged
        if abs(qty) < abs(prev_qty):
            new_state = state.with_update(qty=new_qty, avg_price=state.avg_price, last_price=price, ts=ts)
            return ReducerResult(state=new_state, changed=True, note="position_reduce")

        # 4) crossing zero -> reverse: avg resets to this fill price
        new_state = state.with_update(qty=new_qty, avg_price=price, last_price=price, ts=ts)
        return ReducerResult(state=new_state, changed=True, note="position_reverse")
