# DEPRECATED: Superseded by Rust AccountReducer (rust_pipeline_apply). Retained for parity tests.
from __future__ import annotations

from typing import Any

from state.account import AccountState
from state.errors import ReducerError
from state.reducers.base import ReducerResult
from state._util import get_event_type, get_event_ts, to_decimal


class AccountReducer:
    """Project fill and funding events into AccountState (v1 minimal ledger)."""

    def reduce(self, state: AccountState, event: Any) -> ReducerResult[AccountState]:
        et = get_event_type(event)

        if et == "funding":
            return self._reduce_funding(state, event)

        if et not in ("fill", "trade_fill", "execution_fill"):
            return ReducerResult(state=state, changed=False)

        ts = get_event_ts(event)

        fee = to_decimal(getattr(event, "fee", None), allow_none=True) or to_decimal("0")
        realized = to_decimal(getattr(event, "realized_pnl", None), allow_none=True) or to_decimal("0")
        cash_delta = to_decimal(getattr(event, "cash_delta", None), allow_none=True) or to_decimal("0")
        margin_change = to_decimal(getattr(event, "margin_change", None), allow_none=True) or to_decimal("0")

        new_balance = state.balance + realized + cash_delta - fee
        new_margin_used = state.margin_used + margin_change

        new_state = state.with_update(
            balance=new_balance,
            margin_used=new_margin_used,
            realized_pnl=state.realized_pnl + realized,
            unrealized_pnl=state.unrealized_pnl,
            fees_paid=state.fees_paid + fee,
            ts=ts,
        )
        return ReducerResult(state=new_state, changed=True, note="fill_account")

    def _reduce_funding(self, state: AccountState, event: Any) -> ReducerResult[AccountState]:
        """Apply funding rate settlement to account balance.

        Settlement = position_qty * mark_price * funding_rate
        - Positive rate + long position -> pay (balance decreases)
        - Positive rate + short position -> receive (balance increases)
        - Negative rate reverses the direction
        """
        ts = get_event_ts(event)
        funding_rate = to_decimal(getattr(event, "funding_rate", None), allow_none=True)
        mark_price = to_decimal(getattr(event, "mark_price", None), allow_none=True)
        position_qty = to_decimal(getattr(event, "position_qty", None), allow_none=True)

        if funding_rate is None or mark_price is None or position_qty is None:
            return ReducerResult(state=state, changed=False)
        if position_qty == 0:
            return ReducerResult(state=state, changed=False)

        # funding_payment > 0 means account pays; < 0 means account receives
        funding_payment = position_qty * mark_price * funding_rate
        new_balance = state.balance - funding_payment

        new_state = state.with_update(
            balance=new_balance,
            margin_used=state.margin_used,
            realized_pnl=state.realized_pnl - funding_payment,
            unrealized_pnl=state.unrealized_pnl,
            fees_paid=state.fees_paid + abs(funding_payment),
            ts=ts,
        )
        return ReducerResult(state=new_state, changed=True, note="funding_settlement")
