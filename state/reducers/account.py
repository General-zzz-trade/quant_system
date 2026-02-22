from __future__ import annotations

from typing import Any

from state.account import AccountState
from state.errors import ReducerError
from state.reducers.base import ReducerResult
from state._util import get_event_type, get_event_ts, to_decimal


class AccountReducer:
    """Project fill events into AccountState (v1 minimal ledger)."""

    def reduce(self, state: AccountState, event: Any) -> ReducerResult[AccountState]:
        et = get_event_type(event)
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
