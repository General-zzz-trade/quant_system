# tests/unit/state/test_account_state.py
"""AccountReducer unit tests — covers fill accounting: fees, realized PnL, margin."""
from __future__ import annotations

import pytest
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from state.account import AccountState
from state.reducers.account import AccountReducer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Header:
    event_type: str = "fill"
    ts: Optional[datetime] = None
    event_id: Optional[str] = None

@dataclass(frozen=True)
class FillEvent:
    header: _Header
    fee: str = "0"
    realized_pnl: str = "0"
    cash_delta: str = "0"
    margin_change: str = "0"

def _fill(fee: str = "0", realized_pnl: str = "0", cash_delta: str = "0", margin_change: str = "0") -> FillEvent:
    return FillEvent(
        header=_Header(event_type="fill"),
        fee=fee,
        realized_pnl=realized_pnl,
        cash_delta=cash_delta,
        margin_change=margin_change,
    )

@dataclass(frozen=True)
class NonFillEvent:
    header: _Header = _Header(event_type="market")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reducer() -> AccountReducer:
    return AccountReducer()

@pytest.fixture
def initial_account() -> AccountState:
    return AccountState.initial(currency="USDT", balance=Decimal("10000"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAccountReducer:
    def test_non_fill_event_ignored(self, reducer: AccountReducer, initial_account: AccountState) -> None:
        res = reducer.reduce(initial_account, NonFillEvent())
        assert res.changed is False
        assert res.state is initial_account

    def test_fee_deducted(self, reducer: AccountReducer, initial_account: AccountState) -> None:
        res = reducer.reduce(initial_account, _fill(fee="5"))
        assert res.changed is True
        assert res.state.balance == Decimal("9995")
        assert res.state.fees_paid == Decimal("5")

    def test_realized_pnl_added(self, reducer: AccountReducer, initial_account: AccountState) -> None:
        res = reducer.reduce(initial_account, _fill(realized_pnl="200"))
        assert res.state.balance == Decimal("10200")
        assert res.state.realized_pnl == Decimal("200")

    def test_fee_and_pnl_combined(self, reducer: AccountReducer, initial_account: AccountState) -> None:
        res = reducer.reduce(initial_account, _fill(fee="10", realized_pnl="500"))
        assert res.state.balance == Decimal("10490")  # 10000 + 500 - 10
        assert res.state.fees_paid == Decimal("10")
        assert res.state.realized_pnl == Decimal("500")

    def test_margin_change(self, reducer: AccountReducer, initial_account: AccountState) -> None:
        res = reducer.reduce(initial_account, _fill(margin_change="1000"))
        assert res.state.margin_used == Decimal("1000")
        assert res.state.margin_available == Decimal("9000")  # 10000 - 1000

    def test_multiple_fills_accumulate(self, reducer: AccountReducer, initial_account: AccountState) -> None:
        r1 = reducer.reduce(initial_account, _fill(fee="5", realized_pnl="100"))
        r2 = reducer.reduce(r1.state, _fill(fee="3", realized_pnl="-50"))
        assert r2.state.balance == Decimal("10042")  # 10000 + 100 - 5 + (-50) - 3
        assert r2.state.fees_paid == Decimal("8")
        assert r2.state.realized_pnl == Decimal("50")

    def test_immutability(self, reducer: AccountReducer, initial_account: AccountState) -> None:
        reducer.reduce(initial_account, _fill(fee="100"))
        assert initial_account.balance == Decimal("10000")
        assert initial_account.fees_paid == Decimal("0")

    def test_cash_delta(self, reducer: AccountReducer, initial_account: AccountState) -> None:
        res = reducer.reduce(initial_account, _fill(cash_delta="500"))
        assert res.state.balance == Decimal("10500")
