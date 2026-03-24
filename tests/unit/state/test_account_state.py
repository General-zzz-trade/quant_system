"""Tests for RustAccountState (via state.AccountState alias)."""
from __future__ import annotations

import pytest

from state import AccountState

_SCALE = 100_000_000


class TestAccountStateInitialFactory:
    def test_initial_sets_balance(self):
        acc = AccountState.initial(currency="USDT", balance=10000 * _SCALE)
        assert acc.balance == 10000 * _SCALE
        assert acc.currency == "USDT"

    def test_initial_zeros(self):
        acc = AccountState.initial(currency="USDT", balance=1000 * _SCALE)
        assert acc.margin_used == 0
        assert acc.realized_pnl == 0
        assert acc.unrealized_pnl == 0
        assert acc.fees_paid == 0

    def test_initial_last_ts_none(self):
        acc = AccountState.initial(currency="USDT", balance=1000 * _SCALE)
        assert acc.last_ts is None

    def test_initial_zero_balance(self):
        acc = AccountState.initial(currency="USDT", balance=0)
        assert acc.balance == 0


class TestAccountStateWithUpdate:
    def test_with_update_preserves_currency(self):
        acc = AccountState.initial(currency="USDT", balance=10000 * _SCALE)
        updated = acc.with_update(
            balance=9500 * _SCALE,
            margin_used=500 * _SCALE,
            realized_pnl=100 * _SCALE,
            unrealized_pnl=-50 * _SCALE,
            fees_paid=10 * _SCALE,
            ts="2024-01-01T00:00:00+00:00",
        )
        assert updated.currency == "USDT"

    def test_with_update_ts(self):
        acc = AccountState.initial(currency="USDT", balance=1000 * _SCALE)
        updated = acc.with_update(
            balance=900 * _SCALE,
            margin_used=100 * _SCALE,
            realized_pnl=0,
            unrealized_pnl=0,
            fees_paid=0,
            ts="2024-01-01T00:00:00+00:00",
        )
        assert updated.last_ts == "2024-01-01T00:00:00+00:00"

    def test_immutability(self):
        acc = AccountState.initial(currency="USDT", balance=1000 * _SCALE)
        with pytest.raises(AttributeError):
            acc.balance = 2000 * _SCALE  # type: ignore[misc]

    def test_float_accessors(self):
        acc = AccountState.initial(currency="USDT", balance=10000 * _SCALE)
        assert acc.balance_f == pytest.approx(10000.0)
        assert acc.margin_used_f == pytest.approx(0.0)

    def test_equality(self):
        a1 = AccountState.initial(currency="USDT", balance=1000 * _SCALE)
        a2 = AccountState.initial(currency="USDT", balance=1000 * _SCALE)
        assert a1 == a2
