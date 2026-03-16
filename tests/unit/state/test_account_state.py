"""Tests for state.account.AccountState."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from state.account import AccountState


class TestAccountStateInitialFactory:
    def test_initial_sets_balance(self):
        acc = AccountState.initial(currency="USDT", balance=Decimal("10000"))
        assert acc.balance == Decimal("10000")
        assert acc.currency == "USDT"

    def test_initial_margin_available_equals_balance(self):
        acc = AccountState.initial(currency="USDT", balance=Decimal("5000"))
        assert acc.margin_available == Decimal("5000")

    def test_initial_zeros(self):
        acc = AccountState.initial(currency="USDT", balance=Decimal("1000"))
        assert acc.margin_used == Decimal("0")
        assert acc.realized_pnl == Decimal("0")
        assert acc.unrealized_pnl == Decimal("0")
        assert acc.fees_paid == Decimal("0")

    def test_initial_last_ts_none(self):
        acc = AccountState.initial(currency="USDT", balance=Decimal("1000"))
        assert acc.last_ts is None

    def test_initial_zero_balance(self):
        acc = AccountState.initial(currency="USDT", balance=Decimal("0"))
        assert acc.balance == Decimal("0")
        assert acc.margin_available == Decimal("0")


class TestAccountStateWithUpdate:
    def test_with_update_preserves_currency(self):
        acc = AccountState.initial(currency="USDT", balance=Decimal("10000"))
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        updated = acc.with_update(
            balance=Decimal("9500"),
            margin_used=Decimal("500"),
            realized_pnl=Decimal("100"),
            unrealized_pnl=Decimal("-50"),
            fees_paid=Decimal("10"),
            ts=ts,
        )
        assert updated.currency == "USDT"

    def test_with_update_computes_margin_available(self):
        acc = AccountState.initial(currency="USDT", balance=Decimal("10000"))
        updated = acc.with_update(
            balance=Decimal("10000"),
            margin_used=Decimal("3000"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            fees_paid=Decimal("0"),
            ts=None,
        )
        assert updated.margin_available == Decimal("7000")

    def test_with_update_ts_none_keeps_previous(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        acc = AccountState.initial(currency="USDT", balance=Decimal("1000"))
        acc = acc.with_update(
            balance=Decimal("1000"),
            margin_used=Decimal("0"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            fees_paid=Decimal("0"),
            ts=ts,
        )
        updated = acc.with_update(
            balance=Decimal("900"),
            margin_used=Decimal("100"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            fees_paid=Decimal("0"),
            ts=None,
        )
        assert updated.last_ts == ts

    def test_immutability(self):
        acc = AccountState.initial(currency="USDT", balance=Decimal("1000"))
        with pytest.raises(AttributeError):
            acc.balance = Decimal("2000")  # type: ignore[misc]

    def test_negative_balance(self):
        acc = AccountState.initial(currency="USDT", balance=Decimal("-100"))
        assert acc.balance == Decimal("-100")

    def test_with_update_naive_ts_converted_to_utc(self):
        acc = AccountState.initial(currency="USDT", balance=Decimal("1000"))
        naive_ts = datetime(2024, 6, 15, 12, 0, 0)
        updated = acc.with_update(
            balance=Decimal("1000"),
            margin_used=Decimal("0"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            fees_paid=Decimal("0"),
            ts=naive_ts,
        )
        assert updated.last_ts is not None
        assert updated.last_ts.tzinfo is not None
