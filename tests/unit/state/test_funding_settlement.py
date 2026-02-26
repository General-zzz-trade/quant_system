# tests/unit/state/test_funding_settlement.py
"""Tests for funding rate settlement in AccountReducer."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

import pytest

from state.account import AccountState
from state.reducers.account import AccountReducer


@dataclass(frozen=True)
class _Header:
    event_type: str = "funding"
    ts: Optional[datetime] = None
    event_id: str = "funding-1"


@dataclass(frozen=True)
class FakeFundingEvent:
    """Minimal funding event for testing."""
    header: _Header
    ts: datetime
    symbol: str
    funding_rate: Decimal
    mark_price: Decimal
    position_qty: Decimal


def _make_account(balance: Decimal = Decimal("10000")) -> AccountState:
    return AccountState.initial(currency="USDT", balance=balance)


class TestFundingSettlement:
    def test_long_positive_rate_pays(self) -> None:
        """Long position with positive funding rate: account pays."""
        reducer = AccountReducer()
        account = _make_account(Decimal("10000"))
        ts = datetime(2025, 1, 15, 8, 0, 0, tzinfo=timezone.utc)

        event = FakeFundingEvent(
            header=_Header(ts=ts),
            ts=ts,
            symbol="BTCUSDT",
            funding_rate=Decimal("0.0001"),   # 1 bps
            mark_price=Decimal("40000"),
            position_qty=Decimal("0.5"),      # long 0.5 BTC
        )
        result = reducer.reduce(account, event)
        assert result.changed
        assert result.note == "funding_settlement"

        # payment = 0.5 * 40000 * 0.0001 = 2.0 USDT
        expected_balance = Decimal("10000") - Decimal("2.0")
        assert result.state.balance == expected_balance

    def test_short_positive_rate_receives(self) -> None:
        """Short position with positive funding rate: account receives."""
        reducer = AccountReducer()
        account = _make_account(Decimal("10000"))
        ts = datetime(2025, 1, 15, 8, 0, 0, tzinfo=timezone.utc)

        event = FakeFundingEvent(
            header=_Header(ts=ts),
            ts=ts,
            symbol="BTCUSDT",
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("40000"),
            position_qty=Decimal("-0.5"),     # short 0.5 BTC
        )
        result = reducer.reduce(account, event)
        assert result.changed

        # payment = -0.5 * 40000 * 0.0001 = -2.0 -> receives 2.0
        expected_balance = Decimal("10000") + Decimal("2.0")
        assert result.state.balance == expected_balance

    def test_negative_funding_rate(self) -> None:
        """Negative funding rate: shorts pay, longs receive."""
        reducer = AccountReducer()
        account = _make_account(Decimal("10000"))
        ts = datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)

        event = FakeFundingEvent(
            header=_Header(ts=ts),
            ts=ts,
            symbol="BTCUSDT",
            funding_rate=Decimal("-0.0002"),  # -2 bps
            mark_price=Decimal("40000"),
            position_qty=Decimal("1.0"),      # long
        )
        result = reducer.reduce(account, event)

        # payment = 1.0 * 40000 * (-0.0002) = -8.0 -> receives 8.0
        expected_balance = Decimal("10000") + Decimal("8.0")
        assert result.state.balance == expected_balance

    def test_flat_position_no_settlement(self) -> None:
        """Zero position: no funding settlement."""
        reducer = AccountReducer()
        account = _make_account()
        ts = datetime(2025, 1, 15, 8, 0, 0, tzinfo=timezone.utc)

        event = FakeFundingEvent(
            header=_Header(ts=ts),
            ts=ts,
            symbol="BTCUSDT",
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("40000"),
            position_qty=Decimal("0"),
        )
        result = reducer.reduce(account, event)
        assert not result.changed
        assert result.state.balance == Decimal("10000")

    def test_fees_paid_tracks_absolute_funding(self) -> None:
        """fees_paid should increase by abs(funding_payment) regardless of direction."""
        reducer = AccountReducer()
        account = _make_account()
        ts = datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc)

        # Long pays
        event = FakeFundingEvent(
            header=_Header(ts=ts),
            ts=ts,
            symbol="BTCUSDT",
            funding_rate=Decimal("0.0003"),
            mark_price=Decimal("50000"),
            position_qty=Decimal("2.0"),
        )
        result = reducer.reduce(account, event)
        # payment = 2.0 * 50000 * 0.0003 = 30.0
        assert result.state.fees_paid == Decimal("30.0")

    def test_realized_pnl_includes_funding(self) -> None:
        """realized_pnl should reflect funding payments."""
        reducer = AccountReducer()
        account = _make_account()
        ts = datetime(2025, 1, 15, 8, 0, 0, tzinfo=timezone.utc)

        event = FakeFundingEvent(
            header=_Header(ts=ts),
            ts=ts,
            symbol="BTCUSDT",
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("40000"),
            position_qty=Decimal("0.5"),
        )
        result = reducer.reduce(account, event)
        # payment = 2.0, realized_pnl = -2.0 (loss from paying funding)
        assert result.state.realized_pnl == Decimal("-2.0")

    def test_fill_events_still_work(self) -> None:
        """Ensure fill events still process correctly after funding support."""
        reducer = AccountReducer()
        account = _make_account()

        @dataclass(frozen=True)
        class FakeFill:
            header: _Header = _Header(event_type="fill")
            fee: Decimal = Decimal("1.0")
            realized_pnl: Decimal = Decimal("100.0")
            cash_delta: Decimal = Decimal("0")
            margin_change: Decimal = Decimal("0")

        result = reducer.reduce(account, FakeFill())
        assert result.changed
        assert result.note == "fill_account"
        assert result.state.balance == Decimal("10000") + Decimal("100") - Decimal("1")
