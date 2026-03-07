"""Tests for Rust state type converters (Python Decimal <-> Rust i64 round-trips)."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

pytest.importorskip("_quant_hotpath")

from state.account import AccountState
from state.market import MarketState
from state.position import PositionState
from state.rust_adapters import (
    account_from_rust,
    account_to_rust,
    market_from_rust,
    market_to_rust,
    position_from_rust,
    position_to_rust,
)


def test_market_state_round_trip() -> None:
    state = MarketState(
        symbol="BTCUSDT",
        last_price=Decimal("100.5"),
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=Decimal("12.3"),
        last_ts=datetime(2026, 3, 7, 1, 2, 3, tzinfo=timezone.utc),
    )

    restored = market_from_rust(market_to_rust(state))

    assert restored == state


def test_position_state_round_trip() -> None:
    state = PositionState(
        symbol="BTCUSDT",
        qty=Decimal("1.25"),
        avg_price=Decimal("95"),
        last_price=Decimal("100"),
        last_ts=datetime(2026, 3, 7, 1, 2, 3, tzinfo=timezone.utc),
    )

    restored = position_from_rust(position_to_rust(state))

    assert restored == state


def test_account_state_round_trip() -> None:
    state = AccountState(
        currency="USDT",
        balance=Decimal("10000"),
        margin_used=Decimal("100"),
        margin_available=Decimal("9900"),
        realized_pnl=Decimal("5"),
        unrealized_pnl=Decimal("2"),
        fees_paid=Decimal("1"),
        last_ts=datetime(2026, 3, 7, 1, 2, 3, tzinfo=timezone.utc),
    )

    restored = account_from_rust(account_to_rust(state))

    assert restored == state
