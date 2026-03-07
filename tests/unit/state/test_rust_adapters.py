from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

pytest.importorskip("_quant_hotpath")

from state.account import AccountState
from state.errors import ReducerError
from state.market import MarketState
from state.position import PositionState
from state.rust_adapters import (
    RustAccountReducerAdapter,
    RustMarketReducerAdapter,
    RustPositionReducerAdapter,
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


def test_market_reducer_adapter_updates_python_state() -> None:
    reducer = RustMarketReducerAdapter()
    state = MarketState.empty("BTCUSDT")
    event = SimpleNamespace(
        event_type="market",
        symbol="BTCUSDT",
        open="99",
        high="101",
        low="98",
        close="100",
        volume="2",
        header=SimpleNamespace(ts=datetime(2026, 3, 7, 1, 2, 3, tzinfo=timezone.utc)),
    )

    result = reducer.reduce(state, event)

    assert result.changed is True
    assert result.state.close == Decimal("100")
    assert result.state.last_ts == datetime(2026, 3, 7, 1, 2, 3, tzinfo=timezone.utc)


def test_position_reducer_adapter_matches_position_contract() -> None:
    reducer = RustPositionReducerAdapter()
    state = PositionState.empty("BTCUSDT")
    event = SimpleNamespace(
        event_type="fill",
        symbol="BTCUSDT",
        side="buy",
        qty="2",
        price="100",
        header=SimpleNamespace(ts=datetime(2026, 3, 7, 1, 2, 3, tzinfo=timezone.utc)),
    )

    result = reducer.reduce(state, event)

    assert result.note == "position_add"
    assert result.state.qty == Decimal("2")
    assert result.state.avg_price == Decimal("100")


def test_account_reducer_adapter_handles_funding() -> None:
    reducer = RustAccountReducerAdapter()
    state = AccountState.initial(currency="USDT", balance=Decimal("10000"))
    event = SimpleNamespace(
        event_type="funding",
        funding_rate=Decimal("0.0001"),
        mark_price=Decimal("40000"),
        position_qty=Decimal("0.5"),
        header=SimpleNamespace(ts=datetime(2026, 3, 7, 1, 2, 3, tzinfo=timezone.utc)),
    )

    result = reducer.reduce(state, event)

    assert result.note == "funding_settlement"
    assert result.state.balance == Decimal("9998")
    assert result.state.fees_paid == Decimal("2")


def test_adapter_raises_reducer_error_on_bad_fill() -> None:
    reducer = RustPositionReducerAdapter()

    with pytest.raises(ReducerError, match="missing side"):
        reducer.reduce(
            PositionState.empty("BTCUSDT"),
            SimpleNamespace(event_type="fill", symbol="BTCUSDT", qty="1", price="100", header=None),
        )
