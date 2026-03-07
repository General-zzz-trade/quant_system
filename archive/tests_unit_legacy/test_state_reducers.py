"""Tests for state reducers: AccountReducer, PositionReducer, MarketReducer."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

import pytest

from state.account import AccountState
from state.market import MarketState
from state.position import PositionState
from state.reducers.account import AccountReducer
from state.reducers.market import MarketReducer
from state.reducers.position import PositionReducer

# ─── Timestamp helper ────────────────────────────────────────────────────────

_TS = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


# ─── Event stubs ─────────────────────────────────────────────────────────────

@dataclass
class _FillEvent:
    """Minimal fill event stub understood by the reducers."""
    symbol: str
    side: str
    qty: Decimal
    price: Decimal
    event_type: str = "fill"
    ts: Optional[datetime] = None
    fee: Optional[Decimal] = None
    realized_pnl: Optional[Decimal] = None
    cash_delta: Optional[Decimal] = None
    margin_change: Optional[Decimal] = None


@dataclass
class _MarketEvent:
    """Minimal market bar event stub."""
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    event_type: str = "market"
    ts: Optional[datetime] = None


@dataclass
class _OtherEvent:
    """A non-fill, non-market event stub."""
    symbol: str = "BTCUSDT"
    event_type: str = "control"
    ts: Optional[datetime] = None


def _fill(
    *,
    symbol: str = "BTCUSDT",
    side: str = "buy",
    qty: Decimal = Decimal("1"),
    price: Decimal = Decimal("50000"),
    fee: Optional[Decimal] = None,
    realized_pnl: Optional[Decimal] = None,
    cash_delta: Optional[Decimal] = None,
    margin_change: Optional[Decimal] = None,
    ts: Optional[datetime] = None,
) -> _FillEvent:
    return _FillEvent(
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        fee=fee,
        realized_pnl=realized_pnl,
        cash_delta=cash_delta,
        margin_change=margin_change,
        ts=ts or _TS,
    )


def _bar(
    *,
    symbol: str = "BTCUSDT",
    open: Decimal = Decimal("100"),
    high: Decimal = Decimal("110"),
    low: Decimal = Decimal("90"),
    close: Decimal = Decimal("105"),
    volume: Decimal = Decimal("1000"),
    ts: Optional[datetime] = None,
) -> _MarketEvent:
    return _MarketEvent(
        symbol=symbol,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
        ts=ts or _TS,
    )


# ─── AccountReducer tests ─────────────────────────────────────────────────────

class TestAccountReducer:
    def setup_method(self) -> None:
        self.reducer = AccountReducer()
        self.state = AccountState.initial(currency="USDT", balance=Decimal("10000"))

    def test_ignores_non_fill_event(self) -> None:
        event = _OtherEvent(event_type="control")
        result = self.reducer.reduce(self.state, event)
        assert result.changed is False
        assert result.state is self.state

    def test_ignores_market_event(self) -> None:
        event = _bar()
        result = self.reducer.reduce(self.state, event)
        assert result.changed is False
        assert result.state is self.state

    def test_updates_balance_on_buy_fill_with_realized_pnl(self) -> None:
        event = _fill(realized_pnl=Decimal("200"))
        result = self.reducer.reduce(self.state, event)
        assert result.changed is True
        # balance increases by realized_pnl
        assert result.state.balance == Decimal("10200")
        assert result.state.realized_pnl == Decimal("200")

    def test_balance_reduced_by_fee(self) -> None:
        event = _fill(fee=Decimal("5"))
        result = self.reducer.reduce(self.state, event)
        assert result.changed is True
        assert result.state.balance == Decimal("9995")
        assert result.state.fees_paid == Decimal("5")

    def test_cash_delta_applied(self) -> None:
        event = _fill(cash_delta=Decimal("-1000"))
        result = self.reducer.reduce(self.state, event)
        assert result.changed is True
        assert result.state.balance == Decimal("9000")

    def test_tracks_fill_events_accumulate_correctly(self) -> None:
        e1 = _fill(realized_pnl=Decimal("100"), fee=Decimal("2"))
        e2 = _fill(realized_pnl=Decimal("50"), fee=Decimal("1"))
        r1 = self.reducer.reduce(self.state, e1)
        r2 = self.reducer.reduce(r1.state, e2)
        assert r2.state.balance == Decimal("10000") + Decimal("100") - Decimal("2") + Decimal("50") - Decimal("1")
        assert r2.state.realized_pnl == Decimal("150")
        assert r2.state.fees_paid == Decimal("3")

    def test_fill_with_no_optional_fields_does_not_crash(self) -> None:
        # No fee, realized_pnl, cash_delta, margin_change — all default to 0
        event = _fill()
        result = self.reducer.reduce(self.state, event)
        assert result.changed is True
        assert result.state.balance == Decimal("10000")

    def test_accepts_trade_fill_event_type(self) -> None:
        event = _fill()
        event.event_type = "trade_fill"
        result = self.reducer.reduce(self.state, event)
        assert result.changed is True

    def test_accepts_execution_fill_event_type(self) -> None:
        event = _fill()
        event.event_type = "execution_fill"
        result = self.reducer.reduce(self.state, event)
        assert result.changed is True


# ─── PositionReducer tests ────────────────────────────────────────────────────

class TestPositionReducer:
    def setup_method(self) -> None:
        self.reducer = PositionReducer()
        self.state = PositionState.empty("BTCUSDT")

    def test_ignore_non_fill_events(self) -> None:
        event = _OtherEvent(event_type="market")
        result = self.reducer.reduce(self.state, event)
        assert result.changed is False
        assert result.state is self.state

    def test_ignore_other_symbols(self) -> None:
        event = _fill(symbol="ETHUSDT")
        result = self.reducer.reduce(self.state, event)
        assert result.changed is False
        assert result.state is self.state

    def test_open_long_position(self) -> None:
        event = _fill(side="buy", qty=Decimal("2"), price=Decimal("50000"))
        result = self.reducer.reduce(self.state, event)
        assert result.changed is True
        assert result.state.qty == Decimal("2")
        assert result.state.avg_price == Decimal("50000")

    def test_add_to_long_position(self) -> None:
        # First fill: 2 BTC @ 50000
        e1 = _fill(side="buy", qty=Decimal("2"), price=Decimal("50000"))
        r1 = self.reducer.reduce(self.state, e1)
        # Second fill: 1 BTC @ 52000
        e2 = _fill(side="buy", qty=Decimal("1"), price=Decimal("52000"))
        r2 = self.reducer.reduce(r1.state, e2)
        assert r2.state.qty == Decimal("3")
        # weighted avg: (2*50000 + 1*52000) / 3 = 152000/3
        expected_avg = (Decimal("2") * Decimal("50000") + Decimal("1") * Decimal("52000")) / Decimal("3")
        assert r2.state.avg_price == expected_avg

    def test_reduce_long_position(self) -> None:
        # Open 3 BTC long
        e_open = _fill(side="buy", qty=Decimal("3"), price=Decimal("50000"))
        r_open = self.reducer.reduce(self.state, e_open)
        # Sell 1 BTC
        e_sell = _fill(side="sell", qty=Decimal("1"), price=Decimal("51000"))
        r_sell = self.reducer.reduce(r_open.state, e_sell)
        assert r_sell.changed is True
        assert r_sell.state.qty == Decimal("2")
        # avg_price must remain unchanged when reducing
        assert r_sell.state.avg_price == Decimal("50000")

    def test_close_long_position(self) -> None:
        # Open 2 BTC long
        e_open = _fill(side="buy", qty=Decimal("2"), price=Decimal("50000"))
        r_open = self.reducer.reduce(self.state, e_open)
        # Close with sell of exact same qty
        e_close = _fill(side="sell", qty=Decimal("2"), price=Decimal("51000"))
        r_close = self.reducer.reduce(r_open.state, e_close)
        assert r_close.changed is True
        assert r_close.state.qty == Decimal("0")
        assert r_close.state.avg_price is None

    def test_open_short_position(self) -> None:
        event = _fill(side="sell", qty=Decimal("1"), price=Decimal("50000"))
        result = self.reducer.reduce(self.state, event)
        assert result.changed is True
        assert result.state.qty == Decimal("-1")
        assert result.state.avg_price == Decimal("50000")

    def test_last_price_updated_on_fill(self) -> None:
        event = _fill(side="buy", qty=Decimal("1"), price=Decimal("48000"))
        result = self.reducer.reduce(self.state, event)
        assert result.state.last_price == Decimal("48000")

    def test_position_flat_note(self) -> None:
        e_open = _fill(side="buy", qty=Decimal("1"), price=Decimal("50000"))
        r_open = self.reducer.reduce(self.state, e_open)
        e_close = _fill(side="sell", qty=Decimal("1"), price=Decimal("50000"))
        r_close = self.reducer.reduce(r_open.state, e_close)
        assert r_close.note == "position_flat"

    def test_position_add_note(self) -> None:
        event = _fill(side="buy", qty=Decimal("1"), price=Decimal("50000"))
        result = self.reducer.reduce(self.state, event)
        assert result.note == "position_add"

    def test_position_reduce_note(self) -> None:
        e_open = _fill(side="buy", qty=Decimal("3"), price=Decimal("50000"))
        r_open = self.reducer.reduce(self.state, e_open)
        e_sell = _fill(side="sell", qty=Decimal("1"), price=Decimal("50000"))
        r_sell = self.reducer.reduce(r_open.state, e_sell)
        assert r_sell.note == "position_reduce"


# ─── MarketReducer tests ──────────────────────────────────────────────────────

class TestMarketReducer:
    def setup_method(self) -> None:
        self.reducer = MarketReducer()
        self.state = MarketState.empty("BTCUSDT")

    def test_ignores_non_market_events(self) -> None:
        event = _fill()
        result = self.reducer.reduce(self.state, event)
        assert result.changed is False
        assert result.state is self.state

    def test_ignores_control_events(self) -> None:
        event = _OtherEvent(event_type="control")
        result = self.reducer.reduce(self.state, event)
        assert result.changed is False

    def test_updates_close_price(self) -> None:
        event = _bar(close=Decimal("55000"))
        result = self.reducer.reduce(self.state, event)
        assert result.changed is True
        assert result.state.close == Decimal("55000")
        assert result.state.last_price == Decimal("55000")

    def test_updates_ohlcv_data(self) -> None:
        event = _bar(
            open=Decimal("48000"),
            high=Decimal("60000"),
            low=Decimal("45000"),
            close=Decimal("55000"),
            volume=Decimal("999"),
        )
        result = self.reducer.reduce(self.state, event)
        assert result.changed is True
        assert result.state.open == Decimal("48000")
        assert result.state.high == Decimal("60000")
        assert result.state.low == Decimal("45000")
        assert result.state.close == Decimal("55000")
        assert result.state.volume == Decimal("999")

    def test_market_bar_note(self) -> None:
        event = _bar()
        result = self.reducer.reduce(self.state, event)
        assert result.note == "market_bar"

    def test_ignores_other_symbol(self) -> None:
        event = _bar(symbol="ETHUSDT")
        result = self.reducer.reduce(self.state, event)
        assert result.changed is False
        assert result.state is self.state

    def test_successive_bars_update_state(self) -> None:
        e1 = _bar(close=Decimal("50000"))
        r1 = self.reducer.reduce(self.state, e1)
        e2 = _bar(close=Decimal("52000"), high=Decimal("53000"), low=Decimal("49000"))
        r2 = self.reducer.reduce(r1.state, e2)
        assert r2.state.close == Decimal("52000")
        assert r2.state.high == Decimal("53000")
        assert r2.state.low == Decimal("49000")

    def test_market_bar_event_type_alias(self) -> None:
        event = _bar()
        event.event_type = "market_bar"
        result = self.reducer.reduce(self.state, event)
        assert result.changed is True

    def test_bar_event_type_alias(self) -> None:
        event = _bar()
        event.event_type = "bar"
        result = self.reducer.reduce(self.state, event)
        assert result.changed is True
