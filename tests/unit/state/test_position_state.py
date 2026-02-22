# tests/unit/state/test_position_state.py
"""PositionReducer unit tests — covers open/add/reduce/flat/reverse scenarios."""
from __future__ import annotations

import pytest
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from state.position import PositionState
from state.reducers.position import PositionReducer
from state.errors import ReducerError


# ---------------------------------------------------------------------------
# Helpers: minimal event stubs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Header:
    event_type: str = "fill"
    ts: Optional[datetime] = None
    event_id: Optional[str] = None

@dataclass(frozen=True)
class FillEvent:
    header: _Header
    symbol: str
    side: str
    qty: str
    price: str
    fee: str = "0"
    realized_pnl: str = "0"

def _fill(symbol: str, side: str, qty: str, price: str, ts: Optional[datetime] = None) -> FillEvent:
    return FillEvent(
        header=_Header(event_type="fill", ts=ts),
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
    )

@dataclass(frozen=True)
class NonFillEvent:
    header: _Header = _Header(event_type="market")
    symbol: str = "BTCUSDT"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reducer() -> PositionReducer:
    return PositionReducer()

@pytest.fixture
def empty_pos() -> PositionState:
    return PositionState.empty("BTCUSDT")


# ---------------------------------------------------------------------------
# Tests: basic open position
# ---------------------------------------------------------------------------

class TestOpenPosition:
    def test_buy_opens_long(self, reducer: PositionReducer, empty_pos: PositionState) -> None:
        res = reducer.reduce(empty_pos, _fill("BTCUSDT", "buy", "1.5", "50000"))
        assert res.changed is True
        assert res.state.qty == Decimal("1.5")
        assert res.state.avg_price == Decimal("50000")
        assert res.state.is_flat is False
        assert res.note == "position_add"

    def test_sell_opens_short(self, reducer: PositionReducer, empty_pos: PositionState) -> None:
        res = reducer.reduce(empty_pos, _fill("BTCUSDT", "sell", "2.0", "48000"))
        assert res.changed is True
        assert res.state.qty == Decimal("-2.0")
        assert res.state.avg_price == Decimal("48000")

    def test_last_price_updated(self, reducer: PositionReducer, empty_pos: PositionState) -> None:
        res = reducer.reduce(empty_pos, _fill("BTCUSDT", "buy", "1", "50000"))
        assert res.state.last_price == Decimal("50000")


# ---------------------------------------------------------------------------
# Tests: add to position (same direction)
# ---------------------------------------------------------------------------

class TestAddPosition:
    def test_add_long_weighted_avg(self, reducer: PositionReducer, empty_pos: PositionState) -> None:
        r1 = reducer.reduce(empty_pos, _fill("BTCUSDT", "buy", "1", "50000"))
        r2 = reducer.reduce(r1.state, _fill("BTCUSDT", "buy", "1", "52000"))
        assert r2.state.qty == Decimal("2")
        assert r2.state.avg_price == Decimal("51000")

    def test_add_short_weighted_avg(self, reducer: PositionReducer, empty_pos: PositionState) -> None:
        r1 = reducer.reduce(empty_pos, _fill("BTCUSDT", "sell", "2", "50000"))
        r2 = reducer.reduce(r1.state, _fill("BTCUSDT", "sell", "1", "49000"))
        assert r2.state.qty == Decimal("-3")
        expected_avg = (Decimal("2") * Decimal("50000") + Decimal("1") * Decimal("49000")) / Decimal("3")
        assert r2.state.avg_price == expected_avg


# ---------------------------------------------------------------------------
# Tests: reduce position (partial close)
# ---------------------------------------------------------------------------

class TestReducePosition:
    def test_partial_close_long(self, reducer: PositionReducer, empty_pos: PositionState) -> None:
        r1 = reducer.reduce(empty_pos, _fill("BTCUSDT", "buy", "3", "50000"))
        r2 = reducer.reduce(r1.state, _fill("BTCUSDT", "sell", "1", "51000"))
        assert r2.state.qty == Decimal("2")
        assert r2.state.avg_price == Decimal("50000")  # unchanged on partial close
        assert r2.note == "position_reduce"

    def test_partial_close_short(self, reducer: PositionReducer, empty_pos: PositionState) -> None:
        r1 = reducer.reduce(empty_pos, _fill("BTCUSDT", "sell", "4", "50000"))
        r2 = reducer.reduce(r1.state, _fill("BTCUSDT", "buy", "2", "49000"))
        assert r2.state.qty == Decimal("-2")
        assert r2.state.avg_price == Decimal("50000")


# ---------------------------------------------------------------------------
# Tests: close position (flat)
# ---------------------------------------------------------------------------

class TestFlatPosition:
    def test_close_long_to_flat(self, reducer: PositionReducer, empty_pos: PositionState) -> None:
        r1 = reducer.reduce(empty_pos, _fill("BTCUSDT", "buy", "2", "50000"))
        r2 = reducer.reduce(r1.state, _fill("BTCUSDT", "sell", "2", "51000"))
        assert r2.state.qty == Decimal("0")
        assert r2.state.is_flat is True
        assert r2.state.avg_price is None
        assert r2.note == "position_flat"

    def test_close_short_to_flat(self, reducer: PositionReducer, empty_pos: PositionState) -> None:
        r1 = reducer.reduce(empty_pos, _fill("BTCUSDT", "sell", "1", "50000"))
        r2 = reducer.reduce(r1.state, _fill("BTCUSDT", "buy", "1", "49000"))
        assert r2.state.is_flat is True


# ---------------------------------------------------------------------------
# Tests: reverse position (cross zero)
# ---------------------------------------------------------------------------

class TestReversePosition:
    def test_long_to_short(self, reducer: PositionReducer, empty_pos: PositionState) -> None:
        r1 = reducer.reduce(empty_pos, _fill("BTCUSDT", "buy", "1", "50000"))
        r2 = reducer.reduce(r1.state, _fill("BTCUSDT", "sell", "3", "52000"))
        assert r2.state.qty == Decimal("-2")
        assert r2.state.avg_price == Decimal("52000")
        assert r2.note == "position_reverse"


# ---------------------------------------------------------------------------
# Tests: ignore / error cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_non_fill_event_ignored(self, reducer: PositionReducer, empty_pos: PositionState) -> None:
        res = reducer.reduce(empty_pos, NonFillEvent())
        assert res.changed is False
        assert res.state is empty_pos

    def test_different_symbol_ignored(self, reducer: PositionReducer, empty_pos: PositionState) -> None:
        res = reducer.reduce(empty_pos, _fill("ETHUSDT", "buy", "1", "3000"))
        assert res.changed is False

    def test_missing_qty_raises(self, reducer: PositionReducer, empty_pos: PositionState) -> None:
        @dataclass(frozen=True)
        class BadFill:
            header: _Header = _Header(event_type="fill")
            symbol: str = "BTCUSDT"
            side: str = "buy"
            price: str = "50000"
        with pytest.raises(ReducerError, match="missing qty"):
            reducer.reduce(empty_pos, BadFill())

    def test_missing_side_raises(self, reducer: PositionReducer, empty_pos: PositionState) -> None:
        @dataclass(frozen=True)
        class BadFill:
            header: _Header = _Header(event_type="fill")
            symbol: str = "BTCUSDT"
            qty: str = "1"
            price: str = "50000"
        with pytest.raises(ReducerError, match="missing side"):
            reducer.reduce(empty_pos, BadFill())

    def test_missing_price_raises(self, reducer: PositionReducer, empty_pos: PositionState) -> None:
        @dataclass(frozen=True)
        class BadFill:
            header: _Header = _Header(event_type="fill")
            symbol: str = "BTCUSDT"
            side: str = "buy"
            qty: str = "1"
        with pytest.raises(ReducerError, match="missing price"):
            reducer.reduce(empty_pos, BadFill())

    def test_immutability(self, reducer: PositionReducer, empty_pos: PositionState) -> None:
        """Original state must not be mutated (frozen dataclass)."""
        reducer.reduce(empty_pos, _fill("BTCUSDT", "buy", "1", "50000"))
        assert empty_pos.qty == Decimal("0")
        assert empty_pos.avg_price is None
