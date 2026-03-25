"""Tests for state reducer purity — reducers must not mutate inputs."""
from __future__ import annotations

import pytest

from _quant_hotpath import (  # type: ignore[import-untyped]
    RustMarketReducer,
    RustMarketState,
    RustPositionReducer,
    RustPositionState,
    RustAccountReducer,
    RustAccountState,
)

_SCALE = 100_000_000


# ---------------------------------------------------------------------------
# MarketReducer purity
# ---------------------------------------------------------------------------

class TestMarketReducerPurity:
    def _make_market(self, close=5000):
        return RustMarketState(
            "BTCUSDT",
            close * _SCALE,  # last_price
            close * _SCALE,  # open (actually close in positional)
            (close + 100) * _SCALE,  # high
            (close - 100) * _SCALE,  # low
            1000 * _SCALE,  # volume
            1000,  # last_ts
        )

    def test_reduce_returns_new_object(self):
        mr = RustMarketReducer()
        ms = self._make_market()
        facts = ms.to_dict()
        result = mr.reduce(ms, facts)
        assert result.state is not ms

    def test_reduce_does_not_mutate_input(self):
        mr = RustMarketReducer()
        ms = self._make_market(close=5000)
        original_close = ms.close
        original_ts = ms.last_ts
        facts = ms.to_dict()
        mr.reduce(ms, facts)
        assert ms.close == original_close
        assert ms.last_ts == original_ts

    def test_same_input_produces_consistent_output(self):
        mr = RustMarketReducer()
        ms = self._make_market()
        facts = ms.to_dict()
        r1 = mr.reduce(ms, facts)
        r2 = mr.reduce(ms, facts)
        assert r1.state.close == r2.state.close
        assert r1.state.symbol == r2.state.symbol

    def test_result_has_changed_flag(self):
        mr = RustMarketReducer()
        ms = self._make_market()
        facts = ms.to_dict()
        result = mr.reduce(ms, facts)
        assert isinstance(result.changed, bool)

    def test_result_state_has_correct_symbol(self):
        mr = RustMarketReducer()
        ms = self._make_market()
        result = mr.reduce(ms, ms.to_dict())
        assert result.state.symbol == "BTCUSDT"


# ---------------------------------------------------------------------------
# PositionReducer purity
# ---------------------------------------------------------------------------

class TestPositionReducerPurity:
    def _make_position(self, symbol="BTCUSDT"):
        return RustPositionState(symbol)

    def test_reduce_returns_new_object(self):
        pr = RustPositionReducer()
        ps = self._make_position()
        facts = {"symbol": "BTCUSDT", "qty": 100 * _SCALE, "price": 5000 * _SCALE}
        result = pr.reduce(ps, facts)
        assert result.state is not ps

    def test_reduce_does_not_mutate_input(self):
        pr = RustPositionReducer()
        ps = self._make_position()
        original_qty = ps.qty
        facts = {"symbol": "BTCUSDT", "qty": 100 * _SCALE, "price": 5000 * _SCALE}
        pr.reduce(ps, facts)
        assert ps.qty == original_qty

    def test_same_input_same_output(self):
        pr = RustPositionReducer()
        ps = self._make_position()
        facts = {"symbol": "BTCUSDT", "qty": 100 * _SCALE, "price": 5000 * _SCALE}
        r1 = pr.reduce(ps, facts)
        r2 = pr.reduce(ps, facts)
        assert r1.state.qty == r2.state.qty
        assert r1.state.symbol == r2.state.symbol

    def test_flat_position_is_flat(self):
        ps = self._make_position()
        assert ps.is_flat


# ---------------------------------------------------------------------------
# AccountReducer purity
# ---------------------------------------------------------------------------

class TestAccountReducerPurity:
    def _make_account(self, balance=100_000):
        return RustAccountState(
            "USDT",
            balance * _SCALE,
            0,  # margin_used
            balance * _SCALE,  # margin_available
        )

    def test_reduce_returns_new_object(self):
        ar = RustAccountReducer()
        ac = self._make_account()
        facts = {"symbol": "BTCUSDT", "qty": 100 * _SCALE, "price": 5000 * _SCALE}
        result = ar.reduce(ac, facts)
        assert result.state is not ac

    def test_reduce_does_not_mutate_input(self):
        ar = RustAccountReducer()
        ac = self._make_account(balance=100_000)
        original_balance = ac.balance
        facts = {"symbol": "BTCUSDT", "qty": 100 * _SCALE, "price": 5000 * _SCALE}
        ar.reduce(ac, facts)
        assert ac.balance == original_balance

    def test_same_input_same_output(self):
        ar = RustAccountReducer()
        ac = self._make_account()
        facts = {"symbol": "BTCUSDT", "qty": 100 * _SCALE, "price": 5000 * _SCALE}
        r1 = ar.reduce(ac, facts)
        r2 = ar.reduce(ac, facts)
        assert r1.state.balance == r2.state.balance

    def test_account_has_balance(self):
        ac = self._make_account(balance=50_000)
        assert ac.balance == 50_000 * _SCALE

    def test_account_to_dict_round_trip(self):
        ac = self._make_account(balance=75_000)
        d = ac.to_dict()
        ac2 = RustAccountState.from_dict(d)
        assert ac2.balance == ac.balance
        assert ac2.currency == ac.currency
