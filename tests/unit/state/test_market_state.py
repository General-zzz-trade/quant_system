"""Tests for RustMarketState (via state.MarketState alias)."""
from __future__ import annotations

import pytest

from state import MarketState

_SCALE = 100_000_000


class TestMarketStateEmpty:
    def test_empty_symbol(self):
        m = MarketState.empty("ETHUSDT")
        assert m.symbol == "ETHUSDT"

    def test_empty_fields_none(self):
        m = MarketState.empty("ETHUSDT")
        assert m.last_price is None
        assert m.open is None
        assert m.high is None
        assert m.low is None
        assert m.close is None
        assert m.volume is None
        assert m.last_ts is None


class TestMarketStateWithTick:
    def test_with_tick_sets_last_price(self):
        m = MarketState.empty("ETHUSDT")
        updated = m.with_tick(price=int(3500_50000000), ts="2024-01-01T00:00:00+00:00")
        assert updated.last_price == 350050000000
        assert updated.last_ts == "2024-01-01T00:00:00+00:00"

    def test_with_tick_preserves_ohlcv(self):
        m = MarketState(
            symbol="ETHUSDT",
            open=100 * _SCALE,
            high=110 * _SCALE,
            low=90 * _SCALE,
            close=105 * _SCALE,
            volume=1000 * _SCALE,
        )
        updated = m.with_tick(price=106 * _SCALE, ts="2024-01-01T00:00:00+00:00")
        assert updated.open == 100 * _SCALE
        assert updated.high == 110 * _SCALE
        assert updated.low == 90 * _SCALE
        assert updated.close == 105 * _SCALE
        assert updated.volume == 1000 * _SCALE

    def test_with_tick_ts_none_keeps_none(self):
        m = MarketState.empty("ETHUSDT")
        updated = m.with_tick(price=100 * _SCALE)
        assert updated.last_ts is None


class TestMarketStateWithBar:
    def test_with_bar_sets_ohlcv(self):
        m = MarketState.empty("ETHUSDT")
        updated = m.with_bar(
            o=100 * _SCALE, h=110 * _SCALE,
            l=90 * _SCALE, c=105 * _SCALE,
            v=5000 * _SCALE, ts="2024-01-01T00:00:00+00:00",
        )
        assert updated.open == 100 * _SCALE
        assert updated.high == 110 * _SCALE
        assert updated.low == 90 * _SCALE
        assert updated.close == 105 * _SCALE
        assert updated.volume == 5000 * _SCALE

    def test_with_bar_last_price_equals_close(self):
        m = MarketState.empty("ETHUSDT")
        updated = m.with_bar(
            o=100 * _SCALE, h=110 * _SCALE,
            l=90 * _SCALE, c=105 * _SCALE,
            v=5000 * _SCALE,
        )
        assert updated.last_price == 105 * _SCALE

    def test_frozen_immutability(self):
        m = MarketState.empty("ETHUSDT")
        with pytest.raises(AttributeError):
            m.last_price = 100 * _SCALE  # type: ignore[misc]

    def test_float_accessors(self):
        m = MarketState(
            symbol="ETHUSDT",
            last_price=350050000000,
            open=100 * _SCALE,
            close=105 * _SCALE,
        )
        assert m.last_price_f == pytest.approx(3500.5)
        assert m.open_f == pytest.approx(100.0)
        assert m.close_f == pytest.approx(105.0)

    def test_equality(self):
        m1 = MarketState("ETHUSDT", last_price=100 * _SCALE)
        m2 = MarketState("ETHUSDT", last_price=100 * _SCALE)
        assert m1 == m2
