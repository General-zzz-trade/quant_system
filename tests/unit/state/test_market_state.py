"""Tests for state.market.MarketState."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from state.market import MarketState


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
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        updated = m.with_tick(price=Decimal("3500.50"), ts=ts)
        assert updated.last_price == Decimal("3500.50")
        assert updated.last_ts == ts

    def test_with_tick_preserves_ohlcv(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        m = MarketState(
            symbol="ETHUSDT",
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("90"),
            close=Decimal("105"),
            volume=Decimal("1000"),
        )
        updated = m.with_tick(price=Decimal("106"), ts=ts)
        assert updated.open == Decimal("100")
        assert updated.high == Decimal("110")
        assert updated.low == Decimal("90")
        assert updated.close == Decimal("105")
        assert updated.volume == Decimal("1000")

    def test_with_tick_ts_none_keeps_previous(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        m = MarketState(symbol="ETHUSDT", last_ts=ts)
        updated = m.with_tick(price=Decimal("100"), ts=None)
        assert updated.last_ts == ts


class TestMarketStateWithBar:
    def test_with_bar_sets_ohlcv(self):
        m = MarketState.empty("ETHUSDT")
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        updated = m.with_bar(
            o=Decimal("100"), h=Decimal("110"),
            l=Decimal("90"), c=Decimal("105"),
            v=Decimal("5000"), ts=ts,
        )
        assert updated.open == Decimal("100")
        assert updated.high == Decimal("110")
        assert updated.low == Decimal("90")
        assert updated.close == Decimal("105")
        assert updated.volume == Decimal("5000")

    def test_with_bar_last_price_equals_close(self):
        m = MarketState.empty("ETHUSDT")
        updated = m.with_bar(
            o=Decimal("100"), h=Decimal("110"),
            l=Decimal("90"), c=Decimal("105"),
            v=Decimal("5000"), ts=None,
        )
        assert updated.last_price == Decimal("105")

    def test_with_bar_volume_none(self):
        m = MarketState.empty("ETHUSDT")
        updated = m.with_bar(
            o=Decimal("100"), h=Decimal("110"),
            l=Decimal("90"), c=Decimal("105"),
            v=None, ts=None,
        )
        assert updated.volume is None

    def test_frozen_immutability(self):
        m = MarketState.empty("ETHUSDT")
        with pytest.raises(AttributeError):
            m.last_price = Decimal("100")  # type: ignore[misc]

    def test_with_bar_naive_ts_becomes_utc(self):
        m = MarketState.empty("ETHUSDT")
        naive_ts = datetime(2024, 6, 15, 12, 0, 0)
        updated = m.with_bar(
            o=Decimal("100"), h=Decimal("110"),
            l=Decimal("90"), c=Decimal("105"),
            v=Decimal("1000"), ts=naive_ts,
        )
        assert updated.last_ts is not None
        assert updated.last_ts.tzinfo is not None
