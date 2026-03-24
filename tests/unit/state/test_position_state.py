"""Tests for RustPositionState (via state.PositionState alias)."""
from __future__ import annotations

import pytest

from state import PositionState

_SCALE = 100_000_000


class TestPositionStateEmpty:
    def test_empty_symbol(self):
        p = PositionState.empty("ETHUSDT")
        assert p.symbol == "ETHUSDT"

    def test_empty_qty_zero(self):
        p = PositionState.empty("ETHUSDT")
        assert p.qty == 0

    def test_empty_fields_none(self):
        p = PositionState.empty("ETHUSDT")
        assert p.avg_price is None
        assert p.last_price is None
        assert p.last_ts is None


class TestPositionStateIsFlat:
    def test_is_flat_on_empty(self):
        p = PositionState.empty("ETHUSDT")
        assert p.is_flat is True

    def test_is_flat_with_qty(self):
        p = PositionState(symbol="ETHUSDT", qty=150000000)
        assert p.is_flat is False

    def test_is_flat_negative_qty(self):
        p = PositionState(symbol="ETHUSDT", qty=-150000000)
        assert p.is_flat is False


class TestPositionStateWithUpdate:
    def test_with_update_all_fields(self):
        p = PositionState.empty("ETHUSDT")
        updated = p.with_update(
            qty=250000000,
            avg_price=3500 * _SCALE,
            last_price=3550 * _SCALE,
            ts="2024-01-01T00:00:00+00:00",
        )
        assert updated.qty == 250000000
        assert updated.avg_price == 3500 * _SCALE
        assert updated.last_price == 3550 * _SCALE
        assert updated.last_ts == "2024-01-01T00:00:00+00:00"

    def test_with_update_preserves_symbol(self):
        p = PositionState.empty("BTCUSDT")
        updated = p.with_update(
            qty=10000000, avg_price=50000 * _SCALE,
            last_price=None, ts=None,
        )
        assert updated.symbol == "BTCUSDT"

    def test_with_update_negative_qty_short(self):
        p = PositionState.empty("ETHUSDT")
        updated = p.with_update(
            qty=-5 * _SCALE, avg_price=3000 * _SCALE,
            last_price=3100 * _SCALE, ts=None,
        )
        assert updated.qty == -5 * _SCALE
        assert updated.is_flat is False

    def test_frozen_immutability(self):
        p = PositionState.empty("ETHUSDT")
        with pytest.raises(AttributeError):
            p.qty = 1 * _SCALE  # type: ignore[misc]

    def test_none_avg_price(self):
        p = PositionState(symbol="ETHUSDT", qty=0, avg_price=None)
        assert p.avg_price is None

    def test_float_accessors(self):
        p = PositionState(symbol="ETHUSDT", qty=250000000, avg_price=350000000000)
        assert p.qty_f == pytest.approx(2.5)
        assert p.avg_price_f == pytest.approx(3500.0)

    def test_equality(self):
        p1 = PositionState("ETHUSDT", qty=100 * _SCALE)
        p2 = PositionState("ETHUSDT", qty=100 * _SCALE)
        assert p1 == p2
