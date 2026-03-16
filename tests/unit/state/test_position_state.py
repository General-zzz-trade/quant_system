"""Tests for state.position.PositionState."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from state.position import PositionState


class TestPositionStateEmpty:
    def test_empty_symbol(self):
        p = PositionState.empty("ETHUSDT")
        assert p.symbol == "ETHUSDT"

    def test_empty_qty_zero(self):
        p = PositionState.empty("ETHUSDT")
        assert p.qty == Decimal("0")

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
        p = PositionState(symbol="ETHUSDT", qty=Decimal("1.5"))
        assert p.is_flat is False

    def test_is_flat_negative_qty(self):
        p = PositionState(symbol="ETHUSDT", qty=Decimal("-1.5"))
        assert p.is_flat is False

    def test_is_flat_decimal_zero(self):
        p = PositionState(symbol="ETHUSDT", qty=Decimal("0.00"))
        assert p.is_flat is True


class TestPositionStateWithUpdate:
    def test_with_update_all_fields(self):
        p = PositionState.empty("ETHUSDT")
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        updated = p.with_update(
            qty=Decimal("2.5"),
            avg_price=Decimal("3500"),
            last_price=Decimal("3550"),
            ts=ts,
        )
        assert updated.qty == Decimal("2.5")
        assert updated.avg_price == Decimal("3500")
        assert updated.last_price == Decimal("3550")
        assert updated.last_ts == ts

    def test_with_update_preserves_symbol(self):
        p = PositionState.empty("BTCUSDT")
        updated = p.with_update(
            qty=Decimal("0.1"), avg_price=Decimal("50000"),
            last_price=None, ts=None,
        )
        assert updated.symbol == "BTCUSDT"

    def test_with_update_negative_qty_short(self):
        p = PositionState.empty("ETHUSDT")
        updated = p.with_update(
            qty=Decimal("-5"), avg_price=Decimal("3000"),
            last_price=Decimal("3100"), ts=None,
        )
        assert updated.qty == Decimal("-5")
        assert updated.is_flat is False

    def test_frozen_immutability(self):
        p = PositionState.empty("ETHUSDT")
        with pytest.raises(AttributeError):
            p.qty = Decimal("1")  # type: ignore[misc]

    def test_with_update_ts_none_keeps_previous(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        p = PositionState(symbol="ETHUSDT", qty=Decimal("1"), last_ts=ts)
        updated = p.with_update(
            qty=Decimal("2"), avg_price=None, last_price=None, ts=None,
        )
        assert updated.last_ts == ts

    def test_none_avg_price(self):
        p = PositionState(symbol="ETHUSDT", qty=Decimal("0"), avg_price=None)
        assert p.avg_price is None
