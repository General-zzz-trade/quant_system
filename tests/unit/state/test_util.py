"""Tests for state._util helper functions."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pytest

from state._util import ensure_utc, signed_qty, to_decimal


class TestToDecimal:
    def test_from_float(self):
        result = to_decimal(3.14)
        assert result == Decimal("3.14")

    def test_from_str(self):
        result = to_decimal("99.99")
        assert result == Decimal("99.99")

    def test_from_int(self):
        result = to_decimal(42)
        assert result == Decimal("42")

    def test_from_decimal_passthrough(self):
        d = Decimal("1.5")
        result = to_decimal(d)
        assert result is d

    def test_bool_rejected(self):
        with pytest.raises(TypeError, match="bool"):
            to_decimal(True)

    def test_none_default_zero(self):
        result = to_decimal(None)
        assert result == Decimal("0")

    def test_none_allow_none(self):
        result = to_decimal(None, allow_none=True)
        assert result is None

    def test_invalid_string(self):
        with pytest.raises(TypeError):
            to_decimal("not_a_number")


class TestEnsureUtc:
    def test_naive_becomes_utc(self):
        naive = datetime(2024, 1, 1, 12, 0, 0)
        result = ensure_utc(naive)
        assert result is not None
        assert result.tzinfo == timezone.utc

    def test_utc_passthrough(self):
        utc_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = ensure_utc(utc_dt)
        assert result == utc_dt

    def test_tz_aware_conversion(self):
        eastern = timezone(timedelta(hours=-5))
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=eastern)
        result = ensure_utc(dt)
        assert result is not None
        assert result.tzinfo == timezone.utc
        assert result.hour == 17  # 12 + 5

    def test_none_returns_none(self):
        result = ensure_utc(None)
        assert result is None

    def test_non_datetime_raises(self):
        with pytest.raises(TypeError, match="datetime"):
            ensure_utc("2024-01-01")  # type: ignore[arg-type]


class TestSignedQty:
    def test_buy_positive(self):
        result = signed_qty(Decimal("5"), "buy")
        assert result == Decimal("5")

    def test_sell_negative(self):
        result = signed_qty(Decimal("5"), "sell")
        assert result == Decimal("-5")

    def test_long_positive(self):
        result = signed_qty(Decimal("3"), "long")
        assert result == Decimal("3")

    def test_short_negative(self):
        result = signed_qty(Decimal("3"), "short")
        assert result == Decimal("-3")

    def test_case_insensitive(self):
        assert signed_qty(Decimal("1"), "BUY") == Decimal("1")
        assert signed_qty(Decimal("1"), "SELL") == Decimal("-1")

    def test_no_side_keeps_sign(self):
        result = signed_qty(Decimal("-2"), None)
        assert result == Decimal("-2")

    def test_unknown_side_keeps_value(self):
        result = signed_qty(Decimal("7"), "unknown")
        assert result == Decimal("7")
