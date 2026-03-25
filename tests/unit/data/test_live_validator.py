"""Tests for data.quality.live_validator — LiveBarValidator for live market events."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from data.quality.live_validator import LiveBarValidator


@dataclass
class FakeMarket:
    open: Any = 100.0
    high: Any = 105.0
    low: Any = 95.0
    close: Any = 102.0


@dataclass
class FakeEvent:
    market: Any = None


class TestLiveBarValidatorValid:
    """Valid events should pass validation."""

    def test_valid_event(self) -> None:
        ev = FakeEvent(market=FakeMarket())
        assert LiveBarValidator().validate(ev) is True

    def test_event_without_market_passes(self) -> None:
        ev = FakeEvent(market=None)
        assert LiveBarValidator().validate(ev) is True

    def test_decimal_prices_pass(self) -> None:
        market = FakeMarket(
            open=Decimal("100.0"),
            high=Decimal("105.0"),
            low=Decimal("95.0"),
            close=Decimal("102.0"),
        )
        ev = FakeEvent(market=market)
        assert LiveBarValidator().validate(ev) is True

    def test_integer_prices_pass(self) -> None:
        market = FakeMarket(open=100, high=105, low=95, close=102)
        ev = FakeEvent(market=market)
        assert LiveBarValidator().validate(ev) is True


class TestLiveBarValidatorRejectsInvalid:
    """Invalid OHLC relationships should be rejected."""

    def test_negative_close_rejected(self) -> None:
        market = FakeMarket(close=-1.0)
        ev = FakeEvent(market=market)
        assert LiveBarValidator().validate(ev) is False

    def test_zero_close_rejected(self) -> None:
        market = FakeMarket(close=0.0)
        ev = FakeEvent(market=market)
        assert LiveBarValidator().validate(ev) is False

    def test_high_below_low_rejected(self) -> None:
        market = FakeMarket(high=90.0, low=95.0)
        ev = FakeEvent(market=market)
        assert LiveBarValidator().validate(ev) is False

    def test_high_below_open_rejected(self) -> None:
        market = FakeMarket(open=110.0, high=105.0, low=95.0, close=102.0)
        ev = FakeEvent(market=market)
        assert LiveBarValidator().validate(ev) is False

    def test_high_below_close_rejected(self) -> None:
        market = FakeMarket(open=100.0, high=101.0, low=95.0, close=102.0)
        ev = FakeEvent(market=market)
        assert LiveBarValidator().validate(ev) is False

    def test_low_above_open_rejected(self) -> None:
        market = FakeMarket(open=94.0, high=105.0, low=95.0, close=102.0)
        ev = FakeEvent(market=market)
        assert LiveBarValidator().validate(ev) is False

    def test_low_above_close_rejected(self) -> None:
        market = FakeMarket(open=100.0, high=105.0, low=103.0, close=102.0)
        ev = FakeEvent(market=market)
        assert LiveBarValidator().validate(ev) is False


class TestLiveBarValidatorConfig:
    """Configuration options."""

    def test_allow_zero_close_when_disabled(self) -> None:
        """With require_positive_close=False, zero close is not rejected
        (as long as OHLC relationships are consistent)."""
        market = FakeMarket(open=0.0, high=0.0, low=0.0, close=0.0)
        ev = FakeEvent(market=market)
        validator = LiveBarValidator(require_positive_close=False)
        assert validator.validate(ev) is True

    def test_negative_close_allowed_when_disabled(self) -> None:
        """With require_positive_close=False, negative close is not rejected
        (as long as OHLC relationships are consistent)."""
        market = FakeMarket(open=-5.0, high=-3.0, low=-7.0, close=-5.0)
        ev = FakeEvent(market=market)
        validator = LiveBarValidator(require_positive_close=False)
        assert validator.validate(ev) is True


class TestLiveBarValidatorPartialData:
    """Events with partial/missing price fields should still be handled."""

    def test_none_fields_pass(self) -> None:
        market = FakeMarket(open=None, high=None, low=None, close=None)
        ev = FakeEvent(market=market)
        assert LiveBarValidator().validate(ev) is True

    def test_only_close_provided(self) -> None:
        market = FakeMarket(open=None, high=None, low=None, close=50.0)
        ev = FakeEvent(market=market)
        assert LiveBarValidator().validate(ev) is True

    def test_string_values_treated_as_none(self) -> None:
        market = FakeMarket(open="bad", high="bad", low="bad", close="bad")
        ev = FakeEvent(market=market)
        # _as_float returns None for strings, so all checks skip
        assert LiveBarValidator().validate(ev) is True
