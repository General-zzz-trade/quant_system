"""Tests for EmbargoExecutionAdapter._stamp_price with various order types."""
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace

from execution.sim.embargo import EmbargoExecutionAdapter


def test_stamp_price_simple_namespace():
    order = SimpleNamespace(symbol="BTCUSDT", side="BUY", qty=1, price=Decimal("100"))
    result = EmbargoExecutionAdapter._stamp_price(order, Decimal("105"))
    assert result.price == Decimal("105")
    assert result.symbol == "BTCUSDT"


def test_stamp_price_frozen_slots_dataclass():
    @dataclass(frozen=True, slots=True)
    class FrozenOrder:
        symbol: str
        side: str
        qty: int
        price: Decimal

    order = FrozenOrder(symbol="BTCUSDT", side="BUY", qty=1, price=Decimal("100"))
    result = EmbargoExecutionAdapter._stamp_price(order, Decimal("105"))
    assert result.price == Decimal("105")
    assert result.symbol == "BTCUSDT"
    assert result.side == "BUY"


def test_stamp_price_regular_dataclass():
    @dataclass
    class RegularOrder:
        symbol: str
        side: str
        qty: int
        price: Decimal

    order = RegularOrder(symbol="BTCUSDT", side="SELL", qty=2, price=Decimal("200"))
    result = EmbargoExecutionAdapter._stamp_price(order, Decimal("210"))
    assert result.price == Decimal("210")
    assert result.symbol == "BTCUSDT"
    assert result.qty == 2
