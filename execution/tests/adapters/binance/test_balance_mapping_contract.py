"""Tests for Binance balance mapping."""
from __future__ import annotations

from decimal import Decimal

from execution.adapters.binance.mapper_balance import map_balance


def test_map_balance_basic():
    raw = {"asset": "USDT", "balance": "1000.0", "availableBalance": "800.0"}
    b = map_balance(raw)
    assert b.asset == "USDT"
    assert b.free == Decimal("800.0")


def test_map_balance_zero():
    raw = {"asset": "BTC", "balance": "0", "availableBalance": "0"}
    b = map_balance(raw)
    assert b.free == Decimal("0")
