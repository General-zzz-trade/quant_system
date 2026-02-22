"""Tests for execution mapping (Binance normalize)."""
from __future__ import annotations

from execution.adapters.binance.normalize import (
    normalize_side,
    normalize_order_status,
    normalize_symbol,
)


def test_normalize_side():
    assert normalize_side("BUY") == "buy"
    assert normalize_side("SELL") == "sell"


def test_normalize_order_status():
    assert normalize_order_status("NEW") == "new"
    assert normalize_order_status("FILLED") == "filled"
    assert normalize_order_status("CANCELED") == "cancelled"


def test_normalize_symbol():
    assert normalize_symbol("btcusdt ") == "BTCUSDT"
