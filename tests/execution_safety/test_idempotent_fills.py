"""Tests for idempotent fill processing."""
from __future__ import annotations

from execution.adapters.common.idempotency import make_fill_idem_key


def test_fill_idem_key_stable():
    k1 = make_fill_idem_key(fill_id="f1", trade_id="t1", order_id="o1", symbol="BTCUSDT", venue="binance")
    k2 = make_fill_idem_key(fill_id="f1", trade_id="t1", order_id="o1", symbol="BTCUSDT", venue="binance")
    assert k1 == k2


def test_fill_idem_key_different():
    k1 = make_fill_idem_key(fill_id="f1", trade_id="t1", order_id="o1", symbol="BTCUSDT", venue="binance")
    k2 = make_fill_idem_key(fill_id="f2", trade_id="t2", order_id="o1", symbol="BTCUSDT", venue="binance")
    assert k1 != k2
