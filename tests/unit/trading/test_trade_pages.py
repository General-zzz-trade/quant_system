"""An order's whole trade history is read, page by page (migration step 14b; known defect R9).

Binance returns at most 1000 trades per request. The adapter reads an order's trades in pages from the
last trade id seen, until a page is not full; a history longer than MAX_TRADE_PAGES pages raises, so the
order stays unresolved instead of settling on part of its trades.
"""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from tests.characterization.test_stops_and_fills import OWNER, SlicedVenue, _drain, _executor, _order
from trading.adapters.binance import adapter as binance


def test_an_order_with_three_pages_of_trades_settles_with_every_trade(tmp_path):
    fake, adapter, executor, clock, consumer = _executor(tmp_path, SlicedVenue)
    executor.send_order(_order(1, "buy", "2.5"))                   # 2,500 trades of 0.001
    _drain(executor, consumer)
    assert consumer.position.qty == D("2.5") and fake.positions["BTCUSDT"] == D("2.5")
    assert executor.journal.orders(OWNER, unresolved=True) == []
    requests = [p for p in fake.trade_requests if p.get("orderId")]
    assert [p.get("fromId") for p in requests][-3:] == [None, requests[-2]["fromId"], requests[-1]["fromId"]]
    assert all(int(p["limit"]) == 1000 for p in requests)
    (row,) = executor.journal.orders(OWNER)
    fills = adapter.get_recent_fills(symbol="BTCUSDT", order_id=row["order_id"], limit=100)
    assert len(fills) == 2500 and len({f.trade_id for f in fills}) == 2500          # every trade, once


def test_a_history_longer_than_the_page_cap_keeps_the_order_unresolved(tmp_path, monkeypatch):
    monkeypatch.setattr(binance, "MAX_TRADE_PAGES", 2)
    fake, adapter, executor, clock, consumer = _executor(tmp_path, SlicedVenue)
    executor.send_order(_order(1, "buy", "2.5"))                   # needs 3 pages
    _drain(executor, consumer)
    (row,) = executor.journal.orders(OWNER, unresolved=True)
    assert row["status"] == "unknown" and consumer.position.qty < D("2.5")
    with pytest.raises(RuntimeError, match="more than 2 pages"):
        adapter.get_recent_fills(symbol="BTCUSDT", order_id=row["order_id"], limit=100)


def test_a_query_without_an_order_is_a_single_request_as_before(tmp_path):
    fake, adapter, executor, clock, consumer = _executor(tmp_path, SlicedVenue)
    executor.send_order(_order(1, "buy", "0.15"))
    before = len(fake.trade_requests)
    assert len(adapter.get_recent_fills(symbol="BTCUSDT", limit=100)) == 100
    assert len(fake.trade_requests) == before + 1
