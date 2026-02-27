"""Tests for infra.audit.trade_log."""
from __future__ import annotations

from pathlib import Path

import pytest

from infra.audit.trade_log import TradeAuditLog


class TestTradeAuditLog:

    def test_append_and_query(self, tmp_path: Path) -> None:
        log = TradeAuditLog(tmp_path / "trades.jsonl")

        record = log.append(
            event_type="fill",
            symbol="BTCUSDT",
            side="buy",
            qty="0.01",
            price="50000.00",
            fee="0.20",
        )

        assert record.seq == 1
        assert record.symbol == "BTCUSDT"

        results = log.query()
        assert len(results) == 1
        assert results[0].price == "50000.00"

    def test_sequential_numbering(self, tmp_path: Path) -> None:
        log = TradeAuditLog(tmp_path / "trades.jsonl")
        r1 = log.append(event_type="fill", symbol="X", side="buy", qty="1", price="100")
        r2 = log.append(event_type="fill", symbol="X", side="sell", qty="1", price="101")

        assert r1.seq == 1
        assert r2.seq == 2

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "trades.jsonl"
        log1 = TradeAuditLog(path)
        log1.append(event_type="fill", symbol="X", side="buy", qty="1", price="100")
        log1.append(event_type="fill", symbol="X", side="sell", qty="1", price="101")

        log2 = TradeAuditLog(path)
        assert log2.count == 2

        r3 = log2.append(event_type="fill", symbol="X", side="buy", qty="1", price="102")
        assert r3.seq == 3

    def test_query_with_filters(self, tmp_path: Path) -> None:
        log = TradeAuditLog(tmp_path / "trades.jsonl")
        log.append(event_type="fill", symbol="BTCUSDT", side="buy", qty="1", price="100")
        log.append(event_type="order_new", symbol="ETHUSDT", side="buy", qty="1", price="100")
        log.append(event_type="fill", symbol="ETHUSDT", side="sell", qty="1", price="100")

        fills = log.query(event_type="fill")
        assert len(fills) == 2

        eth = log.query(symbol="ETHUSDT")
        assert len(eth) == 2

    def test_query_after_seq(self, tmp_path: Path) -> None:
        log = TradeAuditLog(tmp_path / "trades.jsonl")
        for i in range(5):
            log.append(event_type="fill", symbol="X", side="buy", qty="1", price=str(i))

        results = log.query(after_seq=3)
        assert len(results) == 2
        assert results[0].seq == 4

    def test_empty_log_query(self, tmp_path: Path) -> None:
        log = TradeAuditLog(tmp_path / "trades.jsonl")
        assert log.query() == []
        assert log.count == 0
