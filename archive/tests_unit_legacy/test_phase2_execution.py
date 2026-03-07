"""Tests for Phase 2 execution algorithms and depth processing."""
from __future__ import annotations

import json
import time
from decimal import Decimal
from typing import Optional

import pytest

from execution.algos.twap import TWAPAlgo, TWAPOrder
from execution.algos.vwap import VWAPAlgo
from execution.algos.iceberg import IcebergAlgo
from execution.adapters.binance.depth_processor import DepthProcessor, OrderBookSnapshot


class TestTWAPAlgo:
    def _submit(self, symbol: str, side: str, qty: Decimal) -> Optional[Decimal]:
        return Decimal("50000")

    def test_creates_correct_slices(self) -> None:
        algo = TWAPAlgo(submit_fn=self._submit)
        order = algo.create("BTCUSDT", "buy", Decimal("1.0"), n_slices=5, duration_sec=100)

        assert order.n_slices == 5
        assert len(order.slices) == 5
        total = sum(s.qty for s in order.slices)
        assert total == Decimal("1.0")

    def test_tick_executes_due_slices(self) -> None:
        algo = TWAPAlgo(submit_fn=self._submit)
        order = algo.create("BTCUSDT", "buy", Decimal("1.0"), n_slices=5, duration_sec=0)

        # All slices should be due immediately (duration=0)
        executed = 0
        for _ in range(10):
            result = algo.tick(order)
            if result and result.status == "executed":
                executed += 1

        assert executed == 5
        assert order.is_complete
        assert order.avg_price == Decimal("50000")


class TestVWAPAlgo:
    def _submit(self, symbol: str, side: str, qty: Decimal) -> Optional[Decimal]:
        return Decimal("50000")

    def test_creates_weighted_slices(self) -> None:
        algo = VWAPAlgo(submit_fn=self._submit)
        profile = [0.3, 0.2, 0.1, 0.1, 0.3]
        order = algo.create(
            "BTCUSDT", "buy", Decimal("1.0"),
            n_slices=5, duration_sec=0, volume_profile=profile,
        )

        assert len(order.slices) == 5
        # First and last slices should have more quantity (higher weight)
        assert order.slices[0].qty > order.slices[2].qty


class TestIcebergAlgo:
    def _submit(self, symbol: str, side: str, qty: Decimal) -> Optional[Decimal]:
        return Decimal("50000")

    def test_creates_clips(self) -> None:
        algo = IcebergAlgo(submit_fn=self._submit)
        order = algo.create(
            "BTCUSDT", "buy", Decimal("1.0"), clip_size=Decimal("0.1"),
        )

        assert len(order.clips) == 10
        assert order.total_qty == Decimal("1.0")

    def test_sequential_execution(self) -> None:
        algo = IcebergAlgo(submit_fn=self._submit)
        order = algo.create(
            "BTCUSDT", "buy", Decimal("0.3"), clip_size=Decimal("0.1"),
        )

        clip1 = algo.tick(order)
        assert clip1 is not None
        assert clip1.status == "filled"

        clip2 = algo.tick(order)
        assert clip2 is not None

        clip3 = algo.tick(order)
        assert clip3 is not None

        assert order.is_complete
        assert order.filled_qty == Decimal("0.3")


class TestDepthProcessor:
    def test_processes_depth_update(self) -> None:
        proc = DepthProcessor()
        raw = json.dumps({
            "data": {
                "e": "depthUpdate",
                "s": "BTCUSDT",
                "E": 1700000000000,
                "u": 12345,
                "b": [["50000.00", "1.5"], ["49999.00", "2.0"]],
                "a": [["50001.00", "1.0"], ["50002.00", "3.0"]],
            }
        })

        snapshot = proc.process_raw(raw)
        assert snapshot is not None
        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.best_bid == Decimal("50000.00")
        assert snapshot.best_ask == Decimal("50001.00")
        assert snapshot.spread == Decimal("1.00")

    def test_mid_price(self) -> None:
        proc = DepthProcessor()
        raw = json.dumps({
            "data": {
                "e": "depthUpdate",
                "s": "BTCUSDT",
                "E": 1700000000000,
                "u": 1,
                "b": [["100.00", "1"]],
                "a": [["102.00", "1"]],
            }
        })
        snap = proc.process_raw(raw)
        assert snap is not None
        assert snap.mid_price == Decimal("101.00")

    def test_invalid_json(self) -> None:
        proc = DepthProcessor()
        assert proc.process_raw("not json") is None

    def test_non_depth_event(self) -> None:
        proc = DepthProcessor()
        raw = json.dumps({"data": {"e": "aggTrade"}})
        assert proc.process_raw(raw) is None
