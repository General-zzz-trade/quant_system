"""Tests for TCA: benchmarks, analyzer, and reporter."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from execution.tca.benchmarks import BenchmarkCalculator
from execution.tca.analyzer import FillRecord, TCAAnalyzer, TCAResult
from execution.tca.report import TCAReporter


# ── Helpers ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Tick:
    ts: datetime
    symbol: str
    price: Decimal
    qty: Decimal
    side: str


def _ts(seconds: int = 0) -> datetime:
    return datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=seconds)


def _tick(price: str, qty: str, seconds: int = 0) -> _Tick:
    return _Tick(
        ts=_ts(seconds),
        symbol="BTCUSDT",
        price=Decimal(price),
        qty=Decimal(qty),
        side="buy",
    )


def _fill(
    price: str,
    qty: str,
    seconds: int = 0,
    algo: str = "twap",
    order_id: str = "ord-001",
) -> FillRecord:
    return FillRecord(
        ts=_ts(seconds),
        symbol="BTCUSDT",
        side="buy",
        qty=Decimal(qty),
        price=Decimal(price),
        fee=Decimal("1"),
        algo=algo,
        order_id=order_id,
    )


# ── Benchmark Tests ─────────────────────────────────────────


class TestBenchmarkCalculator:
    def test_arrival_price(self) -> None:
        calc = BenchmarkCalculator()
        ticks = [_tick("100", "1", 0), _tick("101", "1", 1), _tick("102", "1", 2)]
        result = calc.arrival_price(_ts(1), ticks)
        assert result.name == "arrival"
        assert result.value == Decimal("101")

    def test_arrival_price_empty(self) -> None:
        calc = BenchmarkCalculator()
        result = calc.arrival_price(_ts(0), [])
        assert result.value == Decimal("0")

    def test_vwap_benchmark(self) -> None:
        calc = BenchmarkCalculator()
        ticks = [
            _tick("100", "10", 0),
            _tick("110", "20", 1),
            _tick("200", "5", 100),  # outside window
        ]
        result = calc.vwap_benchmark(ticks, _ts(0), _ts(2))
        # VWAP = (100*10 + 110*20) / (10+20) = 3200/30 = 106.666...
        expected = (Decimal("100") * Decimal("10") + Decimal("110") * Decimal("20")) / Decimal("30")
        assert result.name == "vwap"
        assert result.value == expected

    def test_twap_benchmark(self) -> None:
        calc = BenchmarkCalculator()
        ticks = [_tick("100", "1", 0), _tick("200", "1", 1)]
        result = calc.twap_benchmark(ticks, _ts(0), _ts(1))
        assert result.name == "twap"
        assert result.value == Decimal("150")

    def test_implementation_shortfall_buy(self) -> None:
        calc = BenchmarkCalculator()
        result = calc.implementation_shortfall(
            decision_price=Decimal("100"),
            avg_fill_price=Decimal("101"),
            side="buy",
        )
        assert result.value == Decimal("1")  # Unfavorable

    def test_implementation_shortfall_sell(self) -> None:
        calc = BenchmarkCalculator()
        result = calc.implementation_shortfall(
            decision_price=Decimal("100"),
            avg_fill_price=Decimal("99"),
            side="sell",
        )
        assert result.value == Decimal("1")  # Unfavorable


# ── Analyzer Tests ──────────────────────────────────────────


class TestTCAAnalyzer:
    def test_empty_fills(self) -> None:
        analyzer = TCAAnalyzer()
        result = analyzer.analyze([])
        assert result.fill_count == 0
        assert result.total_qty == Decimal("0")

    def test_single_fill_with_ticks(self) -> None:
        ticks = [_tick("100", "5", 0), _tick("101", "5", 1)]
        fills = [_fill("101", "5", 0)]
        analyzer = TCAAnalyzer()
        result = analyzer.analyze(fills, ticks=ticks)
        assert result.fill_count == 1
        assert result.avg_fill_price == Decimal("101")
        assert "arrival" in result.benchmarks
        assert "vwap" in result.benchmarks

    def test_slippage_positive_for_unfavorable_buy(self) -> None:
        """Buying above arrival price = positive slippage (bad)."""
        ticks = [_tick("100", "5", 0), _tick("100", "5", 1)]
        fills = [_fill("102", "5", 0)]
        analyzer = TCAAnalyzer()
        result = analyzer.analyze(fills, ticks=ticks)
        assert result.slippage_bps["arrival"] > 0

    def test_implementation_shortfall(self) -> None:
        fills = [_fill("101", "5", 0)]
        analyzer = TCAAnalyzer()
        result = analyzer.analyze(fills, decision_price=Decimal("100"))
        assert "implementation_shortfall" in result.benchmarks
        assert result.slippage_bps["implementation_shortfall"] > 0

    def test_multi_fill_aggregation(self) -> None:
        fills = [
            _fill("100", "3", 0),
            _fill("102", "7", 5),
        ]
        analyzer = TCAAnalyzer()
        result = analyzer.analyze(fills)
        expected_vwap = (Decimal("100") * 3 + Decimal("102") * 7) / Decimal("10")
        assert result.avg_fill_price == expected_vwap
        assert result.total_qty == Decimal("10")
        assert result.total_fee == Decimal("2")
        assert result.duration_sec == 5.0


# ── Reporter Tests ──────────────────────────────────────────


class TestTCAReporter:
    def _make_result(
        self,
        algo: str = "twap",
        symbol: str = "BTCUSDT",
        slippage: float = 5.0,
        qty: str = "1",
    ) -> TCAResult:
        return TCAResult(
            order_id="ord-001",
            symbol=symbol,
            side="buy",
            total_qty=Decimal(qty),
            avg_fill_price=Decimal("100"),
            benchmarks={"arrival": Decimal("99.95")},
            slippage_bps={"arrival": slippage},
            total_fee=Decimal("1"),
            algo=algo,
            fill_count=1,
            duration_sec=10.0,
        )

    def test_summary_by_algo(self) -> None:
        reporter = TCAReporter()
        reporter.add_result(self._make_result(algo="twap", slippage=2.0))
        reporter.add_result(self._make_result(algo="twap", slippage=4.0))
        reporter.add_result(self._make_result(algo="vwap", slippage=1.0))
        entries = reporter.summary_by_algo()
        algos = {e.algo: e for e in entries}
        assert "twap" in algos
        assert algos["twap"].n_orders == 2
        assert algos["twap"].avg_slippage_bps == pytest.approx(3.0)
        assert "vwap" in algos
        assert algos["vwap"].n_orders == 1

    def test_summary_by_symbol(self) -> None:
        reporter = TCAReporter()
        reporter.add_result(self._make_result(symbol="BTCUSDT"))
        reporter.add_result(self._make_result(symbol="ETHUSDT"))
        entries = reporter.summary_by_symbol()
        symbols = {e.symbol: e for e in entries}
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols
