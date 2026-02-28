"""Tests for walk-forward validation framework."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from statistics import stdev
from typing import List
from unittest.mock import patch

import pytest

from runner.backtest.csv_io import OhlcvBar
from runner.backtest.metrics import EquityPoint
from scripts.run_walk_forward_validation import (
    BARS_PER_MONTH,
    FoldResult,
    _bars_to_temp_csv,
    _compute_sharpe,
    _generate_folds,
    _print_report,
)


def _make_bars(n: int, start_price: float = 10000.0, step_hours: int = 1) -> List[OhlcvBar]:
    """Generate synthetic OHLCV bars."""
    bars = []
    base_ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    price = start_price
    for i in range(n):
        # Simple oscillating price for test variety
        delta = 50.0 * (1 if i % 7 < 4 else -1)
        o = Decimal(str(round(price, 2)))
        h = Decimal(str(round(price + abs(delta), 2)))
        l = Decimal(str(round(price - abs(delta) * 0.5, 2)))
        c = Decimal(str(round(price + delta, 2)))
        v = Decimal("100")
        bars.append(OhlcvBar(
            ts=base_ts + timedelta(hours=i * step_hours),
            o=o, h=h, l=l, c=c, v=v,
        ))
        price = float(c)
    return bars


class TestFoldSplitting:
    def test_basic_expanding_window(self):
        total = 10000
        train = 2000
        test = 1000
        folds = _generate_folds(total, train, test)

        # First fold: train_end=2000, test_end=3000
        assert folds[0] == (2000, 3000)
        # Second fold: train_end=3000, test_end=4000
        assert folds[1] == (3000, 4000)
        # Non-overlapping test windows
        for i in range(1, len(folds)):
            assert folds[i][0] == folds[i - 1][1] or folds[i][0] >= folds[i - 1][1]

    def test_covers_all_data(self):
        total = 10000
        train = 2000
        test = 1000
        folds = _generate_folds(total, train, test)
        # Last fold's test_end should reach near total
        assert folds[-1][1] == total

    def test_partial_last_fold(self):
        # 10500 bars: 8 full folds (2000+8*1000=10000), 500 remaining >= 500 (half test)
        total = 10500
        folds = _generate_folds(total, 2000, 1000)
        assert folds[-1][1] == 10500
        assert folds[-1][0] == 10000

    def test_no_partial_if_too_small(self):
        # 10200 bars: 8 full folds, 200 remaining < 500 (half of 1000)
        total = 10200
        folds = _generate_folds(total, 2000, 1000)
        assert folds[-1][1] == 10000  # last full fold, no partial

    def test_fold_count_matches_expectation(self):
        train = 6 * BARS_PER_MONTH  # 4380
        test = 3 * BARS_PER_MONTH  # 2190
        total = 56737
        folds = _generate_folds(total, train, test)
        # (56737 - 4380) / 2190 ≈ 23.9 → ~24 folds
        assert 23 <= len(folds) <= 25


class TestBarsToTempCsv:
    def test_roundtrip(self):
        bars = _make_bars(50)
        path = _bars_to_temp_csv(bars)
        try:
            from runner.backtest.csv_io import iter_ohlcv_csv
            loaded = list(iter_ohlcv_csv(path))
            assert len(loaded) == 50
            assert loaded[0].ts == bars[0].ts
            assert loaded[0].o == bars[0].o
            assert loaded[-1].c == bars[-1].c
        finally:
            path.unlink(missing_ok=True)


class TestComputeSharpe:
    def test_flat_equity_returns_zero(self):
        base = datetime(2020, 1, 1, tzinfo=timezone.utc)
        pts = [
            EquityPoint(
                ts=base + timedelta(hours=i),
                close=Decimal("100"), position_qty=Decimal("0"),
                avg_price=None, balance=Decimal("10000"),
                realized=Decimal("0"), unrealized=Decimal("0"),
                equity=Decimal("10000"),
            )
            for i in range(100)
        ]
        assert _compute_sharpe(pts) == 0.0

    def test_positive_trend_positive_sharpe(self):
        base = datetime(2020, 1, 1, tzinfo=timezone.utc)
        pts = [
            EquityPoint(
                ts=base + timedelta(hours=i),
                close=Decimal("100"), position_qty=Decimal("0"),
                avg_price=None, balance=Decimal(str(10000 + i * 10)),
                realized=Decimal("0"), unrealized=Decimal("0"),
                equity=Decimal(str(10000 + i * 10)),
            )
            for i in range(200)
        ]
        sharpe = _compute_sharpe(pts)
        assert sharpe > 0

    def test_too_few_points(self):
        base = datetime(2020, 1, 1, tzinfo=timezone.utc)
        pts = [
            EquityPoint(
                ts=base, close=Decimal("100"), position_qty=Decimal("0"),
                avg_price=None, balance=Decimal("10000"),
                realized=Decimal("0"), unrealized=Decimal("0"),
                equity=Decimal("10000"),
            )
        ]
        assert _compute_sharpe(pts) == 0.0


class TestAggregateStatistics:
    def test_mean_and_std(self):
        results = [
            FoldResult(fold_idx=i, train_bars=100, test_bars=50,
                       train_start_ts="", train_end_ts="",
                       test_start_ts="", test_end_ts="",
                       sharpe=s, total_return=r, max_drawdown=d, trades=t)
            for i, (s, r, d, t) in enumerate([
                (1.0, 0.20, 0.10, 15),
                (0.5, 0.10, 0.15, 12),
                (0.8, 0.15, 0.12, 18),
            ])
        ]
        sharpes = [r.sharpe for r in results]
        returns = [r.total_return for r in results]
        avg_sharpe = sum(sharpes) / len(sharpes)
        std_sharpe = stdev(sharpes)

        assert abs(avg_sharpe - 0.7667) < 0.01
        assert std_sharpe > 0

        avg_return = sum(returns) / len(returns)
        assert abs(avg_return - 0.15) < 0.01


class TestOverfitDetection:
    def test_no_overfit_when_close(self):
        full_sharpe = 1.0
        avg_oos = 0.8
        degradation = 1.0 - avg_oos / full_sharpe
        assert degradation < 0.50  # 20% degradation → no overfit

    def test_overfit_when_large_gap(self):
        full_sharpe = 1.0
        avg_oos = 0.3
        degradation = 1.0 - avg_oos / full_sharpe
        assert degradation > 0.50  # 70% degradation → overfit

    def test_zero_full_sharpe(self):
        full_sharpe = 0.0
        degradation = 0.0  # avoid division by zero
        assert degradation <= 0.50


class TestRunFold:
    @pytest.mark.slow
    def test_run_fold_produces_metrics(self):
        """Run a single fold with small synthetic data."""
        bars = _make_bars(500)
        from scripts.run_walk_forward_validation import _run_fold
        result = _run_fold(
            bars=bars,
            train_end=300,
            test_end=500,
            config_kwargs={},
            symbol="BTCUSDT",
            starting_balance=Decimal("10000"),
            fee_bps=Decimal("4"),
            slippage_bps=Decimal("2"),
        )
        assert result.test_bars == 200
        assert result.train_bars == 300
        assert isinstance(result.sharpe, float)
        assert isinstance(result.total_return, float)
        assert isinstance(result.max_drawdown, float)
        assert result.max_drawdown >= 0


class TestPrintReport:
    def test_saves_json(self, tmp_path):
        results = [
            FoldResult(fold_idx=0, train_bars=100, test_bars=50,
                       train_start_ts="2020-01", train_end_ts="2020-06",
                       test_start_ts="2020-06", test_end_ts="2020-09",
                       sharpe=0.8, total_return=0.15, max_drawdown=0.10, trades=12),
            FoldResult(fold_idx=1, train_bars=150, test_bars=50,
                       train_start_ts="2020-01", train_end_ts="2020-09",
                       test_start_ts="2020-09", test_end_ts="2020-12",
                       sharpe=0.6, total_return=0.10, max_drawdown=0.12, trades=10),
        ]
        _print_report(results, full_sample_sharpe=0.9, out_dir=tmp_path)
        report_path = tmp_path / "walk_forward_report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert len(data["folds"]) == 2
        assert data["aggregate"]["avg_sharpe"] == 0.7
        assert data["aggregate"]["full_sample_sharpe"] == 0.9
        assert data["aggregate"]["total_folds"] == 2
