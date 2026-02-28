"""Tests for scripts/run_walk_forward_validation.py — pure utility functions."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from scripts.run_walk_forward_validation import (
    BARS_PER_MONTH,
    FoldResult,
    _compute_sharpe,
    _generate_folds,
    _print_report,
)
from runner.backtest.metrics import EquityPoint


# ── _generate_folds ─────────────────────────────────────────


def test_generate_folds_basic():
    folds = _generate_folds(total_bars=1000, train_size=200, test_size=100)
    assert len(folds) > 0
    # First fold: train_end=200, test_end=300
    assert folds[0] == (200, 300)
    # Second fold: train_end=300, test_end=400
    assert folds[1] == (300, 400)


def test_generate_folds_exact_fit():
    # 200 train + 4*100 test = 600
    folds = _generate_folds(total_bars=600, train_size=200, test_size=100)
    assert len(folds) == 4
    assert folds[-1] == (500, 600)


def test_generate_folds_partial_last():
    # 200 train + 3*100 test = 500, remaining 80 >= 50 (half of 100)
    folds = _generate_folds(total_bars=580, train_size=200, test_size=100)
    assert folds[-1] == (500, 580)  # partial last fold


def test_generate_folds_too_small_remainder():
    # 200 train + 3*100 test = 500, remaining 20 < 50 (half of 100)
    folds = _generate_folds(total_bars=520, train_size=200, test_size=100)
    assert folds[-1] == (400, 500)  # no partial fold


def test_generate_folds_not_enough_data():
    folds = _generate_folds(total_bars=100, train_size=200, test_size=100)
    assert len(folds) == 0


def test_generate_folds_expanding_window():
    folds = _generate_folds(total_bars=1000, train_size=300, test_size=100)
    # Each fold's train_end grows by test_size
    for i in range(1, len(folds)):
        assert folds[i][0] - folds[i - 1][0] == 100


def test_generate_folds_single_fold():
    folds = _generate_folds(total_bars=300, train_size=200, test_size=100)
    assert len(folds) == 1
    assert folds[0] == (200, 300)


# ── _compute_sharpe ─────────────────────────────────────────


_EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_equity(values, start_hour=0):
    return [
        EquityPoint(
            ts=_EPOCH + timedelta(hours=start_hour + i),
            close=Decimal(str(v)),
            position_qty=Decimal("0"),
            avg_price=None,
            balance=Decimal(str(v)),
            realized=Decimal("0"),
            unrealized=Decimal("0"),
            equity=Decimal(str(v)),
        )
        for i, v in enumerate(values)
    ]


def test_compute_sharpe_empty():
    assert _compute_sharpe([]) == 0.0


def test_compute_sharpe_single_point():
    assert _compute_sharpe(_make_equity([100])) == 0.0


def test_compute_sharpe_flat_equity():
    eq = _make_equity([100] * 10)
    assert _compute_sharpe(eq) == 0.0  # zero std => 0


def test_compute_sharpe_uptrend():
    # Monotonically increasing => positive Sharpe
    eq = _make_equity([100 + i for i in range(50)])
    sharpe = _compute_sharpe(eq)
    assert sharpe > 0


def test_compute_sharpe_downtrend():
    # Monotonically decreasing => negative Sharpe
    eq = _make_equity([100 - i * 0.5 for i in range(50)])
    sharpe = _compute_sharpe(eq)
    assert sharpe < 0


def test_compute_sharpe_annualized():
    # Check that annualization factor is applied
    eq = _make_equity([100 + i * 0.01 for i in range(100)])
    sharpe = _compute_sharpe(eq)
    # Should be a reasonable annualized number, not a tiny per-bar value
    assert abs(sharpe) > 1.0


# ── FoldResult ──────────────────────────────────────────────


def test_fold_result_dataclass():
    r = FoldResult(
        fold_idx=0,
        train_bars=1000,
        test_bars=200,
        train_start_ts="2024-01",
        train_end_ts="2024-06",
        test_start_ts="2024-07",
        test_end_ts="2024-09",
        sharpe=1.5,
        total_return=0.12,
        max_drawdown=0.05,
        trades=42,
    )
    assert r.fold_idx == 0
    assert r.params is None  # optional default


def test_fold_result_with_params():
    r = FoldResult(
        fold_idx=1,
        train_bars=500,
        test_bars=100,
        train_start_ts="2024-01",
        train_end_ts="2024-03",
        test_start_ts="2024-04",
        test_end_ts="2024-06",
        sharpe=0.8,
        total_return=0.05,
        max_drawdown=0.03,
        trades=10,
        params={"atr_stop": 3.0},
    )
    assert r.params == {"atr_stop": 3.0}


# ── _print_report ───────────────────────────────────────────


def test_print_report_no_results(capsys):
    _print_report([], None, None)
    captured = capsys.readouterr()
    assert "No fold results" in captured.out


def test_print_report_with_results(capsys, tmp_path):
    results = [
        FoldResult(0, 1000, 200, "2024-01", "2024-06", "2024-07", "2024-09",
                   sharpe=1.5, total_return=0.10, max_drawdown=0.05, trades=20),
        FoldResult(1, 1200, 200, "2024-01", "2024-09", "2024-10", "2024-12",
                   sharpe=0.8, total_return=0.03, max_drawdown=0.08, trades=15),
    ]
    _print_report(results, full_sample_sharpe=2.0, out_dir=tmp_path)

    captured = capsys.readouterr()
    assert "Avg Sharpe" in captured.out
    assert "Degradation" in captured.out
    assert "Full-sample Sharpe" in captured.out

    # Report JSON saved
    report_path = tmp_path / "walk_forward_report.json"
    assert report_path.exists()
    import json
    data = json.loads(report_path.read_text())
    assert len(data["folds"]) == 2
    assert "aggregate" in data


def test_print_report_overfit_detection(capsys):
    # full_sample=2.0, avg_oos=0.5 => degradation=75% > 50% => YES
    results = [
        FoldResult(0, 1000, 200, "a", "b", "c", "d",
                   sharpe=0.5, total_return=0.01, max_drawdown=0.02, trades=5),
    ]
    _print_report(results, full_sample_sharpe=2.0, out_dir=None)
    captured = capsys.readouterr()
    assert "YES" in captured.out


def test_print_report_no_overfit(capsys):
    # full_sample=1.0, avg_oos=0.9 => degradation=10% < 50% => NO
    results = [
        FoldResult(0, 1000, 200, "a", "b", "c", "d",
                   sharpe=0.9, total_return=0.05, max_drawdown=0.02, trades=10),
    ]
    _print_report(results, full_sample_sharpe=1.0, out_dir=None)
    captured = capsys.readouterr()
    assert "NO" in captured.out


# ── BARS_PER_MONTH ──────────────────────────────────────────


def test_bars_per_month():
    assert BARS_PER_MONTH == 730
