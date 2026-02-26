# tests/unit/research/test_strategy_walk_forward.py
"""Tests for Strategy Walk-Forward Validation."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from research.strategy_walk_forward import (
    StrategyMetrics,
    StrategyWalkForwardRunner,
    WalkForwardReport,
    bollinger_grid,
    compute_metrics,
    ma_cross_grid,
    rsi_grid,
)


class TestComputeMetrics:
    def test_flat_equity_curve(self):
        curve = [100.0] * 50
        m = compute_metrics(curve)
        assert m.total_return == pytest.approx(0.0)
        assert m.max_drawdown == pytest.approx(0.0)

    def test_linear_growth(self):
        curve = [100.0 + i for i in range(100)]
        m = compute_metrics(curve)
        assert m.total_return > 0
        assert m.max_drawdown == pytest.approx(0.0)

    def test_drawdown_detected(self):
        curve = [100.0, 110.0, 90.0, 95.0]
        m = compute_metrics(curve)
        # Peak 110, trough 90, dd = 20/110 = 18.18%
        assert m.max_drawdown == pytest.approx(20.0 / 110.0, rel=1e-3)

    def test_with_trade_pnls(self):
        curve = [100.0, 102.0, 105.0]
        pnls = [2.0, 3.0, -1.0]
        m = compute_metrics(curve, trade_pnls=pnls)
        assert m.trade_count == 3
        assert m.win_rate == pytest.approx(2 / 3)
        assert m.avg_trade_pnl == pytest.approx(4.0 / 3)

    def test_empty_curve(self):
        m = compute_metrics([])
        assert m.total_return == 0.0

    def test_sharpe_positive_for_growth(self):
        curve = [100.0 + i * 0.5 for i in range(200)]
        m = compute_metrics(curve)
        assert m.sharpe_ratio > 0


class TestParameterGrids:
    def test_ma_cross_grid_no_invalid(self):
        grid = ma_cross_grid()
        for params in grid:
            assert params["fast_window"] < params["slow_window"]
        assert len(grid) > 0

    def test_bollinger_grid_size(self):
        grid = bollinger_grid()
        assert len(grid) == 3 * 4  # 3 windows * 4 stds

    def test_rsi_grid_valid(self):
        grid = rsi_grid()
        for params in grid:
            assert params["overbought"] > params["oversold"]
        assert len(grid) > 0


class TestStrategyWalkForwardRunner:
    def _simple_evaluator(self, params: Dict[str, Any], start: int, end: int) -> float:
        """Returns higher metric for larger windows (simulating better smoothing)."""
        window = params.get("fast_window", params.get("window", 10))
        n_bars = end - start
        # Simple metric: longer training period + moderate window = better
        return n_bars / 1000.0 + window / 100.0

    def test_basic_run(self):
        runner = StrategyWalkForwardRunner(
            data_length=1000,
            evaluate_fn=self._simple_evaluator,
            n_folds=3,
        )
        grid = [
            {"fast_window": 10, "slow_window": 30},
            {"fast_window": 20, "slow_window": 50},
        ]
        report = runner.run(grid)
        assert len(report.result.folds) == 3
        assert isinstance(report.fold_metrics, tuple)

    def test_overfit_detection(self):
        # Train metric much higher than test → overfit
        # With expanding window, train always starts at 0; test starts > 0
        def _overfit_eval(params, start, end):
            return 5.0 if start == 0 else 0.1

        runner = StrategyWalkForwardRunner(
            data_length=1000,
            evaluate_fn=_overfit_eval,
            n_folds=3,
        )
        report = runner.run([{"a": 1}])
        assert report.is_overfit

    def test_non_overfit(self):
        def _good_eval(params, start, end):
            return 1.0  # consistent

        runner = StrategyWalkForwardRunner(
            data_length=1000,
            evaluate_fn=_good_eval,
            n_folds=3,
        )
        report = runner.run([{"a": 1}])
        assert not report.is_overfit

    def test_to_dict(self):
        runner = StrategyWalkForwardRunner(
            data_length=500,
            evaluate_fn=lambda p, s, e: 1.0,
            n_folds=2,
        )
        report = runner.run([{"x": 1}])
        d = report.to_dict()
        assert "folds" in d
        assert "is_overfit" in d
        assert "avg_train_metric" in d

    def test_run_and_save(self, tmp_path):
        runner = StrategyWalkForwardRunner(
            data_length=500,
            evaluate_fn=lambda p, s, e: 1.0,
            n_folds=2,
        )
        out = tmp_path / "report.json"
        report = runner.run_and_save([{"x": 1}], out)
        assert out.exists()
        import json
        data = json.loads(out.read_text())
        assert data["n_folds"] == 2
