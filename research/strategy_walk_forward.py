# research/strategy_walk_forward.py
"""Strategy Walk-Forward Validation — integrates walk_forward_optimize with backtest metrics."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from research.walk_forward_optimizer import (
    WalkForwardResult,
    walk_forward_optimize,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StrategyMetrics:
    """Standard strategy performance metrics."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    avg_trade_pnl: float


def compute_metrics(
    equity_curve: Sequence[float],
    trade_pnls: Sequence[float] | None = None,
) -> StrategyMetrics:
    """Compute strategy metrics from an equity curve."""
    if len(equity_curve) < 2:
        return StrategyMetrics(0.0, 0.0, 0.0, 0.0, 0, 0.0)

    # Returns
    returns = [
        (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
        for i in range(1, len(equity_curve))
        if equity_curve[i - 1] != 0
    ]

    total_return = (equity_curve[-1] / equity_curve[0] - 1.0) if equity_curve[0] != 0 else 0.0

    # Sharpe (annualized, assuming daily)
    if returns:
        mean_ret = sum(returns) / len(returns)
        var_ret = sum((r - mean_ret) ** 2 for r in returns) / max(len(returns) - 1, 1)
        std_ret = sqrt(max(var_ret, 0.0))
        sharpe = (mean_ret / std_ret * sqrt(252)) if std_ret > 1e-12 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    # Trade stats
    pnls = list(trade_pnls or [])
    trade_count = len(pnls)
    win_rate = sum(1 for p in pnls if p > 0) / max(trade_count, 1)
    avg_pnl = sum(pnls) / max(trade_count, 1) if pnls else 0.0

    return StrategyMetrics(
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        trade_count=trade_count,
        avg_trade_pnl=avg_pnl,
    )


@dataclass(frozen=True, slots=True)
class WalkForwardReport:
    """Extended walk-forward result with per-fold metrics."""
    result: WalkForwardResult
    fold_metrics: tuple[Dict[str, Any], ...]
    overall_sharpe_train: float
    overall_sharpe_test: float

    @property
    def is_overfit(self) -> bool:
        return self.result.is_overfit

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_train_metric": self.result.avg_train_metric,
            "avg_test_metric": self.result.avg_test_metric,
            "overall_sharpe_train": self.overall_sharpe_train,
            "overall_sharpe_test": self.overall_sharpe_test,
            "is_overfit": self.is_overfit,
            "n_folds": len(self.result.folds),
            "folds": list(self.fold_metrics),
            "best_params_frequency": self.result.best_params_frequency,
        }


# ── Standard parameter grids ────────────────────────────────

def ma_cross_grid(
    fast_range: Sequence[int] = (5, 10, 15, 20),
    slow_range: Sequence[int] = (30, 50, 75, 100),
) -> List[Dict[str, Any]]:
    """Generate MA cross parameter grid."""
    return [
        {"strategy": "ma_cross", "fast_window": f, "slow_window": s}
        for f in fast_range
        for s in slow_range
        if f < s
    ]


def bollinger_grid(
    window_range: Sequence[int] = (10, 20, 30),
    std_range: Sequence[float] = (1.5, 2.0, 2.5, 3.0),
) -> List[Dict[str, Any]]:
    """Generate Bollinger Band parameter grid."""
    return [
        {"strategy": "bollinger", "window": w, "n_std": s}
        for w in window_range
        for s in std_range
    ]


def rsi_grid(
    period_range: Sequence[int] = (7, 14, 21),
    overbought_range: Sequence[int] = (65, 70, 75, 80),
    oversold_range: Sequence[int] = (20, 25, 30, 35),
) -> List[Dict[str, Any]]:
    """Generate RSI parameter grid."""
    return [
        {"strategy": "rsi", "period": p, "overbought": ob, "oversold": os_}
        for p in period_range
        for ob in overbought_range
        for os_ in oversold_range
        if ob > os_
    ]


# ── Main runner ─────────────────────────────────────────────

@dataclass
class StrategyWalkForwardRunner:
    """Runs walk-forward validation with configurable strategy evaluator.

    Usage:
        def my_evaluator(params, start, end) -> float:
            # Run backtest with params on data[start:end]
            # Return Sharpe ratio
            return sharpe

        runner = StrategyWalkForwardRunner(
            data_length=5000,
            evaluate_fn=my_evaluator,
        )
        report = runner.run(ma_cross_grid())
    """

    data_length: int
    evaluate_fn: Callable[[Dict[str, Any], int, int], float]
    n_folds: int = 5
    train_ratio: float = 0.6
    expanding: bool = True
    metric_name: str = "sharpe_ratio"

    def run(
        self,
        param_grid: Sequence[Dict[str, Any]],
    ) -> WalkForwardReport:
        """Execute walk-forward optimization and return detailed report."""
        result = walk_forward_optimize(
            data_length=self.data_length,
            param_grid=param_grid,
            evaluate_fn=self.evaluate_fn,
            n_folds=self.n_folds,
            train_ratio=self.train_ratio,
            expanding=self.expanding,
            metric_higher_is_better=True,
        )

        # Build per-fold metrics
        fold_metrics: List[Dict[str, Any]] = []
        train_sharpes: List[float] = []
        test_sharpes: List[float] = []

        for fold in result.folds:
            fm = {
                "fold_idx": fold.fold_idx,
                "train_range": [fold.train_start, fold.train_end],
                "test_range": [fold.test_start, fold.test_end],
                "best_params": fold.best_params,
                "train_metric": fold.train_metric,
                "test_metric": fold.test_metric,
                "degradation": (
                    1.0 - fold.test_metric / fold.train_metric
                    if fold.train_metric != 0 else 0.0
                ),
            }
            fold_metrics.append(fm)
            train_sharpes.append(fold.train_metric)
            test_sharpes.append(fold.test_metric)

        avg_train = sum(train_sharpes) / max(len(train_sharpes), 1)
        avg_test = sum(test_sharpes) / max(len(test_sharpes), 1)

        report = WalkForwardReport(
            result=result,
            fold_metrics=tuple(fold_metrics),
            overall_sharpe_train=avg_train,
            overall_sharpe_test=avg_test,
        )

        logger.info(
            "Walk-Forward complete: %d folds, train=%.3f test=%.3f overfit=%s",
            len(result.folds), avg_train, avg_test, report.is_overfit,
        )

        return report

    def run_and_save(
        self,
        param_grid: Sequence[Dict[str, Any]],
        output_path: Path,
    ) -> WalkForwardReport:
        """Run and save results to JSON."""
        report = self.run(param_grid)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Walk-Forward report saved to %s", output_path)
        return report
