"""Tests for Phase 2 risk and monitoring modules."""
from __future__ import annotations

from decimal import Decimal

import pytest

from risk.rules.correlation_limit import CorrelationLimitRule
from risk.rules.var_limit import VaRLimitRule
from risk.dynamic.risk_budget import DynamicRiskBudgetManager
from risk.decisions import RiskAction
from research.backtest_analysis import compute_metrics, compute_rolling_sharpe
from monitoring.metrics.pnl_tracker import PnLTracker


class _FakeIntent:
    """Minimal intent event stub for risk rule testing."""
    symbol = "BTCUSDT"
    side = "buy"
    qty = Decimal("0.01")


class TestCorrelationLimitRule:
    def test_allows_low_correlation(self) -> None:
        rule = CorrelationLimitRule(max_avg_correlation=0.7)
        result = rule.evaluate_intent(_FakeIntent(), meta={"portfolio_avg_correlation": 0.5})
        assert result.action == RiskAction.ALLOW

    def test_rejects_high_correlation(self) -> None:
        rule = CorrelationLimitRule(max_avg_correlation=0.7)
        result = rule.evaluate_intent(_FakeIntent(), meta={"portfolio_avg_correlation": 0.8})
        assert result.action == RiskAction.REJECT

    def test_rejects_high_position_correlation(self) -> None:
        rule = CorrelationLimitRule(max_position_correlation=0.85)
        result = rule.evaluate_intent(
            _FakeIntent(), meta={"position_correlation_to_portfolio": 0.9},
        )
        assert result.action == RiskAction.REJECT


class TestVaRLimitRule:
    def test_allows_within_limit(self) -> None:
        rule = VaRLimitRule(max_var_95_pct=5.0)
        result = rule.evaluate_intent(_FakeIntent(), meta={"portfolio_var_95": 3.0})
        assert result.action == RiskAction.ALLOW

    def test_rejects_over_limit(self) -> None:
        rule = VaRLimitRule(max_var_95_pct=5.0)
        result = rule.evaluate_intent(_FakeIntent(), meta={"portfolio_var_95": 7.0})
        assert result.action == RiskAction.REJECT

    def test_rejects_post_trade_var(self) -> None:
        rule = VaRLimitRule(max_var_95_pct=5.0)
        result = rule.evaluate_intent(_FakeIntent(), meta={"post_trade_var_95": 6.0})
        assert result.action == RiskAction.REJECT


class TestDynamicRiskBudget:
    def test_normal_regime(self) -> None:
        mgr = DynamicRiskBudgetManager()
        budget = mgr.update(regime="normal")
        assert budget.scale_factor > 0
        assert budget.regime == "normal"

    def test_volatile_regime_reduces(self) -> None:
        mgr = DynamicRiskBudgetManager()
        normal = mgr.update(regime="normal")
        volatile = mgr.update(regime="volatile")
        assert volatile.scale_factor < normal.scale_factor

    def test_drawdown_halts(self) -> None:
        mgr = DynamicRiskBudgetManager(drawdown_halt_threshold=15.0)
        budget = mgr.update(current_drawdown_pct=20.0)
        assert budget.scale_factor == 0.0

    def test_vol_targeting(self) -> None:
        mgr = DynamicRiskBudgetManager(target_vol=0.15, vol_lookback=10)
        # High vol returns → scale down
        high_vol = [0.05 * ((-1) ** i) for i in range(20)]
        budget = mgr.update(recent_returns=high_vol)
        assert budget.scale_factor < 1.0


class TestBacktestAnalysis:
    def test_positive_returns(self) -> None:
        returns = [0.01 + 0.001 * (i % 3) for i in range(252)]
        metrics = compute_metrics(returns)
        assert metrics.total_return > 0
        assert metrics.sharpe_ratio > 0
        assert metrics.max_drawdown == 0.0

    def test_negative_returns(self) -> None:
        returns = [-0.01] * 100
        metrics = compute_metrics(returns)
        assert metrics.total_return < 0
        assert metrics.max_drawdown > 0

    def test_with_trades(self) -> None:
        returns = [0.01, -0.005] * 50
        trade_pnls = [100, -50, 80, -30, 120]
        metrics = compute_metrics(returns, trade_pnls=trade_pnls)
        assert metrics.total_trades == 5
        assert metrics.win_rate > 0

    def test_empty_returns(self) -> None:
        metrics = compute_metrics([])
        assert metrics.sharpe_ratio == 0

    def test_rolling_sharpe(self) -> None:
        returns = [0.01] * 100
        result = compute_rolling_sharpe(returns, window=20)
        assert len(result) == 100
        assert result[0] is None
        assert result[19] is not None


class TestPnLTracker:
    def test_tracks_pnl(self) -> None:
        tracker = PnLTracker(starting_equity=Decimal("10000"))
        tracker.on_fill(Decimal("100"))
        tracker.on_fill(Decimal("-50"))

        assert tracker.equity == Decimal("10050")

    def test_snapshot(self) -> None:
        tracker = PnLTracker(starting_equity=Decimal("10000"))
        tracker.on_fill(Decimal("500"))
        snap = tracker.snapshot()

        assert snap.equity == Decimal("10500")
        assert snap.realized_pnl == Decimal("500")
        assert snap.trade_count == 1

    def test_drawdown_tracking(self) -> None:
        tracker = PnLTracker(starting_equity=Decimal("10000"))
        tracker.on_fill(Decimal("1000"))
        tracker.snapshot()  # peak = 11000

        tracker.on_fill(Decimal("-500"))
        snap = tracker.snapshot()
        assert snap.drawdown_pct > 0
