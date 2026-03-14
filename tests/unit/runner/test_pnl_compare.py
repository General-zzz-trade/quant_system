"""Tests for PnL comparison tool."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal


from runner.backtest.pnl_compare import PnLPoint, compare_pnl


def _make_points(start_eq: float, returns: list[float]) -> list[PnLPoint]:
    """Generate equity points from a return sequence."""
    points = []
    eq = start_eq
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i, r in enumerate(returns):
        eq = eq * (1 + r)
        points.append(PnLPoint(
            ts=base.replace(hour=i),
            equity=Decimal(str(round(eq, 2))),
            realized=Decimal("0"),
            unrealized=Decimal("0"),
        ))
    return points


class TestPnLComparison:
    def test_identical_curves(self):
        rets = [0.01, -0.005, 0.02, 0.003, -0.01]
        bt = _make_points(10000, rets)
        live = _make_points(10000, rets)
        result = compare_pnl(bt, live)
        assert result.return_divergence_pct < 0.1
        assert result.correlation > 0.99
        assert result.aligned_points == 5

    def test_divergent_curves(self):
        bt = _make_points(10000, [0.01, 0.02, 0.03, 0.04, 0.05])
        live = _make_points(10000, [-0.01, -0.02, -0.03, -0.04, -0.05])
        result = compare_pnl(bt, live)
        assert result.return_divergence_pct > 10
        assert len(result.warnings) > 0

    def test_empty_input(self):
        result = compare_pnl([], [])
        assert result.aligned_points == 0

    def test_max_drawdown(self):
        rets = [0.05, 0.03, -0.15, -0.05, 0.02]
        bt = _make_points(10000, rets)
        live = _make_points(10000, [0.01] * 5)
        result = compare_pnl(bt, live)
        assert result.backtest_max_dd_pct > 5
        assert result.live_max_dd_pct < 1

    def test_tracking_error(self):
        bt = _make_points(10000, [0.01, 0.02, -0.01, 0.03, -0.02])
        live = _make_points(10000, [0.015, 0.018, -0.012, 0.028, -0.022])
        result = compare_pnl(bt, live)
        assert result.tracking_error_pct > 0
        assert result.correlation > 0.9
