"""Tests for drawdown rules."""
from __future__ import annotations

from portfolio.risk_model.tail.drawdown import compute_drawdowns, analyze_drawdowns


def test_compute_drawdowns_no_loss():
    returns = [0.01, 0.02, 0.01, 0.03]
    dds = compute_drawdowns(returns)
    assert all(d >= 0 or abs(d) < 1e-10 for d in dds)


def test_compute_drawdowns_with_loss():
    returns = [0.05, -0.10, 0.02, -0.03]
    dds = compute_drawdowns(returns)
    assert min(dds) < 0


def test_analyze_drawdowns():
    returns = [0.01, 0.02, -0.05, -0.03, 0.01, 0.04]
    stats = analyze_drawdowns(returns)
    assert stats.max_drawdown < 0
    assert stats.max_drawdown_duration > 0
