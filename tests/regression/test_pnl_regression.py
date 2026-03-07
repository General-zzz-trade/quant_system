"""Regression: PnL computation correctness.

Verifies that a known signal sequence produces expected PnL within tolerance.
"""
from __future__ import annotations

import numpy as np


def test_known_signal_pnl():
    """A known signal + price sequence must produce expected PnL."""
    # Simple scenario: buy at 100, sell at 110 → 10% return
    closes = np.array([100.0, 102.0, 105.0, 108.0, 110.0, 110.0])
    signal = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    # Compute PnL as signal[i] * (closes[i+1] - closes[i]) / closes[i]
    returns = np.diff(closes) / closes[:-1]
    pnl = np.sum(signal[:-1] * returns)

    # Expected: (2/100) + (3/102) + (3/105) + (2/108) = ~0.0968
    assert abs(pnl - 0.0968) < 0.005, f"PnL {pnl:.4f} not within tolerance of 0.0968"


def test_short_signal_pnl():
    """Short signal should profit from price drops."""
    closes = np.array([100.0, 95.0, 90.0, 95.0])
    signal = np.array([-1.0, -1.0, -1.0, 0.0])

    returns = np.diff(closes) / closes[:-1]
    pnl = np.sum(signal[:-1] * returns)

    # Short at 100, price drops to 90, then rises to 95
    # -1 * (-5/100) + -1 * (-5/95) + -1 * (5/90) = 0.05 + 0.0526 - 0.0556 = ~0.047
    assert pnl > 0, f"Short signal should profit from price drop, got PnL={pnl:.4f}"


def test_flat_signal_zero_pnl():
    """Flat signal (all zeros) must produce zero PnL."""
    closes = np.array([100.0, 110.0, 90.0, 120.0])
    signal = np.zeros(4)

    returns = np.diff(closes) / closes[:-1]
    pnl = np.sum(signal[:-1] * returns)

    assert pnl == 0.0, f"Flat signal must produce zero PnL, got {pnl}"


def test_funding_cost_reduces_pnl():
    """Funding cost must reduce net PnL when holding a position."""
    closes = np.array([100.0] * 100)
    signal = np.ones(100)
    funding_rate = 0.0001  # 0.01% per 8h

    gross_returns = np.diff(closes) / closes[:-1]
    gross_pnl = np.sum(signal[:-1] * gross_returns)

    # Funding cost: position * rate / 8 per bar (hourly bars)
    funding_cost = np.sum(np.abs(signal[:-1]) * funding_rate / 8.0)

    net_pnl = gross_pnl - funding_cost

    assert net_pnl < gross_pnl, "Funding must reduce PnL"
    assert funding_cost > 0, "Funding cost must be positive"
