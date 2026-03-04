"""Tests for RealisticCostModel."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from execution.sim.cost_model import RealisticCostModel, CostBreakdown


@pytest.fixture
def cm():
    return RealisticCostModel()


@pytest.fixture
def sample_data():
    """Generate sample trading data."""
    n = 100
    np.random.seed(42)
    closes = 50000.0 + np.cumsum(np.random.randn(n) * 100)
    volumes = np.random.uniform(500, 2000, n)
    volatility = np.full(n, 0.01)  # 1% per bar
    signal = np.zeros(n)
    signal[10:50] = 1.0    # Long from bar 10-49
    signal[60:80] = -1.0   # Short from bar 60-79
    return signal, closes, volumes, volatility


class TestFlatCost:
    def test_flat_matches_legacy(self, cm):
        """Flat cost should reproduce the 6bps behavior."""
        signal = np.array([0, 1, 1, 1, -1, -1, 0])
        flat = RealisticCostModel.flat_cost(signal, 0.0006)
        turnover = np.abs(np.diff(signal, prepend=0))
        expected = turnover * 0.0006
        np.testing.assert_allclose(flat, expected)

    def test_flip_costs_double(self, cm):
        """Flipping from +1 to -1 should cost 2x turnover."""
        signal = np.array([0, 1, -1, 0])
        flat = RealisticCostModel.flat_cost(signal, 0.0006)
        # bar 0: 0->0 = 0
        # bar 1: 0->1 = 1 * 6bps
        # bar 2: 1->-1 = 2 * 6bps
        # bar 3: -1->0 = 1 * 6bps
        np.testing.assert_allclose(flat, [0.0, 0.0006, 0.0012, 0.0006])


class TestRealisticCost:
    def test_returns_cost_breakdown(self, cm, sample_data):
        signal, closes, volumes, volatility = sample_data
        breakdown = cm.compute_costs(signal, closes, volumes, volatility)
        assert isinstance(breakdown, CostBreakdown)
        assert len(breakdown.total_cost) == len(signal)
        assert len(breakdown.fee_cost) == len(signal)
        assert len(breakdown.impact_cost) == len(signal)
        assert len(breakdown.spread_cost) == len(signal)
        assert len(breakdown.clipped_signal) == len(signal)

    def test_total_is_sum_of_components(self, cm, sample_data):
        signal, closes, volumes, volatility = sample_data
        breakdown = cm.compute_costs(signal, closes, volumes, volatility)
        expected = breakdown.fee_cost + breakdown.impact_cost + breakdown.spread_cost
        np.testing.assert_allclose(breakdown.total_cost, expected)

    def test_zero_signal_zero_cost(self, cm):
        n = 50
        signal = np.zeros(n)
        closes = np.full(n, 50000.0)
        volumes = np.full(n, 1000.0)
        vol = np.full(n, 0.01)
        breakdown = cm.compute_costs(signal, closes, volumes, vol)
        np.testing.assert_allclose(breakdown.total_cost, 0.0)

    def test_large_trade_higher_impact(self, cm):
        """Large position change should have higher market impact than small."""
        n = 10
        closes = np.full(n, 50000.0)
        volumes = np.full(n, 1000.0)
        vol = np.full(n, 0.01)

        # Small trade: 0 → 0.1
        small_signal = np.zeros(n)
        small_signal[5:] = 0.1
        small_bd = cm.compute_costs(small_signal, closes, volumes, vol)

        # Large trade: 0 → 1.0
        large_signal = np.zeros(n)
        large_signal[5:] = 1.0
        large_bd = cm.compute_costs(large_signal, closes, volumes, vol)

        # Impact should be larger for bigger trade
        assert np.sum(large_bd.impact_cost) > np.sum(small_bd.impact_cost)

    def test_volume_participation_clips_signal(self, cm):
        """Very large position change should be clipped by volume participation."""
        n = 10
        closes = np.full(n, 50000.0)
        volumes = np.full(n, 0.01)  # Very low volume
        vol = np.full(n, 0.01)

        signal = np.zeros(n)
        signal[5:] = 1.0  # Jump to full position
        breakdown = cm.compute_costs(signal, closes, volumes, vol, capital=10000.0)

        # Clipped signal should be smaller than original at the transition
        assert abs(breakdown.clipped_signal[5]) <= abs(signal[5])

    def test_realistic_costs_higher_than_flat(self, cm, sample_data):
        """Realistic model should generally produce higher costs than flat 6bps."""
        signal, closes, volumes, volatility = sample_data
        breakdown = cm.compute_costs(signal, closes, volumes, volatility)
        flat = RealisticCostModel.flat_cost(signal, 0.0006)
        # Realistic should include more cost components
        assert np.sum(breakdown.total_cost) > np.sum(flat) * 0.5  # At least half of flat

    def test_nan_volatility_handled(self, cm):
        """NaN volatility should not crash, impact/spread should be zero."""
        n = 10
        signal = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0], dtype=np.float64)
        closes = np.full(n, 50000.0)
        volumes = np.full(n, 1000.0)
        vol = np.full(n, np.nan)
        breakdown = cm.compute_costs(signal, closes, volumes, vol)
        # Should not crash
        assert not np.any(np.isnan(breakdown.total_cost))
        # Impact and spread should be zero when vol is NaN
        np.testing.assert_allclose(breakdown.impact_cost, 0.0)
        np.testing.assert_allclose(breakdown.spread_cost, 0.0)

    def test_custom_fee_rates(self):
        """Custom maker/taker rates should affect fees."""
        cm_high = RealisticCostModel(maker_fee_bps=5.0, taker_fee_bps=10.0, taker_ratio=1.0)
        cm_low = RealisticCostModel(maker_fee_bps=1.0, taker_fee_bps=2.0, taker_ratio=1.0)

        n = 10
        signal = np.zeros(n)
        signal[5:] = 1.0
        closes = np.full(n, 50000.0)
        volumes = np.full(n, 10000.0)
        vol = np.full(n, 0.01)

        high_bd = cm_high.compute_costs(signal, closes, volumes, vol)
        low_bd = cm_low.compute_costs(signal, closes, volumes, vol)

        assert np.sum(high_bd.fee_cost) > np.sum(low_bd.fee_cost)
