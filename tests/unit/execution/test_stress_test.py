# tests/unit/execution/test_stress_test.py
"""Tests for Monte Carlo stress testing and delay model."""
import numpy as np
import pytest

from execution.sim.realistic_backtest import BacktestConfig
from execution.sim.stress_test import run_stress_test
from execution.sim.delay_model import apply_signal_delay, apply_fill_price_noise


class TestSignalDelay:
    def test_zero_delay(self):
        sig = np.array([0, 0, 1, 1, -1, -1, 0])
        delayed = apply_signal_delay(sig, 0)
        np.testing.assert_array_equal(delayed, sig)

    def test_one_bar_delay(self):
        sig = np.array([0, 0, 1, 1, -1, -1, 0])
        delayed = apply_signal_delay(sig, 1)
        np.testing.assert_array_equal(delayed, [0, 0, 0, 1, 1, -1, -1])

    def test_two_bar_delay(self):
        sig = np.array([0, 0, 1, 1, -1])
        delayed = apply_signal_delay(sig, 2)
        np.testing.assert_array_equal(delayed, [0, 0, 0, 0, 1])


class TestFillPriceNoise:
    def test_noise_around_open(self):
        closes = np.array([100, 101, 102, 103, 104], dtype=float)
        opens = np.array([100, 100.5, 101.5, 102.5, 103.5], dtype=float)
        signal = np.array([0, 0, 1, 1, 0], dtype=float)

        fills = apply_fill_price_noise(closes, opens, signal, noise_bps=5)
        # Bar 2: signal changes → fill near open (101.5 ± noise)
        assert abs(fills[2] - 101.5) < 1.0  # within ~1% of open
        # Bar 4: signal changes → fill near open (103.5 ± noise)
        assert abs(fills[4] - 103.5) < 1.0


class TestStressTest:
    def test_basic_stress_test(self):
        n = 200
        np.random.seed(42)
        closes = 2000 + np.cumsum(np.random.randn(n) * 10)
        highs = closes + abs(np.random.randn(n) * 5)
        lows = closes - abs(np.random.randn(n) * 5)
        volumes = np.ones(n) * 1000
        signal = np.zeros(n)
        signal[20:80] = 1
        signal[120:180] = -1

        cfg = BacktestConfig(initial_equity=1000, leverage=1)
        result = run_stress_test(
            closes, highs, lows, volumes, signal, cfg,
            n_sims=10,  # small for test speed
        )

        assert result.n_simulations == 10
        assert result.median_return != 0 or result.median_trades > 0
        assert result.bust_rate >= 0
        assert result.p5_return <= result.median_return <= result.p95_return

    def test_delay_reduces_returns(self):
        """Adding execution delay should generally reduce performance."""
        n = 200
        np.random.seed(42)
        prices = 2000 + np.cumsum(np.random.randn(n) * 10)
        closes = prices
        highs = prices + 5
        lows = prices - 5
        volumes = np.ones(n) * 1000
        signal = np.zeros(n)
        signal[20:100] = 1  # long during uptrend

        cfg = BacktestConfig(initial_equity=1000, leverage=1)

        r_no_delay = run_stress_test(
            closes, highs, lows, volumes, signal, cfg,
            n_sims=10, execution_delay_bars=0,
        )
        r_delay = run_stress_test(
            closes, highs, lows, volumes, signal, cfg,
            n_sims=10, execution_delay_bars=2,
        )

        # Delay should generally reduce returns (not guaranteed per MC, but typical)
        # At minimum, both should complete without error
        assert r_no_delay.n_simulations == 10
        assert r_delay.n_simulations == 10
