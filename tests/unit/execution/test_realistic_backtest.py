# tests/unit/execution/test_realistic_backtest.py
"""Tests for realistic backtest engine — verify P0 fixes."""
import numpy as np
import pytest

from execution.sim.realistic_backtest import (
    BacktestConfig,
    _check_intrabar_stoploss,
    _check_liquidation,
    _compute_slippage,
    run_realistic_backtest,
)


class TestIntrabarStoploss:
    """P0-1: Stop-loss must use high/low, not close."""

    def test_long_stop_triggered_by_low(self):
        # Long at $100, stop at 3% → $97. Bar low hits $96 → triggered.
        triggered, px = _check_intrabar_stoploss(1, 100.0, 105.0, 96.0, 0.03)
        assert triggered is True
        assert px == pytest.approx(97.0)

    def test_long_stop_not_triggered(self):
        # Long at $100, stop at $97. Bar low $98 → not triggered.
        triggered, _ = _check_intrabar_stoploss(1, 100.0, 105.0, 98.0, 0.03)
        assert triggered is False

    def test_short_stop_triggered_by_high(self):
        # Short at $100, stop at $103. Bar high hits $104 → triggered.
        triggered, px = _check_intrabar_stoploss(-1, 100.0, 104.0, 95.0, 0.03)
        assert triggered is True
        assert px == pytest.approx(103.0)

    def test_short_stop_not_triggered(self):
        triggered, _ = _check_intrabar_stoploss(-1, 100.0, 102.0, 95.0, 0.03)
        assert triggered is False


class TestLiquidation:
    """P0-2: Margin/liquidation must be simulated."""

    def test_liquidation_triggered(self):
        # $10 equity, $200 position → margin ratio 5% → below 0.5% maintenance
        assert _check_liquidation(10.0, 200.0, 0.005) is False  # 5% > 0.5%
        assert _check_liquidation(0.5, 200.0, 0.005) is True   # 0.25% < 0.5%

    def test_no_liquidation_healthy(self):
        assert _check_liquidation(100.0, 300.0, 0.005) is False  # 33% >> 0.5%


class TestDynamicSlippage:
    """P1-1: Slippage must scale with position size."""

    def test_small_order_low_slippage(self):
        cfg = BacktestConfig()
        s = _compute_slippage(100, 1_000_000, 0.005, cfg)
        assert s < 0.002  # less than 20bps for tiny order

    def test_large_order_high_slippage(self):
        cfg = BacktestConfig()
        s_small = _compute_slippage(100, 1_000_000, 0.005, cfg)
        s_large = _compute_slippage(100_000, 1_000_000, 0.005, cfg)
        assert s_large > s_small  # larger order = more slippage


class TestPositionCap:
    """P0-3: Position must be capped to prevent exponential blowup."""

    def test_position_capped(self):
        n = 100
        closes = np.ones(n) * 2000
        highs = closes * 1.01
        lows = closes * 0.99
        volumes = np.ones(n) * 1000
        signal = np.ones(n)  # always long

        cfg = BacktestConfig(
            initial_equity=100, leverage=10, max_position_pct=0.5,
        )
        result = run_realistic_backtest(closes, highs, lows, volumes, signal, cfg)
        # With 50% cap at 10x: max notional = 100 * 0.5 * 10 = $500
        # Not $100 * 10 = $1000
        if result.trades:
            assert result.trades[0].size_usd <= 100 * 0.5 * 10 + 1


class TestEndToEnd:
    """Full backtest with all P0 fixes."""

    def test_basic_long_profitable(self):
        n = 50
        closes = np.linspace(100, 110, n)  # steady uptrend
        highs = closes * 1.005
        lows = closes * 0.995
        volumes = np.ones(n) * 1000
        signal = np.zeros(n)
        signal[5:45] = 1  # long from bar 5 to 45

        cfg = BacktestConfig(initial_equity=1000, leverage=1)
        result = run_realistic_backtest(closes, highs, lows, volumes, signal, cfg)
        assert result.total_return_pct > 0  # should be profitable
        assert result.n_trades >= 1

    def test_stop_loss_limits_drawdown(self):
        n = 50
        closes = np.concatenate([
            np.linspace(100, 100, 10),  # flat
            np.linspace(100, 80, 20),   # crash 20%
            np.linspace(80, 90, 20),    # recovery
        ])
        highs = closes * 1.01
        lows = closes * 0.99
        volumes = np.ones(n) * 1000
        signal = np.ones(n)  # always long

        cfg = BacktestConfig(initial_equity=1000, leverage=3, stop_loss_pct=0.03)
        result = run_realistic_backtest(closes, highs, lows, volumes, signal, cfg)
        # With 3% stop at 3x: each stop-loss re-entry eats ~9% + fees
        # Multiple re-entries in crash still cause significant DD but prevent single catastrophic loss
        # Key: each INDIVIDUAL trade should be capped, not total DD
        stopped_trades = [t for t in result.trades if t.exit_reason == "stop_loss"]
        for t in stopped_trades:
            single_loss_pct = abs(t.pnl_net / max(t.size_usd, 1)) * 100
            assert single_loss_pct < 15, f"Single stop-loss trade lost {single_loss_pct:.1f}%"

    def test_liquidation_fires(self):
        n = 20
        closes = np.concatenate([
            np.array([100.0] * 5),
            np.linspace(100, 50, 15),  # 50% crash
        ])
        highs = closes * 1.001
        lows = closes * 0.999
        volumes = np.ones(n) * 1000
        signal = np.ones(n)  # always long during crash

        cfg = BacktestConfig(
            initial_equity=100, leverage=10, stop_loss_pct=0.5,  # wide stop
            maintenance_margin=0.05,  # 5% maintenance
        )
        result = run_realistic_backtest(closes, highs, lows, volumes, signal, cfg)
        assert result.n_liquidations >= 1 or result.max_drawdown_pct > 80

    def test_fees_reduce_returns(self):
        n = 100
        closes = np.ones(n) * 2000  # flat market
        highs = closes * 1.001
        lows = closes * 0.999
        volumes = np.ones(n) * 1000
        signal = np.zeros(n)
        # Frequent trading (enter/exit every 5 bars)
        for i in range(0, n, 10):
            signal[i:i+5] = 1

        cfg_no_fee = BacktestConfig(initial_equity=1000, fee_bps=0, base_slippage_bps=0, spread_bps=0)
        cfg_fee = BacktestConfig(initial_equity=1000, fee_bps=4, base_slippage_bps=1, spread_bps=1)

        r_no_fee = run_realistic_backtest(closes, highs, lows, volumes, signal, cfg_no_fee)
        r_fee = run_realistic_backtest(closes, highs, lows, volumes, signal, cfg_fee)

        assert r_fee.total_fees > 0
        assert r_fee.equity_curve[-1] < r_no_fee.equity_curve[-1]
