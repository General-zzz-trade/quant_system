"""Tests for scripts.oos_eval — OOS evaluation utilities."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.oos_eval import (
    _group_trades,
    apply_threshold,
    compute_1bar_returns,
    compute_signal_costs,
    evaluate_oos,
)


# ── compute_1bar_returns ────────────────────────────────────────────


class TestCompute1barReturns:
    def test_basic(self):
        closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        idx = np.array([0, 1, 2])
        ret = compute_1bar_returns(closes, idx)
        expected = np.array([101 / 100 - 1, 102 / 101 - 1, 103 / 102 - 1])
        np.testing.assert_allclose(ret, expected, rtol=1e-10)

    def test_out_of_bounds_produces_nan(self):
        closes = np.array([100.0, 101.0, 102.0])
        idx = np.array([1, 2])  # idx=2 → needs close[3] which doesn't exist
        ret = compute_1bar_returns(closes, idx)
        assert not np.isnan(ret[0])
        assert np.isnan(ret[1])

    def test_empty_input(self):
        closes = np.array([100.0, 101.0])
        idx = np.array([], dtype=int)
        ret = compute_1bar_returns(closes, idx)
        assert len(ret) == 0

    def test_single_bar(self):
        closes = np.array([50.0, 55.0])
        idx = np.array([0])
        ret = compute_1bar_returns(closes, idx)
        np.testing.assert_allclose(ret, [0.1], rtol=1e-10)


# ── apply_threshold ─────────────────────────────────────────────────


class TestApplyThreshold:
    def test_zero_threshold_is_sign(self):
        pred = np.array([0.01, -0.02, 0.0, 0.005, -0.001])
        sig = apply_threshold(pred, 0.0)
        expected = np.array([1.0, -1.0, 0.0, 1.0, -1.0])
        np.testing.assert_array_equal(sig, expected)

    def test_threshold_filters_small(self):
        pred = np.array([0.01, -0.02, 0.0005, -0.0003, 0.003])
        sig = apply_threshold(pred, 0.001)
        expected = np.array([1.0, -1.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(sig, expected)

    def test_all_flat_with_large_threshold(self):
        pred = np.array([0.001, -0.001, 0.0005])
        sig = apply_threshold(pred, 1.0)
        np.testing.assert_array_equal(sig, [0.0, 0.0, 0.0])

    def test_output_only_minus1_0_plus1(self):
        rng = np.random.default_rng(42)
        pred = rng.normal(0, 0.01, 1000)
        sig = apply_threshold(pred, 0.005)
        unique = set(sig)
        assert unique <= {-1.0, 0.0, 1.0}


# ── compute_signal_costs ────────────────────────────────────────────


class TestComputeSignalCosts:
    def test_no_trades_zero_cost(self):
        signal = np.array([0.0, 0.0, 0.0])
        costs = compute_signal_costs(signal)
        np.testing.assert_array_equal(costs, [0.0, 0.0, 0.0])

    def test_hold_position_zero_cost(self):
        signal = np.array([1.0, 1.0, 1.0, 1.0])
        costs = compute_signal_costs(signal)
        # fee=4bps + slip=2bps = 6bps per leg
        assert costs[0] == pytest.approx(0.0006)
        np.testing.assert_array_equal(costs[1:], [0.0, 0.0, 0.0])

    def test_reversal_two_legs(self):
        signal = np.array([1.0, -1.0])
        costs = compute_signal_costs(signal, fee_bps=4.0, slippage_bps=2.0)
        assert costs[0] == pytest.approx(0.0006)  # open long
        assert costs[1] == pytest.approx(0.0012)  # reversal = 2 legs

    def test_open_close_one_leg_each(self):
        signal = np.array([0.0, 1.0, 0.0])
        costs = compute_signal_costs(signal, fee_bps=8.0, slippage_bps=2.0)
        # leg cost = (8+2)/10000 = 0.001
        np.testing.assert_array_equal(costs, [0.0, 0.001, 0.001])

    def test_costs_nonnegative(self):
        rng = np.random.default_rng(123)
        signal = rng.choice([-1.0, 0.0, 1.0], size=500)
        costs = compute_signal_costs(signal)
        assert np.all(costs >= 0)

    def test_zero_fees_zero_costs(self):
        signal = np.array([0.0, 1.0, -1.0, 0.0])
        costs = compute_signal_costs(signal, fee_bps=0.0, slippage_bps=0.0)
        np.testing.assert_array_equal(costs, [0.0, 0.0, 0.0, 0.0])

    def test_separate_fee_and_slippage(self):
        signal = np.array([0.0, 1.0, 0.0])
        # fee only
        c1 = compute_signal_costs(signal, fee_bps=4.0, slippage_bps=0.0)
        # slippage only
        c2 = compute_signal_costs(signal, fee_bps=0.0, slippage_bps=4.0)
        # Both should produce the same costs (additive model)
        np.testing.assert_allclose(c1, c2)
        # Combined = sum
        c3 = compute_signal_costs(signal, fee_bps=4.0, slippage_bps=4.0)
        np.testing.assert_allclose(c3, 2 * c1)


# ── _group_trades ──────────────────────────────────────────────────


class TestGroupTrades:
    def test_single_long_trade(self):
        signal = np.array([1.0, 1.0, 1.0, 0.0])
        net_pnl = np.array([0.01, 0.02, -0.005, 0.0])
        trades = _group_trades(signal, net_pnl)
        assert len(trades) == 1
        assert trades[0] == pytest.approx(0.025)

    def test_reversal_creates_two_trades(self):
        signal = np.array([1.0, 1.0, -1.0, -1.0])
        net_pnl = np.array([0.01, 0.02, -0.01, 0.005])
        trades = _group_trades(signal, net_pnl)
        assert len(trades) == 2
        assert trades[0] == pytest.approx(0.03)  # long: 0.01 + 0.02
        assert trades[1] == pytest.approx(-0.005)  # short: -0.01 + 0.005

    def test_flat_in_between(self):
        signal = np.array([1.0, 0.0, 0.0, -1.0])
        net_pnl = np.array([0.01, 0.0, 0.0, -0.005])
        trades = _group_trades(signal, net_pnl)
        assert len(trades) == 2
        assert trades[0] == pytest.approx(0.01)
        assert trades[1] == pytest.approx(-0.005)

    def test_all_flat_no_trades(self):
        signal = np.array([0.0, 0.0, 0.0])
        net_pnl = np.array([0.0, 0.0, 0.0])
        trades = _group_trades(signal, net_pnl)
        assert len(trades) == 0

    def test_unclosed_position(self):
        signal = np.array([1.0, 1.0, 1.0])
        net_pnl = np.array([0.01, 0.01, 0.01])
        trades = _group_trades(signal, net_pnl)
        assert len(trades) == 1
        assert trades[0] == pytest.approx(0.03)


# ── evaluate_oos ────────────────────────────────────────────────────


class TestEvaluateOOS:
    def _make_data(self, n=200, seed=42):
        rng = np.random.default_rng(seed)
        y_pred = rng.normal(0, 0.01, n)
        y_test = rng.normal(0, 0.01, n)
        ret_1bar = rng.normal(0, 0.005, n)
        return y_pred, y_test, ret_1bar

    def test_prediction_quality_uses_5bar(self):
        y_pred, y_test, ret_1bar = self._make_data()
        result = evaluate_oos(y_pred, y_test, ret_1bar)
        pq = result["prediction_quality"]
        assert "direction_accuracy" in pq
        assert "ic" in pq
        assert "mse" in pq
        expected_da = float(np.mean(np.sign(y_pred) == np.sign(y_test)))
        assert pq["direction_accuracy"] == pytest.approx(expected_da)

    def test_net_return_le_gross(self):
        """Net return should always be <= gross return (costs are non-negative)."""
        y_pred, y_test, ret_1bar = self._make_data()
        result = evaluate_oos(y_pred, y_test, ret_1bar)
        for row in result["threshold_scan"]:
            assert row["net_return"] <= row["gross_return"] + 1e-15

    def test_threshold_scan_returns_all_thresholds(self):
        y_pred, y_test, ret_1bar = self._make_data()
        thresholds = (0.0, 0.001, 0.005)
        result = evaluate_oos(y_pred, y_test, ret_1bar, thresholds=thresholds)
        assert len(result["threshold_scan"]) == 3
        scanned = [r["threshold"] for r in result["threshold_scan"]]
        assert scanned == [0.0, 0.001, 0.005]

    def test_best_threshold_in_scan(self):
        y_pred, y_test, ret_1bar = self._make_data()
        result = evaluate_oos(y_pred, y_test, ret_1bar)
        best = result["best_threshold"]
        scanned_thrs = [r["threshold"] for r in result["threshold_scan"]]
        assert best in scanned_thrs

    def test_higher_threshold_lower_exposure(self):
        """Higher threshold should result in lower or equal exposure."""
        rng = np.random.default_rng(99)
        y_pred = rng.normal(0, 0.01, 500)
        y_test = rng.normal(0, 0.01, 500)
        ret_1bar = rng.normal(0, 0.005, 500)
        result = evaluate_oos(y_pred, y_test, ret_1bar, thresholds=(0.0, 0.001, 0.01, 0.1))
        exposures = [r["exposure"] for r in result["threshold_scan"]]
        for i in range(1, len(exposures)):
            assert exposures[i] <= exposures[i - 1] + 1e-10

    def test_zero_fees_net_equals_gross(self):
        """With zero cost, net should equal gross."""
        y_pred, y_test, ret_1bar = self._make_data(n=100)
        result = evaluate_oos(y_pred, y_test, ret_1bar, fee_bps=0.0, slippage_bps=0.0)
        for row in result["threshold_scan"]:
            assert row["net_return"] == pytest.approx(row["gross_return"], abs=1e-15)
            assert row["total_costs"] == pytest.approx(0.0)

    def test_nan_in_ret_1bar_handled(self):
        """NaN in ret_1bar should be excluded, not crash."""
        y_pred = np.array([0.01, -0.01, 0.02, -0.02])
        y_test = np.array([0.01, -0.01, 0.02, -0.02])
        ret_1bar = np.array([0.005, np.nan, 0.01, -0.01])
        result = evaluate_oos(y_pred, y_test, ret_1bar)
        assert result["prediction_quality"]["direction_accuracy"] == pytest.approx(1.0)

    def test_sharpe_uses_active_bars_only(self):
        """Sharpe should not be inflated by flat bars."""
        # Construct data where signal is active on ~50% of bars
        n = 400
        rng = np.random.default_rng(77)
        y_pred = rng.normal(0, 0.01, n)
        y_test = rng.normal(0, 0.01, n)
        ret_1bar = rng.normal(0, 0.005, n)

        result = evaluate_oos(y_pred, y_test, ret_1bar, thresholds=(0.005,))
        row = result["threshold_scan"][0]

        # Recompute Sharpe manually on active bars
        signal = apply_threshold(y_pred, 0.005)
        valid = ~np.isnan(ret_1bar)
        signal_v = signal[valid]
        costs = compute_signal_costs(signal_v, fee_bps=4.0, slippage_bps=2.0)
        net_pnl = signal_v * ret_1bar[valid] - costs
        active = signal_v != 0
        if np.sum(active) > 1:
            active_net = net_pnl[active]
            expected_sharpe = (np.mean(active_net) / np.std(active_net, ddof=1)) * np.sqrt(8760)
        else:
            expected_sharpe = 0.0
        assert row["sharpe_annual"] == pytest.approx(expected_sharpe, rel=1e-8)

    def test_win_rate_is_trade_level(self):
        """Win rate should count trades, not bars."""
        # 1 winning trade (3 bars), 1 losing trade (2 bars)
        y_pred = np.array([0.1, 0.1, 0.1, -0.1, -0.1])
        y_test = np.array([0.01, 0.01, 0.01, -0.01, -0.01])
        ret_1bar = np.array([0.01, 0.01, 0.01, -0.02, -0.02])

        result = evaluate_oos(y_pred, y_test, ret_1bar, thresholds=(0.0,),
                              fee_bps=0.0, slippage_bps=0.0)
        row = result["threshold_scan"][0]
        # Trade 1 (long): 0.01+0.01+0.01 = 0.03 > 0 → win
        # Trade 2 (short): -(-0.02) + -(-0.02) = 0.04 > 0 → win
        # Wait — signal is -1 on those bars, so net_pnl = -1 * -0.02 = 0.02 each
        # Trade 2 PnL = 0.02 + 0.02 = 0.04 → win
        # Both trades win → 100%
        assert row["win_rate"] == pytest.approx(1.0)

    def test_win_rate_trade_level_mixed(self):
        """Mixed winning and losing trades."""
        y_pred = np.array([0.1, 0.1, 0.0, -0.1, -0.1])
        y_test = np.array([0.01, 0.01, 0.0, -0.01, -0.01])
        ret_1bar = np.array([0.01, -0.03, 0.0, 0.01, 0.01])

        result = evaluate_oos(y_pred, y_test, ret_1bar, thresholds=(0.0,),
                              fee_bps=0.0, slippage_bps=0.0)
        row = result["threshold_scan"][0]
        # Trade 1 (long): 0.01 + (-0.03) = -0.02 → loss
        # Trade 2 (short): -1*0.01 + -1*0.01 = -0.02 → loss
        assert row["win_rate"] == pytest.approx(0.0)
