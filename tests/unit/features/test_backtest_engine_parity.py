"""Parity tests: C++ backtest engine vs Python implementation.

Tests each component individually and the full pipeline end-to-end.
"""
import json
import time

import numpy as np
import pytest

from _quant_hotpath import cpp_pred_to_signal, cpp_run_backtest
from scripts.backtest_alpha_v8 import (
    _apply_dd_breaker,
    _apply_monthly_gate,
    _compute_bear_mask,
    _pred_to_signal,
)

# ── Fixtures ──────────────────────────────────────────────────

def _make_synthetic_data(n=5000, seed=42):
    rng = np.random.RandomState(seed)
    base = 40000.0
    returns = rng.normal(0.0002, 0.005, n)
    closes = base * np.cumprod(1 + returns)
    volumes = rng.uniform(100, 1000, n)
    vol_20 = np.full(n, 0.005)
    y_pred = rng.normal(0.0, 0.01, n)
    # Timestamps: hourly from 2023-01-01
    ts = np.arange(n, dtype=np.int64) * 3600_000 + 1672531200000
    # Funding: every 8h
    n_fund = n // 8
    funding_ts = np.arange(n_fund, dtype=np.int64) * 8 * 3600_000 + 1672531200000
    funding_rates = rng.normal(0.0001, 0.0005, n_fund)
    return ts, closes, volumes, vol_20, y_pred, funding_ts, funding_rates


# ── Component 1: pred_to_signal ───────────────────────────────

class TestPredToSignal:
    def test_parity_default(self):
        rng = np.random.RandomState(123)
        y_pred = rng.normal(0, 0.01, 3000)
        py_sig = _pred_to_signal(y_pred, deadzone=0.5, min_hold=24, zscore_window=720)
        cpp_sig = np.asarray(cpp_pred_to_signal(y_pred, 0.5, 24, 720, 180))
        np.testing.assert_allclose(cpp_sig, py_sig, atol=1e-10)

    def test_parity_small_window(self):
        rng = np.random.RandomState(456)
        y_pred = rng.normal(0, 0.02, 1000)
        py_sig = _pred_to_signal(y_pred, deadzone=0.3, min_hold=12, zscore_window=200)
        cpp_sig = np.asarray(cpp_pred_to_signal(y_pred, 0.3, 12, 200, 180))
        np.testing.assert_allclose(cpp_sig, py_sig, atol=1e-10)

    def test_parity_large(self):
        rng = np.random.RandomState(789)
        y_pred = rng.normal(0, 0.015, 13000)
        py_sig = _pred_to_signal(y_pred, deadzone=0.5, min_hold=24, zscore_window=720)
        cpp_sig = np.asarray(cpp_pred_to_signal(y_pred, 0.5, 24, 720, 180))
        np.testing.assert_allclose(cpp_sig, py_sig, atol=1e-10)

    def test_all_zeros(self):
        y_pred = np.zeros(500)
        py_sig = _pred_to_signal(y_pred)
        cpp_sig = np.asarray(cpp_pred_to_signal(y_pred, 0.5, 24, 720, 180))
        np.testing.assert_allclose(cpp_sig, py_sig, atol=1e-10)

    def test_constant_pred(self):
        y_pred = np.full(1000, 0.05)
        py_sig = _pred_to_signal(y_pred)
        cpp_sig = np.asarray(cpp_pred_to_signal(y_pred, 0.5, 24, 720, 180))
        np.testing.assert_allclose(cpp_sig, py_sig, atol=1e-10)


# ── Component 2: bear_mask / monthly_gate ─────────────────────

class TestBearMask:
    def test_bear_mask_parity(self):
        rng = np.random.RandomState(42)
        closes = 40000 * np.cumprod(1 + rng.normal(0, 0.005, 2000))
        py_mask = _compute_bear_mask(closes, ma_window=480)
        # C++ bear mask is tested indirectly via monthly gate
        py_gated = _apply_monthly_gate(np.ones(2000), closes, ma_window=480)
        # In monthly gate, signal is zeroed where bear_mask is True
        expected = np.where(py_mask, 0.0, 1.0)
        np.testing.assert_allclose(py_gated, expected, atol=1e-10)

    def test_monthly_gate_cpp(self):
        rng = np.random.RandomState(42)
        n = 2000
        closes = 40000 * np.cumprod(1 + rng.normal(0, 0.005, n))
        y_pred = rng.normal(0, 0.01, n)
        ts = np.arange(n, dtype=np.int64) * 3600_000 + 1672531200000
        cfg = {"monthly_gate": True, "ma_window": 480, "min_hold": 1,
               "deadzone": 0.5, "zscore_window": 720}
        result = cpp_run_backtest(
            ts, closes, np.empty(0), np.empty(0),
            y_pred, np.empty(0), np.empty(0),
            np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64),
            json.dumps(cfg))
        cpp_sig = np.asarray(result["signal"])
        # Verify bear bars have zero signal
        bear_mask = _compute_bear_mask(closes, 480)
        assert np.all(cpp_sig[bear_mask] == 0.0) or np.sum(bear_mask) == 0


# ── Component 3: DD breaker ──────────────────────────────────

class TestDDBreaker:
    def test_dd_breaker_triggers(self):
        # Create signal that will trigger drawdown
        n = 500
        signal = np.ones(n)
        # Closes that drop sharply
        closes = np.concatenate([
            np.linspace(100, 80, 200),  # -20% drop
            np.linspace(80, 90, 300),
        ])
        py_out = _apply_dd_breaker(signal.copy(), closes, dd_limit=-0.10, cooldown=48)
        # Verify some bars are zeroed
        assert np.sum(py_out == 0.0) > 0


# ── Component 4: full backtest ────────────────────────────────

class TestFullBacktest:
    def test_flat_cost_parity(self):
        ts, closes, volumes, vol_20, y_pred, f_ts, f_rates = _make_synthetic_data(3000)
        cfg = {"deadzone": 0.5, "min_hold": 24, "zscore_window": 720,
               "cost_per_trade": 0.0006}
        result = cpp_run_backtest(
            ts, closes, np.empty(0), np.empty(0),
            y_pred, np.empty(0), np.empty(0),
            np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64),
            json.dumps(cfg))
        # Compare signal with Python
        py_sig = _pred_to_signal(y_pred, deadzone=0.5, min_hold=24, zscore_window=720)
        cpp_sig = np.asarray(result["signal"])
        np.testing.assert_allclose(cpp_sig, py_sig, atol=1e-10)

        # Verify Sharpe is computed
        assert isinstance(result["sharpe"], float)
        assert isinstance(result["max_drawdown"], float)
        assert isinstance(result["total_return"], float)

    def test_flat_cost_pnl_parity(self):
        """Full PnL parity: Python vs C++ with flat cost, no funding."""
        ts, closes, volumes, vol_20, y_pred, f_ts, f_rates = _make_synthetic_data(3000)

        # Python path
        py_sig = _pred_to_signal(y_pred, deadzone=0.5, min_hold=24, zscore_window=720)
        ret_1bar = np.diff(closes) / closes[:-1]
        sig_trade = py_sig[:len(ret_1bar)]
        gross = sig_trade * ret_1bar
        turnover = np.abs(np.diff(sig_trade, prepend=0))
        cost = turnover * 6e-4
        net = gross - cost
        equity_py = np.ones(len(net) + 1) * 10000.0
        for i in range(len(net)):
            equity_py[i + 1] = equity_py[i] * (1 + net[i])

        # C++ path
        cfg = {"deadzone": 0.5, "min_hold": 24, "zscore_window": 720,
               "cost_per_trade": 0.0006, "capital": 10000.0}
        result = cpp_run_backtest(
            ts, closes, np.empty(0), np.empty(0),
            y_pred, np.empty(0), np.empty(0),
            np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64),
            json.dumps(cfg))

        cpp_net = np.asarray(result["net_pnl"])
        cpp_eq = np.asarray(result["equity"])

        np.testing.assert_allclose(cpp_net, net, atol=1e-12)
        np.testing.assert_allclose(cpp_eq, equity_py, atol=1e-6)

    def test_with_funding_parity(self):
        """PnL parity with funding costs."""
        ts, closes, volumes, vol_20, y_pred, f_ts, f_rates = _make_synthetic_data(2000)

        # Python: manual funding merge
        py_sig = _pred_to_signal(y_pred, deadzone=0.5, min_hold=24, zscore_window=720)
        ret_1bar = np.diff(closes) / closes[:-1]
        sig_trade = py_sig[:len(ret_1bar)]
        gross = sig_trade * ret_1bar
        turnover = np.abs(np.diff(sig_trade, prepend=0))
        cost = turnover * 6e-4

        # Funding merge (Python)
        funding_cost = np.zeros(len(sig_trade))
        f_sorted = sorted(zip(f_ts, f_rates))
        f_idx = 0
        current_rate = 0.0
        for i in range(len(sig_trade)):
            bar_ts = ts[i]
            while f_idx < len(f_sorted) and f_sorted[f_idx][0] <= bar_ts:
                current_rate = f_sorted[f_idx][1]
                f_idx += 1
            if sig_trade[i] != 0.0:
                funding_cost[i] = sig_trade[i] * current_rate / 8.0
        net = gross - cost - funding_cost

        # C++ path
        # Sort funding by ts for C++
        sort_idx = np.argsort(f_ts)
        cfg = {"deadzone": 0.5, "min_hold": 24, "zscore_window": 720,
               "cost_per_trade": 0.0006, "capital": 10000.0}
        result = cpp_run_backtest(
            ts, closes, np.empty(0), np.empty(0),
            y_pred, np.empty(0), np.empty(0),
            f_rates[sort_idx].astype(np.float64),
            f_ts[sort_idx].astype(np.int64),
            json.dumps(cfg))

        cpp_net = np.asarray(result["net_pnl"])
        np.testing.assert_allclose(cpp_net, net, atol=1e-12)

    def test_realistic_cost_parity(self):
        """Realistic cost model parity."""
        from execution.sim.cost_model import RealisticCostModel

        ts, closes, volumes, vol_20, y_pred, f_ts, f_rates = _make_synthetic_data(2000)

        # Python path
        py_sig = _pred_to_signal(y_pred, deadzone=0.5, min_hold=24, zscore_window=720)
        cm = RealisticCostModel()
        ret_1bar = np.diff(closes) / closes[:-1]
        sig_trade = py_sig[:len(ret_1bar)]
        breakdown = cm.compute_costs(sig_trade, closes[:len(sig_trade)],
                                     volumes[:len(sig_trade)], vol_20[:len(sig_trade)])
        py_cost = breakdown.total_cost
        py_clipped = breakdown.clipped_signal
        gross = py_clipped * ret_1bar
        net = gross - py_cost

        # C++ path
        cfg = {"deadzone": 0.5, "min_hold": 24, "zscore_window": 720,
               "realistic_cost": True, "capital": 10000.0}
        result = cpp_run_backtest(
            ts, closes, volumes, vol_20,
            y_pred, np.empty(0), np.empty(0),
            np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64),
            json.dumps(cfg))

        cpp_sig = np.asarray(result["signal"])
        cpp_net = np.asarray(result["net_pnl"])
        # C++ signal is full length (n_pred), Python clipped signal is n_pred-1
        np.testing.assert_allclose(cpp_sig[:len(py_clipped)], py_clipped, atol=1e-10)
        np.testing.assert_allclose(cpp_net, net, atol=1e-10)

    def test_monthly_gate_parity(self):
        """Monthly gate parity: uses unified single-pass semantics.

        The Rust backtest now matches the live path: min-hold runs first,
        then gate overrides the output (bypassing min-hold protection).
        The Python reference is computed with the same interleaved approach.
        """
        ts, closes, volumes, vol_20, y_pred, f_ts, f_rates = _make_synthetic_data(3000)
        from shared.signal_postprocess import rolling_zscore

        # Step 1: rolling z-score → discretize (same as Rust zscore_discretize_array)
        z = rolling_zscore(y_pred, window=720, warmup=180)
        raw = np.where(z > 0.5, 1.0, np.where(z < -0.5, -1.0, 0.0))

        # Step 2: bear mask
        bear_mask = _compute_bear_mask(closes, ma_window=480)

        # Step 3: single-pass min-hold + gate override (matches live)
        n = len(raw)
        signal = np.zeros(n)
        signal[0] = raw[0]
        if bear_mask[0]:
            signal[0] = 0.0
        hold_count = 1
        for i in range(1, n):
            desired = raw[i]
            # min-hold enforcement
            if hold_count < 24:
                sig = signal[i - 1]
                hold_count += 1
            else:
                sig = desired
                if desired != signal[i - 1]:
                    hold_count = 1
                else:
                    hold_count += 1
            # gate override (bypasses min-hold)
            if bear_mask[i]:
                if sig != 0.0:
                    sig = 0.0
                    hold_count = 1
            signal[i] = sig

        cfg = {"deadzone": 0.5, "min_hold": 24, "zscore_window": 720,
               "monthly_gate": True, "ma_window": 480, "cost_per_trade": 0.0006}
        result = cpp_run_backtest(
            ts, closes, np.empty(0), np.empty(0),
            y_pred, np.empty(0), np.empty(0),
            np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64),
            json.dumps(cfg))
        cpp_sig = np.asarray(result["signal"])
        np.testing.assert_allclose(cpp_sig, signal, atol=1e-10)

    def test_metrics_monthly_breakdown(self):
        ts, closes, volumes, vol_20, y_pred, f_ts, f_rates = _make_synthetic_data(5000)
        cfg = {"deadzone": 0.5, "min_hold": 24, "zscore_window": 720,
               "cost_per_trade": 0.0006}
        result = cpp_run_backtest(
            ts, closes, np.empty(0), np.empty(0),
            y_pred, np.empty(0), np.empty(0),
            np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64),
            json.dumps(cfg))
        monthly = result["monthly"]
        assert len(monthly) > 0
        for m in monthly:
            assert "month" in m
            assert "return" in m
            assert "sharpe" in m

    def test_metrics_sharpe_parity(self):
        """Sharpe ratio parity between Python and C++."""
        ts, closes, volumes, vol_20, y_pred, f_ts, f_rates = _make_synthetic_data(3000)

        # Python
        py_sig = _pred_to_signal(y_pred, deadzone=0.5, min_hold=24, zscore_window=720)
        ret_1bar = np.diff(closes) / closes[:-1]
        sig_trade = py_sig[:len(ret_1bar)]
        turnover = np.abs(np.diff(sig_trade, prepend=0))
        cost = turnover * 6e-4
        net = sig_trade * ret_1bar - cost
        active = sig_trade != 0
        n_active = int(active.sum())
        if n_active > 1:
            active_pnl = net[active]
            std_a = float(np.std(active_pnl, ddof=1))
            py_sharpe = float(np.mean(active_pnl)) / std_a * np.sqrt(8760) if std_a > 0 else 0.0
        else:
            py_sharpe = 0.0

        # C++
        cfg = {"deadzone": 0.5, "min_hold": 24, "zscore_window": 720,
               "cost_per_trade": 0.0006, "capital": 10000.0}
        result = cpp_run_backtest(
            ts, closes, np.empty(0), np.empty(0),
            y_pred, np.empty(0), np.empty(0),
            np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64),
            json.dumps(cfg))
        np.testing.assert_allclose(result["sharpe"], py_sharpe, atol=1e-6)


# ── Performance ───────────────────────────────────────────────

class TestPerformance:
    @pytest.mark.benchmark
    def test_speed_vs_python(self):
        ts, closes, volumes, vol_20, y_pred, f_ts, f_rates = _make_synthetic_data(13000)
        cfg_json = json.dumps({"deadzone": 0.5, "min_hold": 24, "zscore_window": 720,
                               "cost_per_trade": 0.0006, "capital": 10000.0})

        # Warm up
        cpp_run_backtest(ts, closes, np.empty(0), np.empty(0),
                         y_pred, np.empty(0), np.empty(0),
                         np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64),
                         cfg_json)

        # C++ timing
        t0 = time.perf_counter()
        for _ in range(10):
            cpp_run_backtest(ts, closes, np.empty(0), np.empty(0),
                             y_pred, np.empty(0), np.empty(0),
                             np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64),
                             cfg_json)
        cpp_ms = (time.perf_counter() - t0) / 10 * 1000

        # Python timing
        t0 = time.perf_counter()
        for _ in range(3):
            sig = _pred_to_signal(y_pred, deadzone=0.5, min_hold=24, zscore_window=720)
            ret_1bar = np.diff(closes) / closes[:-1]
            sig_trade = sig[:len(ret_1bar)]
            turnover = np.abs(np.diff(sig_trade, prepend=0))
            cost = turnover * 6e-4
            sig_trade * ret_1bar - cost
        py_ms = (time.perf_counter() - t0) / 3 * 1000

        speedup = py_ms / max(cpp_ms, 0.001)
        print(f"\nC++: {cpp_ms:.2f}ms, Python: {py_ms:.2f}ms, Speedup: {speedup:.1f}x")
        assert cpp_ms < py_ms, f"C++ ({cpp_ms:.2f}ms) should be faster than Python ({py_ms:.2f}ms)"
