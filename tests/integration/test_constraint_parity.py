"""Constraint parity tests — verify live (bar-by-bar) vs batch (array) signal identity.

Ensures that RustInferenceBridge.apply_constraints() (bar-by-bar, used in live)
produces identical results to cpp_pred_to_signal() (batch, used in backtests).

Marks: slow (requires _quant_hotpath).
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, "/quant_system")

pytestmark = [pytest.mark.slow]

_hp = pytest.importorskip("_quant_hotpath")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar_by_bar_constraints(
    preds: list[float],
    *,
    deadzone: float = 0.5,
    min_hold: int = 24,
    zscore_window: int = 720,
    zscore_warmup: int = 180,
    long_only: bool = False,
    trend_follow: bool = False,
    trend_values: list[float] | None = None,
    trend_threshold: float = 0.0,
    max_hold: int = 120,
) -> list[float]:
    """Run predictions through RustInferenceBridge bar-by-bar (live path)."""
    bridge = _hp.RustInferenceBridge(
        zscore_window=zscore_window,
        zscore_warmup=zscore_warmup,
    )
    results = []
    for i, pred in enumerate(preds):
        tv = float("nan")
        if trend_follow and trend_values is not None and i < len(trend_values):
            tv = trend_values[i]
        sig = bridge.apply_constraints(
            "TEST",
            float(pred),
            i,  # hour_key = bar index (unique per bar)
            deadzone=deadzone,
            min_hold=min_hold,
            long_only=long_only,
            trend_follow=trend_follow,
            trend_val=tv,
            trend_threshold=trend_threshold,
            max_hold=max_hold,
        )
        results.append(float(sig))
    return results


def _batch_constraints(
    preds: list[float],
    *,
    deadzone: float = 0.5,
    min_hold: int = 24,
    zscore_window: int = 720,
    zscore_warmup: int = 180,
    long_only: bool = False,
    trend_follow: bool = False,
    trend_values: list[float] | None = None,
    trend_threshold: float = 0.0,
    max_hold: int = 120,
) -> list[float]:
    """Run predictions through cpp_pred_to_signal (batch path)."""
    tv = trend_values if trend_values is not None else []
    return list(_hp.cpp_pred_to_signal(
        preds,
        deadzone,
        min_hold,
        zscore_window,
        zscore_warmup,
        long_only,
        trend_follow,
        tv,
        trend_threshold,
        max_hold,
    ))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLiveVsBatchSignalIdentity:
    """500 random predictions through bar-by-bar vs batch — must be identical."""

    def test_random_normal_preds(self):
        rng = np.random.RandomState(42)
        preds = rng.randn(500).tolist()

        live = _bar_by_bar_constraints(preds, min_hold=12, deadzone=0.5)
        batch = _batch_constraints(preds, min_hold=12, deadzone=0.5)

        np.testing.assert_array_equal(
            np.array(live), np.array(batch),
            err_msg="Live (bar-by-bar) and batch signal paths diverged",
        )

    def test_long_only(self):
        rng = np.random.RandomState(123)
        preds = rng.randn(500).tolist()

        live = _bar_by_bar_constraints(preds, min_hold=12, deadzone=0.5, long_only=True)
        batch = _batch_constraints(preds, min_hold=12, deadzone=0.5, long_only=True)

        np.testing.assert_array_equal(np.array(live), np.array(batch))

    def test_trend_follow(self):
        rng = np.random.RandomState(999)
        preds = rng.randn(500).tolist()
        # Trend values oscillate above/below threshold
        trend_vals = (np.sin(np.linspace(0, 10, 500)) * 0.5).tolist()

        live = _bar_by_bar_constraints(
            preds, min_hold=12, deadzone=0.5,
            trend_follow=True, trend_values=trend_vals, trend_threshold=0.0,
            max_hold=60,
        )
        batch = _batch_constraints(
            preds, min_hold=12, deadzone=0.5,
            trend_follow=True, trend_values=trend_vals, trend_threshold=0.0,
            max_hold=60,
        )

        np.testing.assert_array_equal(np.array(live), np.array(batch))


    def test_trend_follow_short(self):
        """Short trend-hold: bar-by-bar vs batch must be identical for negative signals."""
        rng = np.random.RandomState(777)
        # Generate predictions biased negative to produce short signals
        preds = (rng.randn(500) - 0.5).tolist()
        # Trend values oscillate into negative territory
        trend_vals = (np.sin(np.linspace(0, 10, 500)) * 0.5 - 0.2).tolist()

        live = _bar_by_bar_constraints(
            preds, min_hold=12, deadzone=0.5,
            trend_follow=True, trend_values=trend_vals, trend_threshold=0.0,
            max_hold=60,
        )
        batch = _batch_constraints(
            preds, min_hold=12, deadzone=0.5,
            trend_follow=True, trend_values=trend_vals, trend_threshold=0.0,
            max_hold=60,
        )

        np.testing.assert_array_equal(
            np.array(live), np.array(batch),
            err_msg="Short trend-hold: live vs batch diverged",
        )
        # Verify we actually produced some short signals
        assert any(s < 0 for s in live), "Expected some short signals in test data"


# KNOWN LIMITATION: These parity tests use bar index as hour_key (i.e. hour_key=i),
# which matches 1h-bar backtests. On sub-hourly bars in production, live uses
# ts-based hour_key (int(ts.timestamp())//3600). The Rust bridge handles both
# correctly; the divergence only exists in the Python fallback (_discretize_signal
# in backtest_module.py), which is not tested here. See also: T8 bar-vs-hour docs.

class TestMinHoldTiming:
    """Verify hold counter delays match across both paths."""

    def test_hold_delays_signal_change(self):
        """Known sequence: signal flips should be delayed by min_hold bars."""
        # Create a prediction stream that produces clear signal changes:
        # warmup zeros, then strong positive, then strong negative
        warmup = 180
        min_hold = 24

        preds = [0.0] * warmup  # warmup (z-score outputs 0)
        preds += [1.0] * 50     # strong positive -> signal = +1 after z-score
        preds += [-1.0] * 50    # strong negative -> signal = -1 eventually
        preds += [1.0] * 50     # flip back

        live = _bar_by_bar_constraints(preds, min_hold=min_hold, deadzone=0.3)
        batch = _batch_constraints(preds, min_hold=min_hold, deadzone=0.3)

        live_arr = np.array(live)
        batch_arr = np.array(batch)

        # Both paths must agree exactly
        np.testing.assert_array_equal(live_arr, batch_arr)

        # After warmup + some bars, we should see non-zero signals
        assert np.any(live_arr[warmup + 10:] != 0), "Expected non-zero signals after warmup"

    def test_min_hold_prevents_early_exit(self):
        """Signal that flips immediately should be held for min_hold bars."""
        min_hold = 10

        # After warmup, go long then immediately try to go flat
        preds = [0.0] * 180 + [2.0] * 5 + [0.0] * 30 + [2.0] * 20

        live = _bar_by_bar_constraints(preds, min_hold=min_hold, deadzone=0.3)
        batch = _batch_constraints(preds, min_hold=min_hold, deadzone=0.3)

        np.testing.assert_array_equal(np.array(live), np.array(batch))


class TestPythonFallbackVsRustParity:
    """Verify Python fallback in signal_postprocess matches Rust output."""

    def test_python_fallback_vs_rust_parity(self):
        """500 random preds through Rust vs Python fallback must be close."""
        from unittest.mock import patch
        from scripts import signal_postprocess

        rng = np.random.RandomState(77)
        preds = rng.randn(500)

        # Rust path
        rust_result = signal_postprocess.pred_to_signal(preds, min_hold=12, deadzone=0.5)

        # Force Python fallback
        with patch.object(signal_postprocess, "_HAS_RUST", False):
            py_result = signal_postprocess.pred_to_signal(preds, min_hold=12, deadzone=0.5)

        np.testing.assert_allclose(
            rust_result, py_result, atol=1e-10,
            err_msg="Rust vs Python fallback diverged (default params)",
        )

    def test_long_only_parity(self):
        rng = np.random.RandomState(88)
        preds = rng.randn(500)

        from unittest.mock import patch
        from scripts import signal_postprocess

        rust_result = signal_postprocess.pred_to_signal(preds, min_hold=12, deadzone=0.5, long_only=True)

        with patch.object(signal_postprocess, "_HAS_RUST", False):
            py_result = signal_postprocess.pred_to_signal(preds, min_hold=12, deadzone=0.5, long_only=True)

        np.testing.assert_allclose(rust_result, py_result, atol=1e-10)

    def test_trend_follow_parity(self):
        rng = np.random.RandomState(99)
        preds = rng.randn(500)
        trend_vals = np.sin(np.linspace(0, 10, 500)) * 0.5

        from unittest.mock import patch
        from scripts import signal_postprocess

        rust_result = signal_postprocess.pred_to_signal(
            preds, min_hold=12, deadzone=0.5,
            trend_follow=True, trend_values=trend_vals,
            trend_threshold=0.0, max_hold=60,
        )

        with patch.object(signal_postprocess, "_HAS_RUST", False):
            py_result = signal_postprocess.pred_to_signal(
                preds, min_hold=12, deadzone=0.5,
                trend_follow=True, trend_values=trend_vals,
                trend_threshold=0.0, max_hold=60,
            )

        np.testing.assert_allclose(rust_result, py_result, atol=1e-10)

    def test_various_min_hold(self):
        rng = np.random.RandomState(55)
        preds = rng.randn(500)

        from unittest.mock import patch
        from scripts import signal_postprocess

        for min_hold in [1, 6, 24, 48]:
            rust_result = signal_postprocess.pred_to_signal(preds, min_hold=min_hold, deadzone=0.5)

            with patch.object(signal_postprocess, "_HAS_RUST", False):
                py_result = signal_postprocess.pred_to_signal(preds, min_hold=min_hold, deadzone=0.5)

            np.testing.assert_allclose(
                rust_result, py_result, atol=1e-10,
                err_msg=f"Diverged at min_hold={min_hold}",
            )


class TestMonthlyGateFlip:
    """Verify monthly gate transitions are identical across paths.

    The monthly gate (SMA-based) is handled differently: RustInferenceBridge
    uses check_monthly_gate() while cpp_pred_to_signal doesn't include it.
    Instead, we test that the Python signal_postprocess._compute_bear_mask
    produces correct gate transitions that match expected behavior.
    """

    def test_gate_transitions_consistent(self):
        """Close series crossing SMA(480) produces correct gate flips."""
        from scripts.signal_postprocess import _compute_bear_mask

        n = 1000
        # Price starts above SMA, dips below, then recovers
        closes = np.concatenate([
            np.linspace(100, 110, 500),   # uptrend
            np.linspace(110, 85, 200),    # sharp drop below SMA
            np.linspace(85, 105, 300),    # recovery
        ])
        assert len(closes) == n

        mask = _compute_bear_mask(closes, ma_window=480)

        # First 480 bars: mask should be False (warmup)
        assert not np.any(mask[:480]), "Warmup bars should not be masked"

        # After the drop, some bars should be masked (close < SMA)
        assert np.any(mask[600:700]), "Expected masked bars during price drop"

        # Verify mask is deterministic (run twice)
        mask2 = _compute_bear_mask(closes, ma_window=480)
        np.testing.assert_array_equal(mask, mask2)

    def test_monthly_gate_bridge_deterministic(self):
        """RustInferenceBridge monthly gate is deterministic across two runs."""
        closes = np.concatenate([
            np.linspace(100, 120, 300),
            np.linspace(120, 90, 300),
        ]).tolist()

        def _run_gate():
            bridge = _hp.RustInferenceBridge(zscore_window=720, zscore_warmup=180)
            results = []
            for i, c in enumerate(closes):
                ok = bridge.check_monthly_gate("TEST", c, i, 480)
                results.append(ok)
            return results

        run1 = _run_gate()
        run2 = _run_gate()
        assert run1 == run2, "Monthly gate must be deterministic"
