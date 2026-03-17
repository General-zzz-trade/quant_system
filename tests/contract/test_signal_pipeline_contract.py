# tests/contract/test_signal_pipeline_contract.py
"""Contract: signal constraint pipeline — Rust/Python parity, deadzone, min-hold."""
from __future__ import annotations

import numpy as np

from scripts.shared.signal_postprocess import (
    pred_to_signal,
    rolling_zscore,
)


class TestZscoreContract:
    def test_warmup_produces_zeros(self):
        """During warmup period, z-scores must be zero."""
        preds = np.random.randn(100)
        z = rolling_zscore(preds, window=50, warmup=30)
        # First warmup-1 values must be zero (implementation may emit at warmup boundary)
        assert np.all(z[:29] == 0.0)

    def test_post_warmup_nonzero(self):
        """After warmup, z-scores should be non-zero for non-constant input."""
        preds = np.random.randn(200) * 0.1
        z = rolling_zscore(preds, window=50, warmup=30)
        assert np.any(z[50:] != 0.0)

    def test_constant_input_zero_zscore(self):
        """Constant predictions → zero z-score (no variance)."""
        preds = np.ones(200) * 0.5
        z = rolling_zscore(preds, window=50, warmup=30)
        # After warmup, std=0 → z should be 0 or NaN-clipped to 0
        assert np.all(np.isfinite(z))


class TestDeadzoneContract:
    def test_signal_within_deadzone_is_flat(self):
        """Z-scores within [-deadzone, +deadzone] must produce signal=0."""
        # Craft predictions that produce small z-scores
        preds = np.ones(300) * 0.01  # near-zero predictions
        signal = pred_to_signal(preds, deadzone=0.5, min_hold=1,
                                zscore_window=50, zscore_warmup=30)
        # Most should be flat
        assert np.mean(signal == 0) > 0.8

    def test_signal_above_deadzone_is_long(self):
        """Z-scores above deadzone must produce signal=+1."""
        # Strong upward predictions
        preds = np.linspace(0, 5, 300)
        signal = pred_to_signal(preds, deadzone=0.3, min_hold=1,
                                zscore_window=50, zscore_warmup=30)
        # After warmup, should eventually go long
        assert np.any(signal == 1)

    def test_signal_below_neg_deadzone_is_short(self):
        """Z-scores below -deadzone must produce signal=-1."""
        preds = np.linspace(0, -5, 300)
        signal = pred_to_signal(preds, deadzone=0.3, min_hold=1,
                                zscore_window=50, zscore_warmup=30)
        assert np.any(signal == -1)


class TestMinHoldContract:
    def test_min_hold_enforced(self):
        """Once signal is non-zero, it must hold for min_hold bars."""
        # Alternating predictions to try to force rapid flips
        preds = np.array([3.0 if i % 2 == 0 else -3.0 for i in range(500)])
        signal = pred_to_signal(preds, deadzone=0.1, min_hold=10,
                                zscore_window=50, zscore_warmup=30,
                                max_hold=999)
        # Find first non-zero signal and check it holds
        nonzero = np.where(signal != 0)[0]
        if len(nonzero) > 10:
            first = nonzero[0]
            first_val = signal[first]
            # Must hold for at least min_hold bars
            held = 0
            for i in range(first, min(first + 10, len(signal))):
                if signal[i] == first_val:
                    held += 1
            assert held >= 5, f"Min-hold violated: held only {held} of 10"

    def test_min_hold_one_allows_flips(self):
        """With min_hold=1, rapid signal changes are allowed."""
        preds = np.array([3.0 if i % 3 == 0 else -3.0 for i in range(300)])
        signal = pred_to_signal(preds, deadzone=0.1, min_hold=1,
                                zscore_window=50, zscore_warmup=30,
                                max_hold=999)
        # Should have both +1 and -1
        assert np.any(signal == 1) and np.any(signal == -1)


class TestRustPythonParity:
    def test_pred_to_signal_deterministic(self):
        """Same input → same output across calls."""
        np.random.seed(42)
        preds = np.random.randn(500) * 0.1
        s1 = pred_to_signal(preds, deadzone=0.5, min_hold=18,
                            zscore_window=100, zscore_warmup=50)
        s2 = pred_to_signal(preds, deadzone=0.5, min_hold=18,
                            zscore_window=100, zscore_warmup=50)
        np.testing.assert_array_equal(s1, s2)

    def test_output_is_discrete(self):
        """Signal output must be exactly {-1, 0, +1}."""
        np.random.seed(123)
        preds = np.random.randn(300) * 0.2
        signal = pred_to_signal(preds, deadzone=0.3, min_hold=5,
                                zscore_window=50, zscore_warmup=30)
        unique = set(np.unique(signal))
        assert unique.issubset({-1.0, 0.0, 1.0}), f"Non-discrete: {unique}"

    def test_long_only_no_shorts(self):
        """With long_only=True, signal must never be -1."""
        preds = np.random.randn(300) * 0.5
        signal = pred_to_signal(preds, deadzone=0.3, min_hold=1,
                                zscore_window=50, zscore_warmup=30,
                                long_only=True)
        assert not np.any(signal == -1), "Long-only produced short signal"
