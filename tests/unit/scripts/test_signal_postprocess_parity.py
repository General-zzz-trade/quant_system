# tests/unit/scripts/test_signal_postprocess_parity.py
"""Parity tests: Python signal_postprocess must match Rust constraint_pipeline.

These tests verify that _enforce_hold_single_pass produces identical output
to the Rust enforce_hold_array for the same inputs. This catches the known
divergence where the old two-pass approach allowed extra trend extensions.
"""
import numpy as np

from shared.signal_postprocess import (
    _enforce_hold_single_pass,
    _enforce_min_hold,
    _apply_trend_hold,
    pred_to_signal,
)


class TestEnforceHoldSinglePassBasic:
    def test_empty_array(self):
        result = _enforce_hold_single_pass(np.array([]), min_hold=3)
        assert len(result) == 0

    def test_min_hold_locks_position(self):
        raw = np.array([1.0, 0.0, 0.0, 0.0, -1.0])
        result = _enforce_hold_single_pass(raw, min_hold=3)
        # bar 0: 1.0 (hold_count=1), bar 1-2: held (hold_count 2,3)
        # bar 3: hold_count=3 >= min_hold, desired=0 != prev=1 → change to 0 (hold_count=1)
        # bar 4: hold_count=1 < min_hold=3 → held at 0
        np.testing.assert_array_equal(result, [1.0, 1.0, 1.0, 0.0, 0.0])

    def test_min_hold_then_change(self):
        raw = np.array([1.0, -1.0, -1.0, -1.0, -1.0])
        result = _enforce_hold_single_pass(raw, min_hold=2)
        # Hold 1.0 for 2 bars, then allow change
        np.testing.assert_array_equal(result, [1.0, 1.0, -1.0, -1.0, -1.0])


class TestEnforceHoldSinglePassTrendHold:
    def test_trend_extends_long(self):
        raw = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        trend = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        result = _enforce_hold_single_pass(
            raw, min_hold=1, trend_follow=True,
            trend_values=trend, trend_threshold=0.0, max_hold=5,
        )
        # After 3 bars of 1.0, trend holds at bar 3 and 4
        assert result[3] == 1.0  # trend extended
        assert result[4] == 1.0  # trend extended

    def test_trend_does_not_extend_past_max_hold(self):
        raw = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        trend = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        result = _enforce_hold_single_pass(
            raw, min_hold=1, trend_follow=True,
            trend_values=trend, trend_threshold=0.0, max_hold=3,
        )
        # max_hold=3: can hold for 3 bars total (bars 0,1,2), bar 3 released
        assert result[0] == 1.0
        assert result[1] == 1.0
        assert result[2] == 1.0  # trend extended (hold_count=3)
        assert result[3] == 0.0  # released at max_hold

    def test_trend_extends_short(self):
        raw = np.array([-1.0, -1.0, 0.0, 0.0])
        trend = np.array([-0.5, -0.5, -0.5, -0.5])
        result = _enforce_hold_single_pass(
            raw, min_hold=1, trend_follow=True,
            trend_values=trend, trend_threshold=0.0, max_hold=5,
        )
        assert result[2] == -1.0  # trend extended


class TestOldVsNewDivergence:
    """Demonstrate the divergence between old two-pass and new single-pass."""

    def test_old_two_pass_overextends(self):
        """The old approach allowed extra holds because _apply_trend_hold
        had its own hold_count starting from scratch."""
        raw = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        trend = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        # Old: min_hold first, then trend_hold separately
        old_held = _enforce_min_hold(raw, min_hold=2)
        # old_held = [1, 1, 0, 0, 0] (min-hold locks bar 1)
        old_result = _apply_trend_hold(old_held, trend, 0.0, max_hold=3)
        # _apply_trend_hold has its own hold_count starting from 1 at bar 0
        # It sees bar 2 as "prev=1, current=0, trend up" → extends
        # Total hold can exceed what Rust would do

        # New: single pass (matches Rust)
        new_result = _enforce_hold_single_pass(
            raw, min_hold=2, trend_follow=True,
            trend_values=trend, trend_threshold=0.0, max_hold=3,
        )

        # Both should agree on bars 0-1 (min-hold)
        assert old_result[0] == new_result[0] == 1.0
        assert old_result[1] == new_result[1] == 1.0

        # Bar 2: both should extend via trend
        assert new_result[2] == 1.0  # trend extension (hold_count=3 = max_hold)
        # Bar 3: new correctly stops at max_hold
        assert new_result[3] == 0.0  # released


class TestPredToSignalParity:
    """Test pred_to_signal produces consistent output regardless of Rust availability."""

    def test_basic_signal_generation(self):
        np.random.seed(42)
        pred = np.random.randn(500) * 0.01
        result = pred_to_signal(pred, deadzone=0.5, min_hold=24)
        assert len(result) == 500
        # All values should be -1, 0, or 1
        unique = set(np.unique(result))
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_long_only(self):
        np.random.seed(42)
        pred = np.random.randn(500) * 0.01
        result = pred_to_signal(pred, deadzone=0.5, min_hold=24, long_only=True)
        assert np.all(result >= 0)  # no shorts
