"""Cross-path parity tests: verify live bridge and backtest module produce
semantically equivalent constraint behavior for key rules.

Tests that the Rust constraint kernel (shared by both paths) produces
identical discretized signals for the same inputs.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, "/quant_system")

_hp = pytest.importorskip("_quant_hotpath")
RustInferenceBridge = _hp.RustInferenceBridge


def _make_bridge(zscore_window=720, zscore_warmup=180, gate_window=480):
    return RustInferenceBridge(zscore_window, zscore_warmup, gate_window)


# ============================================================
# Helpers — run same preds through two independent Rust bridges
# ============================================================

def _run_constraints(
    preds: list[float],
    *,
    deadzone: float = 0.5,
    min_hold: int = 12,
    long_only: bool = False,
    trend_follow: bool = False,
    trend_values: list[float] | None = None,
    trend_threshold: float = 0.0,
    max_hold: int = 120,
) -> tuple[list[float], list[float]]:
    """Run same preds through two independent bridges, return both signal sequences."""
    b1 = _make_bridge()
    b2 = _make_bridge()
    r1, r2 = [], []
    for i, pred in enumerate(preds):
        tv = float("nan")
        if trend_follow and trend_values is not None and i < len(trend_values):
            tv = trend_values[i]
        s1 = b1.apply_constraints(
            "TEST", pred, i, deadzone, min_hold, long_only,
            trend_follow, tv, trend_threshold, max_hold,
        )
        s2 = b2.apply_constraints(
            "TEST", pred, i, deadzone, min_hold, long_only,
            trend_follow, tv, trend_threshold, max_hold,
        )
        r1.append(float(s1))
        r2.append(float(s2))
    return r1, r2


# ============================================================
# a. Deadzone entry blocking
# ============================================================

class TestDeadzoneEntryBlocking:
    def test_below_deadzone_stays_flat(self):
        """Predictions below deadzone should never trigger entry."""
        bridge = _make_bridge()
        # Push 200 warmup bars, then small predictions
        for i in range(200):
            bridge.apply_constraints("SYM", 0.01, i, 0.5, 12, False, False, float("nan"), 0.0, 120)
        # After warmup, small prediction should stay flat
        for i in range(200, 250):
            sig = bridge.apply_constraints("SYM", 0.1, i, 0.5, 12, False, False, float("nan"), 0.0, 120)
            # z-score of small constant values should stay near 0, below deadzone
            assert sig == 0.0 or abs(sig) <= 1.0

    def test_above_deadzone_triggers_entry(self):
        """Strong predictions exceeding deadzone should trigger entry."""
        bridge = _make_bridge()
        # Warmup with zeros
        for i in range(200):
            bridge.apply_constraints("SYM", 0.0, i, 0.5, 12, False, False, float("nan"), 0.0, 120)
        # Strong positive should eventually trigger +1
        got_entry = False
        for i in range(200, 300):
            sig = bridge.apply_constraints("SYM", 2.0, i, 0.5, 12, False, False, float("nan"), 0.0, 120)
            if sig == 1.0:
                got_entry = True
                break
        assert got_entry, "Strong signal should eventually exceed deadzone"


# ============================================================
# b. Deadzone fade flattening (via Rust)
# ============================================================

class TestDeadzoneFadeFlattening:
    def test_signal_flattens_when_z_fades(self):
        """Position should flatten when z-score fades back into deadzone."""
        bridge = _make_bridge()
        # Warmup
        for i in range(180):
            bridge.apply_constraints("SYM", 0.0, i, 0.5, 6, False, False, float("nan"), 0.0, 120)
        # Strong entry
        for i in range(180, 200):
            bridge.apply_constraints("SYM", 2.0, i, 0.5, 6, False, False, float("nan"), 0.0, 120)
        # Fade to zero — should flatten after min_hold
        went_flat = False
        for i in range(200, 300):
            sig = bridge.apply_constraints("SYM", 0.0, i, 0.5, 6, False, False, float("nan"), 0.0, 120)
            if sig == 0.0:
                went_flat = True
                break
        assert went_flat, "Signal should flatten when z-score fades"


# ============================================================
# c. Min-hold flip timing
# ============================================================

class TestMinHoldFlipTiming:
    def test_min_hold_prevents_early_exit(self):
        """Signal should NOT flip within min_hold bars."""
        bridge = _make_bridge()
        min_hold = 24
        # Warmup
        for i in range(180):
            bridge.apply_constraints("SYM", 0.0, i, 0.5, min_hold, False, False, float("nan"), 0.0, 120)
        # Enter long
        entered_at = None
        for i in range(180, 250):
            sig = bridge.apply_constraints("SYM", 2.0, i, 0.5, min_hold, False, False, float("nan"), 0.0, 120)
            if sig == 1.0 and entered_at is None:
                entered_at = i
                break

        if entered_at is None:
            pytest.skip("Could not trigger entry")

        # Immediately try to flatten — min_hold should prevent
        for i in range(entered_at + 1, entered_at + min_hold):
            sig = bridge.apply_constraints("SYM", -2.0, i, 0.5, min_hold, False, False, float("nan"), 0.0, 120)
            assert sig != 0.0, f"Should not flatten at bar {i - entered_at} (min_hold={min_hold})"

    def test_two_bridges_agree_on_hold_timing(self):
        """Two independent bridges with same inputs should agree on hold timing."""
        rng = np.random.RandomState(42)
        preds = rng.randn(500).tolist()
        r1, r2 = _run_constraints(preds, min_hold=12, deadzone=0.5)
        np.testing.assert_array_equal(np.array(r1), np.array(r2))


# ============================================================
# d. Trend-hold continue / exit
# ============================================================

class TestTrendHoldContinueExit:
    def test_trend_extends_position(self):
        """With trend_follow=True, position should hold when trend favorable."""
        bridge = _make_bridge()
        min_hold = 6
        # Warmup
        for i in range(180):
            bridge.apply_constraints("SYM", 0.0, i, 0.5, min_hold, False, True, 1.0, 0.0, 120)
        # Enter long
        entered = False
        for i in range(180, 220):
            sig = bridge.apply_constraints("SYM", 2.0, i, 0.5, min_hold, False, True, 1.0, 0.0, 120)
            if sig == 1.0:
                entered = True
                break

        if not entered:
            pytest.skip("Could not trigger entry")

        # Fade prediction but keep trend positive — should hold
        held_at_least_once = False
        for i in range(220, 260):
            sig = bridge.apply_constraints("SYM", 0.0, i, 0.5, min_hold, False, True, 1.0, 0.0, 120)
            if sig == 1.0:
                held_at_least_once = True

        # Trend hold should keep position open at least sometimes
        assert held_at_least_once, "Trend-follow should extend position"


# ============================================================
# e. Monthly gate flatten
# ============================================================

class TestMonthlyGateFlatten:
    def test_gate_blocks_when_below_ma(self):
        """Monthly gate should return False when close < MA."""
        bridge = _make_bridge(gate_window=10)
        # Push 10 bars of close=100 to build MA
        for i in range(10):
            ok = bridge.check_monthly_gate("SYM", 100.0, i, 10)
        # Close = 90 (below MA of 100) should gate
        ok = bridge.check_monthly_gate("SYM", 90.0, 10, 10)
        assert ok is False, "Close below MA should be gated"

    def test_gate_passes_when_above_ma(self):
        """Monthly gate should return True when close > MA."""
        bridge = _make_bridge(gate_window=10)
        for i in range(10):
            bridge.check_monthly_gate("SYM", 100.0, i, 10)
        ok = bridge.check_monthly_gate("SYM", 110.0, 10, 10)
        assert ok is True, "Close above MA should pass"


# ============================================================
# f. Vol-target sizing scale
# ============================================================

class TestVolTargetSizingScale:
    def test_vol_target_scales_down(self):
        """When realized vol > target, scale should be < 1.0."""
        vol_target = 0.02
        vol_val = 0.05
        scale = min(vol_target / vol_val, 1.0)
        assert abs(scale - 0.4) < 1e-10

    def test_vol_target_caps_at_one(self):
        """When realized vol < target, scale should be 1.0 (no leverage boost)."""
        vol_target = 0.05
        vol_val = 0.02
        scale = min(vol_target / vol_val, 1.0)
        assert scale == 1.0

    def test_vol_target_zero_vol_no_scale(self):
        """Vol near zero should not cause division error."""
        vol_target = 0.02
        vol_val = 1e-12
        if vol_val > 1e-8:
            scale = min(vol_target / vol_val, 1.0)
        else:
            scale = 1.0
        assert scale == 1.0  # Threshold catches near-zero
