"""Inference path parity tests: backtest (cpp_pred_to_signal) vs live (RustInferenceBridge).

Validates that the backtest signal path and live signal path produce equivalent
outputs given identical prediction sequences, ensuring walk-forward validated
alpha translates faithfully to live trading.

KNOWN DIVERGENCE (documented and tested):
1. warmup > window: backtest clamps warmup to min(warmup, window), live does not.
   Live never produces signals when warmup > window.

FIXED DIVERGENCE (was present before inference_bridge.rs warmup hold fix):
2. min_hold >= warmup: FIXED. Live now increments hold_counter during warmup
   to match backtest's hold accumulation. Full parity for all min_hold/warmup
   combinations when warmup <= window.
"""
from __future__ import annotations

import pytest

_qh = pytest.importorskip("_quant_hotpath")
cpp_pred_to_signal = _qh.cpp_pred_to_signal
RustInferenceBridge = _qh.RustInferenceBridge


def _live_signal_sequence(
    preds: list[float],
    deadzone: float,
    min_hold: int,
    zscore_window: int,
    zscore_warmup: int,
) -> list[float]:
    """Run predictions through live path one-at-a-time, returning signal sequence."""
    bridge = RustInferenceBridge(zscore_window, zscore_warmup, 480)
    signals = []
    for i, pred in enumerate(preds):
        hour_key = i  # unique hour per bar to avoid dedup
        sig = bridge.apply_constraints(
            "TEST", pred, hour_key, deadzone, min_hold,
            False,   # long_only
            False,   # trend_follow
            float("nan"),  # trend_val
            0.0,     # trend_threshold
            9999,    # max_hold (effectively infinite)
        )
        signals.append(sig)
    return signals


def _backtest_signal_sequence(
    preds: list[float],
    deadzone: float,
    min_hold: int,
    zscore_window: int,
    zscore_warmup: int,
) -> list[float]:
    """Run predictions through backtest path (batch)."""
    return list(cpp_pred_to_signal(preds, deadzone, min_hold, zscore_window, zscore_warmup))


# ── Helpers ──────────────────────────────────────────────────

def _assert_parity(
    preds: list[float],
    deadzone: float,
    min_hold: int,
    zscore_window: int,
    zscore_warmup: int,
    atol: float = 1e-10,
):
    """Assert backtest and live paths produce identical signals."""
    bt = _backtest_signal_sequence(preds, deadzone, min_hold, zscore_window, zscore_warmup)
    live = _live_signal_sequence(preds, deadzone, min_hold, zscore_window, zscore_warmup)

    assert len(bt) == len(live) == len(preds), (
        f"Length mismatch: bt={len(bt)}, live={len(live)}, preds={len(preds)}"
    )

    mismatches = []
    for i, (b, l) in enumerate(zip(bt, live)):
        if abs(b - l) > atol:
            mismatches.append((i, b, l))

    if mismatches:
        detail = "\n".join(
            f"  bar {i}: backtest={b:.6f}, live={l:.6f}"
            for i, b, l in mismatches[:10]
        )
        total = len(mismatches)
        pytest.fail(
            f"{total} signal mismatches (deadzone={deadzone}, min_hold={min_hold}, "
            f"zscore_window={zscore_window}, warmup={zscore_warmup}):\n{detail}"
        )


def _assert_parity_after_convergence(
    preds: list[float],
    deadzone: float,
    min_hold: int,
    zscore_window: int,
    zscore_warmup: int,
    skip_bars: int,
    atol: float = 1e-10,
):
    """Assert parity after skipping the first `skip_bars` (for known-divergent warmup)."""
    bt = _backtest_signal_sequence(preds, deadzone, min_hold, zscore_window, zscore_warmup)
    live = _live_signal_sequence(preds, deadzone, min_hold, zscore_window, zscore_warmup)

    assert len(bt) == len(live) == len(preds)

    mismatches = []
    for i in range(skip_bars, len(preds)):
        if abs(bt[i] - live[i]) > atol:
            mismatches.append((i, bt[i], live[i]))

    if mismatches:
        detail = "\n".join(
            f"  bar {i}: backtest={b:.6f}, live={l:.6f}"
            for i, b, l in mismatches[:10]
        )
        pytest.fail(
            f"{len(mismatches)} post-convergence mismatches "
            f"(after bar {skip_bars}):\n{detail}"
        )


def _make_trending_preds(n: int, seed: int = 42) -> list[float]:
    """Generate a deterministic trending prediction sequence."""
    import random
    rng = random.Random(seed)
    preds = []
    val = 0.0
    for _ in range(n):
        val += rng.gauss(0.01, 0.1)
        preds.append(val)
    return preds


def _make_alternating_preds(n: int, amplitude: float = 0.8) -> list[float]:
    """Generate alternating high/low predictions."""
    return [amplitude if i % 2 == 0 else -amplitude for i in range(n)]


def _make_regime_shift_preds(n: int) -> list[float]:
    """Generate predictions with a clear regime shift in the middle."""
    import random
    rng = random.Random(123)
    half = n // 2
    first = [0.05 + rng.gauss(0, 0.05) for _ in range(half)]
    second = [-0.05 + rng.gauss(0, 0.05) for _ in range(n - half)]
    return first + second


# ── Core parity tests (min_hold <= warmup → full parity) ────

class TestInferenceParity:
    """Verify backtest and live paths produce identical outputs.

    All tests use min_hold < warmup (strictly less) to avoid the known
    divergence in warmup-period hold enforcement.
    """

    @pytest.mark.parametrize("deadzone", [0.3, 0.5, 1.0, 2.0])
    def test_parity_varying_deadzone(self, deadzone):
        preds = _make_trending_preds(200)
        _assert_parity(preds, deadzone=deadzone, min_hold=4, zscore_window=50, zscore_warmup=10)

    @pytest.mark.parametrize("min_hold", [1, 4, 8, 9])
    def test_parity_varying_min_hold(self, min_hold):
        """min_hold < warmup=10 → full parity (strict less-than required)."""
        preds = _make_trending_preds(300)
        _assert_parity(preds, deadzone=0.5, min_hold=min_hold, zscore_window=50, zscore_warmup=10)

    @pytest.mark.parametrize("zscore_window", [10, 50, 200, 720])
    def test_parity_varying_window(self, zscore_window):
        warmup = min(zscore_window // 4, 180)
        min_hold = min(4, warmup - 1) if warmup > 1 else 1  # strict < warmup
        n = max(zscore_window * 3, 300)
        preds = _make_trending_preds(n)
        _assert_parity(preds, deadzone=0.5, min_hold=min_hold, zscore_window=zscore_window, zscore_warmup=warmup)

    @pytest.mark.parametrize("zscore_warmup", [5, 10, 50, 168])
    def test_parity_varying_warmup(self, zscore_warmup):
        preds = _make_trending_preds(500)
        min_hold = min(4, zscore_warmup - 1) if zscore_warmup > 1 else 1  # strict < warmup
        _assert_parity(preds, deadzone=0.5, min_hold=min_hold, zscore_window=200, zscore_warmup=zscore_warmup)

    def test_parity_alternating_preds(self):
        preds = _make_alternating_preds(200)
        _assert_parity(preds, deadzone=0.5, min_hold=4, zscore_window=50, zscore_warmup=10)

    def test_parity_regime_shift(self):
        preds = _make_regime_shift_preds(400)
        _assert_parity(preds, deadzone=0.5, min_hold=8, zscore_window=100, zscore_warmup=20)


# ── Fixed: min_hold >= warmup now has parity ─────────────────

class TestInferenceParityMinHoldExceedsWarmup:
    """Verify parity for min_hold >= warmup cases.

    Previously divergent: live treated hold_counter=0 as "hold satisfied"
    and immediately allowed position change on first post-warmup bar.
    Fixed by incrementing hold_counter during warmup to match backtest's
    hold_count accumulation over warmup bars.
    """

    @pytest.mark.parametrize("min_hold", [10, 12, 24, 48])
    def test_parity_min_hold_exceeds_warmup(self, min_hold):
        """min_hold >= warmup=10 → full parity after fix."""
        preds = _make_trending_preds(300)
        _assert_parity(preds, deadzone=0.5, min_hold=min_hold, zscore_window=50, zscore_warmup=10)

    def test_parity_min_hold_equals_warmup(self):
        """min_hold == warmup: previously off-by-one, now exact parity."""
        preds = _make_trending_preds(200)
        _assert_parity(preds, deadzone=0.5, min_hold=10, zscore_window=50, zscore_warmup=10)

    def test_parity_large_min_hold(self):
        """min_hold >> warmup: hold counter accumulates correctly during warmup."""
        preds = _make_trending_preds(500)
        _assert_parity(preds, deadzone=0.5, min_hold=100, zscore_window=200, zscore_warmup=10)

    def test_warmup_exceeds_window_parity(self):
        """warmup > window: both paths clamp warmup to min(warmup, window).

        After fix: live clamps warmup at init (inference_bridge.rs),
        matching backtest's min(warmup, window) clamp. Full parity.
        """
        preds = _make_trending_preds(100)
        _assert_parity(preds, deadzone=0.5, min_hold=4, zscore_window=20, zscore_warmup=30)


# ── Edge cases ───────────────────────────────────────────────

class TestInferenceParityEdgeCases:
    """Edge cases: constant input, zero std, boundaries, etc."""

    def test_constant_input_zero_std(self):
        """Constant predictions → std=0 → both paths should output 0."""
        preds = [0.5] * 100
        _assert_parity(preds, deadzone=0.5, min_hold=4, zscore_window=20, zscore_warmup=5)

    def test_single_prediction(self):
        _assert_parity([0.8], deadzone=0.5, min_hold=1, zscore_window=10, zscore_warmup=1)

    def test_two_predictions(self):
        _assert_parity([0.8, -0.8], deadzone=0.5, min_hold=1, zscore_window=10, zscore_warmup=1)

    def test_warmup_equals_window(self):
        """warmup == window: need full window before any output."""
        preds = _make_trending_preds(100)
        _assert_parity(preds, deadzone=0.5, min_hold=4, zscore_window=20, zscore_warmup=20)

    def test_window_rollover(self):
        """Buffer wraps around after filling window."""
        preds = _make_trending_preds(200)
        _assert_parity(preds, deadzone=0.5, min_hold=4, zscore_window=20, zscore_warmup=5)

    def test_all_zero_predictions(self):
        preds = [0.0] * 50
        _assert_parity(preds, deadzone=0.5, min_hold=4, zscore_window=20, zscore_warmup=5)

    def test_min_hold_one(self):
        """min_hold=1: signal can change every bar after hold."""
        preds = _make_alternating_preds(50, amplitude=1.0)
        _assert_parity(preds, deadzone=0.3, min_hold=1, zscore_window=20, zscore_warmup=5)

    def test_large_deadzone_no_signals(self):
        """Very large deadzone → z-score never exceeds → all zeros."""
        preds = _make_trending_preds(100)
        _assert_parity(preds, deadzone=100.0, min_hold=4, zscore_window=20, zscore_warmup=5)

    def test_tiny_deadzone_many_signals(self):
        """Very small deadzone → any non-zero z triggers signal."""
        preds = _make_trending_preds(100)
        _assert_parity(preds, deadzone=0.01, min_hold=1, zscore_window=20, zscore_warmup=5)

    def test_near_zero_std_boundary(self):
        """Values very close together but not identical → std near 1e-12."""
        base = 1.0
        eps = 1e-13
        preds = [base + eps * i for i in range(50)]
        _assert_parity(preds, deadzone=0.5, min_hold=4, zscore_window=20, zscore_warmup=5)


# ── Parameter combinations ──────────────────────────────────

class TestInferenceParityParameterCombinations:
    """Cross-product parameter combinations (all with min_hold <= warmup)."""

    @pytest.mark.parametrize("deadzone,min_hold,window,warmup", [
        (0.3, 1, 10, 3),
        (0.5, 4, 50, 10),
        (1.0, 12, 200, 50),   # 12 < 50 ✓
        (2.0, 24, 720, 168),  # 24 < 168 ✓
        (0.5, 8, 100, 25),    # 8 < 25 ✓
        (1.5, 4, 30, 10),     # 4 < 10 ✓
    ])
    def test_param_combo(self, deadzone, min_hold, window, warmup):
        n = max(window * 3, 300)
        preds = _make_trending_preds(n, seed=int(deadzone * 100 + min_hold))
        _assert_parity(preds, deadzone, min_hold, window, warmup)


# ── Large-scale tests ────────────────────────────────────────

@pytest.mark.slow
class TestInferenceParityLargeScale:
    """Large-scale parity tests (10,000+ bars)."""

    def test_10k_bars_trending(self):
        """Production params: window=720, warmup=168, min_hold=12 (12 <= 168)."""
        preds = _make_trending_preds(10000)
        _assert_parity(preds, deadzone=0.5, min_hold=12, zscore_window=720, zscore_warmup=168)

    def test_10k_bars_regime_shift(self):
        preds = _make_regime_shift_preds(10000)
        _assert_parity(preds, deadzone=1.0, min_hold=8, zscore_window=200, zscore_warmup=50)

    def test_15k_bars_production_params(self):
        """Full production config: window=720, warmup=168, min_hold=12."""
        preds = _make_trending_preds(15000, seed=99)
        _assert_parity(preds, deadzone=0.5, min_hold=12, zscore_window=720, zscore_warmup=168)
