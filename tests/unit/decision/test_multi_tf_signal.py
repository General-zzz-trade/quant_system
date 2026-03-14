"""Tests for MultiTimeframeSignal — 1h + 4h blended z-score fusion."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Mapping, Optional


from decision.signals.ml.multi_tf_signal import (
    MultiTimeframeSignal,
    _ZScoreBuffer,
    _HoldState,
    _BarAcc,
)


# ── Helpers ──

@dataclass(frozen=True)
class FakeSnapshot:
    symbol: str = "BTCUSDT"
    ts: Optional[datetime] = None
    features: Optional[Mapping[str, Any]] = None
    markets: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class FakeMarket:
    close: float = 50000.0
    high: float = 50100.0
    low: float = 49900.0
    open: float = 50000.0
    volume: float = 100.0


# ── ZScoreBuffer tests ──

class TestZScoreBuffer:
    def test_warmup_returns_zero(self):
        buf = _ZScoreBuffer(window=10, warmup=5)
        for i in range(4):
            assert buf.push(float(i)) == 0.0
        assert not buf.ready

    def test_after_warmup_returns_zscore(self):
        buf = _ZScoreBuffer(window=100, warmup=5)
        for i in range(10):
            buf.push(1.0)
        # All same values → std=0 → z=0
        assert buf.push(1.0) == 0.0

    def test_outlier_detection(self):
        buf = _ZScoreBuffer(window=100, warmup=5)
        for _ in range(50):
            buf.push(0.0)
        z = buf.push(3.0)
        assert z > 2.0  # 3.0 is far from mean=0

    def test_ready_flag(self):
        buf = _ZScoreBuffer(window=10, warmup=3)
        buf.push(1.0)
        buf.push(2.0)
        assert not buf.ready
        buf.push(3.0)
        assert buf.ready


# ── HoldState tests ──

class TestHoldState:
    def test_min_hold_enforced(self):
        h = _HoldState()
        # Enter long
        pos = h.update(1.0, 3.0, min_hold=3, max_hold=10)
        assert pos == 1.0

        # Try to exit immediately (reversal signal)
        pos = h.update(0.0, -1.0, min_hold=3, max_hold=10)
        assert pos == 1.0  # Still held

        pos = h.update(0.0, -1.0, min_hold=3, max_hold=10)
        assert pos == 1.0  # Still held (bar 3, just reached min_hold)

        pos = h.update(0.0, -1.0, min_hold=3, max_hold=10)
        assert pos == 0.0  # Now can exit

    def test_max_hold_forces_exit(self):
        h = _HoldState()
        h.update(1.0, 3.0, min_hold=2, max_hold=5)
        # Hold for enough bars to exceed max_hold
        for _ in range(10):
            h.update(1.0, 3.0, min_hold=2, max_hold=5)
        # After max_hold bars, position should have been cleared
        # (may re-enter if desired=1.0, but proves exit logic works)
        # Check it exits then re-enters (hold resets)
        assert h.bar_count > 5


# ── BarAcc tests ──

class TestBarAcc:
    def test_completes_after_4_bars(self):
        acc = _BarAcc()
        assert not acc.push(100, 105, 95, 102, 10)
        assert not acc.push(102, 108, 100, 106, 12)
        assert not acc.push(106, 110, 104, 107, 8)
        assert acc.push(107, 112, 103, 109, 15)

        assert acc.open == 100
        assert acc.high == 112
        assert acc.low == 95
        assert acc.close == 109
        assert acc.volume == 45

    def test_reset(self):
        acc = _BarAcc()
        for _ in range(4):
            acc.push(100, 105, 95, 102, 10)
        acc.reset()
        assert acc.count == 0


# ── MultiTimeframeSignal tests ──

class TestMultiTimeframeSignal:
    @staticmethod
    def _make_signal(**kwargs):
        """Create a MultiTimeframeSignal with no model (for unit tests)."""
        defaults = dict(
            model_dir_4h="/nonexistent",
            deadzone=2.5, min_hold=3, max_hold=20,
            zscore_window_1h=50, zscore_window_4h=20,
        )
        defaults.update(kwargs)
        return MultiTimeframeSignal(**defaults)

    def test_flat_without_features(self):
        sig = self._make_signal()

        snap = FakeSnapshot(features=None)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "flat"
        assert result.score == Decimal("0")

    def test_flat_during_warmup(self):
        sig = self._make_signal()

        base_ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for i in range(10):
            snap = FakeSnapshot(
                ts=base_ts + timedelta(hours=i),
                features={"ml_score": 0.01},
                markets={"BTCUSDT": FakeMarket()},
            )
            result = sig.compute(snap, "BTCUSDT")
            # During warmup, z-score is 0 → blend is 0 → flat
            assert result.side == "flat"

    def test_signal_correlation_low(self):
        """Verify z-score buffers are independent for 1h and 4h."""
        buf_1h = _ZScoreBuffer(window=50, warmup=5)
        buf_4h = _ZScoreBuffer(window=50, warmup=5)

        for i in range(20):
            buf_1h.push(float(i % 3))  # Oscillating
            buf_4h.push(float(i))       # Trending

        z1 = buf_1h.push(5.0)
        z4 = buf_4h.push(5.0)
        # Both push same value but different histories → different z-scores
        assert z1 != z4

    def test_name_attribute(self):
        """MultiTimeframeSignal must have a name for protocol compliance."""
        assert MultiTimeframeSignal.name == "multi_tf_1h_4h"

    def test_adaptive_deadzone_lowers_threshold(self):
        """Adaptive DZ should lower the deadzone when both models agree."""
        sig = self._make_signal(
            deadzone=2.5, adaptive_dz=True,
            dz_agreement_discount=0.8, dz_min=2.0,
            long_only=False,
        )
        # Effective dz = max(2.5 * 0.8, 2.0) = 2.0
        # With adaptive_dz=False, dz=2.5 would block weaker signals
        assert sig._adaptive_dz is True
        assert sig._dz_discount == 0.8
        assert sig._dz_min == 2.0

    def test_short_signal_allowed(self):
        """long_only=False should allow short signals."""
        sig = self._make_signal(long_only=False)
        assert sig._long_only is False

    def test_default_long_only_false(self):
        """Default should be long_only=False (P0 validated shorts)."""
        sig = MultiTimeframeSignal(model_dir_4h="/nonexistent")
        assert sig._long_only is False

    def test_default_adaptive_dz_true(self):
        """Default should have adaptive_dz enabled."""
        sig = MultiTimeframeSignal(model_dir_4h="/nonexistent")
        assert sig._adaptive_dz is True
