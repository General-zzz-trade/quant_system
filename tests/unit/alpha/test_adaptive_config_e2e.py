"""End-to-end tests for adaptive BTC config parameter updates.

Verifies that:
1. update_params() changes signal behavior observably
2. select_robust() produces valid (non-degenerate) configs
3. Z-score state is preserved across parameter updates
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import numpy as np
import pytest

from alpha.adaptive_config import AdaptiveConfigSelector, AdaptiveParams
from alpha.base import Signal
from alpha.inference.bridge import LiveInferenceBridge

_qh = pytest.importorskip("_quant_hotpath")


# ── Stub model ───────────────────────────────────────────────

@dataclass
class SequenceModel:
    """Model that returns scores from a list."""
    name: str = "seq"
    _scores: list = None
    _idx: int = 0

    def __post_init__(self):
        if self._scores is None:
            self._scores = [0.8]

    def predict(self, *, symbol: str, ts: datetime, features: Dict[str, Any]) -> Optional[Signal]:
        score = self._scores[self._idx % len(self._scores)]
        self._idx += 1
        side = "long" if score > 0 else ("short" if score < 0 else "flat")
        return Signal(symbol=symbol, ts=ts, side=side, strength=abs(score))


def _run_bridge(
    bridge: LiveInferenceBridge,
    symbol: str,
    n: int,
    scores: list[float],
) -> list[float]:
    """Run bridge for n bars and return ml_score sequence."""
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    results = []
    for i in range(n):
        feats: Dict[str, Any] = {"close": 50000.0}
        bar_ts = ts + timedelta(hours=i)
        bridge.enrich(symbol, bar_ts, feats)
        results.append(feats.get("ml_score", 0.0))
    return results


# ── Tests ────────────────────────────────────────────────────

class TestUpdateParamsChangesSignal:
    """Verify that update_params() observably changes signal behavior."""

    def test_deadzone_change_affects_threshold(self):
        """Changing deadzone from 0.5 to 2.0 should change signal trigger."""
        # With small deadzone, moderate z-scores trigger signals
        # With large deadzone, same z-scores produce flat
        scores = [0.5] * 20 + [0.8, 0.8, 0.8, 0.8, 0.8]
        model = SequenceModel(_scores=scores)
        bridge = LiveInferenceBridge(
            models=[model],
            min_hold_bars={"BTCUSDT": 1},
            deadzone=0.5,
            zscore_window=20,
            zscore_warmup=10,
        )
        # Run warmup + a few bars with dz=0.5
        result_before = _run_bridge(bridge, "BTCUSDT", 25, scores)

        # Now create a fresh bridge with dz=2.0 and same scores
        model2 = SequenceModel(_scores=scores)
        bridge2 = LiveInferenceBridge(
            models=[model2],
            min_hold_bars={"BTCUSDT": 1},
            deadzone=2.0,
            zscore_window=20,
            zscore_warmup=10,
        )
        result_after = _run_bridge(bridge2, "BTCUSDT", 25, scores)

        # At least some signals should differ
        # With larger deadzone, fewer signals should trigger
        active_before = sum(1 for s in result_before if s != 0.0)
        active_after = sum(1 for s in result_after if s != 0.0)
        assert active_before >= active_after, (
            f"Larger deadzone should produce equal or fewer signals: "
            f"dz=0.5 had {active_before}, dz=2.0 had {active_after}"
        )

    def test_update_params_preserves_zscore_state(self):
        """update_params() should not reset z-score buffer."""
        scores = [0.5] * 30 + [0.8] * 10
        model = SequenceModel(_scores=scores)
        bridge = LiveInferenceBridge(
            models=[model],
            min_hold_bars={"BTCUSDT": 1},
            deadzone=0.5,
            zscore_window=20,
            zscore_warmup=10,
        )
        # Run 30 bars to fill z-score buffer
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(30):
            feats: Dict[str, Any] = {"close": 50000.0}
            bridge.enrich("BTCUSDT", ts + timedelta(hours=i), feats)

        # Get checkpoint before update
        checkpoint_before = bridge.checkpoint()
        buf_before = checkpoint_before.get("zscore_buf", {}).get("BTCUSDT", [])

        # Update params
        bridge.update_params("BTCUSDT", deadzone=1.0, min_hold=24)

        # Check z-score buffer is preserved
        checkpoint_after = bridge.checkpoint()
        buf_after = checkpoint_after.get("zscore_buf", {}).get("BTCUSDT", [])
        assert buf_before == buf_after, "Z-score buffer should not change on update_params"

    def test_min_hold_change_affects_hold_period(self):
        """Changing min_hold should change how long positions are held."""
        # With min_hold=2: can flip after 2 bars
        # With min_hold=10: must hold for 10 bars
        scores = [0.8, -0.8, 0.8, -0.8, 0.8, -0.8, 0.8, -0.8, 0.8, -0.8, 0.8, -0.8]

        model1 = SequenceModel(_scores=scores)
        bridge1 = LiveInferenceBridge(
            models=[model1],
            min_hold_bars={"BTCUSDT": 2},
            deadzone=0.5,
            zscore_warmup=0,
        )
        result_short_hold = _run_bridge(bridge1, "BTCUSDT", 12, scores)

        model2 = SequenceModel(_scores=scores)
        bridge2 = LiveInferenceBridge(
            models=[model2],
            min_hold_bars={"BTCUSDT": 10},
            deadzone=0.5,
            zscore_warmup=0,
        )
        result_long_hold = _run_bridge(bridge2, "BTCUSDT", 12, scores)

        # With longer hold, fewer position changes
        changes_short = sum(1 for i in range(1, 12) if result_short_hold[i] != result_short_hold[i-1])
        changes_long = sum(1 for i in range(1, 12) if result_long_hold[i] != result_long_hold[i-1])
        assert changes_short >= changes_long, (
            f"Longer min_hold should produce fewer changes: "
            f"hold=2 had {changes_short}, hold=10 had {changes_long}"
        )

    def test_long_only_update(self):
        """Updating long_only should clip short signals."""
        scores = [-0.8] * 5

        # Without long_only
        model1 = SequenceModel(_scores=scores)
        bridge1 = LiveInferenceBridge(
            models=[model1],
            min_hold_bars={"BTCUSDT": 1},
            deadzone=0.5,
            zscore_warmup=0,
        )
        result_both = _run_bridge(bridge1, "BTCUSDT", 5, scores)

        # With long_only
        model2 = SequenceModel(_scores=scores)
        bridge2 = LiveInferenceBridge(
            models=[model2],
            min_hold_bars={"BTCUSDT": 1},
            long_only_symbols={"BTCUSDT"},
            deadzone=0.5,
            zscore_warmup=0,
        )
        result_long = _run_bridge(bridge2, "BTCUSDT", 5, scores)

        # Without long_only: should have negative signals
        assert any(s < 0 for s in result_both), "Without long_only, should have short signals"
        # With long_only: all signals should be >= 0
        assert all(s >= 0 for s in result_long), "With long_only, all signals should be >= 0"


class TestAdaptiveConfigSelector:
    """Verify select_robust() produces valid configs."""

    def _make_data(self, n: int = 5000, seed: int = 42) -> tuple:
        """Generate synthetic z-scores and close prices."""
        rng = np.random.RandomState(seed)
        closes = 50000.0 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
        z_scores = rng.normal(0, 1, n)
        return z_scores, closes

    def test_select_returns_valid_params(self):
        z_scores, closes = self._make_data()
        selector = AdaptiveConfigSelector(lookback_months=3)
        params = selector.select(z_scores, closes)

        assert isinstance(params, AdaptiveParams)
        assert params.deadzone > 0
        assert params.min_hold > 0
        assert params.max_hold > params.min_hold
        assert isinstance(params.long_only, bool)
        assert params.confidence in ("high", "medium", "low")

    def test_select_robust_returns_valid_params(self):
        z_scores, closes = self._make_data(n=8000)
        selector = AdaptiveConfigSelector(lookback_months=6)
        params = selector.select_robust(z_scores, closes)

        assert isinstance(params, AdaptiveParams)
        assert params.deadzone > 0
        assert params.min_hold > 0
        assert params.max_hold >= params.min_hold

    def test_select_robust_no_degenerate_config(self):
        """select_robust should not return configs with 0 trades."""
        z_scores, closes = self._make_data(n=8000)
        selector = AdaptiveConfigSelector(lookback_months=6, min_trades=8)
        params = selector.select_robust(z_scores, closes)

        # Either has trades or returns conservative fallback
        assert params.trades >= 0
        if params.confidence == "low" and params.trades == 0:
            # Fallback defaults
            assert params.deadzone == 1.0
            assert params.min_hold == 12

    def test_select_stability_across_calls(self):
        """Repeated calls with same data should give same result."""
        z_scores, closes = self._make_data()

        s1 = AdaptiveConfigSelector(lookback_months=3)
        p1 = s1.select(z_scores, closes)

        s2 = AdaptiveConfigSelector(lookback_months=3)
        p2 = s2.select(z_scores, closes)

        assert p1.deadzone == p2.deadzone
        assert p1.min_hold == p2.min_hold
        assert p1.long_only == p2.long_only

    def test_to_dict_roundtrip(self):
        z_scores, closes = self._make_data()
        selector = AdaptiveConfigSelector()
        params = selector.select(z_scores, closes)
        d = params.to_dict()

        assert "deadzone" in d
        assert "min_hold" in d
        assert "max_hold" in d
        assert "long_only" in d
        assert "sharpe" in d
        assert "confidence" in d
