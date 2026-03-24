"""Tests for Alpha inference pipeline: InferenceEngine, LiveInferenceBridge."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


from alpha.base import Signal
from alpha.inference import InferenceEngine
from alpha.inference.bridge import LiveInferenceBridge


# ── Stub model ────────────────────────────────────────────────

@dataclass
class StubAlphaModel:
    name: str = "stub_model"
    _side: str = "long"
    _strength: float = 0.75
    _should_fail: bool = False

    def predict(self, *, symbol: str, ts: datetime, features: Dict[str, Any]) -> Optional[Signal]:
        if self._should_fail:
            raise RuntimeError("model error")
        return Signal(symbol=symbol, ts=ts, side=self._side, strength=self._strength)


# ── InferenceEngine ───────────────────────────────────────────

class TestInferenceEngine:
    def test_run_single_model(self):
        model = StubAlphaModel()
        engine = InferenceEngine(models=[model])
        results = engine.run(symbol="BTCUSDT", ts=datetime(2024, 1, 1), features={"sma": 40000})

        assert len(results) == 1
        r = results[0]
        assert r.model_name == "stub_model"
        assert r.symbol == "BTCUSDT"
        assert r.signal is not None
        assert r.signal.side == "long"
        assert r.signal.strength == 0.75
        assert r.error is None
        assert r.latency_ms >= 0

    def test_run_multiple_models(self):
        m1 = StubAlphaModel(name="m1", _side="long", _strength=0.8)
        m2 = StubAlphaModel(name="m2", _side="short", _strength=0.3)
        engine = InferenceEngine(models=[m1, m2])
        results = engine.run(symbol="ETHUSDT", ts=datetime(2024, 1, 1), features={})

        assert len(results) == 2
        assert results[0].model_name == "m1"
        assert results[1].model_name == "m2"
        assert results[0].signal.side == "long"
        assert results[1].signal.side == "short"

    def test_run_handles_model_error(self):
        model = StubAlphaModel(_should_fail=True)
        engine = InferenceEngine(models=[model])
        results = engine.run(symbol="BTCUSDT", ts=datetime(2024, 1, 1), features={})

        assert len(results) == 1
        assert results[0].signal is None
        assert results[0].error is not None
        assert "model error" in results[0].error

    def test_add_model(self):
        engine = InferenceEngine()
        assert len(engine._models) == 0
        engine.add_model(StubAlphaModel(name="added"))
        results = engine.run(symbol="BTCUSDT", ts=datetime(2024, 1, 1), features={})
        assert len(results) == 1
        assert results[0].model_name == "added"

    def test_empty_models(self):
        engine = InferenceEngine(models=[])
        results = engine.run(symbol="BTCUSDT", ts=datetime(2024, 1, 1), features={})
        assert results == []


# ── LiveInferenceBridge ───────────────────────────────────────

class TestLiveInferenceBridge:
    def test_enrich_adds_ml_score(self):
        model = StubAlphaModel(_side="long", _strength=0.9)
        bridge = LiveInferenceBridge(models=[model])
        features: Dict[str, Any] = {"sma": 40000}
        result = bridge.enrich("BTCUSDT", datetime(2024, 1, 1), features)

        assert "ml_score" in result
        assert result["ml_score"] == 0.9

    def test_enrich_short_signal_negative_score(self):
        model = StubAlphaModel(_side="short", _strength=0.6)
        bridge = LiveInferenceBridge(models=[model])
        result = bridge.enrich("BTCUSDT", datetime(2024, 1, 1), {})

        assert result["ml_score"] == -0.6

    def test_enrich_flat_signal_zero_score(self):
        model = StubAlphaModel(_side="flat", _strength=0.5)
        bridge = LiveInferenceBridge(models=[model])
        result = bridge.enrich("BTCUSDT", datetime(2024, 1, 1), {})

        assert result["ml_score"] == 0.0

    def test_enrich_custom_score_key(self):
        model = StubAlphaModel()
        bridge = LiveInferenceBridge(models=[model], score_key="alpha_signal")
        result = bridge.enrich("BTCUSDT", datetime(2024, 1, 1), {})

        assert "alpha_signal" in result
        assert "ml_score" not in result

    def test_enrich_model_error_no_crash(self):
        model = StubAlphaModel(_should_fail=True)
        bridge = LiveInferenceBridge(models=[model])
        features: Dict[str, Any] = {"sma": 40000}
        result = bridge.enrich("BTCUSDT", datetime(2024, 1, 1), features)

        # Should not crash, ml_score not added
        assert "ml_score" not in result
        assert result["sma"] == 40000

    def test_enrich_none_ts_uses_now(self):
        model = StubAlphaModel()
        bridge = LiveInferenceBridge(models=[model])
        result = bridge.enrich("BTCUSDT", None, {})

        assert "ml_score" in result


# ── Sequencing stub for constraint tests ──────────────────────

@dataclass
class SequenceModel:
    """Model that returns scores from a list, cycling."""
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


def _run_n(
    bridge: LiveInferenceBridge,
    symbol: str,
    n: int,
    features_per_bar: Optional[list] = None,
) -> list[float]:
    from datetime import timedelta
    ts = datetime(2024, 1, 1)
    out = []
    for i in range(n):
        feats: Dict[str, Any] = {}
        if features_per_bar is not None and i < len(features_per_bar):
            feats.update(features_per_bar[i])
        # Each call gets a unique hour so hourly-gated buffers advance
        bar_ts = ts + timedelta(hours=i)
        bridge.enrich(symbol, bar_ts, feats)
        out.append(feats.get("ml_score", 0.0))
    return out


# ── Signal constraint tests ──────────────────────────────────

class TestMinHoldPreventsEarlyChange:
    def test_hold_period_blocks_flip(self):
        """Position should not change within min_hold period."""
        model = SequenceModel(_scores=[0.8, -0.8, 0.8, -0.8, -0.8, -0.8])
        bridge = LiveInferenceBridge(
            models=[model],
            min_hold_bars={"BTCUSDT": 4},
            deadzone=0.5,
            zscore_warmup=0,
        )
        scores = _run_n(bridge, "BTCUSDT", 6)
        # Bars 0-3: +1 held despite alternating raw scores
        assert scores[:4] == [1.0, 1.0, 1.0, 1.0]
        # Bar 4: hold expired → flip to -1
        assert scores[4] == -1.0


class TestMinHoldAllowsChangeAfterExpiry:
    def test_change_allowed_after_min_hold(self):
        model = SequenceModel(_scores=[0.8, 0.8, 0.8, 0.8, 0.8, -0.8, -0.8])
        bridge = LiveInferenceBridge(
            models=[model],
            min_hold_bars={"BTCUSDT": 5},
            deadzone=0.5,
            zscore_warmup=0,
        )
        scores = _run_n(bridge, "BTCUSDT", 7)
        assert scores[0] == 1.0
        assert scores[4] == 1.0
        # Bar 5: hold_count=5 >= 5 and desired=-1 → flip
        assert scores[5] == -1.0
        # Bar 6: hold_count=1 < 5 → keep -1
        assert scores[6] == -1.0


class TestLongOnlyClipsShort:
    def test_short_signal_clipped_to_zero(self):
        model = SequenceModel(_scores=[-0.8, 0.8, -0.8])
        bridge = LiveInferenceBridge(
            models=[model],
            min_hold_bars={"SOLUSDT": 1},
            long_only_symbols={"SOLUSDT"},
            deadzone=0.5,
            zscore_warmup=0,
        )
        scores = _run_n(bridge, "SOLUSDT", 3)
        # raw=-0.8 → long_only clip to 0 → |0| < 0.5 → desired=0
        assert scores[0] == 0.0
        # raw=0.8 → clip=0.8 > 0.5 → desired=+1
        assert scores[1] == 1.0
        # raw=-0.8 → long_only clip to 0 → desired=0
        assert scores[2] == 0.0


class TestNoMinHoldPreservesRawBehavior:
    def test_raw_float_passthrough(self):
        """Without min_hold_bars, raw scores pass through unchanged."""
        model = StubAlphaModel(_side="long", _strength=0.3)
        bridge = LiveInferenceBridge(models=[model])
        result = bridge.enrich("BTCUSDT", datetime(2024, 1, 1), {})
        assert result["ml_score"] == 0.3

    def test_symbols_without_config_get_raw(self):
        """A symbol not in min_hold_bars dict gets raw passthrough."""
        model = SequenceModel(_scores=[-0.8, 0.8])
        bridge = LiveInferenceBridge(
            models=[model],
            min_hold_bars={"BTCUSDT": 24},  # only BTC constrained
            deadzone=0.5,
            zscore_warmup=0,
        )
        ts = datetime(2024, 1, 1)
        feats: Dict[str, Any] = {}
        bridge.enrich("ETHUSDT", ts, feats)
        # ETHUSDT not in min_hold_bars → raw passthrough
        assert feats["ml_score"] == -0.8


# ── Trend hold tests ────────────────────────────────────────

class TestTrendHoldKeepsPosition:
    def test_trend_hold_extends_position(self):
        """When model says exit but trend is favorable, keep position."""
        # Bar 0: score=0.8 → desired=+1, enter long
        # Bar 1: score=0.8 → desired=+1, hold
        # Bar 2: score=0.1 → desired=0 BUT trend=0.5>0 → trend hold
        # Bar 3: score=0.1 → desired=0 BUT trend=0.3>0 → trend hold
        # Bar 4: score=0.1 → desired=0, trend=-0.1<0 → exit
        model = SequenceModel(_scores=[0.8, 0.8, 0.1, 0.1, 0.1])
        bridge = LiveInferenceBridge(
            models=[model],
            min_hold_bars={"BTCUSDT": 2},
            long_only_symbols={"BTCUSDT"},
            deadzone=0.5,
            zscore_warmup=0,
            trend_follow=True,
            trend_indicator="trend",
            trend_threshold=0.0,
            max_hold=120,
        )
        features_per_bar = [
            {"trend": 0.5},
            {"trend": 0.5},
            {"trend": 0.5},
            {"trend": 0.3},
            {"trend": -0.1},
        ]
        scores = _run_n(bridge, "BTCUSDT", 5, features_per_bar)
        assert scores[0] == 1.0  # enter long
        assert scores[1] == 1.0  # min hold
        assert scores[2] == 1.0  # trend hold (trend=0.5>0)
        assert scores[3] == 1.0  # trend hold (trend=0.3>0)
        assert scores[4] == 0.0  # trend negative → exit

    def test_trend_hold_respects_max_hold(self):
        """Trend hold cannot exceed max_hold bars."""
        model = SequenceModel(_scores=[0.8, 0.8, 0.1, 0.1, 0.1, 0.1])
        bridge = LiveInferenceBridge(
            models=[model],
            min_hold_bars={"BTCUSDT": 2},
            long_only_symbols={"BTCUSDT"},
            deadzone=0.5,
            zscore_warmup=0,
            trend_follow=True,
            trend_indicator="trend",
            trend_threshold=0.0,
            max_hold=4,  # force exit at bar 4
        )
        features_per_bar = [{"trend": 1.0}] * 6
        scores = _run_n(bridge, "BTCUSDT", 6, features_per_bar)
        assert scores[0] == 1.0
        assert scores[1] == 1.0  # min hold
        assert scores[2] == 1.0  # trend hold (count=3<4)
        assert scores[3] == 1.0  # trend hold (count=4, but check is hold_count < max_hold)
        # Bar 4: hold_count=4 >= max_hold=4 → trend hold NOT applied → exit
        assert scores[4] == 0.0

    def test_no_trend_hold_when_disabled(self):
        """Without trend_follow=True, positions exit normally."""
        model = SequenceModel(_scores=[0.8, 0.8, 0.1])
        bridge = LiveInferenceBridge(
            models=[model],
            min_hold_bars={"BTCUSDT": 2},
            long_only_symbols={"BTCUSDT"},
            deadzone=0.5,
            zscore_warmup=0,
            trend_follow=False,
        )
        features_per_bar = [{"trend": 1.0}] * 3
        scores = _run_n(bridge, "BTCUSDT", 3, features_per_bar)
        assert scores[0] == 1.0
        assert scores[1] == 1.0
        assert scores[2] == 0.0  # no trend hold → exit


# ── Monthly gate tests ────────────────────────────────────

class TestMonthlyGate:
    def test_gate_blocks_below_ma(self):
        """When close < MA(window), signal is forced to 0."""
        # Use a small window (3) for easy testing
        model = SequenceModel(_scores=[0.8] * 6)
        bridge = LiveInferenceBridge(
            models=[model],
            min_hold_bars={"BTCUSDT": 1},
            deadzone=0.5,
            zscore_warmup=0,
            monthly_gate=True,
            monthly_gate_window=3,
        )
        # Prices: declining → close < MA → gated
        # Bar 0-2: filling deque (len < window) → allowed (optimistic warmup)
        # Bar 3: close=97, MA(3)=mean(100,99,98)=99 → 97<99 → gated
        # Bar 4: close=96, MA(3)=mean(99,98,97)=98 → 96<98 → gated
        prices = [100, 99, 98, 97, 96, 95]
        features_per_bar = [{"close": p} for p in prices]
        scores = _run_n(bridge, "BTCUSDT", 6, features_per_bar)
        # Bars 0-2: warmup (optimistic → signal passes), Bars 3-5: gated
        assert all(s == 0.0 for s in scores[3:])

    def test_gate_allows_above_ma(self):
        """When close > MA(window), signal passes through."""
        model = SequenceModel(_scores=[0.8] * 6)
        bridge = LiveInferenceBridge(
            models=[model],
            min_hold_bars={"BTCUSDT": 1},
            deadzone=0.5,
            zscore_warmup=0,
            monthly_gate=True,
            monthly_gate_window=3,
        )
        # Prices: rising → close > MA → allowed
        # Bar 0: deque=[100], len=1 < 3 → gated
        # Bar 1: deque=[100,101], len=2 < 3 → gated
        # Bar 2: deque=[100,101,102], MA=101, close=102>101 → allowed
        # Bar 3: close=103, MA=mean(101,102,103-wait, deque is maxlen=3)
        prices = [100, 101, 102, 103, 104, 105]
        features_per_bar = [{"close": p} for p in prices]
        scores = _run_n(bridge, "BTCUSDT", 6, features_per_bar)
        # First 2 bars: warmup (optimistic → signal passes through)
        assert scores[0] == 1.0
        assert scores[1] == 1.0
        # Bar 2+: above MA → signal passes
        assert scores[2] == 1.0
        assert scores[3] == 1.0
        assert scores[4] == 1.0
        assert scores[5] == 1.0

    def test_gate_disabled_by_default(self):
        """monthly_gate=False (default) does not affect signals."""
        model = SequenceModel(_scores=[0.8] * 4)
        bridge = LiveInferenceBridge(
            models=[model],
            min_hold_bars={"BTCUSDT": 1},
            deadzone=0.5,
            zscore_warmup=0,
            monthly_gate=False,  # default
        )
        # Declining prices — without gate, signal should still be +1
        prices = [100, 99, 98, 97]
        features_per_bar = [{"close": p} for p in prices]
        scores = _run_n(bridge, "BTCUSDT", 4, features_per_bar)
        assert all(s == 1.0 for s in scores)


# ── Z-score normalization tests ──────────────────────────────

class TestZscoreNormalization:
    def test_warmup_returns_flat(self):
        """During z-score warmup, position stays flat."""
        model = SequenceModel(_scores=[0.8] * 10)
        bridge = LiveInferenceBridge(
            models=[model],
            min_hold_bars={"BTCUSDT": 1},
            deadzone=0.5,
            zscore_warmup=5,
        )
        scores = _run_n(bridge, "BTCUSDT", 4)
        # All within warmup (4 < 5) → flat
        assert all(s == 0.0 for s in scores)

    def test_zscore_triggers_after_warmup(self):
        """After warmup, z-score normalization produces non-zero signals."""
        # Feed constant 0.5 for warmup, then a spike at 0.8
        # z-score of 0.8 relative to mean=0.5 should be large
        n_warmup = 10
        baseline = [0.5] * n_warmup + [0.8, 0.8, 0.8]
        model = SequenceModel(_scores=baseline)
        bridge = LiveInferenceBridge(
            models=[model],
            min_hold_bars={"BTCUSDT": 1},
            deadzone=0.5,
            zscore_window=20,
            zscore_warmup=n_warmup,
        )
        scores = _run_n(bridge, "BTCUSDT", n_warmup + 3)
        # Warmup bars: flat
        assert all(s == 0.0 for s in scores[:n_warmup])
        # After warmup: 0.8 is well above mean(0.5), z >> 0.5 → +1
        assert scores[n_warmup] == 1.0

    def test_zscore_long_only_clips_negative_z(self):
        """With long_only, negative z-scores are clipped to 0."""
        n_warmup = 10
        baseline = [0.5] * n_warmup + [-0.8, -0.8]
        model = SequenceModel(_scores=baseline)
        bridge = LiveInferenceBridge(
            models=[model],
            min_hold_bars={"BTCUSDT": 1},
            long_only_symbols={"BTCUSDT"},
            deadzone=0.5,
            zscore_window=20,
            zscore_warmup=n_warmup,
        )
        scores = _run_n(bridge, "BTCUSDT", n_warmup + 2)
        # After warmup: -0.8 has very negative z → clipped to 0 by long_only → flat
        assert scores[n_warmup] == 0.0
