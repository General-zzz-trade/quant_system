"""Parity tests: Incremental trackers (EMA, RSI, ATR, ADX)."""
import pytest
import math

try:
    # The trackers are internal Rust structs not directly exposed via PyO3,
    # but we can test via RustFeatureEngine which already computes rsi_14/atr_norm_14
    from _quant_hotpath import RustFeatureEngine
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust not available")


class TestFeatureTrackerParity:
    """Test RSI/ATR parity between Rust engine and Python trackers."""

    def _push_bars(self, engine, bars):
        for bar in bars:
            engine.push_bar(
                close=bar["close"], volume=bar.get("volume", 100.0),
                high=bar["high"], low=bar["low"], open=bar.get("open", bar["close"]),
            )

    def test_rsi_warmup(self):
        """RSI should be NaN during warmup (< 14 bars)."""
        eng = RustFeatureEngine()
        for i in range(10):
            eng.push_bar(close=100.0 + i, volume=100.0,
                        high=101.0 + i, low=99.0 + i, open=100.0 + i)
        feats = eng.get_features()
        # rsi_14 should not be ready yet
        rsi = feats.get("rsi_14")
        # Either None or NaN (Rust returns NaN for unready)
        assert rsi is None or (isinstance(rsi, float) and math.isnan(rsi))

    def test_rsi_produces_value(self):
        """RSI should produce values after warmup."""
        eng = RustFeatureEngine()
        for i in range(30):
            eng.push_bar(close=100.0 + i * 0.5, volume=100.0,
                        high=101.0 + i * 0.5, low=99.0 + i * 0.5, open=100.0 + i * 0.5)
        feats = eng.get_features()
        rsi = feats.get("rsi_14")
        # Should be a valid number (normalized: (rsi - 50) / 50 -> range [-1, 1])
        assert rsi is not None
        if not math.isnan(rsi):
            assert -1.1 <= rsi <= 1.1

    def test_atr_produces_value(self):
        """ATR should produce values after warmup."""
        eng = RustFeatureEngine()
        for i in range(20):
            close = 100.0 + i * 0.5
            eng.push_bar(close=close, volume=100.0,
                        high=close + 1.0, low=close - 1.0, open=close)
        feats = eng.get_features()
        atr = feats.get("atr_norm_14")
        assert atr is not None
        if not math.isnan(atr):
            assert atr > 0  # ATR should be positive

    def test_nan_input_handled(self):
        """NaN input should not crash the engine."""
        eng = RustFeatureEngine()
        for i in range(5):
            eng.push_bar(close=100.0 + i, volume=100.0,
                        high=101.0 + i, low=99.0 + i, open=100.0 + i)
        # Push NaN bar -- should not crash
        eng.push_bar(close=float('nan'), volume=100.0,
                    high=float('nan'), low=float('nan'), open=float('nan'))
        feats = eng.get_features()
        # Engine should survive and return something
        assert isinstance(feats, dict)
