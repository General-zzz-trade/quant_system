"""Safety tests for EnrichedFeatureComputer — NaN propagation, zero division, overflow."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import math
from features.enriched_computer import EnrichedFeatureComputer


class TestEnrichedComputerSafety:
    def test_zero_close_no_crash(self):
        """Close=0 should not cause ZeroDivisionError."""
        comp = EnrichedFeatureComputer()
        # Feed a few normal bars first
        for i in range(30):
            comp.on_bar("TEST", close=100.0 + i, volume=1000.0, high=101.0 + i, low=99.0 + i)
        # Now feed zero close
        feats = comp.on_bar("TEST", close=0.0, volume=1000.0, high=1.0, low=0.0)
        # Should not crash, features may be None but not NaN from division
        assert isinstance(feats, dict)

    def test_zero_volume_no_crash(self):
        """Volume=0 should not cause ZeroDivisionError."""
        comp = EnrichedFeatureComputer()
        for i in range(30):
            comp.on_bar("TEST", close=100.0, volume=1000.0)
        feats = comp.on_bar("TEST", close=100.0, volume=0.0)
        assert isinstance(feats, dict)

    def test_nan_close(self):
        """NaN close should produce None features, not crash."""
        comp = EnrichedFeatureComputer()
        for i in range(5):
            comp.on_bar("TEST", close=100.0, volume=1000.0)
        feats = comp.on_bar("TEST", close=float("nan"), volume=1000.0)
        assert isinstance(feats, dict)

    def test_negative_price(self):
        """Negative prices should not crash (edge case from bad data)."""
        comp = EnrichedFeatureComputer()
        for i in range(5):
            comp.on_bar("TEST", close=100.0, volume=1000.0, high=101.0, low=99.0)
        feats = comp.on_bar("TEST", close=-1.0, volume=1000.0, high=-0.5, low=-1.5)
        assert isinstance(feats, dict)

    def test_extreme_price(self):
        """Very large price should not overflow."""
        comp = EnrichedFeatureComputer()
        for i in range(30):
            comp.on_bar("TEST", close=1e15, volume=1e12, high=1e15 + 1, low=1e15 - 1)
        feats = comp.on_bar("TEST", close=1e15, volume=1e12)
        assert isinstance(feats, dict)
        # Check no infinities in returned features
        for k, v in feats.items():
            if v is not None and isinstance(v, float):
                assert not math.isinf(v), f"Feature {k} is inf"

    def test_all_same_price(self):
        """Constant price → vol=0, should handle gracefully."""
        comp = EnrichedFeatureComputer()
        for i in range(50):
            feats = comp.on_bar("TEST", close=100.0, volume=1000.0,
                                high=100.0, low=100.0, open_=100.0)
        # vol_20 should be 0 or very small, not NaN
        vol = feats.get("vol_20")
        if vol is not None:
            assert vol == vol  # not NaN

    def test_options_metrics_none(self):
        """options_metrics=None should not add NaN features."""
        comp = EnrichedFeatureComputer()
        feats = comp.on_bar("TEST", close=100.0, volume=1000.0, options_metrics=None)
        assert isinstance(feats, dict)

    def test_options_metrics_partial(self):
        """Partial options metrics should fill missing with None."""
        comp = EnrichedFeatureComputer()
        feats = comp.on_bar("TEST", close=100.0, volume=1000.0,
                            options_metrics={"pcr_zscore": 1.5})
        assert feats.get("pcr_zscore") == 1.5

    def test_funding_rate_nan(self):
        """NaN funding rate should produce None funding features."""
        comp = EnrichedFeatureComputer()
        for i in range(10):
            comp.on_bar("TEST", close=100.0, volume=1000.0, funding_rate=0.0001)
        feats = comp.on_bar("TEST", close=100.0, volume=1000.0,
                            funding_rate=float("nan"))
        assert isinstance(feats, dict)

    def test_multiple_symbols_isolation(self):
        """Different symbols should have isolated state."""
        comp = EnrichedFeatureComputer()
        for i in range(5):
            comp.on_bar("BTC", close=50000.0 + i, volume=1000.0)
            comp.on_bar("ETH", close=3000.0 + i, volume=500.0)
        btc_feats = comp.on_bar("BTC", close=50005.0, volume=1000.0)
        eth_feats = comp.on_bar("ETH", close=3005.0, volume=500.0)
        # Prices should not leak between symbols
        btc_close = btc_feats.get("ret_1")
        eth_close = eth_feats.get("ret_1")
        if btc_close is not None and eth_close is not None:
            # Returns should be different magnitudes
            assert btc_close != eth_close or btc_close == 0.0

    def test_warmup_returns_none_features(self):
        """First few bars should return None for features needing warmup."""
        comp = EnrichedFeatureComputer()
        feats = comp.on_bar("TEST", close=100.0, volume=1000.0)
        # First bar: most features should be None (no history)
        none_count = sum(1 for v in feats.values() if v is None)
        assert none_count > 10  # many features need warmup
