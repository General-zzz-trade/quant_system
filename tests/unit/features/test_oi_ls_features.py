"""Tests for OI and LS Ratio features in EnrichedFeatureComputer."""
from __future__ import annotations

import pytest

from features.enriched_computer import EnrichedFeatureComputer, ENRICHED_FEATURE_NAMES


class TestOIFeatures:
    def test_feature_names_include_oi(self):
        for name in ["oi_change_pct", "oi_change_ma8", "oi_close_divergence"]:
            assert name in ENRICHED_FEATURE_NAMES

    def test_oi_none_without_data(self):
        comp = EnrichedFeatureComputer()
        feats = comp.on_bar("BTC", close=100.0, volume=10.0)
        assert feats["oi_change_pct"] is None
        assert feats["oi_change_ma8"] is None
        assert feats["oi_close_divergence"] is None

    def test_oi_change_pct(self):
        comp = EnrichedFeatureComputer()
        comp.on_bar("BTC", close=100.0, volume=10.0, open_interest=1000.0)
        feats = comp.on_bar("BTC", close=101.0, volume=10.0, open_interest=1100.0)
        assert feats["oi_change_pct"] == pytest.approx(0.10)

    def test_oi_change_ma8(self):
        comp = EnrichedFeatureComputer()
        for i in range(12):
            comp.on_bar("BTC", close=100.0 + i, volume=10.0,
                        open_interest=1000.0 + i * 10)
        feats = comp.on_bar("BTC", close=112.0, volume=10.0, open_interest=1130.0)
        assert feats["oi_change_ma8"] is not None

    def test_oi_close_divergence_opposite(self):
        comp = EnrichedFeatureComputer()
        # Need 2 bars for ret_1, 2 bars for OI change
        comp.on_bar("BTC", close=100.0, volume=10.0, high=101.0, low=99.0,
                    open_=100.0, open_interest=1000.0)
        comp.on_bar("BTC", close=99.0, volume=10.0, high=100.0, low=98.0,
                    open_=100.0, open_interest=1000.0)
        # Price down, OI up → divergence = positive (opposite signs)
        feats = comp.on_bar("BTC", close=97.0, volume=10.0, high=99.0, low=96.0,
                            open_=99.0, open_interest=1100.0)
        # ret_1 is negative (97 < 99), OI change is positive (1100 > 1000)
        # divergence = -(-1 * 1) = 1.0
        assert feats["oi_close_divergence"] == pytest.approx(1.0)

    def test_oi_close_divergence_same_dir(self):
        comp = EnrichedFeatureComputer()
        comp.on_bar("BTC", close=100.0, volume=10.0, high=101.0, low=99.0,
                    open_=100.0, open_interest=1000.0)
        # Price up, OI up → divergence = negative (same direction)
        feats = comp.on_bar("BTC", close=110.0, volume=10.0, high=111.0, low=109.0,
                            open_=100.0, open_interest=1100.0)
        assert feats["oi_close_divergence"] == pytest.approx(-1.0)


class TestLSRatioFeatures:
    def test_feature_names_include_ls(self):
        for name in ["ls_ratio", "ls_ratio_zscore_24", "ls_extreme"]:
            assert name in ENRICHED_FEATURE_NAMES

    def test_ls_none_without_data(self):
        comp = EnrichedFeatureComputer()
        feats = comp.on_bar("BTC", close=100.0, volume=10.0)
        assert feats["ls_ratio"] is None
        assert feats["ls_ratio_zscore_24"] is None

    def test_ls_ratio_stored(self):
        comp = EnrichedFeatureComputer()
        feats = comp.on_bar("BTC", close=100.0, volume=10.0, ls_ratio=1.5)
        assert feats["ls_ratio"] == 1.5

    def test_ls_zscore_after_warmup(self):
        comp = EnrichedFeatureComputer()
        import random
        random.seed(42)
        for i in range(24):
            comp.on_bar("BTC", close=100.0, volume=10.0,
                        ls_ratio=1.0 + random.gauss(0, 0.1))
        feats = comp.on_bar("BTC", close=100.0, volume=10.0, ls_ratio=1.0)
        assert feats["ls_ratio_zscore_24"] is not None

    def test_ls_extreme_flag(self):
        comp = EnrichedFeatureComputer()
        for i in range(24):
            comp.on_bar("BTC", close=100.0, volume=10.0, ls_ratio=1.0)
        # Push extreme ratio (should have std ~0 from constant, so use varying)
        comp2 = EnrichedFeatureComputer()
        import random
        random.seed(42)
        for i in range(24):
            comp2.on_bar("BTC", close=100.0, volume=10.0,
                         ls_ratio=1.0 + random.gauss(0, 0.05))
        feats = comp2.on_bar("BTC", close=100.0, volume=10.0, ls_ratio=2.0)
        assert feats["ls_extreme"] == 1.0
