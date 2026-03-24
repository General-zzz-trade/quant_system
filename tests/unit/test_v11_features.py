"""Tests for V11 features: liquidation, mempool, macro, sentiment."""
from __future__ import annotations

import pytest
from features.enriched_computer import EnrichedFeatureComputer, ENRICHED_FEATURE_NAMES


class TestV11FeatureNames:
    """Verify V11 feature names are registered."""

    V11_FEATURES = [
        "liquidation_volume_zscore_24",
        "liquidation_imbalance",
        "liquidation_volume_ratio",
        "liquidation_cluster_flag",
        "mempool_fee_zscore_24",
        "mempool_size_zscore_24",
        "fee_urgency_ratio",
        "dxy_change_5d",
        "spx_btc_corr_30d",
        "spx_overnight_ret",
        "vix_zscore_14",
        "social_volume_zscore_24",
        "social_sentiment_score",
        "social_volume_price_div",
    ]

    def test_all_v11_in_feature_names(self):
        for name in self.V11_FEATURES:
            assert name in ENRICHED_FEATURE_NAMES, f"{name} not in ENRICHED_FEATURE_NAMES"

    def test_feature_catalog_has_not_regressed(self):
        assert len(ENRICHED_FEATURE_NAMES) >= 105
        assert len(ENRICHED_FEATURE_NAMES) == len(set(ENRICHED_FEATURE_NAMES))


class TestLiquidationFeatures:
    """Test liquidation feature computation."""

    def test_liq_features_none_without_data(self):
        comp = EnrichedFeatureComputer()
        feats = comp.on_bar("BTCUSDT", close=50000, volume=100)
        assert feats["liquidation_volume_zscore_24"] is None
        assert feats["liquidation_imbalance"] is None
        assert feats["liquidation_volume_ratio"] is None
        assert feats["liquidation_cluster_flag"] is None

    def test_liq_imbalance_with_data(self):
        comp = EnrichedFeatureComputer()
        liq = {"liq_total_volume": 1000, "liq_buy_volume": 700, "liq_sell_volume": 300, "liq_count": 5}
        feats = comp.on_bar("BTCUSDT", close=50000, volume=100, liquidation_metrics=liq)
        # imbalance = (700 - 300) / 1000 = 0.4
        assert feats["liquidation_imbalance"] == pytest.approx(0.4)

    def test_liq_volume_ratio(self):
        comp = EnrichedFeatureComputer()
        liq = {"liq_total_volume": 500, "liq_buy_volume": 250, "liq_sell_volume": 250, "liq_count": 2}
        feats = comp.on_bar("BTCUSDT", close=50000, volume=100, quote_volume=10000, liquidation_metrics=liq)
        # ratio = 500 / 10000 = 0.05
        assert feats["liquidation_volume_ratio"] == pytest.approx(0.05)

    def test_liq_zscore_after_24_bars(self):
        comp = EnrichedFeatureComputer()
        for i in range(25):
            liq = {"liq_total_volume": 100 + i, "liq_buy_volume": 50, "liq_sell_volume": 50, "liq_count": 1}
            feats = comp.on_bar("BTCUSDT", close=50000, volume=100, liquidation_metrics=liq)
        assert feats["liquidation_volume_zscore_24"] is not None


class TestMempoolFeatures:
    """Test mempool feature computation."""

    def test_mempool_none_without_data(self):
        comp = EnrichedFeatureComputer()
        feats = comp.on_bar("BTCUSDT", close=50000, volume=100)
        assert feats["mempool_fee_zscore_24"] is None
        assert feats["fee_urgency_ratio"] is None

    def test_fee_urgency_ratio(self):
        comp = EnrichedFeatureComputer()
        mp = {"fastest_fee": 100, "economy_fee": 10, "mempool_size": 50000}
        feats = comp.on_bar("BTCUSDT", close=50000, volume=100, mempool_metrics=mp)
        assert feats["fee_urgency_ratio"] == pytest.approx(10.0)

    def test_mempool_zscore_after_24_bars(self):
        comp = EnrichedFeatureComputer()
        for i in range(25):
            mp = {"fastest_fee": 50 + i, "economy_fee": 5, "mempool_size": 40000 + i * 1000}
            feats = comp.on_bar("BTCUSDT", close=50000, volume=100, mempool_metrics=mp)
        assert feats["mempool_fee_zscore_24"] is not None
        assert feats["mempool_size_zscore_24"] is not None


class TestMacroFeatures:
    """Test macro feature computation."""

    def test_macro_none_without_data(self):
        comp = EnrichedFeatureComputer()
        feats = comp.on_bar("BTCUSDT", close=50000, volume=100)
        assert feats["dxy_change_5d"] is None
        assert feats["vix_zscore_14"] is None

    def test_dxy_change_5d(self):
        comp = EnrichedFeatureComputer()
        for i in range(7):
            macro = {"date": f"2024-01-0{i+1}", "dxy": 100 + i, "spx": 4500 + i * 10, "vix": 15 + i}
            feats = comp.on_bar("BTCUSDT", close=50000, volume=100, macro_metrics=macro)
        # dxy_change_5d = (106 - 101) / 101
        assert feats["dxy_change_5d"] is not None
        assert feats["dxy_change_5d"] == pytest.approx(5 / 101, rel=0.01)

    def test_spx_overnight_ret(self):
        comp = EnrichedFeatureComputer()
        macro1 = {"date": "2024-01-01", "spx": 4500}
        comp.on_bar("BTCUSDT", close=50000, volume=100, macro_metrics=macro1)
        macro2 = {"date": "2024-01-02", "spx": 4550}
        feats = comp.on_bar("BTCUSDT", close=50000, volume=100, macro_metrics=macro2)
        # overnight_ret = (4550 - 4500) / 4500
        assert feats["spx_overnight_ret"] == pytest.approx(50 / 4500, rel=0.01)

    def test_vix_zscore_14(self):
        comp = EnrichedFeatureComputer()
        for i in range(15):
            macro = {"date": f"2024-01-{i+1:02d}", "vix": 15.0 + i * 0.5}
            feats = comp.on_bar("BTCUSDT", close=50000, volume=100, macro_metrics=macro)
        assert feats["vix_zscore_14"] is not None

    def test_macro_date_dedup(self):
        """Same date should not push VIX buffer twice."""
        comp = EnrichedFeatureComputer()
        macro = {"date": "2024-01-01", "vix": 15.0}
        comp.on_bar("BTCUSDT", close=50000, volume=100, macro_metrics=macro)
        comp.on_bar("BTCUSDT", close=50100, volume=100, macro_metrics=macro)
        # Only 1 entry in vix_buf
        state = comp._states["BTCUSDT"]
        assert len(state._vix_buf) == 1


class TestSentimentFeatures:
    """Test sentiment feature computation."""

    def test_sentiment_none_without_data(self):
        comp = EnrichedFeatureComputer()
        feats = comp.on_bar("BTCUSDT", close=50000, volume=100)
        assert feats["social_volume_zscore_24"] is None
        assert feats["social_sentiment_score"] is None
        assert feats["social_volume_price_div"] is None

    def test_sentiment_score_passthrough(self):
        comp = EnrichedFeatureComputer()
        sent = {"social_volume": 50, "sentiment_score": 0.75}
        feats = comp.on_bar("BTCUSDT", close=50000, volume=100, sentiment_metrics=sent)
        assert feats["social_sentiment_score"] == pytest.approx(0.75)



class TestFeatureHookV11:
    """Test FeatureComputeHook with new V11 sources."""

    def test_hook_accepts_new_sources(self):
        from engine.feature_hook import FeatureComputeHook
        from types import SimpleNamespace
        from datetime import datetime
        comp = EnrichedFeatureComputer()
        hook = FeatureComputeHook(
            comp,
            liquidation_source=lambda: {"liq_total_volume": 100, "liq_buy_volume": 50, "liq_sell_volume": 50,
                "liq_count": 1},
            mempool_source=lambda: {"fastest_fee": 50, "economy_fee": 5, "mempool_size": 40000},
            macro_source=lambda: {"date": "2024-01-01", "dxy": 100, "spx": 4500, "vix": 15},
            sentiment_source=lambda: {"social_volume": 50, "sentiment_score": 0.75},
        )
        assert hook._liquidation_source is not None
        assert hook._mempool_source is not None
        assert hook._macro_source is not None
        assert hook._sentiment_source is not None
        # Verify sources flow through to Rust path
        ev = SimpleNamespace(event_type="MARKET", symbol="BTCUSDT",
                             close=50000.0, volume=100.0, high=50100.0,
                             low=49900.0, open=49950.0, ts=datetime(2024, 1, 1))
        feats = hook.on_event(ev)
        assert feats is not None
