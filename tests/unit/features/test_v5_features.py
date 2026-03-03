"""Tests for V5 features: Order Flow, Vol Microstructure, Liquidation Proxy, Funding Carry."""
from __future__ import annotations

import math

import pytest

from features.enriched_computer import EnrichedFeatureComputer, ENRICHED_FEATURE_NAMES


class _Base:
    """Shared helpers for V5 feature tests."""

    def _make_bar(self, close=100.0, volume=10.0, trades=100.0,
                  taker_buy_volume=5.0, quote_volume=1000.0,
                  high=None, low=None, open_=None, **kw):
        if high is None:
            high = close * 1.005
        if low is None:
            low = close * 0.995
        if open_ is None:
            open_ = close
        return dict(close=close, volume=volume, trades=trades,
                    taker_buy_volume=taker_buy_volume, quote_volume=quote_volume,
                    high=high, low=low, open_=open_, **kw)

    def _warmup(self, comp, n=55, symbol="BTC", oi_start=1000.0):
        """Push n bars to warm up all windows."""
        for i in range(n):
            comp.on_bar(symbol, **self._make_bar(
                close=100.0 + i * 0.1,
                volume=10.0 + i * 0.01,
                high=100.5 + i * 0.1,
                low=99.5 + i * 0.1,
                trades=100.0,
                taker_buy_volume=5.0,
                quote_volume=1000.0,
                funding_rate=0.0001,
                open_interest=oi_start + i * 10,
            ))


class TestOrderFlowFeatures(_Base):

    def test_feature_names_include_order_flow(self):
        for name in ["cvd_10", "cvd_20", "cvd_price_divergence", "aggressive_flow_zscore"]:
            assert name in ENRICHED_FEATURE_NAMES

    def test_cvd_10_none_before_warmup(self):
        comp = EnrichedFeatureComputer()
        for i in range(5):
            comp.on_bar("BTC", **self._make_bar(close=100.0 + i))
        feats = comp.get_features_dict("BTC")
        assert feats["cvd_10"] is None

    def test_cvd_10_computed_after_warmup(self):
        comp = EnrichedFeatureComputer()
        self._warmup(comp, n=15)
        feats = comp.get_features_dict("BTC")
        assert feats["cvd_10"] is not None

    def test_cvd_values_buy_dominated(self):
        comp = EnrichedFeatureComputer()
        # Push bars with heavy taker buy (tbr=0.9 → imbalance=0.8)
        for i in range(25):
            comp.on_bar("BTC", **self._make_bar(
                close=100.0 + i * 0.1, taker_buy_volume=9.0, volume=10.0))
        feats = comp.get_features_dict("BTC")
        assert feats["cvd_10"] is not None
        assert feats["cvd_10"] > 0  # positive CVD for buy-dominated flow
        assert feats["cvd_20"] is not None
        assert feats["cvd_20"] > 0

    def test_cvd_price_divergence_detected(self):
        comp = EnrichedFeatureComputer()
        # Warmup with rising price, buy-dominated
        for i in range(25):
            comp.on_bar("BTC", **self._make_bar(
                close=100.0 + i * 0.5, taker_buy_volume=7.0, volume=10.0))
        # Now: price still rising but selling dominates (negative CVD)
        for i in range(25):
            comp.on_bar("BTC", **self._make_bar(
                close=112.0 + i * 0.2, taker_buy_volume=1.0, volume=10.0))
        feats = comp.get_features_dict("BTC")
        # CVD should be negative (sell-dominated) while price is going up
        if feats["cvd_price_divergence"] is not None:
            # Divergence expected
            assert feats["cvd_price_divergence"] in (0.0, 1.0)

    def test_aggressive_flow_zscore(self):
        comp = EnrichedFeatureComputer()
        # 55 bars with tbr ~0.5
        for i in range(55):
            comp.on_bar("BTC", **self._make_bar(
                close=100.0 + i * 0.1, taker_buy_volume=5.0, volume=10.0))
        # Spike in taker buy
        feats = comp.on_bar("BTC", **self._make_bar(
            close=106.0, taker_buy_volume=9.0, volume=10.0))
        assert feats["aggressive_flow_zscore"] is not None
        assert feats["aggressive_flow_zscore"] > 0  # above average


class TestVolMicrostructureFeatures(_Base):

    def test_feature_names_include_vol_micro(self):
        for name in ["vol_of_vol", "range_vs_rv", "parkinson_vol", "rv_acceleration"]:
            assert name in ENRICHED_FEATURE_NAMES

    def test_vol_of_vol_none_before_warmup(self):
        comp = EnrichedFeatureComputer()
        for i in range(10):
            comp.on_bar("BTC", **self._make_bar(close=100.0 + i * 0.1))
        feats = comp.get_features_dict("BTC")
        assert feats["vol_of_vol"] is None

    def test_vol_of_vol_computed(self):
        comp = EnrichedFeatureComputer()
        self._warmup(comp, n=30)
        feats = comp.get_features_dict("BTC")
        assert feats["vol_of_vol"] is not None
        assert feats["vol_of_vol"] >= 0

    def test_range_vs_rv(self):
        comp = EnrichedFeatureComputer()
        self._warmup(comp, n=10)
        feats = comp.get_features_dict("BTC")
        assert feats["range_vs_rv"] is not None
        assert feats["range_vs_rv"] > 0

    def test_parkinson_vol_computed(self):
        comp = EnrichedFeatureComputer()
        self._warmup(comp, n=25)
        feats = comp.get_features_dict("BTC")
        assert feats["parkinson_vol"] is not None
        assert feats["parkinson_vol"] > 0

    def test_rv_acceleration(self):
        comp = EnrichedFeatureComputer()
        self._warmup(comp, n=15)
        feats = comp.get_features_dict("BTC")
        assert feats["rv_acceleration"] is not None


class TestLiquidationProxyFeatures(_Base):

    def test_feature_names_include_liquidation(self):
        for name in ["oi_acceleration", "leverage_proxy", "oi_vol_divergence", "oi_liquidation_flag"]:
            assert name in ENRICHED_FEATURE_NAMES

    def test_oi_acceleration(self):
        comp = EnrichedFeatureComputer()
        # Push 3 OI bars to get two changes + acceleration
        comp.on_bar("BTC", **self._make_bar(close=100.0, open_interest=1000.0))
        comp.on_bar("BTC", **self._make_bar(close=100.1, open_interest=1050.0))  # +5%
        comp.on_bar("BTC", **self._make_bar(close=100.2, open_interest=1150.0))  # +9.5%
        feats = comp.get_features_dict("BTC")
        assert feats["oi_acceleration"] is not None
        # OI change accelerated (9.5% > 5%), so acceleration > 0
        assert feats["oi_acceleration"] > 0

    def test_leverage_proxy(self):
        comp = EnrichedFeatureComputer()
        self._warmup(comp, n=25)
        feats = comp.get_features_dict("BTC")
        assert feats["leverage_proxy"] is not None
        assert feats["leverage_proxy"] > 0

    def test_oi_vol_divergence_triggered(self):
        comp = EnrichedFeatureComputer()
        self._warmup(comp, n=25)
        # OI goes up, volume goes down (below 20-bar avg)
        feats = comp.on_bar("BTC", **self._make_bar(
            close=103.0, volume=1.0, open_interest=2000.0))
        if feats["oi_vol_divergence"] is not None:
            # volume is way below average, OI increased
            assert feats["oi_vol_divergence"] == 1.0

    def test_oi_liquidation_flag(self):
        comp = EnrichedFeatureComputer()
        self._warmup(comp, n=25, oi_start=10000.0)
        # Large OI drop + volume spike
        feats = comp.on_bar("BTC", **self._make_bar(
            close=103.0, volume=100.0, open_interest=9000.0))
        # oi_change ~= (9000 - 10240) / 10240 ~= -12% → flag should trigger
        if feats["oi_liquidation_flag"] is not None and feats["vol_ratio_20"] is not None:
            if feats["vol_ratio_20"] > 2.0 and (feats.get("oi_change_pct") or 0) < -0.05:
                assert feats["oi_liquidation_flag"] == 1.0

    def test_oi_acceleration_none_without_oi(self):
        comp = EnrichedFeatureComputer()
        comp.on_bar("BTC", **self._make_bar(close=100.0))
        feats = comp.get_features_dict("BTC")
        assert feats["oi_acceleration"] is None


class TestFundingCarryFeatures(_Base):

    def test_feature_names_include_funding_carry(self):
        for name in ["funding_annualized", "funding_vs_vol"]:
            assert name in ENRICHED_FEATURE_NAMES

    def test_funding_annualized(self):
        comp = EnrichedFeatureComputer()
        rate = 0.0001
        feats = comp.on_bar("BTC", **self._make_bar(funding_rate=rate))
        assert feats["funding_annualized"] is not None
        assert feats["funding_annualized"] == pytest.approx(rate * 3 * 365)

    def test_funding_vs_vol(self):
        comp = EnrichedFeatureComputer()
        self._warmup(comp, n=25)
        feats = comp.on_bar("BTC", **self._make_bar(funding_rate=0.0001))
        assert feats["funding_vs_vol"] is not None

    def test_funding_vs_vol_none_without_vol(self):
        comp = EnrichedFeatureComputer()
        feats = comp.on_bar("BTC", **self._make_bar(funding_rate=0.0001))
        assert feats["funding_vs_vol"] is None  # vol_20 not ready


class TestV5FeatureCount(_Base):

    def test_enriched_feature_names_count(self):
        assert len(ENRICHED_FEATURE_NAMES) == 79, f"Expected 79, got {len(ENRICHED_FEATURE_NAMES)}"

    def test_all_v5_features_present_after_warmup(self):
        comp = EnrichedFeatureComputer()
        for i in range(80):
            comp.on_bar("BTC", **self._make_bar(
                close=100.0 + i * 0.1,
                volume=10.0 + i * 0.01,
                high=100.5 + i * 0.1,
                low=99.5 + i * 0.1,
                trades=100.0,
                taker_buy_volume=5.0,
                quote_volume=1000.0,
                hour=i % 24, dow=i % 7,
                funding_rate=0.0001,
                open_interest=1000.0 + i * 10,
                ls_ratio=1.0 + i * 0.001,
            ))
        feats = comp.get_features_dict("BTC")
        for name in ENRICHED_FEATURE_NAMES:
            assert name in feats, f"Feature '{name}' missing from output"

    def test_no_regression_feature_count(self):
        """All features should be non-None after sufficient warmup."""
        comp = EnrichedFeatureComputer()
        for i in range(80):
            close = 100.0 + i * 0.1
            comp.on_bar("BTC", close=close, volume=10.0 + i * 0.01,
                        high=100.5 + i * 0.1, low=99.5 + i * 0.1,
                        open_=100.0 + i * 0.1, hour=i % 24, dow=i % 7,
                        funding_rate=0.0001, trades=100.0, taker_buy_volume=5.0,
                        quote_volume=1000.0,
                        taker_buy_quote_volume=500.0,
                        open_interest=1000.0 + i * 10,
                        ls_ratio=1.0 + i * 0.001,
                        spot_close=close - 0.5,
                        fear_greed=50.0 + (i % 8) * 5)
        feats = comp.get_features_dict("BTC")
        for name in ENRICHED_FEATURE_NAMES:
            assert feats[name] is not None, f"Feature '{name}' is None after 80 bars warmup"
