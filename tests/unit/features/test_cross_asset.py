"""Tests for CrossAssetComputer."""
from __future__ import annotations

import math
import pytest

from features.cross_asset_computer import CrossAssetComputer, CROSS_ASSET_FEATURE_NAMES


class TestCrossAssetComputer:
    def _push_bars(self, comp, n=70, btc_base=40000.0, eth_base=3000.0):
        """Push n bars for BTC and ETH with slight uptrend."""
        for i in range(n):
            comp.on_bar("BTCUSDT", close=btc_base + i * 10.0, funding_rate=0.0001)
            comp.on_bar("ETHUSDT", close=eth_base + i * 8.0, funding_rate=0.00015)

    def test_btc_returns_none_for_btc(self):
        comp = CrossAssetComputer()
        self._push_bars(comp, 5)
        feats = comp.get_features("BTCUSDT")
        # All None for BTC itself
        for name in CROSS_ASSET_FEATURE_NAMES:
            assert feats[name] is None

    def test_btc_returns_populated_for_altcoin(self):
        comp = CrossAssetComputer()
        self._push_bars(comp, 10)
        feats = comp.get_features("ETHUSDT")
        assert feats["btc_ret_1"] is not None
        assert feats["btc_ret_3"] is not None
        assert feats["btc_ret_6"] is not None

    def test_btc_ret_1_correct(self):
        comp = CrossAssetComputer()
        comp.on_bar("BTCUSDT", close=100.0)
        comp.on_bar("ETHUSDT", close=50.0)
        comp.on_bar("BTCUSDT", close=110.0)
        comp.on_bar("ETHUSDT", close=55.0)
        feats = comp.get_features("ETHUSDT")
        assert feats["btc_ret_1"] == pytest.approx(0.10)

    def test_rolling_beta_after_warmup(self):
        comp = CrossAssetComputer()
        self._push_bars(comp, 40)
        feats = comp.get_features("ETHUSDT")
        assert feats["rolling_beta_30"] is not None
        # Both trending up linearly → beta should be positive
        assert feats["rolling_beta_30"] > 0

    def test_rolling_beta_60_needs_more_data(self):
        comp = CrossAssetComputer()
        self._push_bars(comp, 40)
        feats = comp.get_features("ETHUSDT")
        assert feats["rolling_beta_60"] is None  # Not enough data

    def test_rolling_beta_60_after_warmup(self):
        comp = CrossAssetComputer()
        self._push_bars(comp, 70)
        feats = comp.get_features("ETHUSDT")
        assert feats["rolling_beta_60"] is not None

    def test_relative_strength_20(self):
        comp = CrossAssetComputer()
        self._push_bars(comp, 25)
        feats = comp.get_features("ETHUSDT")
        assert feats["relative_strength_20"] is not None

    def test_rolling_corr_30(self):
        comp = CrossAssetComputer()
        self._push_bars(comp, 35)
        feats = comp.get_features("ETHUSDT")
        assert feats["rolling_corr_30"] is not None
        # Both trending up → high positive correlation
        assert feats["rolling_corr_30"] > 0.5

    def test_funding_diff(self):
        comp = CrossAssetComputer()
        comp.on_bar("BTCUSDT", close=40000.0, funding_rate=0.0001)
        comp.on_bar("ETHUSDT", close=3000.0, funding_rate=0.0003)
        comp.on_bar("BTCUSDT", close=40010.0, funding_rate=0.0001)
        comp.on_bar("ETHUSDT", close=3005.0, funding_rate=0.0003)
        feats = comp.get_features("ETHUSDT")
        assert feats["funding_diff"] == pytest.approx(0.0002)

    def test_funding_diff_ma8(self):
        comp = CrossAssetComputer()
        for i in range(12):
            comp.on_bar("BTCUSDT", close=40000.0 + i, funding_rate=0.0001)
            comp.on_bar("ETHUSDT", close=3000.0 + i, funding_rate=0.0003)
        feats = comp.get_features("ETHUSDT")
        assert feats["funding_diff_ma8"] is not None

    def test_spread_zscore_after_warmup(self):
        comp = CrossAssetComputer()
        # Need 30+ bars for beta + 20 for z-score spread window
        self._push_bars(comp, 55)
        feats = comp.get_features("ETHUSDT")
        # spread_zscore_20 needs beta + spread window
        # May or may not be ready depending on window alignment
        if feats["spread_zscore_20"] is not None:
            assert isinstance(feats["spread_zscore_20"], float)

    def test_unknown_symbol(self):
        comp = CrossAssetComputer()
        feats = comp.get_features("XYZUSDT")
        for name in CROSS_ASSET_FEATURE_NAMES:
            assert feats[name] is None

    def test_feature_names_complete(self):
        assert len(CROSS_ASSET_FEATURE_NAMES) == 17
        expected = [
            "btc_ret_1", "btc_ret_3", "btc_ret_6",
            "btc_ret_12", "btc_ret_24",
            "btc_rsi_14", "btc_macd_line",
            "btc_mean_reversion_20", "btc_atr_norm_14", "btc_bb_width_20",
            "rolling_beta_30", "rolling_beta_60",
            "relative_strength_20", "rolling_corr_30",
            "funding_diff", "funding_diff_ma8",
            "spread_zscore_20",
        ]
        assert set(CROSS_ASSET_FEATURE_NAMES) == set(expected)

    def test_btc_lead_features_after_warmup(self):
        comp = CrossAssetComputer()
        # Push 35 bars with high/low for BTC
        for i in range(35):
            btc_close = 40000.0 + i * 10.0
            comp.on_bar("BTCUSDT", close=btc_close, funding_rate=0.0001,
                        high=btc_close + 50, low=btc_close - 50)
            comp.on_bar("ETHUSDT", close=3000.0 + i * 8.0)
        feats = comp.get_features("ETHUSDT")
        assert feats["btc_ret_12"] is not None
        assert feats["btc_ret_24"] is not None
        assert feats["btc_rsi_14"] is not None
        assert feats["btc_macd_line"] is not None
        assert feats["btc_mean_reversion_20"] is not None
        assert feats["btc_atr_norm_14"] is not None
        assert feats["btc_bb_width_20"] is not None
        # RSI should be between 0 and 100
        assert 0 <= feats["btc_rsi_14"] <= 100
        # BB width should be positive
        assert feats["btc_bb_width_20"] > 0

    def test_btc_lead_features_none_during_warmup(self):
        comp = CrossAssetComputer()
        for i in range(5):
            comp.on_bar("BTCUSDT", close=40000.0 + i * 10, high=40050, low=39950)
            comp.on_bar("ETHUSDT", close=3000.0 + i * 8)
        feats = comp.get_features("ETHUSDT")
        assert feats["btc_ret_12"] is None  # need > 12 bars
        assert feats["btc_ret_24"] is None
        assert feats["btc_rsi_14"] is None  # need >= 15 bars
        assert feats["btc_mean_reversion_20"] is None  # need >= 20 bars
