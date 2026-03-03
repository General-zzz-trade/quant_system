"""Tests for kline microstructure + funding deep features in EnrichedFeatureComputer."""
from __future__ import annotations

import pytest

from features.enriched_computer import EnrichedFeatureComputer, ENRICHED_FEATURE_NAMES


class TestMicrostructureFeatures:
    """Test the 8 kline microstructure features."""

    def _make_bar(self, close=100.0, volume=10.0, trades=100.0,
                  taker_buy_volume=5.0, quote_volume=1000.0, **kw):
        return dict(close=close, volume=volume, trades=trades,
                    taker_buy_volume=taker_buy_volume,
                    quote_volume=quote_volume, **kw)

    def _warmup(self, comp, n=25, symbol="BTC"):
        """Push n bars to warm up EMAs."""
        for i in range(n):
            comp.on_bar(symbol, **self._make_bar(
                close=100.0 + i * 0.1,
                volume=10.0 + i * 0.01,
                trades=100.0,
                taker_buy_volume=5.0,
                quote_volume=1000.0,
            ))

    def test_feature_names_include_microstructure(self):
        expected = [
            "trade_intensity", "taker_buy_ratio", "taker_buy_ratio_ma10",
            "taker_imbalance", "avg_trade_size", "avg_trade_size_ratio",
            "volume_per_trade", "trade_count_regime",
        ]
        for name in expected:
            assert name in ENRICHED_FEATURE_NAMES, f"{name} not in ENRICHED_FEATURE_NAMES"

    def test_microstructure_none_without_trades(self):
        comp = EnrichedFeatureComputer()
        feats = comp.on_bar("BTC", close=100.0, volume=10.0)
        assert feats["trade_intensity"] is None
        assert feats["taker_buy_ratio"] is None
        assert feats["taker_imbalance"] is None

    def test_taker_buy_ratio_range(self):
        comp = EnrichedFeatureComputer()
        self._warmup(comp)
        feats = comp.on_bar("BTC", **self._make_bar(taker_buy_volume=7.0, volume=10.0))
        assert feats["taker_buy_ratio"] == pytest.approx(0.7)
        assert -1.0 <= feats["taker_imbalance"] <= 1.0

    def test_taker_imbalance_symmetric(self):
        comp = EnrichedFeatureComputer()
        self._warmup(comp)
        # All buy
        feats_buy = comp.on_bar("BTC", **self._make_bar(taker_buy_volume=10.0, volume=10.0))
        assert feats_buy["taker_imbalance"] == pytest.approx(1.0)

        comp2 = EnrichedFeatureComputer()
        self._warmup(comp2)
        # All sell (0 taker buy)
        feats_sell = comp2.on_bar("BTC", **self._make_bar(taker_buy_volume=0.0, volume=10.0))
        assert feats_sell["taker_imbalance"] == pytest.approx(-1.0)

    def test_trade_intensity_above_average(self):
        comp = EnrichedFeatureComputer()
        # Warm up with 100 trades per bar
        self._warmup(comp, n=25)
        # Spike: 200 trades
        feats = comp.on_bar("BTC", **self._make_bar(trades=200.0))
        assert feats["trade_intensity"] is not None
        assert feats["trade_intensity"] > 1.5  # should be ~2x

    def test_trade_intensity_below_average(self):
        comp = EnrichedFeatureComputer()
        self._warmup(comp, n=25)
        # Low activity: 50 trades
        feats = comp.on_bar("BTC", **self._make_bar(trades=50.0))
        assert feats["trade_intensity"] is not None
        assert feats["trade_intensity"] < 0.8

    def test_avg_trade_size(self):
        comp = EnrichedFeatureComputer()
        self._warmup(comp)
        feats = comp.on_bar("BTC", **self._make_bar(
            trades=100.0, quote_volume=2000.0))
        assert feats["avg_trade_size"] == pytest.approx(20.0)

    def test_avg_trade_size_ratio_whale(self):
        comp = EnrichedFeatureComputer()
        self._warmup(comp, n=25)  # avg_trade_size ~10
        # Whale trade: much higher avg size
        feats = comp.on_bar("BTC", **self._make_bar(
            trades=100.0, quote_volume=5000.0))
        assert feats["avg_trade_size_ratio"] is not None
        assert feats["avg_trade_size_ratio"] > 1.5

    def test_volume_per_trade_normalized(self):
        comp = EnrichedFeatureComputer()
        self._warmup(comp)
        feats = comp.on_bar("BTC", **self._make_bar(volume=10.0, trades=100.0))
        assert feats["volume_per_trade"] is not None

    def test_trade_count_regime(self):
        comp = EnrichedFeatureComputer()
        self._warmup(comp, n=25)
        feats = comp.on_bar("BTC", **self._make_bar())
        assert feats["trade_count_regime"] is not None
        # Stable trades → regime ~1.0
        assert 0.8 < feats["trade_count_regime"] < 1.2

    def test_taker_buy_ratio_ma10_smoothed(self):
        comp = EnrichedFeatureComputer()
        # Push 15 bars to warm up EMA(10)
        for i in range(15):
            comp.on_bar("BTC", **self._make_bar(
                close=100.0 + i,
                taker_buy_volume=5.0 + i * 0.1,
                volume=10.0,
            ))
        feats = comp.on_bar("BTC", **self._make_bar())
        assert feats["taker_buy_ratio_ma10"] is not None

    def test_zero_volume_safe(self):
        comp = EnrichedFeatureComputer()
        self._warmup(comp)
        feats = comp.on_bar("BTC", **self._make_bar(volume=0.0, trades=100.0))
        # taker_buy_ratio uses volume as denominator — 0 volume → fallback 0.5 in push
        # but get_features checks volume > 0 → None
        assert feats["taker_buy_ratio"] is None
        # No crash — that's the key assertion
        assert feats["trade_intensity"] is not None


class TestFundingDeepFeatures:
    """Test the 5 funding deep features."""

    def test_feature_names_include_funding_deep(self):
        expected = [
            "funding_zscore_24", "funding_momentum", "funding_extreme",
            "funding_cumulative_8", "funding_sign_persist",
        ]
        for name in expected:
            assert name in ENRICHED_FEATURE_NAMES, f"{name} not in ENRICHED_FEATURE_NAMES"

    def _push_funding(self, comp, n, rate=0.0001, symbol="BTC"):
        for i in range(n):
            comp.on_bar(symbol, close=100.0 + i * 0.01,
                        volume=10.0, funding_rate=rate)

    def test_funding_zscore_warmup(self):
        comp = EnrichedFeatureComputer()
        # Need 24 funding bars
        self._push_funding(comp, 20)
        feats = comp.on_bar("BTC", close=100.0, volume=10.0, funding_rate=0.0001)
        assert feats["funding_zscore_24"] is None  # not enough

        self._push_funding(comp, 4)
        feats = comp.on_bar("BTC", close=100.0, volume=10.0, funding_rate=0.0005)
        assert feats["funding_zscore_24"] is not None

    def test_funding_extreme_flag(self):
        comp = EnrichedFeatureComputer()
        # Push 24 bars with stable rate
        for i in range(24):
            comp.on_bar("BTC", close=100.0, volume=10.0, funding_rate=0.0001)
        # Push extreme rate
        feats = comp.on_bar("BTC", close=100.0, volume=10.0, funding_rate=0.01)
        assert feats["funding_extreme"] == 1.0

    def test_funding_extreme_not_triggered(self):
        comp = EnrichedFeatureComputer()
        # Use varying rates so std > 0
        import random
        random.seed(42)
        for i in range(24):
            rate = 0.0001 + random.gauss(0, 0.00002)
            comp.on_bar("BTC", close=100.0, volume=10.0, funding_rate=rate)
        # Normal rate near mean — should NOT be extreme
        feats = comp.on_bar("BTC", close=100.0, volume=10.0, funding_rate=0.0001)
        assert feats["funding_extreme"] == 0.0

    def test_funding_momentum(self):
        comp = EnrichedFeatureComputer()
        # Push 10 bars (enough for funding_ma8)
        for i in range(10):
            comp.on_bar("BTC", close=100.0, volume=10.0, funding_rate=0.0001)
        # Push higher rate
        feats = comp.on_bar("BTC", close=100.0, volume=10.0, funding_rate=0.001)
        assert feats["funding_momentum"] is not None
        assert feats["funding_momentum"] > 0  # rate > ma

    def test_funding_cumulative_8(self):
        comp = EnrichedFeatureComputer()
        rate = 0.0001
        for i in range(8):
            comp.on_bar("BTC", close=100.0, volume=10.0, funding_rate=rate)
        feats = comp.on_bar("BTC", close=100.0, volume=10.0, funding_rate=rate)
        # deque maxlen=8 — last 8 are all `rate`
        assert feats["funding_cumulative_8"] == pytest.approx(8 * rate)

    def test_funding_sign_persist(self):
        comp = EnrichedFeatureComputer()
        # 5 positive funding bars
        for i in range(5):
            comp.on_bar("BTC", close=100.0, volume=10.0, funding_rate=0.0001)
        feats = comp.on_bar("BTC", close=100.0, volume=10.0, funding_rate=0.0001)
        assert feats["funding_sign_persist"] == 6.0

    def test_funding_sign_persist_resets(self):
        comp = EnrichedFeatureComputer()
        for i in range(5):
            comp.on_bar("BTC", close=100.0, volume=10.0, funding_rate=0.0001)
        # Switch sign
        feats = comp.on_bar("BTC", close=100.0, volume=10.0, funding_rate=-0.0001)
        assert feats["funding_sign_persist"] == 1.0

    def test_no_funding_all_none(self):
        comp = EnrichedFeatureComputer()
        feats = comp.on_bar("BTC", close=100.0, volume=10.0)
        assert feats["funding_zscore_24"] is None
        assert feats["funding_momentum"] is None
        assert feats["funding_cumulative_8"] is None
        assert feats["funding_sign_persist"] is None


class TestFeatureCountConsistency:
    """Verify total feature count matches ENRICHED_FEATURE_NAMES."""

    def test_all_features_present(self):
        comp = EnrichedFeatureComputer()
        # Push enough bars for all features to be computed
        for i in range(70):
            comp.on_bar("BTC", close=100.0 + i * 0.1, volume=10.0,
                        high=100.5 + i * 0.1, low=99.5 + i * 0.1,
                        open_=100.0 + i * 0.1, hour=i % 24, dow=i % 7,
                        funding_rate=0.0001,
                        trades=100.0, taker_buy_volume=5.0,
                        quote_volume=1000.0,
                        taker_buy_quote_volume=500.0)
        feats = comp.get_features_dict("BTC")
        for name in ENRICHED_FEATURE_NAMES:
            assert name in feats, f"Feature '{name}' missing from output"

    def test_feature_count_matches(self):
        """All ENRICHED_FEATURE_NAMES features should be in output dict."""
        comp = EnrichedFeatureComputer()
        for i in range(80):
            close = 100.0 + i * 0.1
            comp.on_bar("BTC", close=close, volume=10.0 + i * 0.01,
                        high=100.5 + i * 0.1, low=99.5 + i * 0.1,
                        open_=100.0 + i * 0.1, hour=i % 24, dow=i % 7,
                        funding_rate=0.0001,
                        trades=100.0, taker_buy_volume=5.0 + i * 0.02,
                        quote_volume=1000.0,
                        taker_buy_quote_volume=500.0 + i * 0.01,
                        open_interest=1000.0 + i * 10,
                        ls_ratio=1.0 + i * 0.001,
                        spot_close=close - 0.5,
                        fear_greed=50.0 + (i % 8) * 5)
        feats = comp.get_features_dict("BTC")
        for name in ENRICHED_FEATURE_NAMES:
            assert feats[name] is not None, f"Feature '{name}' is None after 80 bars warmup"


class TestV8AlphaRebuildFeatures:
    """Test the 6 V8 Alpha Rebuild V3 features."""

    def _make_bar(self, close=100.0, volume=10.0, trades=100.0,
                  taker_buy_volume=5.0, quote_volume=1000.0,
                  taker_buy_quote_volume=500.0, **kw):
        return dict(close=close, volume=volume, trades=trades,
                    taker_buy_volume=taker_buy_volume,
                    quote_volume=quote_volume,
                    taker_buy_quote_volume=taker_buy_quote_volume, **kw)

    def _warmup(self, comp, n=25, symbol="BTC"):
        for i in range(n):
            comp.on_bar(symbol, **self._make_bar(
                close=100.0 + i * 0.1,
                volume=10.0 + i * 0.01,
            ))

    def test_feature_names_include_v8(self):
        expected = [
            "taker_bq_ratio", "vwap_dev_20", "volume_momentum_10",
            "mom_vol_divergence", "basis_carry_adj", "vol_regime_adaptive",
        ]
        for name in expected:
            assert name in ENRICHED_FEATURE_NAMES, f"{name} not in ENRICHED_FEATURE_NAMES"

    def test_taker_bq_ratio_basic(self):
        comp = EnrichedFeatureComputer()
        self._warmup(comp)
        feats = comp.on_bar("BTC", **self._make_bar(
            taker_buy_quote_volume=700.0, quote_volume=1000.0))
        assert feats["taker_bq_ratio"] == pytest.approx(0.7)

    def test_taker_bq_ratio_none_when_zero(self):
        comp = EnrichedFeatureComputer()
        feats = comp.on_bar("BTC", close=100.0, volume=10.0,
                            taker_buy_quote_volume=0.0, quote_volume=0.0)
        assert feats["taker_bq_ratio"] is None

    def test_vwap_dev_20_warmup(self):
        comp = EnrichedFeatureComputer()
        # Need 20 bars for VWAP window to be full
        for i in range(19):
            comp.on_bar("BTC", **self._make_bar(close=100.0, volume=10.0))
        feats = comp.on_bar("BTC", **self._make_bar(close=100.0, volume=10.0))
        assert feats["vwap_dev_20"] is not None

    def test_vwap_dev_20_above_vwap(self):
        comp = EnrichedFeatureComputer()
        # Push 20 bars at price 100
        for i in range(20):
            comp.on_bar("BTC", **self._make_bar(close=100.0, volume=10.0))
        # Push bar with higher close → deviation should be positive
        feats = comp.on_bar("BTC", **self._make_bar(close=105.0, volume=10.0))
        assert feats["vwap_dev_20"] is not None
        assert feats["vwap_dev_20"] > 0

    def test_volume_momentum_10(self):
        comp = EnrichedFeatureComputer()
        # Warm up with enough bars
        for i in range(25):
            comp.on_bar("BTC", **self._make_bar(
                close=100.0 + i * 0.5,  # uptrend
                volume=10.0,
            ))
        feats = comp.on_bar("BTC", **self._make_bar(
            close=115.0, volume=15.0))  # high volume confirms momentum
        assert feats["volume_momentum_10"] is not None
        # Uptrend + high volume → positive
        assert feats["volume_momentum_10"] > 0

    def test_mom_vol_divergence_agreement(self):
        comp = EnrichedFeatureComputer()
        # Warm up
        for i in range(25):
            comp.on_bar("BTC", **self._make_bar(
                close=100.0 + i * 0.1, volume=10.0))
        # Price up + volume above average → agreement (+1)
        feats = comp.on_bar("BTC", **self._make_bar(
            close=103.0, volume=20.0))  # price up, vol above avg
        assert feats["mom_vol_divergence"] is not None
        assert feats["mom_vol_divergence"] == 1.0

    def test_mom_vol_divergence_divergence(self):
        comp = EnrichedFeatureComputer()
        for i in range(25):
            comp.on_bar("BTC", **self._make_bar(
                close=100.0 + i * 0.1, volume=10.0))
        # Price up + volume below average → divergence (-1)
        feats = comp.on_bar("BTC", **self._make_bar(
            close=103.0, volume=2.0))  # price up, vol below avg
        assert feats["mom_vol_divergence"] is not None
        assert feats["mom_vol_divergence"] == -1.0

    def test_basis_carry_adj(self):
        comp = EnrichedFeatureComputer()
        for i in range(5):
            comp.on_bar("BTC", **self._make_bar(
                close=100.0 + i, spot_close=99.5 + i,
                funding_rate=0.0001))
        feats = comp.on_bar("BTC", **self._make_bar(
            close=110.0, spot_close=109.5, funding_rate=0.0002))
        assert feats["basis_carry_adj"] is not None
        # basis ≈ (110 - 109.5) / 109.5 ≈ 0.00457
        # funding * 3 = 0.0006
        # total ≈ 0.00517
        expected_basis = (110.0 - 109.5) / 109.5
        expected = expected_basis + 0.0002 * 3.0
        assert feats["basis_carry_adj"] == pytest.approx(expected, rel=1e-3)

    def test_basis_carry_adj_none_without_data(self):
        comp = EnrichedFeatureComputer()
        feats = comp.on_bar("BTC", close=100.0, volume=10.0)
        assert feats["basis_carry_adj"] is None

    def test_vol_regime_adaptive_warmup(self):
        comp = EnrichedFeatureComputer()
        # Need vol_regime EMA(5) ready + 30 history entries
        # vol_regime needs vol_5 and vol_20 → need ~30+ bars minimum
        for i in range(35):
            comp.on_bar("BTC", **self._make_bar(
                close=100.0 + i * 0.1, volume=10.0))
        feats = comp.get_features_dict("BTC")
        # May still be None if not enough vol_regime history yet
        # After 35 bars: vol_5 ready after ~6, vol_20 after ~21
        # vol_regime history starts accumulating after ~21
        # Need 30 entries → ~51 bars total
        assert feats["vol_regime_adaptive"] is None  # not ready yet

    def test_vol_regime_adaptive_ready(self):
        comp = EnrichedFeatureComputer()
        for i in range(55):
            comp.on_bar("BTC", **self._make_bar(
                close=100.0 + i * 0.1, volume=10.0))
        feats = comp.get_features_dict("BTC")
        assert feats["vol_regime_adaptive"] is not None
        assert feats["vol_regime_adaptive"] in (-1.0, 0.0, 1.0)
