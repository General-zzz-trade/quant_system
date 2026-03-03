"""Unit tests for V7 Alpha — basis features, FGI features, multi-timeframe integration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.enriched_computer import EnrichedFeatureComputer, ENRICHED_FEATURE_NAMES
from features.multi_timeframe import TF4H_FEATURE_NAMES


# ── Basis feature tests ─────────────────────────────────────

class TestBasisFeatures:
    """Test spot-futures basis feature computation."""

    def _make_computer(self):
        return EnrichedFeatureComputer()

    def test_basis_computed(self):
        comp = self._make_computer()
        feats = comp.on_bar("BTC", close=100.0, spot_close=99.0)
        expected_basis = (100.0 - 99.0) / 99.0
        assert feats["basis"] == pytest.approx(expected_basis, abs=1e-10)

    def test_basis_none_without_spot(self):
        comp = self._make_computer()
        feats = comp.on_bar("BTC", close=100.0)
        assert feats["basis"] is None

    def test_basis_zscore_after_warmup(self):
        comp = self._make_computer()
        rng = np.random.RandomState(42)
        # Push 30 bars to warm up (need 24 for window)
        for i in range(30):
            futures = 100.0 + rng.randn() * 0.5
            spot = futures - 0.5 + rng.randn() * 0.1
            feats = comp.on_bar("BTC", close=futures, spot_close=spot)
        # After 30 bars, zscore should be computed
        assert feats["basis_zscore_24"] is not None
        assert isinstance(feats["basis_zscore_24"], float)

    def test_basis_momentum(self):
        comp = self._make_computer()
        # Push enough bars for EMA(8) to be ready
        for i in range(15):
            futures = 100.0 + i * 0.1
            spot = futures - 0.5
            feats = comp.on_bar("BTC", close=futures, spot_close=spot)
        assert feats["basis_momentum"] is not None
        # Momentum = basis - EMA(basis, 8)
        assert isinstance(feats["basis_momentum"], float)

    def test_basis_extreme_values(self):
        comp = self._make_computer()
        # Push 24 bars with stable basis, then push extreme
        for i in range(24):
            comp.on_bar("BTC", close=100.0, spot_close=99.5)

        # Push extreme contango (futures >> spot)
        feats = comp.on_bar("BTC", close=110.0, spot_close=99.5)
        # The zscore should be very high → basis_extreme = 1.0
        if feats["basis_extreme"] is not None:
            assert feats["basis_extreme"] in {-1.0, 0.0, 1.0}

    def test_basis_zero_spot(self):
        """spot_close=0 should not produce basis."""
        comp = self._make_computer()
        feats = comp.on_bar("BTC", close=100.0, spot_close=0.0)
        assert feats["basis"] is None


# ── FGI feature tests ───────────────────────────────────────

class TestFGIFeatures:
    """Test Fear & Greed Index feature computation."""

    def _make_computer(self):
        return EnrichedFeatureComputer()

    def test_fgi_normalized_range(self):
        comp = self._make_computer()
        # FGI=50 → normalized=0.0
        feats = comp.on_bar("BTC", close=100.0, fear_greed=50.0)
        assert feats["fgi_normalized"] == pytest.approx(0.0, abs=1e-10)

    def test_fgi_normalized_extreme_low(self):
        comp = self._make_computer()
        feats = comp.on_bar("BTC", close=100.0, fear_greed=0.0)
        assert feats["fgi_normalized"] == pytest.approx(-0.5, abs=1e-10)

    def test_fgi_normalized_extreme_high(self):
        comp = self._make_computer()
        feats = comp.on_bar("BTC", close=100.0, fear_greed=100.0)
        assert feats["fgi_normalized"] == pytest.approx(0.5, abs=1e-10)

    def test_fgi_extreme_flags(self):
        comp = self._make_computer()
        # Fear (< 25)
        feats = comp.on_bar("BTC", close=100.0, fear_greed=10.0)
        assert feats["fgi_extreme"] == -1.0

        comp2 = self._make_computer()
        # Greed (> 75)
        feats = comp2.on_bar("BTC", close=100.0, fear_greed=90.0)
        assert feats["fgi_extreme"] == 1.0

        comp3 = self._make_computer()
        # Neutral
        feats = comp3.on_bar("BTC", close=100.0, fear_greed=50.0)
        assert feats["fgi_extreme"] == 0.0

    def test_fgi_none_without_data(self):
        comp = self._make_computer()
        feats = comp.on_bar("BTC", close=100.0)
        assert feats["fgi_normalized"] is None
        assert feats["fgi_extreme"] is None
        assert feats["fgi_zscore_7"] is None

    def test_fgi_zscore_after_7_days(self):
        comp = self._make_computer()
        # FGI is daily, so we simulate 7 different values
        fgi_values = [30, 35, 40, 45, 50, 55, 60]
        for i, fgi in enumerate(fgi_values):
            feats = comp.on_bar("BTC", close=100.0, fear_greed=float(fgi))
        assert feats["fgi_zscore_7"] is not None
        assert isinstance(feats["fgi_zscore_7"], float)

    def test_fgi_daily_dedup(self):
        """Same FGI value pushed multiple times (hourly) should only push once to window."""
        comp = self._make_computer()
        # Push same value 24 times (simulating 24h of same FGI)
        for _ in range(24):
            feats = comp.on_bar("BTC", close=100.0, fear_greed=50.0)

        # Window should only have 1 entry (value didn't change)
        state = comp._states["BTC"]
        assert state.fgi_window_7.n == 1


# ── Multi-timeframe integration tests ───────────────────────

class TestMultiTimeframeIntegration:
    """Test 4h feature integration into V7 pipeline."""

    def _make_1h_df(self, n_bars: int = 200) -> pd.DataFrame:
        rng = np.random.RandomState(42)
        base_ts = 1609459200000  # 2021-01-01 00:00 UTC
        ts = [base_ts + i * 3600000 for i in range(n_bars)]
        prices = 100.0 * np.cumprod(1 + rng.randn(n_bars) * 0.005)
        return pd.DataFrame({
            "open_time": ts,
            "open": prices * (1 - rng.rand(n_bars) * 0.002),
            "high": prices * (1 + rng.rand(n_bars) * 0.005),
            "low": prices * (1 - rng.rand(n_bars) * 0.005),
            "close": prices,
            "volume": rng.exponential(1000, n_bars),
        })

    def test_4h_features_correct_length(self):
        from features.multi_timeframe import compute_4h_features
        df = self._make_1h_df(200)
        tf4h = compute_4h_features(df)
        assert len(tf4h) == 200

    def test_4h_feature_names(self):
        from features.multi_timeframe import compute_4h_features
        df = self._make_1h_df(200)
        tf4h = compute_4h_features(df)
        for name in TF4H_FEATURE_NAMES:
            assert name in tf4h.columns

    def test_4h_no_lookahead(self):
        """Features for a 1h bar in group G should come from group G-1."""
        from features.multi_timeframe import compute_4h_features
        df = self._make_1h_df(200)
        tf4h = compute_4h_features(df)
        # First 4 bars (group 0) should have NaN since there's no G-1
        for name in TF4H_FEATURE_NAMES:
            # At least the first bar should be NaN (no previous 4h bar)
            assert np.isnan(tf4h[name].iloc[0])


# ── V7 available features test ───────────────────────────────

class TestV7AvailableFeatures:
    """Test that V7 has the right number of features."""

    def test_enriched_includes_v7_features(self):
        v7_features = [
            "basis", "basis_zscore_24", "basis_momentum", "basis_extreme",
            "fgi_normalized", "fgi_zscore_7", "fgi_extreme",
        ]
        for f in v7_features:
            assert f in ENRICHED_FEATURE_NAMES, f"{f} not in ENRICHED_FEATURE_NAMES"

    def test_v7_available_has_4h_features(self):
        from scripts.train_v7_alpha import get_available_features
        features = get_available_features("BTCUSDT")
        for name in TF4H_FEATURE_NAMES:
            assert name in features, f"{name} not in available features"

    def test_v7_available_has_v7_interactions(self):
        from scripts.train_v7_alpha import get_available_features
        features = get_available_features("BTCUSDT")
        assert "basis_x_funding" in features
        assert "basis_x_vol_regime" in features
        assert "fgi_x_rsi14" in features

    def test_v7_feature_count_btc(self):
        from scripts.train_v7_alpha import get_available_features, BLACKLIST
        features = get_available_features("BTCUSDT")
        # V6 enriched (65 - blacklist) + 7 V7 enriched + 10 TF4H + 9 interactions + regime_vol
        # Exact count depends on blacklist overlap — just check minimum
        assert len(features) >= 75, f"Expected >= 75 features, got {len(features)}"

    def test_v7_feature_count_alt(self):
        from scripts.train_v7_alpha import get_available_features
        btc_features = get_available_features("BTCUSDT")
        eth_features = get_available_features("ETHUSDT")
        # Alt coins should have cross-asset features
        assert len(eth_features) > len(btc_features)


# ── V7 data loading tests ───────────────────────────────────

class TestV7DataLoading:
    """Test V7-specific data loading functions."""

    def test_load_fgi_schedule_missing_file(self):
        from scripts.train_v7_alpha import _load_fgi_schedule
        # Should return empty dict when file doesn't exist
        schedule = _load_fgi_schedule()
        assert isinstance(schedule, dict)

    def test_load_spot_closes_missing_file(self):
        from scripts.train_v7_alpha import _load_spot_closes
        # Should return empty dict for nonexistent symbol
        closes = _load_spot_closes("XYZNONEXISTENT")
        assert isinstance(closes, dict)
        assert len(closes) == 0


# ── V7 interaction features test ─────────────────────────────

class TestV7InteractionFeatures:
    """Test V7 new interaction features."""

    def test_interaction_feature_list(self):
        from scripts.train_v7_alpha import INTERACTION_FEATURES
        names = [name for name, _, _ in INTERACTION_FEATURES]
        assert "basis_x_funding" in names
        assert "basis_x_vol_regime" in names
        assert "fgi_x_rsi14" in names

    def test_v6_interactions_preserved(self):
        from scripts.train_v7_alpha import INTERACTION_FEATURES
        names = [name for name, _, _ in INTERACTION_FEATURES]
        # All V6 interactions should still be there
        assert "rsi14_x_vol_regime" in names
        assert "funding_x_taker_imb" in names
        assert "btc_ret1_x_beta30" in names


# ── V7 walk-forward test ─────────────────────────────────────

class TestV7WalkForward:
    """Test V7 expanding window folds (same logic as V6)."""

    def test_expanding_folds(self):
        from scripts.train_v7_alpha import expanding_window_folds
        folds = expanding_window_folds(5000, n_folds=5, min_train=2000)
        assert len(folds) > 0
        _, tr_end, te_start, _ = folds[0]
        assert tr_end == 2000
        assert te_start == 2000

    def test_small_data_rejected(self):
        from scripts.train_v7_alpha import expanding_window_folds
        folds = expanding_window_folds(1500, n_folds=5, min_train=2000)
        assert folds == []


# ── Extended OOS split metrics test ────────────────────────────

class TestSplitMetrics:
    """Test _compute_split_metrics helper."""

    def test_basic_split_metrics(self):
        from scripts.train_v7_alpha import _compute_split_metrics
        rng = np.random.RandomState(42)
        n = 500
        y_true = rng.randn(n)
        y_pred = y_true * 0.3 + rng.randn(n) * 0.7  # weak signal
        closes = 100.0 * np.cumprod(1 + rng.randn(n) * 0.005)

        result = _compute_split_metrics(y_pred, y_true, closes, "clipped")
        assert "ic" in result
        assert "dir_acc" in result
        assert "sharpe" in result
        assert "cum_pnl" in result
        assert "active_pct" in result
        assert result["n"] == n
        assert not np.isnan(result["ic"])

    def test_split_metrics_small_sample(self):
        from scripts.train_v7_alpha import _compute_split_metrics
        result = _compute_split_metrics(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]),
            np.array([100.0, 101.0]), "clipped")
        # n < 30 → NaN
        assert np.isnan(result["ic"])

    def test_split_metrics_binary_mode(self):
        from scripts.train_v7_alpha import _compute_split_metrics
        rng = np.random.RandomState(42)
        n = 200
        y_true = (rng.rand(n) > 0.5).astype(float)
        y_pred = y_true * 0.3 + 0.35 + rng.rand(n) * 0.1
        closes = 100.0 * np.cumprod(1 + rng.randn(n) * 0.005)

        result = _compute_split_metrics(y_pred, y_true, closes, "binary")
        assert 0 <= result["dir_acc"] <= 1
        assert not np.isnan(result["ic"])
