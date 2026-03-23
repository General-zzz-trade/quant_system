"""Tests for multi-resolution feature engine."""
from __future__ import annotations

import unittest.mock

import numpy as np
import pandas as pd
import pytest

from features.multi_resolution import (
    FAST_FEATURE_NAMES,
    SLOW_FEATURE_NAMES,
    SLOW_4H_FEATURE_NAMES,
    resample_to_hourly,
    compute_multi_resolution_features,
    get_all_feature_names,
    _compute_fast_features,
)


def _make_1m_df(n_bars: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create synthetic 1m OHLCV data."""
    rng = np.random.RandomState(seed)
    base_ts = 1640995200000  # 2022-01-01 00:00:00 UTC in ms
    timestamps = np.arange(n_bars) * 60_000 + base_ts

    close = 50000.0 + np.cumsum(rng.randn(n_bars) * 10)
    open_ = close + rng.randn(n_bars) * 5
    high = np.maximum(close, open_) + np.abs(rng.randn(n_bars) * 3)
    low = np.minimum(close, open_) - np.abs(rng.randn(n_bars) * 3)
    volume = np.abs(rng.randn(n_bars)) * 100 + 50
    trades = (rng.rand(n_bars) * 500 + 100).astype(int)
    tbv = volume * (0.3 + rng.rand(n_bars) * 0.4)
    qv = volume * close / 1000

    return pd.DataFrame({
        "open_time": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "quote_volume": qv,
        "trades": trades,
        "taker_buy_volume": tbv,
        "taker_buy_quote_volume": qv * 0.5,
    })


class TestResampleToHourly:
    def test_basic_aggregation(self):
        df = _make_1m_df(180)  # 3 hours
        hourly = resample_to_hourly(df)
        assert len(hourly) == 3

    def test_ohlcv_correctness(self):
        df = _make_1m_df(120)  # 2 hours
        hourly = resample_to_hourly(df)

        # First hour: bars 0-59
        first_hour = df.iloc[:60]
        row = hourly.iloc[0]
        assert row["open"] == pytest.approx(first_hour["open"].iloc[0])
        assert row["high"] == pytest.approx(first_hour["high"].max())
        assert row["low"] == pytest.approx(first_hour["low"].min())
        assert row["close"] == pytest.approx(first_hour["close"].iloc[-1])
        assert row["volume"] == pytest.approx(first_hour["volume"].sum())

    def test_volume_fields_summed(self):
        df = _make_1m_df(60)  # 1 hour
        hourly = resample_to_hourly(df)
        assert hourly["trades"].iloc[0] == pytest.approx(df["trades"].sum())
        assert hourly["taker_buy_volume"].iloc[0] == pytest.approx(df["taker_buy_volume"].sum())

    def test_partial_hour(self):
        df = _make_1m_df(90)  # 1.5 hours
        hourly = resample_to_hourly(df)
        assert len(hourly) == 2  # partial second hour still grouped

    def test_timestamp_preserved(self):
        df = _make_1m_df(120)
        hourly = resample_to_hourly(df)
        # First hour's open_time should be the first bar's open_time
        assert hourly["open_time"].iloc[0] == df["open_time"].iloc[0]


class TestFastFeatures:
    def test_all_fast_features_present(self):
        df = _make_1m_df(300)
        fast = _compute_fast_features(df)
        for name in FAST_FEATURE_NAMES:
            assert name in fast.columns, f"Missing fast feature: {name}"

    def test_return_shapes(self):
        df = _make_1m_df(100)
        fast = _compute_fast_features(df)
        assert len(fast) == 100

    def test_ret_values(self):
        df = _make_1m_df(50)
        fast = _compute_fast_features(df)
        close = df["close"].values
        expected = close[1] / close[0] - 1.0
        assert fast["ret_1"].iloc[1] == pytest.approx(expected, rel=1e-10)

    def test_ret_nan_at_start(self):
        df = _make_1m_df(50)
        fast = _compute_fast_features(df)
        # ret_10 should be NaN for first 10 bars
        assert np.isnan(fast["ret_10"].iloc[0])
        assert np.isnan(fast["ret_10"].iloc[9])
        assert not np.isnan(fast["ret_10"].iloc[10])

    def test_taker_imbalance_range(self):
        df = _make_1m_df(100)
        fast = _compute_fast_features(df)
        ti = fast["taker_imbalance"].values
        valid = ti[~np.isnan(ti)]
        assert np.all(valid >= -1.0)
        assert np.all(valid <= 1.0)

    def test_body_ratio_range(self):
        df = _make_1m_df(100)
        fast = _compute_fast_features(df)
        br = fast["body_ratio"].values
        valid = br[~np.isnan(br)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_rsi_range(self):
        df = _make_1m_df(100)
        fast = _compute_fast_features(df)
        rsi = fast["rsi_6"].values
        valid = rsi[~np.isnan(rsi)]
        assert len(valid) > 0
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 100.0)

    def test_vol_positive(self):
        df = _make_1m_df(100)
        fast = _compute_fast_features(df)
        for col in ("vol_5", "vol_20"):
            vals = fast[col].values
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                assert np.all(valid >= 0.0)


class TestMultiResolutionFeatures:
    def _mock_compute_batch(self, symbol, df, **kwargs):
        """Mock compute_features_batch to avoid C++ issues in tests."""
        n = len(df)
        # Return minimal DataFrame with columns that slow features need
        cols = [
            "rsi_14", "atr_norm_14", "close_vs_ma20", "bb_width_20",
            "basis", "funding_zscore_24", "fgi_normalized", "vol_20",
            "mean_reversion_20", "ret_24", "close",
        ]
        data = {c: np.random.randn(n) for c in cols}
        data["close"] = df["close"].values.astype(np.float64)
        return pd.DataFrame(data, index=df.index)

    def _mock_compute_4h(self, df_1h):
        """Mock compute_4h_features."""
        from features.batch_feature_engine import TF4H_FEATURE_NAMES
        n = len(df_1h)
        data = {c: np.random.randn(n) for c in TF4H_FEATURE_NAMES}
        return pd.DataFrame(data, index=df_1h.index)

    def test_all_features_present(self):
        """All fast + slow + 4h features should be in output."""
        df = _make_1m_df(600)  # 10 hours
        with unittest.mock.patch("features.multi_resolution.compute_features_batch",
                                  side_effect=self._mock_compute_batch):
            with unittest.mock.patch("features.multi_resolution.compute_4h_features",
                                      side_effect=self._mock_compute_4h):
                feat = compute_multi_resolution_features(df, "BTCUSDT")
        for name in FAST_FEATURE_NAMES:
            assert name in feat.columns, f"Missing: {name}"
        for name in SLOW_FEATURE_NAMES:
            assert name in feat.columns, f"Missing: {name}"
        for name in SLOW_4H_FEATURE_NAMES:
            assert name in feat.columns, f"Missing: {name}"
        assert "close" in feat.columns

    def test_output_length_matches_input(self):
        df = _make_1m_df(300)
        with unittest.mock.patch("features.multi_resolution.compute_features_batch",
                                  side_effect=self._mock_compute_batch):
            with unittest.mock.patch("features.multi_resolution.compute_4h_features",
                                      side_effect=self._mock_compute_4h):
                feat = compute_multi_resolution_features(df, "BTCUSDT")
        assert len(feat) == len(df)

    def test_no_lookahead_slow_features(self):
        """Slow features in first hour should be NaN (no completed previous hour)."""
        df = _make_1m_df(300)  # 5 hours
        with unittest.mock.patch("features.multi_resolution.compute_features_batch",
                                  side_effect=self._mock_compute_batch):
            with unittest.mock.patch("features.multi_resolution.compute_4h_features",
                                      side_effect=self._mock_compute_4h):
                feat = compute_multi_resolution_features(df, "BTCUSDT")
        for name in SLOW_FEATURE_NAMES:
            first_hour = feat[name].iloc[:60]
            assert first_hour.isna().all(), f"{name} should be NaN in first hour"

    def test_fast_only_mode(self):
        df = _make_1m_df(100)
        feat = compute_multi_resolution_features(
            df, "BTCUSDT", include_slow=False, include_4h=False)
        for name in FAST_FEATURE_NAMES:
            assert name in feat.columns
        for name in SLOW_FEATURE_NAMES:
            assert name not in feat.columns

    def test_get_all_feature_names(self):
        names = get_all_feature_names()
        expected = len(FAST_FEATURE_NAMES) + len(SLOW_FEATURE_NAMES) + len(SLOW_4H_FEATURE_NAMES)
        assert len(names) == expected
        assert "ret_1" in names
        assert "slow_rsi_14" in names
        assert "tf4h_close_vs_ma20" in names
        # No duplicates
        assert len(names) == len(set(names))


class TestCppFastFeatures:
    """Verify Rust fast features produce correct output."""

    def test_rust_import(self):
        from _quant_hotpath import cpp_compute_fast_1m_features
        assert callable(cpp_compute_fast_1m_features)

    def test_all_features_present(self):
        df = _make_1m_df(300)
        from features.multi_resolution import _compute_fast_features_cpp
        result = _compute_fast_features_cpp(df)
        for name in FAST_FEATURE_NAMES:
            assert name in result.columns, f"Missing: {name}"

    def test_shape_matches(self):
        df = _make_1m_df(200)
        from features.multi_resolution import _compute_fast_features_cpp
        result = _compute_fast_features_cpp(df)
        assert len(result) == len(df)
        assert len(result.columns) == len(FAST_FEATURE_NAMES)

    def test_returns_finite(self):
        df = _make_1m_df(500)
        from features.multi_resolution import _compute_fast_features_cpp
        result = _compute_fast_features_cpp(df)
        for col in ("ret_1", "ret_3", "ret_5", "ret_10"):
            vals = result[col].values[20:]
            valid = ~np.isnan(vals)
            assert valid.sum() > 0, f"{col} all NaN"
