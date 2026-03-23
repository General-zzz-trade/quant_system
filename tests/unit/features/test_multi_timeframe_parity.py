"""Parity tests: C++ multi_timeframe vs Python multi_timeframe."""
import numpy as np
import pandas as pd
import pytest

from features.batch_feature_engine import compute_4h_features, TF4H_FEATURE_NAMES

try:
    from _quant_hotpath import cpp_compute_4h_features, cpp_4h_feature_names
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ ext not built")


def _make_ohlcv(n_bars=500, seed=42):
    """Generate synthetic 1h OHLCV data."""
    rng = np.random.RandomState(seed)
    base_ts = 1700000000000  # ms
    timestamps = np.arange(n_bars) * 3600 * 1000 + base_ts

    # Random walk for close
    returns = rng.randn(n_bars) * 0.002
    close = 50000.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.randn(n_bars)) * 0.001)
    low = close * (1 - np.abs(rng.randn(n_bars)) * 0.001)
    opn = close * (1 + rng.randn(n_bars) * 0.0005)
    volume = np.abs(rng.randn(n_bars)) * 1000 + 100

    df = pd.DataFrame({
        "open_time": timestamps,
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return df


class TestFeatureNamesParity:
    def test_names_match(self):
        cpp_names = cpp_4h_feature_names()
        assert tuple(cpp_names) == TF4H_FEATURE_NAMES


class Test4HFeaturesParity:
    def test_values_match(self):
        df = _make_ohlcv(n_bars=1000)

        # Python path
        py_result = compute_4h_features(df)

        # C++ path
        ts = df["open_time"].values.astype(np.int64)
        opens = df["open"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        closes = df["close"].values.astype(np.float64)
        volumes = df["volume"].values.astype(np.float64)

        cpp_raw = cpp_compute_4h_features(ts, opens, highs, lows, closes, volumes)
        cpp_df = pd.DataFrame(cpp_raw, columns=list(TF4H_FEATURE_NAMES))
        cpp_df = cpp_df.ffill()

        assert py_result.shape == cpp_df.shape, \
            f"Shape mismatch: py={py_result.shape}, cpp={cpp_df.shape}"

        for col in TF4H_FEATURE_NAMES:
            py_col = py_result[col].values
            cpp_col = cpp_df[col].values

            # Compare non-NaN values
            both_valid = ~np.isnan(py_col) & ~np.isnan(cpp_col)
            if both_valid.sum() == 0:
                continue

            np.testing.assert_allclose(
                cpp_col[both_valid], py_col[both_valid],
                atol=1e-10, rtol=1e-10,
                err_msg=f"Column {col} mismatch"
            )

            # NaN positions should match (before ffill won't be exact due to
            # potential edge differences, so check after ffill)
            py_nan = np.isnan(py_col)
            cpp_nan = np.isnan(cpp_col)
            assert np.array_equal(py_nan, cpp_nan), \
                f"NaN mismatch in {col}: py has {py_nan.sum()}, cpp has {cpp_nan.sum()}"

    def test_small_data(self):
        df = _make_ohlcv(n_bars=200)
        py_result = compute_4h_features(df)

        ts = df["open_time"].values.astype(np.int64)
        cpp_raw = cpp_compute_4h_features(
            ts,
            df["open"].values.astype(np.float64),
            df["high"].values.astype(np.float64),
            df["low"].values.astype(np.float64),
            df["close"].values.astype(np.float64),
            df["volume"].values.astype(np.float64),
        )
        cpp_df = pd.DataFrame(cpp_raw, columns=list(TF4H_FEATURE_NAMES))
        cpp_df = cpp_df.ffill()

        assert py_result.shape == cpp_df.shape

    def test_returns_columns(self):
        """Specifically test return columns which are simplest."""
        df = _make_ohlcv(n_bars=200)
        py_result = compute_4h_features(df)

        ts = df["open_time"].values.astype(np.int64)
        cpp_raw = cpp_compute_4h_features(
            ts,
            df["open"].values.astype(np.float64),
            df["high"].values.astype(np.float64),
            df["low"].values.astype(np.float64),
            df["close"].values.astype(np.float64),
            df["volume"].values.astype(np.float64),
        )
        cpp_df = pd.DataFrame(cpp_raw, columns=list(TF4H_FEATURE_NAMES))
        cpp_df = cpp_df.ffill()

        for col in ["tf4h_ret_1", "tf4h_ret_3", "tf4h_ret_6"]:
            both_valid = ~np.isnan(py_result[col].values) & ~np.isnan(cpp_df[col].values)
            if both_valid.sum() > 0:
                np.testing.assert_allclose(
                    cpp_df[col].values[both_valid],
                    py_result[col].values[both_valid],
                    atol=1e-12, err_msg=f"{col}"
                )

    def test_large_data(self):
        """Test with realistic data size."""
        df = _make_ohlcv(n_bars=5000, seed=99)
        py_result = compute_4h_features(df)

        ts = df["open_time"].values.astype(np.int64)
        cpp_raw = cpp_compute_4h_features(
            ts,
            df["open"].values.astype(np.float64),
            df["high"].values.astype(np.float64),
            df["low"].values.astype(np.float64),
            df["close"].values.astype(np.float64),
            df["volume"].values.astype(np.float64),
        )
        cpp_df = pd.DataFrame(cpp_raw, columns=list(TF4H_FEATURE_NAMES))
        cpp_df = cpp_df.ffill()

        for col in TF4H_FEATURE_NAMES:
            both_valid = ~np.isnan(py_result[col].values) & ~np.isnan(cpp_df[col].values)
            if both_valid.sum() > 0:
                np.testing.assert_allclose(
                    cpp_df[col].values[both_valid],
                    py_result[col].values[both_valid],
                    atol=1e-8, rtol=1e-8,
                    err_msg=f"Column {col} mismatch on large data"
                )

    def test_anti_lookahead(self):
        """Verify that 1h bars only see previous 4h bar's features."""
        df = _make_ohlcv(n_bars=200)
        ts = df["open_time"].values.astype(np.int64)
        four_hours_ms = 4 * 3600 * 1000
        group_keys = ts // four_hours_ms

        cpp_raw = np.asarray(cpp_compute_4h_features(
            ts,
            df["open"].values.astype(np.float64),
            df["high"].values.astype(np.float64),
            df["low"].values.astype(np.float64),
            df["close"].values.astype(np.float64),
            df["volume"].values.astype(np.float64),
        ), dtype=np.float64)

        # First bar of each new group should use G-1's features
        # which means bars in the first 4h group should be NaN
        # (since there's no G-1 for the first group)
        first_group = group_keys[0]
        first_group_mask = group_keys == first_group
        # All features for first group should be NaN (no G-1 exists)
        for f in range(10):
            vals = cpp_raw[first_group_mask, f]
            assert np.all(np.isnan(vals)), \
                f"Feature {f} should be NaN for first 4h group"
