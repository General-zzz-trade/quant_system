"""Parity test: C++ batch feature engine vs Python EnrichedFeatureComputer.

Verifies that cpp_compute_all_features() produces identical results to
the Python iterrows() pipeline for every feature on real market data.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from features.enriched_computer import ENRICHED_FEATURE_NAMES

# V11 features that need external data (sentiment has no historical data)
_SENTIMENT_FEATURES = {
    "social_volume_zscore_24", "social_sentiment_score", "social_volume_price_div",
}
# All features except sentiment (which has no batch historical data)
_CPP_FEATURE_NAMES = [f for f in ENRICHED_FEATURE_NAMES if f not in _SENTIMENT_FEATURES]

# Load data once
DATA_PATH = Path("data_files/BTCUSDT_1h.csv")
N_TEST_BARS = 2000


def _skip_if_no_data():
    if not DATA_PATH.exists():
        pytest.skip("No BTCUSDT_1h.csv data file for parity test")


@pytest.fixture(scope="module")
def python_features():
    """Compute features using Python pipeline."""
    _skip_if_no_data()
    from scripts.backtest_alpha_v8 import compute_oos_features
    df = pd.read_csv(DATA_PATH).head(N_TEST_BARS)
    return compute_oos_features("BTCUSDT", df)


@pytest.fixture(scope="module")
def cpp_features():
    """Compute features using C++ batch engine."""
    _skip_if_no_data()
    try:
        from features.batch_feature_engine import compute_features_batch, _USING_CPP
        if not _USING_CPP:
            pytest.skip("C++ feature engine not compiled")
    except ImportError:
        pytest.skip("C++ feature engine not available")

    df = pd.read_csv(DATA_PATH).head(N_TEST_BARS)
    return compute_features_batch("BTCUSDT", df, include_v11=True)


@pytest.fixture(scope="module")
def feature_names():
    return list(_CPP_FEATURE_NAMES)


class TestFeatureEngineParity:
    """Compare C++ and Python feature outputs."""

    def test_same_columns(self, python_features, cpp_features, feature_names):
        """Both engines produce the same feature columns."""
        for name in feature_names:
            assert name in python_features.columns, f"Missing in Python: {name}"
            assert name in cpp_features.columns, f"Missing in C++: {name}"

    def test_same_shape(self, python_features, cpp_features):
        """Same number of rows."""
        assert len(python_features) == len(cpp_features)

    @pytest.mark.parametrize("feat_name", list(_CPP_FEATURE_NAMES))
    def test_feature_parity(self, python_features, cpp_features, feat_name):
        """Each feature matches between C++ and Python within tolerance."""
        py_vals = python_features[feat_name].values.astype(np.float64)
        cpp_vals = cpp_features[feat_name].values.astype(np.float64)

        # Skip warmup bars (first 65)
        py_vals = py_vals[65:]
        cpp_vals = cpp_vals[65:]

        # Check NaN patterns match
        py_nan = np.isnan(py_vals)
        cpp_nan = np.isnan(cpp_vals)

        # Python uses None which becomes NaN in DataFrame
        nan_mismatch = py_nan != cpp_nan
        n_mismatch = nan_mismatch.sum()
        if n_mismatch > 0:
            # Find first mismatch for debugging
            first_idx = np.argmax(nan_mismatch) + 65
            pytest.fail(
                f"NaN pattern mismatch for {feat_name}: "
                f"{n_mismatch} bars differ. "
                f"First at bar {first_idx}: "
                f"Python={'NaN' if py_nan[first_idx - 65] else py_vals[first_idx - 65]}, "
                f"C++={'NaN' if cpp_nan[first_idx - 65] else cpp_vals[first_idx - 65]}"
            )

        # Compare non-NaN values
        valid = ~py_nan
        if not valid.any():
            return  # Feature is all NaN (e.g., cross_tf_regime_sync)

        py_valid = py_vals[valid]
        cpp_valid = cpp_vals[valid]

        # Use relative tolerance for large values, absolute for small
        atol = 1e-10
        rtol = 1e-10
        close = np.allclose(py_valid, cpp_valid, atol=atol, rtol=rtol)
        if not close:
            diffs = np.abs(py_valid - cpp_valid)
            max_diff = diffs.max()
            max_idx = np.argmax(diffs)
            pytest.fail(
                f"Value mismatch for {feat_name}: "
                f"max_diff={max_diff:.2e} at position {max_idx}. "
                f"Python={py_valid[max_idx]}, C++={cpp_valid[max_idx]}"
            )

    def test_close_column(self, python_features, cpp_features):
        """Close column is passed through correctly."""
        np.testing.assert_array_equal(
            python_features["close"].values,
            cpp_features["close"].values,
        )


class TestPerformance:
    """Benchmark C++ vs Python feature computation."""

    def test_cpp_faster_than_python(self):
        """C++ should be significantly faster than Python."""
        _skip_if_no_data()
        import time

        try:
            from features.batch_feature_engine import compute_features_batch, _USING_CPP
            if not _USING_CPP:
                pytest.skip("C++ not available")
        except ImportError:
            pytest.skip("C++ not available")

        from scripts.backtest_alpha_v8 import compute_oos_features

        df = pd.read_csv(DATA_PATH).head(N_TEST_BARS)

        # Time Python
        t0 = time.perf_counter()
        compute_oos_features("BTCUSDT", df)
        py_time = time.perf_counter() - t0

        # Time C++
        t0 = time.perf_counter()
        compute_features_batch("BTCUSDT", df)
        cpp_time = time.perf_counter() - t0

        speedup = py_time / cpp_time if cpp_time > 0 else float("inf")
        print(f"\nPython: {py_time:.3f}s, C++: {cpp_time:.3f}s, speedup: {speedup:.1f}x")

        # C++ should be faster (schedule loading dominates for small N)
        assert speedup > 1.5, f"Expected at least 1.5x speedup, got {speedup:.1f}x"
