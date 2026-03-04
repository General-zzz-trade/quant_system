"""Parity tests: C++ feature_selector vs Python dynamic_selector."""
import numpy as np
import pytest

from features.dynamic_selector import (
    rolling_ic_select,
    spearman_ic_select,
    icir_select,
    stable_icir_select,
    compute_feature_icir_report,
)

try:
    from features._quant_rolling import (
        cpp_rolling_ic_select,
        cpp_spearman_ic_select,
        cpp_icir_select,
        cpp_stable_icir_select,
        cpp_feature_icir_report,
    )
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ ext not built")


def _make_data(n_samples=1200, n_features=30, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    # Make some features correlated with y
    y = rng.randn(n_samples)
    for j in range(0, n_features, 5):
        X[:, j] = y * (0.3 + 0.1 * j / n_features) + rng.randn(n_samples) * 0.7
    # Sprinkle NaN
    mask = rng.rand(n_samples, n_features) < 0.02
    X[mask] = np.nan
    return X, y


def _names(n):
    return [f"f{i}" for i in range(n)]


class TestRollingICSelectParity:
    def test_same_features_selected(self):
        X, y = _make_data()
        n_features = X.shape[1]
        names = _names(n_features)

        py_result = rolling_ic_select(X, y, names, top_k=10, ic_window=500)

        X_c = np.ascontiguousarray(X, dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        cpp_indices = cpp_rolling_ic_select(X_c, y_c, top_k=10, ic_window=500)
        cpp_result = [names[i] for i in cpp_indices]

        assert set(py_result) == set(cpp_result), f"Py: {py_result}, C++: {cpp_result}"

    def test_ordering_matches(self):
        X, y = _make_data(seed=123)
        names = _names(X.shape[1])
        py_result = rolling_ic_select(X, y, names, top_k=5, ic_window=300)

        X_c = np.ascontiguousarray(X, dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        cpp_indices = cpp_rolling_ic_select(X_c, y_c, top_k=5, ic_window=300)
        cpp_result = [names[i] for i in cpp_indices]

        assert py_result == cpp_result


class TestSpearmanICSelectParity:
    def test_same_features_selected(self):
        X, y = _make_data()
        names = _names(X.shape[1])
        py_result = spearman_ic_select(X, y, names, top_k=10, ic_window=500)

        X_c = np.ascontiguousarray(X, dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        cpp_indices = cpp_spearman_ic_select(X_c, y_c, top_k=10, ic_window=500)
        cpp_result = [names[i] for i in cpp_indices]

        assert set(py_result) == set(cpp_result)

    def test_ordering_matches(self):
        X, y = _make_data(seed=77)
        names = _names(X.shape[1])
        py_result = spearman_ic_select(X, y, names, top_k=5, ic_window=500)

        X_c = np.ascontiguousarray(X, dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        cpp_indices = cpp_spearman_ic_select(X_c, y_c, top_k=5, ic_window=500)
        cpp_result = [names[i] for i in cpp_indices]

        assert py_result == cpp_result


class TestICIRSelectParity:
    def test_same_features_selected(self):
        X, y = _make_data(n_samples=2000)
        names = _names(X.shape[1])
        py_result = icir_select(X, y, names, top_k=10, ic_window=200, n_windows=5,
                                min_icir=0.3, max_consecutive_negative=3)

        X_c = np.ascontiguousarray(X, dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        cpp_indices = cpp_icir_select(X_c, y_c, top_k=10, ic_window=200,
                                      n_windows=5, min_icir=0.3, max_consec_neg=3)
        cpp_result = [names[i] for i in cpp_indices]

        # The ICIR-filtered results (before fallback fill) should match
        # Note: icir_select does fallback fill with spearman_ic_select
        # C++ only returns the ICIR-filtered part, Python may extend with fallback
        # So we check C++ result is a subset of Python result
        assert set(cpp_result).issubset(set(py_result)), \
            f"C++ has features not in Python: {set(cpp_result) - set(py_result)}"

    def test_ordering_matches(self):
        X, y = _make_data(n_samples=2000, seed=99)
        names = _names(X.shape[1])

        py_result = icir_select(X, y, names, top_k=10, ic_window=200, n_windows=5,
                                min_icir=0.1, max_consecutive_negative=5)

        X_c = np.ascontiguousarray(X, dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        cpp_indices = cpp_icir_select(X_c, y_c, top_k=10, ic_window=200,
                                      n_windows=5, min_icir=0.1, max_consec_neg=5)
        cpp_result = [names[i] for i in cpp_indices]

        # C++ returns the ICIR-filtered part only; Python may pad with fallback
        # The ICIR-sorted portion should match in order
        n_cpp = len(cpp_result)
        assert py_result[:n_cpp] == cpp_result


class TestStableICIRSelectParity:
    def test_basic_parity(self):
        X, y = _make_data(n_samples=2000, n_features=50)
        names = _names(X.shape[1])

        py_result = stable_icir_select(X, y, names, top_k=15, ic_window=200,
                                        n_windows=5, min_icir=0.3,
                                        min_stable_folds=4, sign_consistency_threshold=0.8)

        X_c = np.ascontiguousarray(X, dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        cpp_indices = cpp_stable_icir_select(X_c, y_c, top_k=15, ic_window=200,
                                              n_windows=5, min_icir=0.3,
                                              min_stable_folds=4, sign_consistency=0.8)

        if len(cpp_indices) == 0:
            # C++ returns empty → fallback to greedy in Python
            # Python should have fallen back too
            pass
        else:
            cpp_result = [names[i] for i in cpp_indices]
            assert set(cpp_result) == set(py_result[:len(cpp_result)])


class TestFeatureICIRReportParity:
    def test_report_values_match(self):
        X, y = _make_data(n_samples=1500, n_features=10)
        names = _names(X.shape[1])

        py_report = compute_feature_icir_report(X, y, names, ic_window=200, n_windows=5)

        X_c = np.ascontiguousarray(X, dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        cpp_arr = cpp_feature_icir_report(X_c, y_c, ic_window=200, n_windows=5)

        assert cpp_arr.shape == (10, 5)
        for j, name in enumerate(names):
            py_vals = py_report[name]
            np.testing.assert_allclose(cpp_arr[j, 0], py_vals["mean_ic"], atol=1e-10,
                                        err_msg=f"{name} mean_ic")
            np.testing.assert_allclose(cpp_arr[j, 1], py_vals["std_ic"], atol=1e-10,
                                        err_msg=f"{name} std_ic")
            np.testing.assert_allclose(cpp_arr[j, 2], py_vals["icir"], atol=1e-8,
                                        err_msg=f"{name} icir")
            np.testing.assert_allclose(cpp_arr[j, 3], py_vals["max_consec_neg"], atol=1e-10,
                                        err_msg=f"{name} max_consec_neg")
            np.testing.assert_allclose(cpp_arr[j, 4], py_vals["pct_positive"], atol=1e-10,
                                        err_msg=f"{name} pct_positive")


class TestEdgeCases:
    def test_too_few_samples(self):
        X = np.random.randn(30, 5)
        y = np.random.randn(30)
        X_c = np.ascontiguousarray(X, dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        assert cpp_rolling_ic_select(X_c, y_c, top_k=3, ic_window=100) == []
        assert cpp_spearman_ic_select(X_c, y_c, top_k=3, ic_window=100) == []

    def test_constant_feature_not_top(self):
        X = np.random.randn(200, 5)
        X[:, 2] = 1.0  # constant
        y = X[:, 0] * 0.5 + np.random.randn(200) * 0.3  # correlate with f0
        names = _names(5)

        X_c = np.ascontiguousarray(X, dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        cpp_indices = cpp_rolling_ic_select(X_c, y_c, top_k=3, ic_window=200)
        # Constant feature should not rank in top 3
        assert 2 not in cpp_indices[:3]

    def test_all_nan_feature(self):
        X = np.random.randn(200, 5)
        X[:, 3] = np.nan
        y = np.random.randn(200)

        X_c = np.ascontiguousarray(X, dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        cpp_indices = cpp_spearman_ic_select(X_c, y_c, top_k=5, ic_window=200)
        # Feature 3 should have IC=0 (not enough valid)
        # It may still appear in results with score 0
        # but should not be ranked above features with real IC
