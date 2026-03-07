"""Tests for C++ cross-sectional features: momentum_rank, rolling_beta, relative_strength."""
from __future__ import annotations

import math
import random

import pytest

NAN = float("nan")

try:
    from features._quant_rolling import (
        cpp_momentum_rank,
        cpp_rolling_beta,
        cpp_relative_strength,
    )
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ extension not built")


def _isnan(v):
    return isinstance(v, float) and math.isnan(v)


# ---------------------------------------------------------------------------
# Momentum Rank
# ---------------------------------------------------------------------------
class TestCppMomentumRank:
    def test_basic_ranking(self):
        T = 5
        matrix = [
            [0.01] * T,  # worst performer
            [0.02] * T,  # middle
            [0.03] * T,  # best performer
        ]
        result = cpp_momentum_rank(matrix, lookback=3)
        assert len(result) == 3
        assert len(result[0]) == T
        # First 3 periods: NaN (lookback=3, need index >= 3)
        for m in range(3):
            for t in range(3):
                assert _isnan(result[m][t])
        # At t=3 and t=4: symbol 0 = rank 0 (worst), symbol 2 = rank 1 (best)
        for t in [3, 4]:
            assert result[0][t] == pytest.approx(0.0)
            assert result[1][t] == pytest.approx(0.5)
            assert result[2][t] == pytest.approx(1.0)

    def test_two_symbols(self):
        matrix = [
            [0.05, 0.05, 0.05],  # better
            [-0.01, -0.01, -0.01],  # worse
        ]
        result = cpp_momentum_rank(matrix, lookback=2)
        assert result[0][2] == pytest.approx(1.0)
        assert result[1][2] == pytest.approx(0.0)

    def test_with_nan_values(self):
        matrix = [
            [0.01, NAN, 0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02, 0.02, 0.02],
        ]
        result = cpp_momentum_rank(matrix, lookback=3)
        # Symbol 0 has 1 NaN in lookback window [1,2,3] → 2 valid out of 3, >= 3//2=1
        assert not _isnan(result[0][3])
        assert not _isnan(result[1][3])

    def test_too_many_nans(self):
        matrix = [
            [NAN, NAN, NAN, 0.01, 0.01],
            [0.02, 0.02, 0.02, 0.02, 0.02],
        ]
        result = cpp_momentum_rank(matrix, lookback=4)
        # Symbol 0 at t=4: window [1,2,3,4] has 2 valid, need 4//2=2 → valid
        assert not _isnan(result[0][4])

    def test_single_valid_symbol(self):
        """When only 1 symbol has enough data, result is NaN (need >= 2)."""
        matrix = [
            [NAN] * 5,
            [0.01] * 5,
        ]
        result = cpp_momentum_rank(matrix, lookback=3)
        assert _isnan(result[0][3])
        assert _isnan(result[1][3])

    def test_empty_input(self):
        result = cpp_momentum_rank([], lookback=3)
        assert result == []

    def test_invalid_lookback(self):
        with pytest.raises(Exception):
            cpp_momentum_rank([[0.01]], lookback=0)
        with pytest.raises(Exception):
            cpp_momentum_rank([[0.01]], lookback=-1)

    def test_matches_python(self):
        """Verify C++ (via dispatch) matches Python implementation."""
        import features.cross_sectional as cs
        saved = cs._USING_CPP
        cs._USING_CPP = False

        random.seed(42)
        symbols = [f"SYM{i}" for i in range(10)]
        T = 200
        returns = {}
        for s in symbols:
            returns[s] = [random.gauss(0.001, 0.02) for _ in range(T)]
            for _ in range(10):
                idx = random.randint(0, T - 1)
                returns[s][idx] = None

        py_result = cs.momentum_rank(returns, lookback=20)
        cs._USING_CPP = saved

        # Use the high-level dispatch which handles None↔NaN
        cpp_result = cs.momentum_rank(returns, lookback=20)

        for s in symbols:
            for i in range(T):
                py_val = py_result[s][i]
                cpp_val = cpp_result[s][i]
                if py_val is None and cpp_val is None:
                    continue
                assert py_val is not None and cpp_val is not None, f"None mismatch at {s}[{i}]"
                assert cpp_val == pytest.approx(py_val, rel=1e-9), f"mismatch at {s}[{i}]"

    def test_dispatch_integration(self):
        """Verify the high-level function dispatches to C++."""
        from features.cross_sectional import momentum_rank
        returns = {
            "A": [0.01, 0.02, 0.01, 0.03, 0.01],
            "B": [-0.01, -0.02, -0.01, -0.03, -0.01],
        }
        result = momentum_rank(returns, lookback=3)
        assert "A" in result and "B" in result
        assert result["A"][3] is not None


# ---------------------------------------------------------------------------
# Rolling Beta
# ---------------------------------------------------------------------------
class TestCppRollingBeta:
    def test_perfect_correlation(self):
        """Asset = 2x market → beta = 2.0."""
        n = 100
        market = [0.01 * (i % 5 - 2) for i in range(n)]
        asset = [2 * m for m in market]
        result = cpp_rolling_beta(asset, market, window=20)
        assert _isnan(result[18])  # Not enough data
        for i in range(30, n):
            assert not _isnan(result[i])
            assert result[i] == pytest.approx(2.0, abs=0.01), f"mismatch at {i}"

    def test_negative_beta(self):
        """Inverse asset → beta = -1.0."""
        n = 100
        market = [0.01 * (i % 7 - 3) for i in range(n)]
        asset = [-m for m in market]
        result = cpp_rolling_beta(asset, market, window=20)
        for i in range(30, n):
            assert result[i] == pytest.approx(-1.0, abs=0.01)

    def test_zero_market_variance(self):
        """Constant market returns → beta is NaN."""
        asset = [0.01] * 50
        market = [0.0] * 50
        result = cpp_rolling_beta(asset, market, window=10)
        for v in result:
            assert _isnan(v)

    def test_with_nans(self):
        n = 60
        market = [0.01 * (i % 5 - 2) for i in range(n)]
        asset = [1.5 * m for m in market]
        asset_opt = [NAN if i % 10 == 0 else a for i, a in enumerate(asset)]
        result = cpp_rolling_beta(asset_opt, market, window=20)
        non_nan = [v for v in result if not _isnan(v)]
        assert len(non_nan) > 0

    def test_matches_python(self):
        """Verify C++ (via dispatch) matches Python implementation."""
        import features.cross_sectional as cs
        saved = cs._USING_CPP
        cs._USING_CPP = False

        random.seed(123)
        n = 500
        market = [random.gauss(0.001, 0.02) for _ in range(n)]
        asset = [1.3 * m + random.gauss(0, 0.005) for m in market]
        asset_opt = [a if random.random() > 0.05 else None for a in asset]
        market_opt = [m if random.random() > 0.05 else None for m in market]

        py_result = cs.rolling_beta(asset_opt, market_opt, window=60)
        cs._USING_CPP = saved

        cpp_result = cs.rolling_beta(asset_opt, market_opt, window=60)
        assert len(cpp_result) == len(py_result)
        for i in range(len(py_result)):
            if py_result[i] is None and cpp_result[i] is None:
                continue
            assert py_result[i] is not None and cpp_result[i] is not None, f"None mismatch at {i}"
            assert cpp_result[i] == pytest.approx(py_result[i], rel=1e-6), f"mismatch at {i}"

    def test_invalid_window(self):
        with pytest.raises(Exception):
            cpp_rolling_beta([0.01], [0.01], 0)

    def test_dispatch_integration(self):
        from features.cross_sectional import rolling_beta
        asset = [0.01 * i for i in range(30)]
        market = [0.005 * i for i in range(30)]
        result = rolling_beta(asset, market, window=10)
        assert len(result) == 30


# ---------------------------------------------------------------------------
# Relative Strength
# ---------------------------------------------------------------------------
class TestCppRelativeStrength:
    def test_equal_returns(self):
        """Same returns → RS = 1.0."""
        rets = [0.01, 0.02, -0.01, 0.015, 0.005]
        result = cpp_relative_strength(rets, rets, window=3)
        assert _isnan(result[0])
        assert _isnan(result[1])
        assert result[2] == pytest.approx(1.0)
        assert result[3] == pytest.approx(1.0)
        assert result[4] == pytest.approx(1.0)

    def test_outperformance(self):
        """Target returns double benchmark → RS > 1."""
        target = [0.02, 0.02, 0.02, 0.02]
        benchmark = [0.01, 0.01, 0.01, 0.01]
        result = cpp_relative_strength(target, benchmark, window=3)
        assert not _isnan(result[2])
        assert result[2] > 1.0

    def test_underperformance(self):
        """Target negative, benchmark positive → RS < 1."""
        target = [-0.01, -0.01, -0.01, -0.01]
        benchmark = [0.01, 0.01, 0.01, 0.01]
        result = cpp_relative_strength(target, benchmark, window=3)
        assert not _isnan(result[2])
        assert result[2] < 1.0

    def test_nan_in_window(self):
        """Any NaN in window → NaN output."""
        target = [0.01, NAN, 0.01, 0.01]
        benchmark = [0.01, 0.01, 0.01, 0.01]
        result = cpp_relative_strength(target, benchmark, window=3)
        assert _isnan(result[2])
        assert _isnan(result[3])

    def test_zero_benchmark_cumret(self):
        """Benchmark cumulative return = 0 → NaN."""
        target = [0.01, 0.01, 0.01]
        benchmark = [-1.0, 0.01, 0.01]
        result = cpp_relative_strength(target, benchmark, window=3)
        assert _isnan(result[2])

    def test_matches_python(self):
        """Verify C++ (via dispatch) matches Python implementation."""
        import features.cross_sectional as cs
        saved = cs._USING_CPP
        cs._USING_CPP = False

        random.seed(456)
        n = 300
        target = [random.gauss(0.001, 0.02) for _ in range(n)]
        benchmark = [random.gauss(0.0005, 0.015) for _ in range(n)]
        target_opt = [t if random.random() > 0.05 else None for t in target]
        benchmark_opt = [b if random.random() > 0.05 else None for b in benchmark]

        py_result = cs.relative_strength(target_opt, benchmark_opt, window=20)
        cs._USING_CPP = saved

        cpp_result = cs.relative_strength(target_opt, benchmark_opt, window=20)
        assert len(cpp_result) == len(py_result)
        for i in range(len(py_result)):
            if py_result[i] is None and cpp_result[i] is None:
                continue
            assert py_result[i] is not None and cpp_result[i] is not None, f"None mismatch at {i}"
            assert cpp_result[i] == pytest.approx(py_result[i], rel=1e-10), f"mismatch at {i}"

    def test_invalid_window(self):
        with pytest.raises(Exception):
            cpp_relative_strength([0.01], [0.01], 0)

    def test_dispatch_integration(self):
        from features.cross_sectional import relative_strength
        target = [0.01 * i for i in range(30)]
        benchmark = [0.005 * i for i in range(30)]
        result = relative_strength(target, benchmark, window=10)
        assert len(result) == 30
