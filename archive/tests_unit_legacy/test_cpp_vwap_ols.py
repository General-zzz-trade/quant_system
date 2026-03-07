"""Tests for C++ VWAPWindow, batch VWAP, OFI, volatility, price_impact, and OLS."""
from __future__ import annotations

from datetime import datetime
from math import sqrt

import pytest

from features.types import Bar

try:
    from features._quant_rolling import (
        VWAPWindow,
        cpp_vwap,
        cpp_order_flow_imbalance,
        cpp_rolling_volatility,
        cpp_price_impact,
        cpp_ols,
    )
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ extension not built")


def _make_bars(closes, opens=None, highs=None, lows=None, volumes=None):
    n = len(closes)
    if opens is None:
        opens = closes
    if highs is None:
        highs = [c + 1.0 for c in closes]
    if lows is None:
        lows = [c - 1.0 for c in closes]
    if volumes is None:
        volumes = [100.0] * n
    return [
        Bar(ts=datetime(2024, 1, 1), open=o, high=h, low=l, close=c, volume=v)
        for o, c, h, l, v in zip(opens, closes, highs, lows, volumes)
    ]


# ---------------------------------------------------------------------------
# VWAPWindow (stateful, O(1) per push)
# ---------------------------------------------------------------------------
class TestVWAPWindow:
    def test_empty(self):
        vw = VWAPWindow(3)
        assert vw.vwap is None
        assert not vw.full
        assert vw.n == 0

    def test_partial(self):
        vw = VWAPWindow(3)
        vw.push(100.0, 10.0)
        assert vw.n == 1
        assert not vw.full
        assert vw.vwap == pytest.approx(100.0)

    def test_full(self):
        vw = VWAPWindow(3)
        vw.push(100.0, 10.0)
        vw.push(101.0, 20.0)
        vw.push(102.0, 30.0)
        assert vw.full
        expected = (100*10 + 101*20 + 102*30) / (10+20+30)
        assert vw.vwap == pytest.approx(expected)

    def test_eviction(self):
        vw = VWAPWindow(2)
        vw.push(100.0, 10.0)
        vw.push(200.0, 20.0)
        vw.push(300.0, 30.0)
        # Window now has (200, 20) and (300, 30)
        expected = (200*20 + 300*30) / (20+30)
        assert vw.vwap == pytest.approx(expected)

    def test_zero_volume(self):
        vw = VWAPWindow(2)
        vw.push(100.0, 0.0)
        vw.push(200.0, 0.0)
        assert vw.vwap is None  # no volume → no VWAP

    def test_size_property(self):
        vw = VWAPWindow(5)
        assert vw.size == 5

    def test_invalid_size(self):
        with pytest.raises(Exception):
            VWAPWindow(0)
        with pytest.raises(Exception):
            VWAPWindow(-1)

    def test_sum_tracking(self):
        vw = VWAPWindow(3)
        vw.push(10.0, 5.0)
        vw.push(20.0, 10.0)
        assert vw.sum_pv == pytest.approx(10*5 + 20*10)
        assert vw.sum_v == pytest.approx(5 + 10)


# ---------------------------------------------------------------------------
# Batch VWAP
# ---------------------------------------------------------------------------
class TestCppVwap:
    def test_basic(self):
        closes = [100.0, 101.0, 102.0, 103.0, 104.0]
        volumes = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = cpp_vwap(closes, volumes, 3)
        assert len(result) == 5
        assert result[0] is None
        assert result[1] is None
        # i=2: (100*10 + 101*20 + 102*30) / (10+20+30)
        expected_2 = (100*10 + 101*20 + 102*30) / 60
        assert result[2] == pytest.approx(expected_2)

    def test_matches_python_vwap(self):
        """Verify C++ matches Python implementation."""
        from features.technical.microstructure import vwap
        closes = [44.0, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10]
        volumes = [100.0, 150.0, 120.0, 200.0, 180.0, 90.0, 110.0]
        bars = _make_bars(closes, volumes=volumes)

        # Force Python path
        import features.technical.microstructure as m
        saved = m._USING_CPP
        m._USING_CPP = False
        py_result = vwap(bars, window=3)
        m._USING_CPP = saved

        cpp_result = cpp_vwap(closes, volumes, 3)
        assert len(cpp_result) == len(py_result)
        for i, (c, p) in enumerate(zip(cpp_result, py_result)):
            if c is None and p is None:
                continue
            assert c == pytest.approx(p, rel=1e-10), f"mismatch at {i}"

    def test_window_1(self):
        closes = [10.0, 20.0, 30.0]
        volumes = [1.0, 2.0, 3.0]
        result = cpp_vwap(closes, volumes, 1)
        assert result[0] == pytest.approx(10.0)
        assert result[1] == pytest.approx(20.0)
        assert result[2] == pytest.approx(30.0)

    def test_zero_volume_window(self):
        closes = [100.0, 200.0, 300.0]
        volumes = [0.0, 0.0, 0.0]
        result = cpp_vwap(closes, volumes, 2)
        assert result[0] is None
        assert result[1] is None  # zero volume → None
        assert result[2] is None

    def test_length_mismatch_raises(self):
        with pytest.raises(Exception):
            cpp_vwap([1.0, 2.0], [1.0], 1)


# ---------------------------------------------------------------------------
# Order Flow Imbalance
# ---------------------------------------------------------------------------
class TestCppOFI:
    def test_all_up_bars(self):
        opens = [100.0, 101.0, 102.0, 103.0]
        closes = [101.0, 102.0, 103.0, 104.0]  # all close >= open
        volumes = [10.0, 10.0, 10.0, 10.0]
        result = cpp_order_flow_imbalance(opens, closes, volumes, 3)
        assert result[0] is None
        assert result[1] is None
        assert result[2] == pytest.approx(1.0)  # all buys
        assert result[3] == pytest.approx(1.0)

    def test_all_down_bars(self):
        opens = [104.0, 103.0, 102.0, 101.0]
        closes = [103.0, 102.0, 101.0, 100.0]  # all close < open
        volumes = [10.0, 10.0, 10.0, 10.0]
        result = cpp_order_flow_imbalance(opens, closes, volumes, 3)
        assert result[2] == pytest.approx(-1.0)

    def test_mixed_bars(self):
        opens = [100.0, 102.0]  # bar0: up, bar1: down
        closes = [101.0, 101.0]
        volumes = [10.0, 10.0]
        result = cpp_order_flow_imbalance(opens, closes, volumes, 2)
        # sv = [+10, -10], OFI = 0/20 = 0
        assert result[1] == pytest.approx(0.0)

    def test_matches_python(self):
        from features.technical.microstructure import order_flow_imbalance
        opens = [100.0, 101.0, 99.0, 102.0, 98.0]
        closes = [101.0, 100.0, 100.0, 101.0, 99.0]
        volumes = [10.0, 20.0, 15.0, 25.0, 30.0]
        bars = _make_bars(closes, opens=opens, volumes=volumes)

        import features.technical.microstructure as m
        saved = m._USING_CPP
        m._USING_CPP = False
        py_result = order_flow_imbalance(bars, window=3)
        m._USING_CPP = saved

        cpp_result = cpp_order_flow_imbalance(opens, closes, volumes, 3)
        for i, (c, p) in enumerate(zip(cpp_result, py_result)):
            if c is None and p is None:
                continue
            assert c == pytest.approx(p, rel=1e-10), f"mismatch at {i}"


# ---------------------------------------------------------------------------
# Price Impact
# ---------------------------------------------------------------------------
class TestCppPriceImpact:
    def test_basic(self):
        closes = [100.0, 101.0, 103.0, 102.0, 105.0]
        volumes = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = cpp_price_impact(closes, volumes, 2)
        assert result[0] is None
        assert result[1] is None
        assert result[2] is not None

    def test_matches_python(self):
        from features.technical.microstructure import price_impact
        closes = [44.0, 44.34, 44.09, 43.61, 44.33, 44.83]
        volumes = [100.0, 150.0, 120.0, 200.0, 180.0, 90.0]
        bars = _make_bars(closes, volumes=volumes)

        import features.technical.microstructure as m
        saved = m._USING_CPP
        m._USING_CPP = False
        py_result = price_impact(bars, window=3)
        m._USING_CPP = saved

        cpp_result = cpp_price_impact(closes, volumes, 3)
        assert len(cpp_result) == len(py_result)
        for i, (c, p) in enumerate(zip(cpp_result, py_result)):
            if c is None and p is None:
                continue
            assert c is not None and p is not None, f"None mismatch at {i}"
            assert c == pytest.approx(p, rel=1e-9), f"mismatch at {i}"


# ---------------------------------------------------------------------------
# OLS Regression
# ---------------------------------------------------------------------------
class TestCppOLS:
    def test_perfect_line(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        slope, r2 = cpp_ols(x, y)
        assert slope == pytest.approx(2.0, rel=1e-10)
        assert r2 == pytest.approx(1.0, rel=1e-10)

    def test_negative_slope(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        slope, r2 = cpp_ols(x, y)
        assert slope == pytest.approx(-2.0, rel=1e-10)
        assert r2 == pytest.approx(1.0, rel=1e-10)

    def test_with_offset(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [3.0, 5.0, 7.0, 9.0, 11.0]  # y = 2x + 1
        slope, r2 = cpp_ols(x, y)
        assert slope == pytest.approx(2.0, rel=1e-10)
        assert r2 == pytest.approx(1.0, rel=1e-10)

    def test_no_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.0, -1.0, 1.0, -1.0]  # oscillating
        slope, r2 = cpp_ols(x, y)
        # Not perfectly zero but low r2
        assert abs(r2) < 0.5

    def test_constant_x(self):
        x = [5.0, 5.0, 5.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0]
        slope, r2 = cpp_ols(x, y)
        assert slope == pytest.approx(0.0)  # zero variance in x

    def test_empty(self):
        slope, r2 = cpp_ols([], [])
        assert slope == 0.0
        assert r2 == 0.0

    def test_single_point(self):
        slope, r2 = cpp_ols([1.0], [2.0])
        assert slope == pytest.approx(0.0)

    def test_matches_python_ols(self):
        """Verify C++ matches the Python _ols implementation."""
        from features.microstructure.kyle_lambda import _ols
        x = [0.5, -0.3, 1.2, -0.8, 0.1, 2.0, -1.5, 0.7]
        y = [0.01, -0.005, 0.02, -0.015, 0.002, 0.04, -0.03, 0.012]
        py_slope, py_r2 = _ols(x, y)
        cpp_slope, cpp_r2 = cpp_ols(x, y)
        assert cpp_slope == pytest.approx(py_slope, rel=1e-9)
        assert cpp_r2 == pytest.approx(py_r2, rel=1e-9)

    def test_length_mismatch_raises(self):
        with pytest.raises(Exception):
            cpp_ols([1.0, 2.0], [1.0])

    def test_large_values_numerical_stability(self):
        x = [1e8 + i for i in range(100)]
        y = [2 * (1e8 + i) + 3 for i in range(100)]
        slope, r2 = cpp_ols(x, y)
        assert slope == pytest.approx(2.0, rel=1e-6)
        assert r2 == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Integration: microstructure.py dispatches to C++ when available
# ---------------------------------------------------------------------------
class TestMicrostructureIntegration:
    def test_vwap_dispatch(self):
        from features.technical.microstructure import vwap
        closes = [100.0, 101.0, 102.0, 103.0, 104.0]
        volumes = [10.0, 20.0, 30.0, 40.0, 50.0]
        bars = _make_bars(closes, volumes=volumes)
        result = vwap(bars, window=3)
        assert len(result) == 5
        assert result[0] is None
        assert result[1] is None
        assert result[2] is not None

    def test_ofi_dispatch(self):
        from features.technical.microstructure import order_flow_imbalance
        opens = [100.0, 101.0, 102.0, 103.0]
        closes = [101.0, 102.0, 103.0, 104.0]
        volumes = [10.0, 10.0, 10.0, 10.0]
        bars = _make_bars(closes, opens=opens, volumes=volumes)
        result = order_flow_imbalance(bars, window=3)
        assert len(result) == 4
        assert result[2] == pytest.approx(1.0)

    def test_price_impact_dispatch(self):
        from features.technical.microstructure import price_impact
        closes = [100.0, 101.0, 103.0, 102.0, 105.0]
        volumes = [10.0, 20.0, 30.0, 40.0, 50.0]
        bars = _make_bars(closes, volumes=volumes)
        result = price_impact(bars, window=2)
        assert len(result) == 5

    def test_kyle_lambda_uses_cpp_ols(self):
        """KyleLambdaEstimator should use C++ OLS when available."""
        from features.microstructure.kyle_lambda import KyleLambdaEstimator
        from dataclasses import dataclass
        from decimal import Decimal

        @dataclass
        class MockTick:
            price: Decimal
            qty: Decimal
            side: str

        ticks = [
            MockTick(price=Decimal("100"), qty=Decimal("1"), side="buy"),
            MockTick(price=Decimal("101"), qty=Decimal("2"), side="buy"),
            MockTick(price=Decimal("100.5"), qty=Decimal("1.5"), side="sell"),
            MockTick(price=Decimal("102"), qty=Decimal("3"), side="buy"),
            MockTick(price=Decimal("101"), qty=Decimal("2"), side="sell"),
        ]
        est = KyleLambdaEstimator(window=10)
        result = est.estimate(ticks)
        assert result.n_observations == 4
        # Just verify it runs and returns reasonable values
        assert isinstance(result.kyle_lambda, float)
        assert isinstance(result.r_squared, float)
        assert 0.0 <= result.r_squared <= 1.0 + 1e-10
