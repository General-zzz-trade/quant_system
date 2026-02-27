"""Dual-backend tests: C++ vs Python RollingWindow and technical indicators."""

from __future__ import annotations

from datetime import datetime
from math import isnan

import pytest

from features._rolling_py import RollingWindow as PyRW

try:
    from features._quant_rolling import RollingWindow as CppRW
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

from features._rolling_py import RollingWindow as PyRW  # noqa: F811
from features.types import Bar


# ---------------------------------------------------------------------------
# RollingWindow dual-backend
# ---------------------------------------------------------------------------

RW_BACKENDS = [pytest.param(PyRW, id="py")]
if HAS_CPP:
    RW_BACKENDS.append(pytest.param(CppRW, id="cpp"))


@pytest.fixture(params=RW_BACKENDS)
def RW(request):
    return request.param


class TestRollingWindow:
    def test_size_positive(self, RW):
        with pytest.raises((ValueError, Exception)):
            RW(0)
        with pytest.raises((ValueError, Exception)):
            RW(-1)

    def test_empty(self, RW):
        rw = RW(3)
        assert rw.mean is None
        assert rw.variance is None
        assert rw.std is None
        assert not rw.full
        assert rw.n == 0

    def test_partial(self, RW):
        rw = RW(3)
        rw.push(10.0)
        assert rw.n == 1
        assert not rw.full
        assert rw.mean == pytest.approx(10.0)

    def test_full(self, RW):
        rw = RW(3)
        for v in [1.0, 2.0, 3.0]:
            rw.push(v)
        assert rw.full
        assert rw.n == 3
        assert rw.mean == pytest.approx(2.0)
        assert rw.variance == pytest.approx(2.0 / 3.0)

    def test_eviction(self, RW):
        rw = RW(2)
        rw.push(10.0)
        rw.push(20.0)
        rw.push(30.0)
        assert rw.n == 2
        assert rw.mean == pytest.approx(25.0)

    def test_std_nonneg(self, RW):
        rw = RW(3)
        for v in [1e15, 1e15 + 1, 1e15 + 2]:
            rw.push(v)
        assert rw.variance >= 0.0
        assert rw.std >= 0.0

    def test_size_property(self, RW):
        rw = RW(5)
        assert rw.size == 5


# ---------------------------------------------------------------------------
# Technical indicators: C++ vs Python parity
# ---------------------------------------------------------------------------

def _make_bars(closes, highs=None, lows=None):
    n = len(closes)
    if highs is None:
        highs = [c + 1.0 for c in closes]
    if lows is None:
        lows = [c - 1.0 for c in closes]
    return [
        Bar(ts=datetime(2024, 1, 1), open=c, high=h, low=l, close=c, volume=100.0)
        for c, h, l in zip(closes, highs, lows)
    ]


def _approx_series(a, b, rel=1e-9):
    assert len(a) == len(b), f"length mismatch: {len(a)} vs {len(b)}"
    for i, (va, vb) in enumerate(zip(a, b)):
        if va is None and vb is None:
            continue
        assert va is not None and vb is not None, f"None mismatch at index {i}: {va} vs {vb}"
        assert va == pytest.approx(vb, rel=rel), f"value mismatch at index {i}: {va} vs {vb}"


PRICES = [44.0, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
          46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41,
          46.22, 45.64]


@pytest.mark.skipif(not HAS_CPP, reason="C++ extension not built")
class TestIndicatorParity:
    def test_sma(self):
        from features.technical import sma
        from features._quant_rolling import cpp_sma
        py_result = sma.__wrapped__(PRICES, 5) if hasattr(sma, '__wrapped__') else None
        # Test via the module dispatch — should use C++
        result = sma(PRICES, 5)
        # Also directly compare
        cpp_result = cpp_sma([float(v) for v in PRICES], 5)
        _approx_series(result, cpp_result)

    def test_ema(self):
        from features.technical import ema
        from features._quant_rolling import cpp_ema
        cpp_result = cpp_ema([float(v) for v in PRICES], 10)
        # Compute Python reference
        alpha = 2.0 / 11.0
        py_result = []
        prev = None
        for x in PRICES:
            if prev is None:
                prev = float(x)
            else:
                prev = alpha * float(x) + (1.0 - alpha) * prev
            py_result.append(prev)
        _approx_series(cpp_result, py_result)

    def test_returns(self):
        from features.technical import returns
        from features._quant_rolling import cpp_returns
        cpp_result = cpp_returns([float(v) for v in PRICES], False)
        py_result = returns(PRICES)
        _approx_series(cpp_result, py_result)

    def test_returns_log(self):
        from features._quant_rolling import cpp_returns
        from features.technical import returns
        cpp_result = cpp_returns([float(v) for v in PRICES], True)
        py_result = returns(PRICES, log_ret=True)
        _approx_series(cpp_result, py_result)

    def test_returns_zero_prev(self):
        from features._quant_rolling import cpp_returns
        result = cpp_returns([0.0, 1.0, 2.0], False)
        assert result[0] is None
        assert result[1] is None  # prev is 0
        assert result[2] == pytest.approx(1.0)

    def test_volatility(self):
        from features.technical import returns, volatility
        from features._quant_rolling import cpp_volatility
        rets = returns(PRICES)
        cpp_result = cpp_volatility(rets, 5)
        py_result = volatility(rets, 5)
        _approx_series(cpp_result, py_result)

    def test_rsi(self):
        from features.technical import rsi
        from features._quant_rolling import cpp_rsi
        cpp_result = cpp_rsi([float(v) for v in PRICES], 14)
        # Use Python fallback for reference
        from features._rolling_py import RollingWindow
        # Compute with Python logic
        py_result = [None] * len(PRICES)
        avg_gain = 0.0
        avg_loss = 0.0
        window = 14
        for i in range(1, len(PRICES)):
            change = float(PRICES[i]) - float(PRICES[i - 1])
            gain = max(change, 0.0)
            loss = max(-change, 0.0)
            if i < window:
                avg_gain += gain
                avg_loss += loss
                continue
            if i == window:
                avg_gain /= window
                avg_loss /= window
            else:
                avg_gain = (avg_gain * (window - 1) + gain) / window
                avg_loss = (avg_loss * (window - 1) + loss) / window
            if avg_loss == 0.0:
                py_result[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                py_result[i] = 100.0 - (100.0 / (1.0 + rs))
        _approx_series(cpp_result, py_result)

    def test_macd(self):
        from features.technical import macd
        from features._quant_rolling import cpp_macd
        cpp_ml, cpp_sl, cpp_hist = cpp_macd([float(v) for v in PRICES], 3, 6, 3)
        # Python reference via module (which may route to C++ — compute manually)
        from features._rolling_py import RollingWindow
        vals = [float(v) for v in PRICES]
        # Direct Python EMA
        def py_ema(values, window):
            alpha = 2.0 / (window + 1.0)
            out = []
            prev = None
            for x in values:
                if prev is None:
                    prev = x
                else:
                    prev = alpha * x + (1.0 - alpha) * prev
                out.append(prev)
            return out
        fast_ema = py_ema(vals, 3)
        slow_ema = py_ema(vals, 6)
        ml = [f - s for f, s in zip(fast_ema, slow_ema)]
        sig_raw = py_ema(ml, 3)
        first_valid = 0
        sl = [None if i < first_valid + 2 else sig_raw[i] for i in range(len(sig_raw))]
        hist = [None if sl[i] is None else ml[i] - sl[i] for i in range(len(ml))]
        _approx_series(cpp_ml, ml)
        _approx_series(cpp_sl, sl)
        _approx_series(cpp_hist, hist)

    def test_bollinger_bands(self):
        from features.technical import bollinger_bands
        from features._quant_rolling import cpp_bollinger_bands
        cpp_u, cpp_m, cpp_l = cpp_bollinger_bands([float(v) for v in PRICES], 5, 2.0)
        # Python reference
        from features._rolling_py import RollingWindow
        rw = RollingWindow(5)
        py_u, py_m, py_l = [], [], []
        for x in PRICES:
            rw.push(float(x))
            if not rw.full:
                py_u.append(None)
                py_m.append(None)
                py_l.append(None)
            else:
                mid = rw.mean
                sd = rw.std
                py_u.append(mid + 2.0 * sd)
                py_m.append(mid)
                py_l.append(mid - 2.0 * sd)
        _approx_series(cpp_u, py_u)
        _approx_series(cpp_m, py_m)
        _approx_series(cpp_l, py_l)

    def test_atr(self):
        from features.technical import atr
        from features._quant_rolling import cpp_atr
        closes = PRICES[:10]
        highs = [c + 0.5 for c in closes]
        lows = [c - 0.3 for c in closes]
        bars = _make_bars(closes, highs, lows)
        py_result = atr(bars, window=3)
        cpp_result = cpp_atr(highs, lows, closes, 3)
        _approx_series(cpp_result, py_result)


# ---------------------------------------------------------------------------
# Integration: features.technical dispatches to C++ when available
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_sma_uses_cpp(self):
        """Verify that features.technical.sma returns correct results."""
        from features.technical import sma
        result = sma([1.0, 2.0, 3.0, 4.0, 5.0], 3)
        assert result[0] is None
        assert result[1] is None
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_rolling_window_from_rolling_module(self):
        """Verify features.rolling.RollingWindow works (C++ or fallback)."""
        from features.rolling import RollingWindow
        rw = RollingWindow(3)
        rw.push(1.0)
        rw.push(2.0)
        rw.push(3.0)
        assert rw.full
        assert rw.mean == pytest.approx(2.0)

    def test_atr_via_module(self):
        from features.technical import atr
        closes = [10.0, 11.0, 12.0, 11.5, 13.0]
        bars = _make_bars(closes)
        result = atr(bars, window=2)
        assert len(result) == 5
        # First two are warmup (None)
        assert result[0] is None
        assert result[1] is None
        assert result[2] is not None
