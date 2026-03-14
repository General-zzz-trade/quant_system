"""Tests for Rust PyO3 technical indicators.

Verifies correctness of the Rust implementations in _quant_hotpath.
"""
import math
import pytest

from _quant_hotpath import (
    RollingWindow as RustRW,
    VWAPWindow as RustVW,
    cpp_sma,
    cpp_ema,
    cpp_returns,
    cpp_rsi,
    cpp_macd,
    cpp_atr,
    cpp_ols,
    cpp_vwap,
    cpp_order_flow_imbalance,
    cpp_rolling_volatility,
    cpp_price_impact,
)


# ── Helpers ──

def _approx_series(a, b, rel=1e-10):
    """Compare two FeatureSeries (List[Optional[float]])."""
    assert len(a) == len(b), f"length mismatch: {len(a)} vs {len(b)}"
    for i, (va, vb) in enumerate(zip(a, b)):
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            pytest.fail(f"None mismatch at {i}: {va} vs {vb}")
        assert va == pytest.approx(vb, rel=rel), f"mismatch at {i}: {va} vs {vb}"


def _make_prices(n=200, seed=42):
    import random
    rng = random.Random(seed)
    prices = [50000.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + rng.gauss(0, 0.005)))
    return prices


def _make_ohlcv(n=200, seed=42):
    import random
    rng = random.Random(seed)
    closes = [50000.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.005)))
    highs = [c * (1 + abs(rng.gauss(0, 0.002))) for c in closes]
    lows = [c * (1 - abs(rng.gauss(0, 0.002))) for c in closes]
    opens = [c * (1 + rng.gauss(0, 0.001)) for c in closes]
    volumes = [abs(rng.gauss(0, 1)) * 1000 + 100 for _ in closes]
    return opens, highs, lows, closes, volumes


# ── RollingWindow ──

class TestRollingWindow:
    def test_basic(self):
        rw = RustRW(3)
        assert not rw.full
        assert rw.n == 0
        assert rw.mean is None

        rw.push(1.0)
        rw.push(2.0)
        rw.push(3.0)
        assert rw.full
        assert rw.n == 3
        assert rw.mean == pytest.approx(2.0)
        assert rw.variance == pytest.approx(2.0 / 3.0)
        assert rw.std == pytest.approx(math.sqrt(2.0 / 3.0))

    def test_eviction(self):
        rw = RustRW(2)
        rw.push(10.0)
        rw.push(20.0)
        assert rw.mean == pytest.approx(15.0)
        rw.push(30.0)
        assert rw.mean == pytest.approx(25.0)

    def test_size_property(self):
        rw = RustRW(5)
        assert rw.size == 5

    def test_invalid_size(self):
        with pytest.raises(Exception):
            RustRW(0)
        with pytest.raises(Exception):
            RustRW(-1)



# ── VWAPWindow ──

class TestVWAPWindow:
    def test_basic(self):
        vw = RustVW(3)
        vw.push(100.0, 10.0)
        vw.push(200.0, 20.0)
        vw.push(300.0, 30.0)
        expected = (100 * 10 + 200 * 20 + 300 * 30) / (10 + 20 + 30)
        assert vw.vwap == pytest.approx(expected)
        assert vw.full

    def test_eviction(self):
        vw = RustVW(2)
        vw.push(100.0, 10.0)
        vw.push(200.0, 20.0)
        vw.push(300.0, 30.0)
        expected = (200 * 20 + 300 * 30) / (20 + 30)
        assert vw.vwap == pytest.approx(expected)

    def test_zero_volume(self):
        vw = RustVW(2)
        vw.push(100.0, 0.0)
        vw.push(200.0, 0.0)
        assert vw.vwap is None


# ── SMA ──

class TestSMA:
    def test_basic(self):
        result = cpp_sma([1.0, 2.0, 3.0, 4.0, 5.0], 3)
        assert result[0] is None
        assert result[1] is None
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_window_sizes(self):
        prices = _make_prices(100)
        for w in [2, 5, 10, 50]:
            result = cpp_sma(prices, w)
            assert len(result) == 100
            assert all(r is None for r in result[:w - 1])
            assert all(r is not None for r in result[w - 1:])


# ── EMA ──

class TestEMA:
    def test_basic(self):
        result = cpp_ema([1.0, 2.0, 3.0], 2)
        assert result[0] == pytest.approx(1.0)
        alpha = 2.0 / 3.0
        assert result[1] == pytest.approx(alpha * 2.0 + (1 - alpha) * 1.0)


# ── Returns ──

class TestReturns:
    def test_simple(self):
        result = cpp_returns([100.0, 110.0, 99.0], False)
        assert result[0] is None
        assert result[1] == pytest.approx(0.1)
        assert result[2] == pytest.approx(-0.1)

    def test_log(self):
        result = cpp_returns([100.0, 110.0], True)
        assert result[1] == pytest.approx(math.log(1.1))

    def test_zero_price(self):
        result = cpp_returns([0.0, 100.0], False)
        assert result[1] is None


# ── RSI ──

class TestRSI:
    def test_edge_all_up(self):
        vals = list(range(1, 21))
        result = cpp_rsi(vals, 14)
        for v in result[14:]:
            assert v == pytest.approx(100.0)

    def test_edge_all_down(self):
        vals = list(range(20, 0, -1))
        result = cpp_rsi(vals, 14)
        for v in result[14:]:
            assert v == pytest.approx(0.0)


# ── MACD ──

class TestMACD:
    def test_invalid(self):
        with pytest.raises(Exception):
            cpp_macd([1.0, 2.0], 26, 12, 9)


# ── ATR ──

class TestATR:
    def test_parity_manual(self):
        opens, highs, lows, closes, volumes = _make_ohlcv(200)
        rust_result = cpp_atr(highs, lows, closes, 14)

        n = len(closes)
        trs = []
        for i in range(n):
            if i == 0:
                tr = highs[i] - lows[i]
            else:
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
            trs.append(tr)

        py_result = [None] * n
        atr_prev = 0.0
        for i in range(n):
            if i < 14:
                atr_prev += trs[i]
                continue
            if i == 14:
                atr_prev /= 14
                py_result[i] = atr_prev
                continue
            atr_prev = (atr_prev * 13 + trs[i]) / 14
            py_result[i] = atr_prev

        _approx_series(py_result, rust_result)


# ── OLS ──

class TestOLS:
    def test_perfect_fit(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        slope, r2 = cpp_ols(x, y)
        assert slope == pytest.approx(2.0, rel=1e-10)
        assert r2 == pytest.approx(1.0, rel=1e-10)

    def test_zero_variance(self):
        x = [1.0, 1.0, 1.0]
        y = [2.0, 3.0, 4.0]
        slope, r2 = cpp_ols(x, y)
        assert slope == 0.0
        assert r2 == 0.0

    def test_empty(self):
        slope, r2 = cpp_ols([], [])
        assert slope == 0.0
        assert r2 == 0.0

    def test_known_values(self):
        import random
        rng = random.Random(42)
        x = [rng.gauss(0, 1) for _ in range(100)]
        y = [2.0 * xi + rng.gauss(0, 0.1) for xi in x]
        slope, r2 = cpp_ols(x, y)
        assert slope == pytest.approx(2.0, abs=0.05)
        assert r2 > 0.99


# ── VWAP (batch) ──

class TestVWAP:
    def test_basic(self):
        closes = [100.0, 200.0, 300.0]
        volumes = [10.0, 20.0, 30.0]
        result = cpp_vwap(closes, volumes, 2)
        assert result[0] is None
        expected = (100 * 10 + 200 * 20) / (10 + 20)
        assert result[1] == pytest.approx(expected)

    def test_parity(self):
        _, _, _, closes, volumes = _make_ohlcv(200)
        rust_result = cpp_vwap(closes, volumes, 20)
        n = len(closes)
        py_result = []
        for i in range(n):
            if i < 19:
                py_result.append(None)
            else:
                start = i - 19
                spv = sum(closes[j] * volumes[j] for j in range(start, i + 1))
                sv = sum(volumes[j] for j in range(start, i + 1))
                py_result.append(spv / sv if sv > 0 else None)
        _approx_series(py_result, rust_result)


# ── Order Flow Imbalance ──

class TestOrderFlowImbalance:
    def test_basic(self):
        opens = [100.0, 100.0, 100.0]
        closes = [110.0, 90.0, 110.0]
        volumes = [10.0, 10.0, 10.0]
        result = cpp_order_flow_imbalance(opens, closes, volumes, 2)
        assert result[0] is None
        assert result[1] == pytest.approx(0.0)

    def test_parity(self):
        opens, _, _, closes, volumes = _make_ohlcv(200)
        rust_result = cpp_order_flow_imbalance(opens, closes, volumes, 20)
        n = len(closes)
        sv = [(1.0 if closes[i] >= opens[i] else -1.0) * volumes[i] for i in range(n)]
        py_result = []
        sum_sv = 0.0
        sum_abs = 0.0
        for i in range(n):
            sum_sv += sv[i]
            sum_abs += abs(sv[i])
            if i >= 20:
                drop = i - 20
                sum_sv -= sv[drop]
                sum_abs -= abs(sv[drop])
            if i < 19:
                py_result.append(None)
            elif sum_abs > 0:
                py_result.append(sum_sv / sum_abs)
            else:
                py_result.append(0.0)
        _approx_series(py_result, rust_result)


# ── Rolling Volatility ──

class TestRollingVolatility:
    def test_parity(self):
        prices = _make_prices(200)
        rets = cpp_returns(prices, False)
        rust_result = cpp_rolling_volatility(rets, 20)
        assert all(r is None for r in rust_result[:19])
        valid = [r for r in rust_result[20:] if r is not None]
        assert len(valid) > 100
        assert all(v >= 0 for v in valid)


# ── Price Impact ──

class TestPriceImpact:
    def test_parity(self):
        _, _, _, closes, volumes = _make_ohlcv(200)
        rust_result = cpp_price_impact(closes, volumes, 20)
        assert all(r is None for r in rust_result[:20])
        valid = [r for r in rust_result[20:] if r is not None]
        assert len(valid) > 100
        assert all(v >= 0 for v in valid)


# ── Dispatch integration ──

class TestDispatchIntegration:
    def test_sma_dispatches_to_rust(self):
        """Verify that features.technical.sma calls Rust directly."""
        from features.technical import sma
        result = sma([1.0, 2.0, 3.0], 2)
        assert result[1] == pytest.approx(1.5)

    def test_rolling_dispatches_to_rust(self):
        """Verify that features.rolling imports from Rust."""
        from features.rolling import RollingWindow
        rw = RollingWindow(5)
        rw.push(1.0)
        assert rw.n == 1

    def test_kyle_lambda_imports_rust(self):
        """Verify microstructure OLS imports from Rust directly."""
        from features.microstructure.kyle_lambda import _cpp_ols
        assert callable(_cpp_ols)
