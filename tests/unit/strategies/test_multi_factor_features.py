"""Tests for MultiFactorFeatureComputer — RSI, MACD, BB, ATR, SMA, warmup."""
from __future__ import annotations

import pytest

from strategies.multi_factor.feature_computer import (
    MultiFactorFeatureComputer,
    MultiFactorFeatures,
    _RollingSum,
)


# ── Helpers ──────────────────────────────────────────────────

def _feed_bars(fc: MultiFactorFeatureComputer, prices: list[float]) -> list[MultiFactorFeatures]:
    """Feed a list of close prices as flat bars (O=H=L=C), return all features."""
    results = []
    for p in prices:
        results.append(fc.on_bar(open=p, high=p, low=p, close=p, volume=100.0))
    return results


def _feed_ohlcv(fc: MultiFactorFeatureComputer, bars: list[tuple]) -> list[MultiFactorFeatures]:
    """Feed (o, h, l, c, v) tuples."""
    results = []
    for o, h, l, c, v in bars:
        results.append(fc.on_bar(open=o, high=h, low=l, close=c, volume=v))
    return results


def _make_rising_prices(start: float, step: float, n: int) -> list[float]:
    return [start + i * step for i in range(n)]


def _make_falling_prices(start: float, step: float, n: int) -> list[float]:
    return [start - i * step for i in range(n)]


# ── _RollingSum ──────────────────────────────────────────────

class TestRollingSum:
    def test_mean_before_full(self):
        rs = _RollingSum(5)
        rs.push(10.0)
        rs.push(20.0)
        assert rs.mean is None
        assert not rs.full

    def test_mean_when_full(self):
        rs = _RollingSum(3)
        for v in [10.0, 20.0, 30.0]:
            rs.push(v)
        assert rs.full
        assert rs.mean == pytest.approx(20.0)

    def test_std_constant(self):
        rs = _RollingSum(4)
        for _ in range(4):
            rs.push(5.0)
        assert rs.std == pytest.approx(0.0)

    def test_std_known(self):
        rs = _RollingSum(4)
        for v in [2.0, 4.0, 6.0, 8.0]:
            rs.push(v)
        # mean=5, var = (9+1+1+9)/4 = 5, std = sqrt(5) ≈ 2.2361
        assert rs.std == pytest.approx(2.2360679, rel=1e-4)

    def test_rolling_eviction(self):
        rs = _RollingSum(3)
        for v in [1.0, 2.0, 3.0, 10.0]:
            rs.push(v)
        # Window is now [2, 3, 10]
        assert rs.mean == pytest.approx(5.0)


# ── Warmup period ────────────────────────────────────────────

class TestWarmup:
    def test_sma_none_before_warmup(self):
        fc = MultiFactorFeatureComputer(sma_fast_window=5, sma_slow_window=10)
        results = _feed_bars(fc, [100.0] * 4)
        assert results[-1].sma_fast is None  # 4 bars < window 5

    def test_sma_ready_at_window(self):
        fc = MultiFactorFeatureComputer(sma_fast_window=5, sma_slow_window=10)
        results = _feed_bars(fc, [100.0] * 5)
        assert results[-1].sma_fast == pytest.approx(100.0)

    def test_rsi_none_before_warmup(self):
        fc = MultiFactorFeatureComputer(rsi_window=14)
        results = _feed_bars(fc, _make_rising_prices(100, 1, 10))
        assert results[-1].rsi is None

    def test_rsi_ready_after_warmup(self):
        fc = MultiFactorFeatureComputer(rsi_window=14)
        # Need rsi_window changes (rsi_window+1 bars) for first RSI
        results = _feed_bars(fc, _make_rising_prices(100, 1, 16))
        assert results[-1].rsi is not None

    def test_macd_none_during_warmup(self):
        fc = MultiFactorFeatureComputer(macd_fast=12, macd_slow=26, macd_signal=9)
        results = _feed_bars(fc, [100.0] * 30)
        # MACD needs macd_slow + macd_signal - 1 = 34 bars
        assert results[-1].macd is None

    def test_close_and_volume_always_present(self):
        fc = MultiFactorFeatureComputer()
        f = fc.on_bar(open=100, high=105, low=95, close=102, volume=50)
        assert f.close == 102.0
        assert f.volume == 50.0


# ── RSI computation ──────────────────────────────────────────

class TestRSI:
    def test_rsi_all_gains(self):
        fc = MultiFactorFeatureComputer(rsi_window=5)
        prices = _make_rising_prices(100, 2, 7)  # 6 changes, all gains
        results = _feed_bars(fc, prices)
        # After 5 changes (6 bars), RSI should be 100 (all gains)
        rsi = results[5].rsi
        assert rsi is not None
        assert rsi == pytest.approx(100.0)

    def test_rsi_all_losses(self):
        fc = MultiFactorFeatureComputer(rsi_window=5)
        prices = _make_falling_prices(120, 2, 7)
        results = _feed_bars(fc, prices)
        rsi = results[5].rsi
        assert rsi is not None
        assert rsi == pytest.approx(0.0)

    def test_rsi_range_0_100(self):
        fc = MultiFactorFeatureComputer(rsi_window=14)
        # Mixed prices
        prices = [100 + (i % 7) * 2 - 6 for i in range(50)]
        results = _feed_bars(fc, prices)
        for f in results:
            if f.rsi is not None:
                assert 0.0 <= f.rsi <= 100.0

    def test_rsi_constant_price(self):
        fc = MultiFactorFeatureComputer(rsi_window=5)
        prices = [100.0] * 8
        results = _feed_bars(fc, prices)
        # All changes are 0 => avg_gain=0, avg_loss=0 => RSI=100 (division guard)
        rsi = results[5].rsi
        assert rsi is not None
        assert rsi == pytest.approx(100.0)


# ── MACD ─────────────────────────────────────────────────────

class TestMACD:
    def test_macd_sign_rising(self):
        fc = MultiFactorFeatureComputer(macd_fast=3, macd_slow=6, macd_signal=3)
        # Warmup: slow + signal - 1 = 8 bars needed
        prices = _make_rising_prices(100, 1, 12)
        results = _feed_bars(fc, prices)
        last = results[-1]
        assert last.macd is not None
        # Fast EMA > Slow EMA for rising prices => MACD > 0
        assert last.macd > 0

    def test_macd_sign_falling(self):
        fc = MultiFactorFeatureComputer(macd_fast=3, macd_slow=6, macd_signal=3)
        prices = _make_falling_prices(200, 1, 12)
        results = _feed_bars(fc, prices)
        last = results[-1]
        assert last.macd is not None
        assert last.macd < 0

    def test_macd_histogram_consistency(self):
        fc = MultiFactorFeatureComputer(macd_fast=3, macd_slow=6, macd_signal=3)
        prices = _make_rising_prices(100, 0.5, 15)
        results = _feed_bars(fc, prices)
        for f in results:
            if f.macd is not None:
                assert f.macd_signal is not None
                assert f.macd_hist == pytest.approx(f.macd - f.macd_signal)


# ── Bollinger Bands ──────────────────────────────────────────

class TestBollingerBands:
    def test_bb_none_before_warmup(self):
        fc = MultiFactorFeatureComputer(bb_window=20)
        results = _feed_bars(fc, [100.0] * 19)
        assert results[-1].bb_upper is None

    def test_bb_constant_price(self):
        fc = MultiFactorFeatureComputer(bb_window=5, bb_std=2.0)
        results = _feed_bars(fc, [100.0] * 6)
        last = results[-1]
        assert last.bb_middle == pytest.approx(100.0)
        # std=0, so upper = lower = middle
        assert last.bb_upper == pytest.approx(100.0)
        assert last.bb_lower == pytest.approx(100.0)
        assert last.bb_pct == pytest.approx(0.5)  # width=0 fallback

    def test_bb_pct_at_middle(self):
        fc = MultiFactorFeatureComputer(bb_window=5, bb_std=2.0)
        prices = [98.0, 99.0, 100.0, 101.0, 102.0, 100.0]
        results = _feed_bars(fc, prices)
        last = results[-1]  # close=100, which is the mean
        assert last.bb_pct is not None
        assert 0.3 <= last.bb_pct <= 0.7  # near middle

    def test_bb_pct_range(self):
        fc = MultiFactorFeatureComputer(bb_window=5, bb_std=2.0)
        prices = [100.0, 101.0, 99.0, 102.0, 98.0, 100.0]
        results = _feed_bars(fc, prices)
        for f in results:
            if f.bb_pct is not None:
                assert -0.5 <= f.bb_pct <= 1.5  # can exceed 0-1 for extreme moves


# ── ATR ──────────────────────────────────────────────────────

class TestATR:
    def test_atr_none_during_warmup(self):
        fc = MultiFactorFeatureComputer(atr_window=5)
        results = _feed_bars(fc, [100.0] * 4)
        assert results[-1].atr is None

    def test_atr_with_high_low_spread(self):
        fc = MultiFactorFeatureComputer(atr_window=3)
        bars = [
            (100, 110, 90, 100, 1),  # TR = 20
            (100, 115, 85, 100, 1),  # TR = 30
            (100, 105, 95, 100, 1),  # TR = 10
            (100, 108, 92, 100, 1),  # First ATR = avg(20,30,10) = 20, then Wilder smooth
        ]
        results = _feed_ohlcv(fc, bars)
        # ATR becomes available at bar index 3 (atr_count == atr_w + 1 == 4)
        assert results[3].atr is not None
        assert results[3].atr > 0

    def test_atr_pct(self):
        fc = MultiFactorFeatureComputer(atr_window=3, atr_pct_window=10)
        bars = [(100, 110, 90, 100, 1)] * 5
        results = _feed_ohlcv(fc, bars)
        last = results[-1]
        if last.atr is not None and last.atr_pct is not None:
            assert last.atr_pct == pytest.approx(last.atr / last.close)


# ── Feature dict completeness ───────────────────────────────

class TestFeatureCompleteness:
    def test_all_fields_present(self):
        fc = MultiFactorFeatureComputer(
            sma_fast_window=3, sma_slow_window=5, sma_trend_window=10,
            rsi_window=3, macd_fast=3, macd_slow=5, macd_signal=3,
            bb_window=3, atr_window=3, atr_pct_window=5, ma_slope_window=3,
        )
        prices = _make_rising_prices(100, 0.5, 80)
        results = _feed_bars(fc, prices)
        last = results[-1]
        expected_fields = [
            "sma_fast", "sma_slow", "rsi", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_middle", "bb_lower", "bb_pct", "atr", "atr_pct",
            "close", "volume",
        ]
        for field in expected_fields:
            assert getattr(last, field) is not None, f"{field} should not be None after warmup"

    def test_reset_clears_state(self):
        fc = MultiFactorFeatureComputer(sma_fast_window=3)
        _feed_bars(fc, [100.0] * 5)
        fc.reset()
        result = fc.on_bar(open=100, high=100, low=100, close=100, volume=1)
        assert result.sma_fast is None  # reset cleared the buffer


# ── Edge cases ───────────────────────────────────────────────

class TestEdgeCases:
    def test_single_bar(self):
        fc = MultiFactorFeatureComputer()
        f = fc.on_bar(open=100, high=105, low=95, close=102, volume=50)
        assert f.close == 102.0
        assert f.sma_fast is None
        assert f.rsi is None
        assert f.macd is None

    def test_constant_prices_no_crash(self):
        fc = MultiFactorFeatureComputer(
            sma_fast_window=3, sma_slow_window=5,
            rsi_window=3, bb_window=3, atr_window=3,
        )
        results = _feed_bars(fc, [100.0] * 20)
        # Should not crash; all values either None or valid
        for f in results:
            if f.sma_fast is not None:
                assert f.sma_fast == pytest.approx(100.0)
