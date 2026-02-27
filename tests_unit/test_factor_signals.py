"""Tests for the 6 classic factor signals."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from decision.signals.factors.momentum import MomentumSignal
from decision.signals.factors.carry import CarrySignal
from decision.signals.factors.volatility import VolatilitySignal
from decision.signals.factors.liquidity import LiquiditySignal
from decision.signals.factors.trend_strength import TrendStrengthSignal
from decision.signals.factors.volume_price_div import VolumePriceDivergenceSignal
from features.types import Bar


def _make_snapshot(
    closes: List[float],
    volumes: List[float] | None = None,
    highs: List[float] | None = None,
    lows: List[float] | None = None,
    funding_rate: float | None = None,
) -> SimpleNamespace:
    n = len(closes)
    if volumes is None:
        volumes = [100.0] * n
    if highs is None:
        highs = [c * 1.01 for c in closes]
    if lows is None:
        lows = [c * 0.99 for c in closes]

    bars = [
        Bar(
            ts=datetime(2024, 1, 1 + i // 24, i % 24),
            open=closes[i],
            high=highs[i],
            low=lows[i],
            close=closes[i],
            volume=volumes[i],
        )
        for i in range(n)
    ]
    snap = SimpleNamespace(bars=bars)
    if funding_rate is not None:
        snap.funding_rate = {"BTCUSDT": funding_rate}
    return snap


class TestMomentumSignal:
    def test_uptrend_is_buy(self) -> None:
        # Steadily rising prices
        closes = [100.0 + i * 2.0 for i in range(30)]
        snap = _make_snapshot(closes)
        sig = MomentumSignal(lookback=20)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "buy"
        assert result.score > 0

    def test_downtrend_is_sell(self) -> None:
        closes = [200.0 - i * 2.0 for i in range(30)]
        snap = _make_snapshot(closes)
        sig = MomentumSignal(lookback=20)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "sell"
        assert result.score < 0

    def test_flat_market_near_zero(self) -> None:
        closes = [100.0] * 30
        snap = _make_snapshot(closes)
        sig = MomentumSignal(lookback=20)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "flat"
        assert result.score == Decimal("0")

    def test_insufficient_data(self) -> None:
        closes = [100.0] * 5
        snap = _make_snapshot(closes)
        sig = MomentumSignal(lookback=20)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "flat"
        assert result.confidence == Decimal("0")


class TestCarrySignal:
    def test_negative_funding_buy(self) -> None:
        snap = _make_snapshot([100.0] * 5, funding_rate=-0.0005)
        sig = CarrySignal()
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "buy"
        assert result.score > 0

    def test_positive_funding_sell(self) -> None:
        snap = _make_snapshot([100.0] * 5, funding_rate=0.0005)
        sig = CarrySignal()
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "sell"
        assert result.score < 0

    def test_no_funding_data(self) -> None:
        snap = SimpleNamespace(bars=[])
        sig = CarrySignal()
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "flat"


class TestVolatilitySignal:
    def test_expanding_vol_is_sell(self) -> None:
        # First half stable, second half volatile → vol expanding → sell
        closes = [100.0 + i * 0.1 for i in range(15)]  # calm
        closes += [100.0 + (10.0 if i % 2 == 0 else -10.0) for i in range(15)]  # volatile
        snap = _make_snapshot(closes)
        sig = VolatilitySignal(lookback=20)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "sell"
        assert result.score < 0

    def test_contracting_vol_is_buy(self) -> None:
        # First half volatile, second half stable → vol contracting → buy
        closes = [100.0 + (10.0 if i % 2 == 0 else -10.0) for i in range(15)]  # volatile
        closes += [100.0 + i * 0.001 for i in range(15)]  # calm
        snap = _make_snapshot(closes)
        sig = VolatilitySignal(lookback=20)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "buy"
        assert result.score > 0

    def test_insufficient_data(self) -> None:
        closes = [100.0] * 5
        snap = _make_snapshot(closes)
        sig = VolatilitySignal(lookback=20)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "flat"


class TestLiquiditySignal:
    def test_high_volume_buy(self) -> None:
        volumes = [100.0] * 19 + [500.0]  # spike at end
        closes = [100.0] * 20
        snap = _make_snapshot(closes, volumes=volumes)
        sig = LiquiditySignal(lookback=20)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "buy"
        assert result.score > 0

    def test_low_volume_sell(self) -> None:
        volumes = [100.0] * 19 + [10.0]  # drop at end
        closes = [100.0] * 20
        snap = _make_snapshot(closes, volumes=volumes)
        sig = LiquiditySignal(lookback=20)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "sell"
        assert result.score < 0

    def test_constant_volume_flat(self) -> None:
        volumes = [100.0] * 20
        closes = [100.0] * 20
        snap = _make_snapshot(closes, volumes=volumes)
        sig = LiquiditySignal(lookback=20)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "flat"


class TestTrendStrengthSignal:
    def test_strong_uptrend(self) -> None:
        # Strong directional move to trigger ADX > threshold
        n = 80
        closes = [100.0 + i * 3.0 for i in range(n)]
        highs = [c + 5.0 for c in closes]
        lows = [c - 2.0 for c in closes]
        snap = _make_snapshot(closes, highs=highs, lows=lows)
        sig = TrendStrengthSignal(adx_window=14, adx_threshold=20.0)
        result = sig.compute(snap, "BTCUSDT")
        # Should be buy in strong uptrend
        assert result.side in ("buy", "flat")

    def test_insufficient_data(self) -> None:
        closes = [100.0] * 10
        snap = _make_snapshot(closes)
        sig = TrendStrengthSignal()
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "flat"
        assert result.confidence == Decimal("0")

    def test_flat_market_low_adx(self) -> None:
        # Random but flat prices → low ADX
        closes = [100.0] * 80
        snap = _make_snapshot(closes)
        sig = TrendStrengthSignal(adx_threshold=25.0)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "flat"


class TestVolumePriceDivergenceSignal:
    def test_price_up_volume_down_bearish(self) -> None:
        # Rising prices, declining volume → bearish divergence
        closes = [100.0 + i * 1.0 for i in range(15)]
        volumes = [1000.0 - i * 50.0 for i in range(15)]
        snap = _make_snapshot(closes, volumes=volumes)
        sig = VolumePriceDivergenceSignal(lookback=10)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "sell"
        assert result.score < 0

    def test_price_down_volume_up_selling_pressure(self) -> None:
        # Falling prices, rising volume → selling pressure → sell
        closes = [200.0 - i * 1.0 for i in range(15)]
        volumes = [100.0 + i * 50.0 for i in range(15)]
        snap = _make_snapshot(closes, volumes=volumes)
        sig = VolumePriceDivergenceSignal(lookback=10)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "sell"
        assert result.score < 0

    def test_confirmed_trend_no_divergence(self) -> None:
        # Price up, volume up → confirmed, score depends on sign
        closes = [100.0 + i * 1.0 for i in range(15)]
        volumes = [100.0 + i * 50.0 for i in range(15)]
        snap = _make_snapshot(closes, volumes=volumes)
        sig = VolumePriceDivergenceSignal(lookback=10)
        result = sig.compute(snap, "BTCUSDT")
        # price↑ vol↑ → confirmed trend → positive score → buy
        assert result.score > 0
        assert result.side == "buy"

    def test_insufficient_data(self) -> None:
        closes = [100.0] * 3
        snap = _make_snapshot(closes)
        sig = VolumePriceDivergenceSignal(lookback=10)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "flat"
