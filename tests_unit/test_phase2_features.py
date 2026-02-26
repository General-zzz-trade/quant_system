"""Tests for Phase 2 feature engineering modules."""
from __future__ import annotations

from decimal import Decimal
from typing import List, Optional

import pytest

from features.types import Bar
from features.technical.microstructure import (
    vwap,
    order_flow_imbalance,
    volatility_cone,
    price_impact,
)
from features.cross_sectional import (
    momentum_rank,
    relative_strength,
    rolling_beta,
)
from features.pipeline import FeaturePipeline
from features.store import FeatureStore


def _bars(n: int = 30, base: float = 100.0) -> List[Bar]:
    from datetime import datetime, timezone, timedelta
    bars = []
    for i in range(n):
        close = base + i * 0.5
        bars.append(Bar(
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i),
            open=Decimal(str(close - 0.2)),
            high=Decimal(str(close + 1.0)),
            low=Decimal(str(close - 1.0)),
            close=Decimal(str(close)),
            volume=Decimal(str(100 + i * 10)),
        ))
    return bars


class TestVWAP:
    def test_basic(self) -> None:
        bars = _bars(30)
        result = vwap(bars, window=5)
        assert len(result) == 30
        assert result[0] is None  # warmup
        assert result[4] is not None
        assert isinstance(result[4], float)

    def test_empty(self) -> None:
        assert vwap([], window=5) == []


class TestOrderFlowImbalance:
    def test_range(self) -> None:
        bars = _bars(20)
        result = order_flow_imbalance(bars, window=5)
        for v in result:
            if v is not None:
                assert -1.0 <= v <= 1.0


class TestVolatilityCone:
    def test_multiple_windows(self) -> None:
        returns = [0.01 * ((-1) ** i) for i in range(100)]
        result = volatility_cone(returns, windows=(5, 10, 20))
        assert set(result.keys()) == {5, 10, 20}
        assert len(result[5]) == 100


class TestMomentumRank:
    def test_ranks_between_0_and_1(self) -> None:
        returns = {
            "A": [0.01] * 30,
            "B": [-0.01] * 30,
            "C": [0.005] * 30,
        }
        result = momentum_rank(returns, lookback=10)
        # After warmup, values should be 0, 0.5, or 1
        for sym in returns:
            vals = [v for v in result[sym] if v is not None]
            for v in vals:
                assert 0.0 <= v <= 1.0


class TestRelativeStrength:
    def test_outperforming(self) -> None:
        target = [0.02] * 30
        bench = [0.01] * 30
        result = relative_strength(target, bench, window=10)
        # Target outperforms → RS > 1
        vals = [v for v in result if v is not None]
        assert all(v > 1.0 for v in vals)


class TestRollingBeta:
    def test_self_beta_is_one(self) -> None:
        returns = [0.01 * ((-1) ** i) for i in range(100)]
        result = rolling_beta(returns, returns, window=20)
        vals = [v for v in result if v is not None]
        assert all(abs(v - 1.0) < 0.01 for v in vals)


class TestFeaturePipeline:
    def test_run_stores_results(self) -> None:
        store = FeatureStore()
        pipeline = FeaturePipeline(store=store)
        pipeline.add_step("close_prices", lambda bars: [float(b.close) for b in bars])

        bars = _bars(10)
        results = pipeline.run(bars, symbol="TEST", timeframe="1h")

        assert "close_prices" in results
        assert len(results["close_prices"]) == 10
        assert store.has(symbol="TEST", timeframe="1h", name="close_prices")
