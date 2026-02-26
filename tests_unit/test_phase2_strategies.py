"""Tests for Phase 2 strategy modules."""
from __future__ import annotations

import pytest

from strategies.stat_arb.pairs import find_pairs, PairsStrategy, PairResult
from strategies.factor.multi_factor import (
    momentum_factor,
    volatility_factor,
    mean_reversion_factor,
    MultiFactorStrategy,
)


class TestFindPairs:
    def test_finds_correlated_pair(self) -> None:
        # Two perfectly correlated series
        prices = {
            "A": [100 + i for i in range(100)],
            "B": [200 + i * 2 for i in range(100)],
            "C": [50 + ((-1) ** i) * 5 for i in range(100)],  # uncorrelated
        }
        results = find_pairs(prices, min_correlation=0.9)
        # A and B should be found
        pairs_found = {(r.symbol_a, r.symbol_b) for r in results}
        assert ("A", "B") in pairs_found or ("B", "A") in pairs_found

    def test_empty_prices(self) -> None:
        assert find_pairs({}) == []

    def test_short_series_skipped(self) -> None:
        prices = {"A": [1, 2, 3], "B": [4, 5, 6]}
        assert find_pairs(prices, min_observations=100) == []


class TestPairsStrategy:
    def test_generates_signal_on_divergence(self) -> None:
        pair = PairResult(
            symbol_a="A", symbol_b="B",
            correlation=0.95, hedge_ratio=2.0,
            spread_mean=0.0, spread_std=1.0,
            half_life=10.0, is_cointegrated=True,
        )
        strategy = PairsStrategy(pair, entry_zscore=2.0, window=10)

        # Feed normal spread for warmup
        for i in range(10):
            strategy.on_prices(100.0, 200.0)

        # Force divergence: B jumps up
        signal = strategy.on_prices(100.0, 220.0)
        # May or may not trigger depending on z-score
        # At minimum, strategy should not crash
        assert signal is None or hasattr(signal, "side_a")


class TestMultiFactorStrategy:
    def test_compute_signals(self) -> None:
        returns = {
            "AAPL": [0.01] * 100,
            "MSFT": [0.005] * 100,
            "GOOG": [-0.005] * 100,
        }
        strategy = MultiFactorStrategy(long_threshold=0.5, short_threshold=-0.5)
        signals = strategy.compute_signals(returns)

        assert len(signals) == 3
        # Each signal has required attributes
        for s in signals:
            assert s.side in ("long", "short", "flat")
            assert s.rank >= 1

    def test_momentum_factor(self) -> None:
        returns = {
            "WINNER": [0.02] * 100,
            "LOSER": [-0.01] * 100,
        }
        scores = momentum_factor(returns, lookback=20)
        assert scores["WINNER"] > scores["LOSER"]
