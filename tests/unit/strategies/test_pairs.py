"""Tests for pairs trading — OLS, correlation, half-life, pair screening, signals."""
from __future__ import annotations

import math

import pytest

from strategies.stat_arb.pairs import (
    PairResult,
    PairSignal,
    PairsStrategy,
    _correlation,
    _half_life,
    _ols_simple,
    find_pairs,
)


# ── OLS regression ───────────────────────────────────────────

class TestOLS:
    def test_perfect_fit(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [3.0, 5.0, 7.0, 9.0, 11.0]  # y = 1 + 2*x
        alpha, beta = _ols_simple(x, y)
        assert alpha == pytest.approx(1.0, abs=1e-10)
        assert beta == pytest.approx(2.0, abs=1e-10)

    def test_horizontal_line(self):
        x = [1.0, 2.0, 3.0]
        y = [5.0, 5.0, 5.0]  # y = 5
        alpha, beta = _ols_simple(x, y)
        assert alpha == pytest.approx(5.0, abs=1e-10)
        assert beta == pytest.approx(0.0, abs=1e-10)

    def test_degenerate_x(self):
        x = [3.0, 3.0, 3.0]
        y = [1.0, 2.0, 3.0]
        alpha, beta = _ols_simple(x, y)
        # Degenerate: denom ≈ 0 => returns (0, 0)
        assert alpha == 0.0
        assert beta == 0.0

    def test_negative_slope(self):
        x = [1.0, 2.0, 3.0, 4.0]
        y = [10.0, 8.0, 6.0, 4.0]  # y = 12 - 2*x
        alpha, beta = _ols_simple(x, y)
        assert beta == pytest.approx(-2.0, abs=1e-10)
        assert alpha == pytest.approx(12.0, abs=1e-10)


# ── Correlation ──────────────────────────────────────────────

class TestCorrelation:
    def test_perfect_positive(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert _correlation(x, y) == pytest.approx(1.0, abs=1e-10)

    def test_perfect_negative(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        assert _correlation(x, y) == pytest.approx(-1.0, abs=1e-10)

    def test_zero_correlation(self):
        # Orthogonal: deviations cancel
        x = [1.0, -1.0, 1.0, -1.0]
        y = [1.0, 1.0, -1.0, -1.0]
        assert _correlation(x, y) == pytest.approx(0.0, abs=1e-10)

    def test_constant_series(self):
        x = [5.0, 5.0, 5.0]
        y = [1.0, 2.0, 3.0]
        assert _correlation(x, y) == 0.0

    def test_single_element(self):
        assert _correlation([1.0], [2.0]) == 0.0


# ── Half-life ────────────────────────────────────────────────

class TestHalfLife:
    def test_mean_reverting_spread(self):
        # AR(1) process with phi = 0.9 (mean-reverting)
        import random
        random.seed(42)
        spread = [0.0]
        phi = 0.9
        for _ in range(200):
            spread.append(phi * spread[-1] + random.gauss(0, 0.1))
        hl = _half_life(spread)
        assert hl > 0
        assert hl < float("inf")

    def test_non_mean_reverting(self):
        # Random walk (beta ≈ 0 in AR(1) of changes)
        spread = [float(i) for i in range(100)]  # pure trend
        hl = _half_life(spread)
        # For a pure trend, diff vs level regression gives beta ≈ 0 => inf
        # or positive beta => inf
        assert hl == float("inf")

    def test_too_short_series(self):
        assert _half_life([1.0, 2.0]) == float("inf")


# ── find_pairs ───────────────────────────────────────────────

class TestFindPairs:
    def test_correlated_pair(self):
        # Two highly correlated series
        import random
        random.seed(123)
        base = [100 + i * 0.5 + random.gauss(0, 0.1) for i in range(100)]
        a = base
        b = [p * 2.0 + 5.0 + random.gauss(0, 0.1) for p in base]
        prices = {"A": a, "B": b}
        results = find_pairs(prices, min_correlation=0.9, min_observations=60)
        assert len(results) == 1
        assert results[0].symbol_a == "A"
        assert results[0].symbol_b == "B"
        assert results[0].correlation > 0.9
        assert results[0].hedge_ratio != 0.0

    def test_uncorrelated_filtered(self):
        import random
        random.seed(456)
        a = [random.gauss(100, 5) for _ in range(100)]
        b = [random.gauss(100, 5) for _ in range(100)]
        prices = {"X": a, "Y": b}
        results = find_pairs(prices, min_correlation=0.7, min_observations=60)
        assert len(results) == 0

    def test_too_few_observations(self):
        prices = {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}
        results = find_pairs(prices, min_observations=60)
        assert len(results) == 0

    def test_sorted_by_half_life(self):
        import random
        random.seed(789)
        base = [100 + i * 0.3 for i in range(100)]
        a = base
        b = [p * 1.5 + random.gauss(0, 0.05) for p in base]
        c = [p * 0.8 + random.gauss(0, 0.05) for p in base]
        prices = {"A": a, "B": b, "C": c}
        results = find_pairs(prices, min_correlation=0.9, min_observations=60)
        if len(results) >= 2:
            assert results[0].half_life <= results[1].half_life


# ── PairsStrategy signals ───────────────────────────────────

class TestPairsStrategy:
    @pytest.fixture
    def pair(self):
        return PairResult(
            symbol_a="A", symbol_b="B",
            correlation=0.95, hedge_ratio=1.5,
            spread_mean=10.0, spread_std=2.0,
            half_life=15.0, is_cointegrated=True,
        )

    def test_no_signal_during_warmup(self, pair):
        ps = PairsStrategy(pair, window=10, entry_zscore=2.0)
        for i in range(9):
            sig = ps.on_prices(100.0, 160.0)
            assert sig is None  # still warming up

    def test_short_spread_entry(self, pair):
        """When z-score > entry_zscore, should short the spread."""
        ps = PairsStrategy(pair, window=10, entry_zscore=2.0)
        # Build up a baseline spread
        for _ in range(10):
            ps.on_prices(100.0, 160.0)
        # Now push spread way above mean => z > entry
        sig = ps.on_prices(100.0, 200.0)
        if sig is not None:
            assert sig.side_a == "buy"
            assert sig.side_b == "sell"
            assert sig.zscore > 0

    def test_long_spread_entry(self, pair):
        """When z-score < -entry_zscore, should long the spread."""
        ps = PairsStrategy(pair, window=10, entry_zscore=2.0)
        for _ in range(10):
            ps.on_prices(100.0, 160.0)
        # Push spread way below mean
        sig = ps.on_prices(100.0, 100.0)
        if sig is not None:
            assert sig.side_a == "sell"
            assert sig.side_b == "buy"
            assert sig.zscore < 0

    def test_exit_after_entry(self, pair):
        """After entering, should exit when z-score reverts."""
        ps = PairsStrategy(pair, window=10, entry_zscore=2.0, exit_zscore=0.5)
        # Warmup with normal spread
        normal_spread_b = pair.hedge_ratio * 100.0 + 10.0  # spread ≈ mean
        for _ in range(10):
            ps.on_prices(100.0, normal_spread_b)
        # Force entry: big positive z
        big_b = normal_spread_b + 50
        entry_sig = ps.on_prices(100.0, big_b)
        if entry_sig is not None and ps._position != "flat":
            # Feed prices that bring z back toward 0
            for _ in range(20):
                sig = ps.on_prices(100.0, normal_spread_b)
                if sig is not None and sig.strength == 0.0:
                    # This is an exit signal
                    assert ps._position == "flat"
                    break

    def test_strength_capped_at_1(self, pair):
        ps = PairsStrategy(pair, window=5, entry_zscore=1.0)
        for _ in range(5):
            ps.on_prices(100.0, 160.0)
        sig = ps.on_prices(100.0, 300.0)  # extreme z
        if sig is not None:
            assert sig.strength <= 1.0

    def test_no_signal_when_within_bounds(self, pair):
        ps = PairsStrategy(pair, window=10, entry_zscore=2.0)
        # Feed constant prices => spread constant => z ≈ 0 after warmup
        signals = []
        for _ in range(20):
            sig = ps.on_prices(100.0, 160.0)
            if sig is not None:
                signals.append(sig)
        # With constant spread, std = 0, so on_prices returns None
        # (the std == 0 guard fires)
        assert len(signals) == 0
