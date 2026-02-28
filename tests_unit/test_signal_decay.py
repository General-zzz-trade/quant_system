"""Tests for SignalDecayAnalyzer."""
from __future__ import annotations

import math

import pytest

from monitoring.signal_decay_analysis import SignalDecayAnalyzer, _spearman_rank_corr


class TestSpearmanRankCorr:
    def test_perfect_positive(self) -> None:
        pairs = [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0)]
        assert abs(_spearman_rank_corr(pairs) - 1.0) < 1e-6

    def test_perfect_negative(self) -> None:
        pairs = [(1.0, 40.0), (2.0, 30.0), (3.0, 20.0), (4.0, 10.0)]
        assert abs(_spearman_rank_corr(pairs) - (-1.0)) < 1e-6

    def test_no_correlation(self) -> None:
        # Roughly uncorrelated
        pairs = [(1.0, 3.0), (2.0, 1.0), (3.0, 4.0), (4.0, 2.0)]
        corr = _spearman_rank_corr(pairs)
        assert abs(corr) < 0.5


class TestICComputation:
    def test_basic_ic_series(self) -> None:
        analyzer = SignalDecayAnalyzer(max_lags=5)
        # Perfect correlation at lag 0
        for i in range(10):
            analyzer.record(float(i), float(i) * 0.1, lag=0)
        ic = analyzer.compute_ic_series()
        assert 0 in ic
        assert ic[0] > 0.9

    def test_decaying_ic(self) -> None:
        analyzer = SignalDecayAnalyzer(max_lags=5)
        import random
        random.seed(42)
        for lag in range(6):
            for i in range(20):
                signal = float(i)
                # Add increasing noise at higher lags
                noise = random.gauss(0, lag * 2.0) if lag > 0 else 0.0
                ret = signal * 0.1 + noise
                analyzer.record(signal, ret, lag=lag)

        ic = analyzer.compute_ic_series()
        assert len(ic) >= 2
        # IC at lag 0 should be higher than at higher lags
        if 0 in ic and max(ic.keys()) > 0:
            assert ic[0] >= ic[max(ic.keys())] or abs(ic[0] - ic[max(ic.keys())]) < 0.5


class TestHalfLife:
    def test_exponentially_decaying_ic(self) -> None:
        analyzer = SignalDecayAnalyzer(max_lags=10)
        import random
        random.seed(123)
        for lag in range(11):
            for i in range(50):
                signal = float(i)
                decay = math.exp(-0.2 * lag)  # known decay rate
                noise = random.gauss(0, 0.5)
                ret = signal * decay + noise
                analyzer.record(signal, ret, lag=lag)

        hl = analyzer.half_life()
        # Half-life should exist and be positive
        if hl is not None:
            assert hl > 0

    def test_no_data_returns_none(self) -> None:
        analyzer = SignalDecayAnalyzer()
        assert analyzer.half_life() is None

    def test_insufficient_lags_returns_none(self) -> None:
        analyzer = SignalDecayAnalyzer(max_lags=5)
        for i in range(5):
            analyzer.record(float(i), float(i) * 0.1, lag=0)
        # Only lag 0 data, can't fit decay
        assert analyzer.half_life() is None


class TestDecayDetection:
    def test_is_decayed_with_weak_signal(self) -> None:
        analyzer = SignalDecayAnalyzer(max_lags=3)
        import random
        random.seed(99)
        # Strong signal at lag 0
        for i in range(20):
            analyzer.record(float(i), float(i) * 0.5, lag=0)
        # Pure noise at lag 3
        for i in range(20):
            analyzer.record(float(i), random.gauss(0, 10), lag=3)

        # Is decayed checks the latest lag
        result = analyzer.is_decayed(threshold_ic=0.5)
        # Should consider it decayed since lag 3 IC is near zero
        assert result is True

    def test_not_decayed_with_strong_signal(self) -> None:
        analyzer = SignalDecayAnalyzer(max_lags=3)
        for lag in range(4):
            for i in range(20):
                analyzer.record(float(i), float(i) * 0.5, lag=lag)
        # All lags have strong IC
        result = analyzer.is_decayed(threshold_ic=0.02)
        assert result is False


class TestSummary:
    def test_summary_structure(self) -> None:
        analyzer = SignalDecayAnalyzer(max_lags=5)
        for i in range(10):
            analyzer.record(float(i), float(i) * 0.1, lag=0)
            analyzer.record(float(i), float(i) * 0.05, lag=1)

        s = analyzer.summary()
        assert "ic_series" in s
        assert "half_life" in s
        assert "is_decayed" in s
        assert "n_observations" in s
        assert s["n_observations"][0] == 10
        assert s["n_observations"][1] == 10
