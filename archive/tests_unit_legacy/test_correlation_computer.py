"""Unit tests for CorrelationComputer."""
from __future__ import annotations

import math

import pytest

from risk.correlation_computer import CorrelationComputer, _pearson_corr


class TestPearsonCorr:

    def test_perfect_positive(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert abs(_pearson_corr(x, y) - 1.0) < 1e-10

    def test_perfect_negative(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        assert abs(_pearson_corr(x, y) - (-1.0)) < 1e-10

    def test_uncorrelated(self):
        x = [1.0, -1.0, 1.0, -1.0]
        y = [1.0, 1.0, -1.0, -1.0]
        assert abs(_pearson_corr(x, y)) < 1e-10

    def test_insufficient_data(self):
        assert _pearson_corr([1.0], [2.0]) is None
        assert _pearson_corr([], []) is None

    def test_constant_series_returns_none(self):
        assert _pearson_corr([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]) is None


class TestCorrelationComputer:

    def test_update_builds_returns(self):
        cc = CorrelationComputer(window=10)
        prices = [100.0, 101.0, 102.0, 103.0]
        for p in prices:
            cc.update("BTC", p)

        assert len(cc._returns["BTC"]) == 3  # n-1 returns

    def test_update_skips_first_price(self):
        cc = CorrelationComputer(window=10)
        cc.update("BTC", 100.0)
        assert "BTC" not in cc._returns

    def test_window_truncation(self):
        cc = CorrelationComputer(window=5)
        for i in range(20):
            cc.update("BTC", 100.0 + i)
        assert len(cc._returns["BTC"]) == 5

    def test_portfolio_avg_correlation_none_if_few_symbols(self):
        cc = CorrelationComputer(window=10)
        for i in range(10):
            cc.update("BTC", 100.0 + i)
        assert cc.portfolio_avg_correlation(["BTC"]) is None

    def test_portfolio_avg_correlation_identical_series(self):
        cc = CorrelationComputer(window=100)
        for i in range(50):
            p = 100.0 + i * 0.5
            cc.update("A", p)
            cc.update("B", p)  # identical price → identical returns

        corr = cc.portfolio_avg_correlation(["A", "B"])
        assert corr is not None
        assert abs(corr - 1.0) < 1e-6

    def test_portfolio_avg_correlation_anticorrelated_series(self):
        cc = CorrelationComputer(window=100)
        # Alternating: when A goes up, B goes down and vice versa
        base_a = 100.0
        base_b = 100.0
        for i in range(50):
            sign = 1.0 if i % 2 == 0 else -1.0
            base_a *= (1.0 + sign * 0.01)
            base_b *= (1.0 - sign * 0.01)
            cc.update("A", base_a)
            cc.update("B", base_b)

        corr = cc.portfolio_avg_correlation(["A", "B"])
        assert corr is not None
        assert corr < -0.9

    def test_position_correlation(self):
        cc = CorrelationComputer(window=100)
        for i in range(50):
            p = 100.0 + i * 0.5
            cc.update("BTC", p)
            cc.update("ETH", p * 1.01)  # highly correlated
            cc.update("SOL", p * 0.99)  # highly correlated

        corr = cc.position_correlation("SOL", ["BTC", "ETH"])
        assert corr is not None
        assert corr > 0.9

    def test_position_correlation_none_when_no_data(self):
        cc = CorrelationComputer(window=10)
        assert cc.position_correlation("BTC", ["ETH"]) is None

    def test_three_symbols_average(self):
        cc = CorrelationComputer(window=100)
        for i in range(50):
            cc.update("A", 100.0 + i)
            cc.update("B", 100.0 + i * 2)
            cc.update("C", 100.0 + i * 3)

        corr = cc.portfolio_avg_correlation(["A", "B", "C"])
        assert corr is not None
        # All trending up → should be highly positively correlated
        assert corr > 0.9
