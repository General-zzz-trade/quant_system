"""Tests for correlation gate — pre-trade correlation checks."""
from __future__ import annotations

import math
import random

import pytest

from risk.correlation_computer import CorrelationComputer
from risk.correlation_gate import CorrelationGate, CorrelationGateConfig
from risk.decisions import RiskAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feed_correlated(computer: CorrelationComputer, symbols: list[str],
                     n: int = 100, correlation: float = 0.9, seed: int = 42):
    """Feed correlated price series into the computer."""
    rng = random.Random(seed)
    base_returns = [rng.gauss(0, 0.02) for _ in range(n)]

    for sym in symbols:
        price = 100.0
        for i in range(n):
            noise = rng.gauss(0, 0.02 * (1 - correlation))
            ret = base_returns[i] * correlation + noise
            price *= (1 + ret)
            computer.update(sym, price)


def _feed_independent(computer: CorrelationComputer, symbols: list[str],
                      n: int = 100, seed: int = 42):
    """Feed independent price series into the computer."""
    rng = random.Random(seed)
    for sym in symbols:
        price = 100.0
        for _ in range(n):
            ret = rng.gauss(0, 0.02)
            price *= (1 + ret)
            computer.update(sym, price)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCorrelationGate:

    def test_high_correlation_rejects(self):
        """3 correlated symbols + 4th correlated → REJECT."""
        computer = CorrelationComputer(window=60)
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
        _feed_correlated(computer, symbols, n=100, correlation=0.95)

        gate = CorrelationGate(computer, CorrelationGateConfig(
            max_avg_correlation=0.5,
            max_position_correlation=0.5,
        ))

        result = gate.should_allow("SOLUSDT", ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
        assert result.action == RiskAction.REJECT
        assert len(result.violations) > 0

    def test_independent_symbols_allows(self):
        """Independent symbols → ALLOW."""
        computer = CorrelationComputer(window=60)
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
        _feed_independent(computer, symbols, n=100)

        gate = CorrelationGate(computer, CorrelationGateConfig(
            max_avg_correlation=0.7,
            max_position_correlation=0.85,
        ))

        result = gate.should_allow("SOLUSDT", ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
        assert result.action == RiskAction.ALLOW

    def test_no_existing_positions_allows(self):
        """No existing positions → always ALLOW."""
        computer = CorrelationComputer(window=60)
        gate = CorrelationGate(computer)
        result = gate.should_allow("BTCUSDT", [])
        assert result.action == RiskAction.ALLOW

    def test_insufficient_data_allows(self):
        """Insufficient data for new symbol → ALLOW (don't block)."""
        computer = CorrelationComputer(window=60)
        # Feed existing symbols but not the new one
        _feed_independent(computer, ["BTCUSDT", "ETHUSDT"], n=100)

        gate = CorrelationGate(computer, CorrelationGateConfig(min_data_points=20))
        result = gate.should_allow("SOLUSDT", ["BTCUSDT", "ETHUSDT"])
        assert result.action == RiskAction.ALLOW

    def test_single_existing_position(self):
        """Single existing position — checks position correlation only."""
        computer = CorrelationComputer(window=60)
        _feed_correlated(computer, ["BTCUSDT", "ETHUSDT"], n=100, correlation=0.99)

        gate = CorrelationGate(computer, CorrelationGateConfig(
            max_position_correlation=0.5,
        ))
        result = gate.should_allow("ETHUSDT", ["BTCUSDT"])
        assert result.action == RiskAction.REJECT

    def test_avg_correlation_threshold(self):
        """Portfolio avg correlation exceeds threshold → REJECT."""
        computer = CorrelationComputer(window=60)
        symbols = ["A", "B", "C", "D"]
        _feed_correlated(computer, symbols, n=100, correlation=0.95)

        gate = CorrelationGate(computer, CorrelationGateConfig(
            max_avg_correlation=0.3,
            max_position_correlation=1.0,  # high to not trigger
        ))
        result = gate.should_allow("D", ["A", "B", "C"])
        assert result.action == RiskAction.REJECT
        assert any("avg correlation" in v.message for v in result.violations)

    def test_config_defaults(self):
        """Default config values."""
        config = CorrelationGateConfig()
        assert config.max_avg_correlation == 0.7
        assert config.max_position_correlation == 0.85
        assert config.min_data_points == 20

    def test_few_data_points_for_symbol(self):
        """Symbol with data but below min_data_points → ALLOW."""
        computer = CorrelationComputer(window=60)
        _feed_independent(computer, ["BTCUSDT"], n=100)
        # Add just a few data points for SOL
        for i in range(5):
            computer.update("SOLUSDT", 100 + i)

        gate = CorrelationGate(computer, CorrelationGateConfig(min_data_points=20))
        result = gate.should_allow("SOLUSDT", ["BTCUSDT"])
        assert result.action == RiskAction.ALLOW
