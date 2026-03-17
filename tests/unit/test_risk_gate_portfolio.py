# tests/unit/test_risk_gate_portfolio.py
"""Tests for RiskGate portfolio-level notional checks."""
from __future__ import annotations

from dataclasses import dataclass

from execution.safety.risk_gate import RiskGate, RiskGateConfig


@dataclass
class _Cmd:
    symbol: str = "BTCUSDT"
    qty: float = 1.0
    price: float = 50000.0


@dataclass
class _Pos:
    qty: float = 0.0
    mark_price: float = 50000.0


def test_single_symbol_under_limit():
    gate = RiskGate(config=RiskGateConfig(max_position_notional=200_000))
    result = gate.check(_Cmd(qty=1.0, price=50000.0))
    assert result.allowed


def test_single_symbol_over_position_limit():
    gate = RiskGate(
        config=RiskGateConfig(max_position_notional=40_000),
        get_positions=lambda: {},
    )
    result = gate.check(_Cmd(qty=1.0, price=50000.0))
    assert not result.allowed
    assert "position_notional" in result.reason


def test_portfolio_notional_exceeded():
    positions = {
        "BTCUSDT": _Pos(qty=5.0, mark_price=50000.0),
        "ETHUSDT": _Pos(qty=100.0, mark_price=3000.0),
    }
    gate = RiskGate(
        config=RiskGateConfig(
            max_position_notional=500_000,
            max_order_notional=100_000,
            max_portfolio_notional=400_000,
        ),
        get_positions=lambda: positions,
    )
    # Existing: BTC 250k + ETH 300k = 550k, new order 50k → 600k > 400k
    result = gate.check(_Cmd(symbol="SOLUSDT", qty=1.0, price=50000.0))
    assert not result.allowed
    assert "portfolio_notional" in result.reason


def test_portfolio_notional_within_limit():
    positions = {
        "BTCUSDT": _Pos(qty=1.0, mark_price=50000.0),
    }
    gate = RiskGate(
        config=RiskGateConfig(
            max_position_notional=500_000,
            max_order_notional=500_000,
            max_portfolio_notional=500_000,
        ),
        get_positions=lambda: positions,
    )
    result = gate.check(_Cmd(symbol="ETHUSDT", qty=10.0, price=3000.0))
    assert result.allowed


def test_kill_switch_blocks():
    gate = RiskGate(is_killed=lambda: True)
    result = gate.check(_Cmd())
    assert not result.allowed
    assert "kill_switch" in result.reason


def test_order_notional_exceeded():
    gate = RiskGate(config=RiskGateConfig(max_order_notional=10_000))
    result = gate.check(_Cmd(qty=1.0, price=50000.0))
    assert not result.allowed
    assert "order_notional" in result.reason


class TestRiskGateMarketOrderPrice:
    def test_market_order_with_mark_price_accepted(self):
        """Mark price should be preferred for notional calculation."""
        from types import SimpleNamespace
        from execution.safety.risk_gate import RiskGate, RiskGateConfig
        gate = RiskGate(RiskGateConfig(max_order_notional=10000))
        cmd = SimpleNamespace(qty=1.0, mark_price=3000.0)
        result = gate.check(cmd)
        assert result.allowed is True

    def test_market_order_without_any_price_rejected(self):
        """Market orders with no price field must be rejected."""
        from types import SimpleNamespace
        from execution.safety.risk_gate import RiskGate, RiskGateConfig
        gate = RiskGate(RiskGateConfig())
        cmd = SimpleNamespace(qty=1.0)
        result = gate.check(cmd)
        assert result.allowed is False

    def test_mark_price_preferred_over_order_price(self):
        """mark_price takes precedence over price."""
        from types import SimpleNamespace
        from execution.safety.risk_gate import _get_price
        cmd = SimpleNamespace(mark_price=3100.0, price=3000.0)
        assert _get_price(cmd) == 3100.0
