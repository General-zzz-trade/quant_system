"""Unit tests for RiskGate pre-execution risk checks."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from execution.safety.risk_gate import RiskGate, RiskGateConfig


def _order(symbol="BTCUSDT", qty=0.1, price=40000.0):
    return SimpleNamespace(symbol=symbol, qty=qty, price=price)


class TestRiskGate:

    def test_allowed_basic_order(self):
        gate = RiskGate(config=RiskGateConfig())
        r = gate.check(_order())
        assert r.allowed

    def test_kill_switch_blocks(self):
        gate = RiskGate(
            config=RiskGateConfig(),
            is_killed=lambda: True,
        )
        r = gate.check(_order())
        assert not r.allowed
        assert "kill_switch" in r.reason

    def test_kill_switch_not_triggered_allows(self):
        gate = RiskGate(
            config=RiskGateConfig(),
            is_killed=lambda: False,
        )
        r = gate.check(_order())
        assert r.allowed

    def test_max_open_orders_blocks(self):
        gate = RiskGate(
            config=RiskGateConfig(max_open_orders=5),
            get_open_order_count=lambda: 5,
        )
        r = gate.check(_order())
        assert not r.allowed
        assert "max_open_orders" in r.reason

    def test_order_notional_blocks(self):
        gate = RiskGate(
            config=RiskGateConfig(max_order_notional=1000.0),
        )
        r = gate.check(_order(qty=1.0, price=2000.0))  # 2000 > 1000
        assert not r.allowed
        assert "order_notional" in r.reason

    def test_position_notional_blocks(self):
        positions = {"BTCUSDT": SimpleNamespace(qty=2.0)}
        gate = RiskGate(
            config=RiskGateConfig(max_position_notional=99_000.0),
            get_positions=lambda: positions,
        )
        # existing: 2.0 * 40000 = 80k, new: 0.5 * 40000 = 20k, total = 100k > 99k
        r = gate.check(_order(qty=0.5, price=40000.0))
        assert not r.allowed
        assert "position_notional" in r.reason

    def test_portfolio_notional_blocks(self):
        positions = {
            "BTCUSDT": SimpleNamespace(qty=1.0),
            "ETHUSDT": SimpleNamespace(qty=1.0),
        }
        gate = RiskGate(
            config=RiskGateConfig(
                max_position_notional=999_999.0,  # high enough to pass position check
                max_portfolio_notional=50_000.0,
            ),
            get_positions=lambda: positions,
        )
        # BTC: 1*40000=40k, ETH: 1*40000=40k, order: 0.1*40000=4k → total=84k > 50k
        r = gate.check(_order(qty=0.1, price=40000.0))
        assert not r.allowed
        assert "portfolio_notional" in r.reason

    def test_no_price_allows(self):
        gate = RiskGate(config=RiskGateConfig(max_order_notional=1.0))
        cmd = SimpleNamespace(symbol="BTC", qty=999.0)  # no price attr
        r = gate.check(cmd)
        assert r.allowed

    def test_no_qty_allows(self):
        gate = RiskGate(config=RiskGateConfig(max_order_notional=1.0))
        cmd = SimpleNamespace(symbol="BTC", price=999.0)  # no qty attr
        r = gate.check(cmd)
        assert r.allowed

    def test_qty_attribute_variants(self):
        gate = RiskGate(config=RiskGateConfig(max_order_notional=100.0))
        # "quantity" attr
        r = gate.check(SimpleNamespace(symbol="X", quantity=1.0, price=200.0))
        assert not r.allowed
        # "size" attr
        r = gate.check(SimpleNamespace(symbol="X", size=1.0, price=200.0))
        assert not r.allowed

    def test_no_callbacks_allows(self):
        gate = RiskGate(config=RiskGateConfig())
        r = gate.check(_order())
        assert r.allowed
