"""Tests for MakerExecGate."""

from unittest.mock import MagicMock
from runner.gates.maker_exec_gate import MakerExecConfig, MakerExecGate


def _make_ev(side="buy"):
    ev = MagicMock()
    ev.metadata = {"side": side}
    return ev


class TestMakerExecGate:
    def test_buy_limit_at_bbo(self):
        gate = MakerExecGate(MakerExecConfig(offset_ticks=0, tick_size=0.01))
        ctx = {"best_bid": 2000.00, "best_ask": 2000.05}
        r = gate.check(_make_ev("buy"), ctx)
        assert r.allowed
        assert ctx["maker_exec"]["limit_price"] == 2000.00
        assert ctx["maker_exec"]["exec_type"] == "limit"

    def test_sell_limit_at_bbo(self):
        gate = MakerExecGate(MakerExecConfig(offset_ticks=0))
        ctx = {"best_bid": 2000.00, "best_ask": 2000.05}
        r = gate.check(_make_ev("sell"), ctx)
        assert r.allowed
        assert ctx["maker_exec"]["limit_price"] == 2000.05

    def test_buy_offset_1_tick(self):
        gate = MakerExecGate(MakerExecConfig(offset_ticks=1, tick_size=0.01))
        ctx = {"best_bid": 2000.00, "best_ask": 2000.05}
        gate.check(_make_ev("buy"), ctx)
        # bid + 1 tick = 2000.01, but must not cross (ask - tick = 2000.04)
        assert ctx["maker_exec"]["limit_price"] == 2000.01

    def test_tight_spread_fallback_taker(self):
        """1-tick spread → can't improve, fall back to taker."""
        gate = MakerExecGate(MakerExecConfig(min_spread_ticks=2, tick_size=0.01))
        ctx = {"best_bid": 2000.00, "best_ask": 2000.01}
        r = gate.check(_make_ev("buy"), ctx)
        assert r.allowed
        assert "maker_exec" not in ctx  # no maker annotation

    def test_no_bbo_fallback(self):
        gate = MakerExecGate()
        ctx = {}
        r = gate.check(_make_ev("buy"), ctx)
        assert r.allowed
        assert "maker_exec" not in ctx

    def test_disabled(self):
        gate = MakerExecGate(MakerExecConfig(enabled=False))
        ctx = {"best_bid": 2000.0, "best_ask": 2000.05}
        r = gate.check(_make_ev("buy"), ctx)
        assert r.allowed
        assert "maker_exec" not in ctx

    def test_buy_doesnt_cross_spread(self):
        """Buy limit must be < ask."""
        gate = MakerExecGate(MakerExecConfig(offset_ticks=10, tick_size=0.01))
        ctx = {"best_bid": 2000.00, "best_ask": 2000.05}
        gate.check(_make_ev("buy"), ctx)
        # 2000.00 + 10×0.01 = 2000.10, but clamped to ask - tick = 2000.04
        assert ctx["maker_exec"]["limit_price"] == 2000.04

    def test_stats_tracking(self):
        gate = MakerExecGate()
        gate.record_fill(was_maker=True)
        gate.record_fill(was_maker=True)
        gate.record_fill(was_maker=False)
        assert gate.maker_rate == 2 / 3
        assert gate.stats["maker_fills"] == 2

    def test_callback_bbo(self):
        gate = MakerExecGate(get_bbo=lambda s: (2000.0, 2000.10))
        ctx = {"symbol": "ETHUSDT"}
        gate.check(_make_ev("buy"), ctx)
        assert ctx["maker_exec"]["limit_price"] == 2000.0
