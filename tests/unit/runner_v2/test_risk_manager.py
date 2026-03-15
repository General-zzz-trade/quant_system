"""Tests for RiskManager."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from runner.risk_manager import RiskManager


def _make_signal(qty=0.1, notional=500.0):
    return SimpleNamespace(qty=qty, notional=notional)


class TestRiskManagerCheck:
    def test_allows_normal_signal(self):
        ks = MagicMock()
        ks.is_killed.return_value = False
        rm = RiskManager(kill_switch=ks, max_position=1.0,
                         max_notional=10000.0, max_open_orders=5)
        allowed, reason = rm.check(_make_signal(), osm_open_count=0)
        assert allowed is True
        assert reason == ""

    def test_blocks_when_kill_switch_active(self):
        ks = MagicMock()
        ks.is_killed.return_value = True
        rm = RiskManager(kill_switch=ks, max_position=1.0,
                         max_notional=10000.0, max_open_orders=5)
        allowed, reason = rm.check(_make_signal(), osm_open_count=0)
        assert allowed is False
        assert "kill" in reason.lower()

    def test_blocks_oversized_position(self):
        ks = MagicMock()
        ks.is_killed.return_value = False
        rm = RiskManager(kill_switch=ks, max_position=0.05,
                         max_notional=500.0, max_open_orders=5)
        allowed, reason = rm.check(_make_signal(qty=1.0), osm_open_count=0)
        assert allowed is False
        assert "position" in reason.lower()

    def test_blocks_oversized_notional(self):
        ks = MagicMock()
        ks.is_killed.return_value = False
        rm = RiskManager(kill_switch=ks, max_position=10.0,
                         max_notional=100.0, max_open_orders=5)
        allowed, reason = rm.check(_make_signal(notional=500.0), osm_open_count=0)
        assert allowed is False
        assert "notional" in reason.lower()

    def test_blocks_too_many_open_orders(self):
        ks = MagicMock()
        ks.is_killed.return_value = False
        rm = RiskManager(kill_switch=ks, max_position=1.0,
                         max_notional=10000.0, max_open_orders=3)
        allowed, reason = rm.check(_make_signal(), osm_open_count=3)
        assert allowed is False
        assert "open_orders" in reason.lower()


class TestRiskManagerKill:
    def test_kill_activates_kill_switch(self):
        ks = MagicMock()
        rm = RiskManager(kill_switch=ks, max_position=1.0,
                         max_notional=10000.0, max_open_orders=5)
        rm.kill("test reason")
        ks.activate.assert_called_once()


class TestRiskManagerCheckpoint:
    def test_checkpoint_returns_dict(self):
        ks = MagicMock()
        ks.get_state.return_value = {"killed": False}
        rm = RiskManager(kill_switch=ks, max_position=1.0,
                         max_notional=10000.0, max_open_orders=5)
        state = rm.checkpoint()
        assert isinstance(state, dict)
        assert "kill_switch" in state
