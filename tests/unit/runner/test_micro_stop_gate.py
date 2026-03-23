"""Tests for MicroStopGate."""

import time
from unittest.mock import MagicMock
from runner.gates.micro_stop_gate import MicroStopConfig, MicroStopGate


def _make_ev():
    return MagicMock()


class TestMicroStopGate:
    def test_no_position_passes(self):
        gate = MicroStopGate()
        r = gate.check(_make_ev(), {"symbol": "ETHUSDT", "price": 2000.0})
        assert r.allowed

    def test_initial_stop_long(self):
        cfg = MicroStopConfig(initial_stop_pct=0.020)
        gate = MicroStopGate(cfg)
        stop = gate.on_new_position("ETHUSDT", 1, 2000.0)
        assert abs(stop - 1960.0) < 0.01  # 2000 × (1 - 0.02)

    def test_initial_stop_short(self):
        cfg = MicroStopConfig(initial_stop_pct=0.020)
        gate = MicroStopGate(cfg)
        stop = gate.on_new_position("ETHUSDT", -1, 2000.0)
        assert abs(stop - 2040.0) < 0.01  # 2000 × (1 + 0.02)

    def test_long_stop_triggered(self):
        gate = MicroStopGate(MicroStopConfig(initial_stop_pct=0.020))
        gate.on_new_position("ETHUSDT", 1, 2000.0)
        r = gate.check(_make_ev(), {"symbol": "ETHUSDT", "price": 1955.0})
        assert not r.allowed
        assert "long_stop" in r.reason

    def test_long_stop_not_triggered(self):
        gate = MicroStopGate(MicroStopConfig(initial_stop_pct=0.020))
        gate.on_new_position("ETHUSDT", 1, 2000.0)
        r = gate.check(_make_ev(), {"symbol": "ETHUSDT", "price": 1970.0})
        assert r.allowed

    def test_breakeven_phase(self):
        cfg = MicroStopConfig(
            initial_stop_pct=0.003,
            breakeven_trigger_pct=0.001,
            breakeven_buffer_pct=0.0002,
        )
        gate = MicroStopGate(cfg)
        gate.on_new_position("ETHUSDT", 1, 2000.0)

        # Price moves up 0.15% → triggers breakeven
        gate.check(_make_ev(), {"symbol": "ETHUSDT", "price": 2003.0})
        assert gate.get_phase("ETHUSDT") == "BREAKEVEN"

        # Stop should now be near entry + buffer
        stop = gate.get_stop_price("ETHUSDT")
        assert stop > 2000.0  # above entry

    def test_trailing_phase(self):
        cfg = MicroStopConfig(
            breakeven_trigger_pct=0.001,
            trail_trigger_pct=0.002,
            trail_distance_pct=0.0015,
        )
        gate = MicroStopGate(cfg)
        gate.on_new_position("ETHUSDT", 1, 2000.0)

        # Push into trailing
        gate.check(_make_ev(), {"symbol": "ETHUSDT", "price": 2005.0})
        assert gate.get_phase("ETHUSDT") == "TRAILING"

        # Peak at 2005, trail at 2005 × (1 - 0.0015) ≈ 2001.99
        stop = gate.get_stop_price("ETHUSDT")
        assert 2001.0 < stop < 2003.0

    def test_trailing_ratchets_up(self):
        cfg = MicroStopConfig(
            breakeven_trigger_pct=0.001,
            trail_trigger_pct=0.002,
            trail_distance_pct=0.001,
        )
        gate = MicroStopGate(cfg)
        gate.on_new_position("ETHUSDT", 1, 2000.0)

        gate.check(_make_ev(), {"symbol": "ETHUSDT", "price": 2005.0})
        stop1 = gate.get_stop_price("ETHUSDT")

        gate.check(_make_ev(), {"symbol": "ETHUSDT", "price": 2010.0})
        stop2 = gate.get_stop_price("ETHUSDT")
        assert stop2 > stop1  # trailing ratchets up

    def test_short_stop_triggered(self):
        gate = MicroStopGate(MicroStopConfig(initial_stop_pct=0.020))
        gate.on_new_position("ETHUSDT", -1, 2000.0)
        r = gate.check(_make_ev(), {"symbol": "ETHUSDT", "price": 2045.0})
        assert not r.allowed

    def test_realtime_check(self):
        gate = MicroStopGate(MicroStopConfig(initial_stop_pct=0.020))
        gate.on_new_position("ETHUSDT", 1, 2000.0)

        assert not gate.check_realtime("ETHUSDT", 1970.0)
        assert gate.check_realtime("ETHUSDT", 1955.0)

    def test_max_hold_time(self):
        cfg = MicroStopConfig(max_hold_s=0.1)  # 100ms for test
        gate = MicroStopGate(cfg)
        gate.on_new_position("ETHUSDT", 1, 2000.0)

        time.sleep(0.15)
        r = gate.check(_make_ev(), {"symbol": "ETHUSDT", "price": 2001.0})
        assert not r.allowed
        assert "max_hold" in r.reason

    def test_reset_symbol(self):
        gate = MicroStopGate()
        gate.on_new_position("ETHUSDT", 1, 2000.0)
        gate.reset_symbol("ETHUSDT")
        assert gate.get_phase("ETHUSDT") == "NONE"

    def test_max_loss_clamp(self):
        cfg = MicroStopConfig(initial_stop_pct=0.05, max_loss_pct=0.025)
        gate = MicroStopGate(cfg)
        gate.on_new_position("ETHUSDT", 1, 2000.0)
        # initial_stop would be 0.05 but max_loss clamps to 0.025
        stop = gate.get_stop_price("ETHUSDT")
        expected = 2000.0 * (1 - 0.025)
        assert abs(stop - expected) < 0.1

    def test_multiple_symbols(self):
        gate = MicroStopGate()
        gate.on_new_position("ETHUSDT", 1, 2000.0)
        gate.on_new_position("BTCUSDT", -1, 90000.0)
        assert gate.get_phase("ETHUSDT") == "INITIAL"
        assert gate.get_phase("BTCUSDT") == "INITIAL"

