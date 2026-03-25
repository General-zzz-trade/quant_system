"""Tests for VPINEntryGate."""

import time
from unittest.mock import MagicMock
from strategy.gates.vpin_entry_gate import VPINEntryConfig, VPINEntryGate


def _make_ev(signal=1):
    ev = MagicMock()
    ev.metadata = {"signal": signal}
    return ev


class TestVPINEntryGate:
    def test_safe_vpin_passes(self):
        gate = VPINEntryGate()
        r = gate.check(_make_ev(1), {"vpin": 0.1, "ob_imbalance": 0.0})
        assert r.allowed
        assert r.scale == 1.0

    def test_danger_vpin_delays(self):
        gate = VPINEntryGate()
        r = gate.check(_make_ev(1), {"vpin": 0.8, "symbol": "ETH"})
        assert r.allowed
        assert r.scale == 0.0  # delay
        assert "vpin_danger" in r.reason

    def test_caution_vpin_reduces(self):
        gate = VPINEntryGate()
        r = gate.check(_make_ev(1), {"vpin": 0.55})
        assert r.allowed
        assert r.scale == 0.3  # between caution(0.5) and danger(0.7)

    def test_imbalance_boost_aligned(self):
        """Buy + positive imbalance (bid-heavy) → boost."""
        gate = VPINEntryGate(VPINEntryConfig(imbalance_boost=0.2))
        r = gate.check(_make_ev(1), {"vpin": 0.1, "ob_imbalance": 0.5})
        assert r.scale == 1.3

    def test_imbalance_no_boost_opposed(self):
        """Buy + negative imbalance (ask-heavy) → no boost."""
        gate = VPINEntryGate(VPINEntryConfig(imbalance_boost=0.2))
        r = gate.check(_make_ev(1), {"vpin": 0.1, "ob_imbalance": -0.5})
        assert r.scale == 1.0

    def test_sell_imbalance_boost(self):
        """Sell + negative imbalance (ask-heavy) → boost."""
        gate = VPINEntryGate(VPINEntryConfig(imbalance_boost=0.2))
        r = gate.check(_make_ev(-1), {"vpin": 0.1, "ob_imbalance": -0.5})
        assert r.scale == 1.3

    def test_wide_spread_reduces(self):
        gate = VPINEntryGate(VPINEntryConfig(max_spread_bps=3.0))
        r = gate.check(_make_ev(1), {"vpin": 0.1, "spread_bps": 5.0, "symbol": "ETH"})
        assert r.scale == 0.3

    def test_max_delay_forces_entry(self):
        cfg = VPINEntryConfig(max_delay_s=0.1)
        gate = VPINEntryGate(cfg)
        # First call: high VPIN → delay
        gate.check(_make_ev(1), {"vpin": 0.8, "symbol": "ETH"})
        time.sleep(0.15)
        # Second call: still high VPIN but max delay reached
        r = gate.check(_make_ev(1), {"vpin": 0.8, "symbol": "ETH"})
        assert r.allowed
        assert r.scale == 0.7  # forced but cautious
        assert "max_delay" in r.reason

    def test_disabled(self):
        gate = VPINEntryGate(VPINEntryConfig(enabled=False))
        r = gate.check(_make_ev(1), {"vpin": 0.9})
        assert r.scale == 1.0

    def test_stats(self):
        gate = VPINEntryGate()
        gate.check(_make_ev(1), {"vpin": 0.8, "symbol": "ETH"})
        gate.check(_make_ev(1), {"vpin": 0.1, "ob_imbalance": 0.5})
        s = gate.stats
        assert s["total_checks"] == 2
        assert s["delayed"] == 1

    def test_no_signal_passes(self):
        gate = VPINEntryGate()
        r = gate.check(_make_ev(0), {"vpin": 0.1})
        assert r.allowed
        assert r.scale == 1.0
