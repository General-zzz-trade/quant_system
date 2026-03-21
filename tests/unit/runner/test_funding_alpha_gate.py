"""Tests for FundingAlphaGate."""

from unittest.mock import MagicMock
from runner.gates.funding_alpha_gate import FundingAlphaGate


def _make_ev(signal=0):
    ev = MagicMock()
    ev.metadata = {"signal": signal}
    return ev


class TestFundingAlphaGate:
    def test_no_signal_no_scale(self):
        gate = FundingAlphaGate()
        r = gate.check(_make_ev(0), {"funding_rate": 0.0001})
        assert r.allowed
        assert r.scale == 1.0

    def test_long_receives_negative_funding(self):
        """Long + negative funding = longs RECEIVE → boost."""
        gate = FundingAlphaGate()
        r = gate.check(_make_ev(1), {"funding_rate": -0.0001})
        assert r.allowed
        assert r.scale == 1.5

    def test_long_pays_positive_funding(self):
        """Long + positive funding = longs PAY → reduce."""
        gate = FundingAlphaGate()
        r = gate.check(_make_ev(1), {"funding_rate": 0.0001})
        assert r.allowed
        assert r.scale == 0.3

    def test_short_receives_positive_funding(self):
        """Short + positive funding = shorts RECEIVE → boost."""
        gate = FundingAlphaGate()
        r = gate.check(_make_ev(-1), {"funding_rate": 0.0001})
        assert r.allowed
        assert r.scale == 1.5

    def test_short_pays_negative_funding(self):
        """Short + negative funding = shorts PAY → reduce."""
        gate = FundingAlphaGate()
        r = gate.check(_make_ev(-1), {"funding_rate": -0.0001})
        assert r.allowed
        assert r.scale == 0.3

    def test_high_funding_opposed_blocks(self):
        """Extreme funding against us → block trade."""
        gate = FundingAlphaGate()
        r = gate.check(_make_ev(1), {"funding_rate": 0.001})
        assert not r.allowed

    def test_high_funding_aligned_boosts(self):
        """Extreme funding in our favor → 2x boost."""
        gate = FundingAlphaGate()
        r = gate.check(_make_ev(-1), {"funding_rate": 0.001})
        assert r.allowed
        assert r.scale == 2.0

    def test_zero_funding_neutral(self):
        gate = FundingAlphaGate()
        r = gate.check(_make_ev(1), {"funding_rate": 0.0})
        assert r.allowed
        assert r.scale == 1.0

    def test_disabled(self):
        gate = FundingAlphaGate(enabled=False)
        r = gate.check(_make_ev(1), {"funding_rate": 0.001})
        assert r.allowed
        assert r.scale == 1.0

    def test_callback_funding(self):
        gate = FundingAlphaGate(get_funding_rate=lambda s: -0.0002)
        r = gate.check(_make_ev(1), {"symbol": "ETHUSDT"})
        assert r.scale == 1.5  # long + negative funding = receives

    def test_funding_impact(self):
        gate = FundingAlphaGate(leverage=100.0)
        gate.check(_make_ev(1), {"funding_rate": 0.0001})
        assert abs(gate.funding_impact_per_8h - 1.0) < 0.01  # 0.01% × 100 = 1%
