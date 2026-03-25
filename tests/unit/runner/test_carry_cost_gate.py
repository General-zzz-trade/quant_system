"""Tests for CarryCostGate."""

from unittest.mock import MagicMock
from strategy.gates.carry_cost_gate import CarryCostGate, CarryCostConfig


def _make_ev(signal=0):
    ev = MagicMock()
    ev.metadata = {"signal": signal}
    return ev


class TestCarryCostGate:
    def test_no_signal_passthrough(self):
        gate = CarryCostGate()
        r = gate.check(_make_ev(0), {"funding_rate": 0.001})
        assert r.allowed
        assert r.scale == 1.0

    def test_long_receives_negative_funding_boost(self):
        """Long + negative funding = receive carry → boost."""
        gate = CarryCostGate()
        # funding_rate = -0.001 per 8h → annualized = -0.001 * 3 * 365 * 100 = -109.5%
        r = gate.check(_make_ev(1), {
            "funding_rate": -0.001,
            "basis": 0.0,
        })
        assert r.allowed
        assert r.scale == 1.15  # favorable

    def test_long_pays_extreme_funding_reduce(self):
        """Long + very high positive funding = extreme cost → reduce."""
        gate = CarryCostGate()
        # funding_rate = 0.001 per 8h → annualized = 109.5%
        r = gate.check(_make_ev(1), {
            "funding_rate": 0.001,
            "basis": 0.0,
        })
        assert r.allowed
        assert r.scale == 0.4  # extreme

    def test_long_pays_moderate_funding_reduce(self):
        """Long + moderate positive funding → costly reduce."""
        gate = CarryCostGate()
        # funding_rate = 0.0001 → annualized = 10.95%
        r = gate.check(_make_ev(1), {
            "funding_rate": 0.0001,
            "basis": 0.0,
        })
        assert r.allowed
        assert r.scale == 0.7  # costly

    def test_short_receives_positive_funding(self):
        """Short + positive funding = shorts receive → favorable."""
        gate = CarryCostGate()
        r = gate.check(_make_ev(-1), {
            "funding_rate": 0.001,
            "basis": 0.0,
        })
        assert r.allowed
        assert r.scale == 1.15

    def test_small_carry_no_change(self):
        """Small carry cost → no adjustment."""
        gate = CarryCostGate()
        # funding_rate = 0.00001 → annualized = 1.095%
        r = gate.check(_make_ev(1), {
            "funding_rate": 0.00001,
            "basis": 0.0,
        })
        assert r.allowed
        assert r.scale == 1.0

    def test_disabled(self):
        gate = CarryCostGate(CarryCostConfig(enabled=False))
        r = gate.check(_make_ev(1), {"funding_rate": 0.01})
        assert r.allowed
        assert r.scale == 1.0

    def test_missing_data_defaults(self):
        gate = CarryCostGate()
        r = gate.check(_make_ev(1), {})
        assert r.allowed
        assert r.scale == 1.0  # zero funding + zero basis = zero carry

    def test_basis_contribution(self):
        """Basis adds to carry cost for longs."""
        gate = CarryCostGate()
        # basis=0.1 → 10% premium, annualized: 0.1 * 365 = 36.5%
        r = gate.check(_make_ev(1), {
            "funding_rate": 0.0,
            "basis": 0.1,  # 10% premium → 36.5% annual carry cost
        })
        assert r.allowed
        assert r.scale == 0.4  # extreme carry cost

    def test_stats(self):
        gate = CarryCostGate()
        gate.check(_make_ev(1), {"funding_rate": -0.001, "basis": 0.0})
        gate.check(_make_ev(1), {"funding_rate": 0.001, "basis": 0.0})
        stats = gate.stats
        assert stats["total_checks"] == 2
        assert stats["boosted"] == 1
        assert stats["reduced"] == 1
