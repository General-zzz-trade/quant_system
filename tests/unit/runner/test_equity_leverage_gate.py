# tests/unit/runner/test_equity_leverage_gate.py
"""Unit tests for EquityLeverageGate."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from runner.gates.equity_leverage_gate import (
    EquityLeverageGate,
    _bracket_leverage,
    _z_scale,
)
from runner.gate_chain import GateResult


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------

class TestBracketLeverage:
    def test_500_equity_returns_1_5(self):
        assert _bracket_leverage(500.0) == 1.5

    def test_5000_equity_returns_1_5(self):
        assert _bracket_leverage(5_000.0) == 1.5

    def test_10000_equity_returns_1_5(self):
        assert _bracket_leverage(10_000.0) == 1.5

    def test_19999_equity_returns_1_5(self):
        assert _bracket_leverage(19_999.0) == 1.5

    def test_20000_equity_returns_1_0(self):
        assert _bracket_leverage(20_000.0) == 1.0

    def test_25000_equity_returns_1_0(self):
        assert _bracket_leverage(25_000.0) == 1.0

    def test_50000_equity_returns_1_0(self):
        assert _bracket_leverage(50_000.0) == 1.0

    def test_100000_equity_returns_1_0(self):
        assert _bracket_leverage(100_000.0) == 1.0

    def test_zero_equity_returns_1_5(self):
        # 0 is in first bracket [0, 5000)
        assert _bracket_leverage(0.0) == 1.5

    def test_custom_brackets(self):
        custom = [(0, 1000, 2.0), (1000, float("inf"), 0.5)]
        assert _bracket_leverage(500.0, custom) == 2.0
        assert _bracket_leverage(1500.0, custom) == 0.5


class TestZScale:
    def test_z_above_2_returns_1_5(self):
        assert _z_scale(2.1) == 1.5
        assert _z_scale(3.0) == 1.5
        assert _z_scale(-2.5) == 1.5

    def test_z_exactly_2_returns_1_0(self):
        # boundary: >2.0 → 1.5; <=2.0 checks next condition >1.0 → 1.0
        assert _z_scale(2.0) == 1.0

    def test_z_above_1_returns_1_0(self):
        assert _z_scale(1.5) == 1.0
        assert _z_scale(-1.2) == 1.0

    def test_z_above_0_5_returns_0_7(self):
        assert _z_scale(0.8) == 0.7
        assert _z_scale(-0.6) == 0.7

    def test_z_below_0_5_returns_0_5(self):
        assert _z_scale(0.3) == 0.5
        assert _z_scale(0.0) == 0.5
        assert _z_scale(-0.4) == 0.5

    def test_z_exactly_0_5_returns_0_5(self):
        # boundary: >0.5 needed for 0.7; exactly 0.5 → 0.5
        assert _z_scale(0.5) == 0.5


# ---------------------------------------------------------------------------
# Gate integration tests
# ---------------------------------------------------------------------------

def _make_event(z: float = 0.0, symbol: str = "ETHUSDT") -> MagicMock:
    ev = MagicMock()
    ev.symbol = symbol
    ev.metadata = {"z_score": z}
    return ev


class TestEquityLeverageGate:
    def test_always_allowed(self):
        gate = EquityLeverageGate(get_equity=lambda: 500.0)
        ev = _make_event(z=1.0)
        result = gate.check(ev, {})
        assert result.allowed is True

    def test_500_equity_z_2_5_gives_2_25(self):
        """$500 (bracket 1.5x) × z=2.5 (scale 1.5) = 2.25."""
        gate = EquityLeverageGate(get_equity=lambda: 500.0)
        ev = _make_event(z=2.5)
        result = gate.check(ev, {})
        assert result.scale == pytest.approx(2.25)

    def test_500_equity_z_0_3_gives_0_75(self):
        """$500 (bracket 1.5x) × z=0.3 (scale 0.5) = 0.75."""
        gate = EquityLeverageGate(get_equity=lambda: 500.0)
        ev = _make_event(z=0.3)
        result = gate.check(ev, {})
        assert result.scale == pytest.approx(0.75)

    def test_25k_equity_z_1_5_gives_1_0(self):
        """$25K (bracket 1.0x) × z=1.5 (scale 1.0) = 1.0."""
        gate = EquityLeverageGate(get_equity=lambda: 25_000.0)
        ev = _make_event(z=1.5)
        result = gate.check(ev, {})
        assert result.scale == pytest.approx(1.0)

    def test_25k_equity_z_2_5_gives_1_5(self):
        """$25K (bracket 1.0x) × z=2.5 (scale 1.5) = 1.5."""
        gate = EquityLeverageGate(get_equity=lambda: 25_000.0)
        ev = _make_event(z=2.5)
        result = gate.check(ev, {})
        assert result.scale == pytest.approx(1.5)

    def test_25k_equity_z_0_3_gives_0_5(self):
        """$25K (bracket 1.0x) × z=0.3 (scale 0.5) = 0.5."""
        gate = EquityLeverageGate(get_equity=lambda: 25_000.0)
        ev = _make_event(z=0.3)
        result = gate.check(ev, {})
        assert result.scale == pytest.approx(0.5)

    def test_z_2_0_boundary(self):
        """z=2.0 hits z_scale=1.0, not 1.5."""
        gate = EquityLeverageGate(get_equity=lambda: 500.0)
        ev = _make_event(z=2.0)
        result = gate.check(ev, {})
        # 1.5 (bracket) × 1.0 (z_scale) = 1.5
        assert result.scale == pytest.approx(1.5)

    def test_equity_from_context_fallback(self):
        """When get_equity=None, reads equity from context."""
        gate = EquityLeverageGate()  # no get_equity
        ev = _make_event(z=2.5)
        result = gate.check(ev, {"equity": 500.0})
        assert result.scale == pytest.approx(2.25)

    def test_z_from_context_fallback(self):
        """When event metadata has no z_score, reads from context."""
        gate = EquityLeverageGate(get_equity=lambda: 500.0)
        ev = MagicMock()
        ev.symbol = "ETHUSDT"
        ev.metadata = {}
        result = gate.check(ev, {"z_score": 2.5})
        assert result.scale == pytest.approx(2.25)

    def test_z_from_context_key_z(self):
        """Also handles 'z' key in context."""
        gate = EquityLeverageGate(get_equity=lambda: 500.0)
        ev = MagicMock()
        ev.symbol = "ETHUSDT"
        ev.metadata = {}
        result = gate.check(ev, {"z": 2.5})
        assert result.scale == pytest.approx(2.25)

    def test_get_equity_exception_falls_back_to_context(self):
        """If get_equity() raises, fall back to context equity."""
        def bad_equity():
            raise RuntimeError("DB down")

        gate = EquityLeverageGate(get_equity=bad_equity)
        ev = _make_event(z=2.5)
        result = gate.check(ev, {"equity": 500.0})
        assert result.scale == pytest.approx(2.25)

    def test_no_equity_zero_uses_default_bracket(self):
        """equity=0 falls into first bracket → 1.5x."""
        gate = EquityLeverageGate()
        ev = _make_event(z=1.5)
        result = gate.check(ev, {})
        # 1.5 (bracket) × 1.0 (z=1.5 → scale 1.0) = 1.5
        assert result.scale == pytest.approx(1.5)

    def test_gate_name(self):
        assert EquityLeverageGate.name == "EquityLeverage"

    def test_result_is_gate_result(self):
        gate = EquityLeverageGate(get_equity=lambda: 1000.0)
        ev = _make_event(z=1.0)
        result = gate.check(ev, {})
        assert isinstance(result, GateResult)
