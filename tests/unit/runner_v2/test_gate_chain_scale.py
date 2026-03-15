# tests/unit/runner_v2/test_gate_chain_scale.py
"""Tests for gate chain scaling — verify no object.__setattr__ hack on frozen events."""
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace


from runner.gate_chain import _apply_scale, GateChain, GateResult


@dataclass(frozen=True)
class FrozenOrder:
    symbol: str = "BTCUSDT"
    qty: Decimal = Decimal("1.0")


@dataclass
class MutableOrder:
    symbol: str = "BTCUSDT"
    qty: Decimal = Decimal("1.0")


class TestApplyScaleFrozen:
    def test_frozen_returns_new_instance(self):
        ev = FrozenOrder(qty=Decimal("1.0"))
        result = _apply_scale(ev, 0.5, "test")
        # Must return NEW instance (not mutate frozen)
        assert result is not ev
        assert result.qty == Decimal("0.5")
        # Original unchanged
        assert ev.qty == Decimal("1.0")

    def test_frozen_preserves_other_fields(self):
        ev = FrozenOrder(symbol="ETHUSDT", qty=Decimal("2.0"))
        result = _apply_scale(ev, 0.25, "test")
        assert result.symbol == "ETHUSDT"
        assert result.qty == Decimal("0.50")


class TestApplyScaleMutable:
    def test_mutable_mutates_in_place(self):
        ev = MutableOrder(qty=Decimal("1.0"))
        result = _apply_scale(ev, 0.5, "test")
        assert result is ev  # same object
        assert ev.qty == Decimal("0.5")


class TestApplyScaleSimpleNamespace:
    def test_namespace_mutates_in_place(self):
        ev = SimpleNamespace(symbol="BTCUSDT", qty=Decimal("1.0"))
        result = _apply_scale(ev, 0.5, "test")
        assert result is ev
        assert ev.qty == Decimal("0.5")


class TestGateChainWithFrozen:
    def test_chain_returns_scaled_frozen_event(self):
        class HalfScaleGate:
            name = "half"
            def check(self, ev, context):
                return GateResult(allowed=True, scale=0.5)

        chain = GateChain([HalfScaleGate()])
        ev = FrozenOrder(qty=Decimal("1.0"))
        result = chain.process(ev, {})
        assert result is not None
        assert result.qty == Decimal("0.5")
        # Original unchanged
        assert ev.qty == Decimal("1.0")
