"""Tests for runner.gate_chain — modular order gate pipeline."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from runner.gate_chain import (
    AlphaHealthGate,
    CorrelationCheckGate,
    ExecQualityGate,
    GateChain,
    GateResult,
    PortfolioAllocatorGate,
    PortfolioRiskGate,
    RegimeSizerGate,
    RiskSizeGate,
    WeightRecGate,
    _apply_scale,
    build_gate_chain,
)


def _order_ev(symbol="BTCUSDT", qty=1.0, price=50000.0, side="buy"):
    return SimpleNamespace(
        event_type=SimpleNamespace(value="ORDER"),
        symbol=symbol,
        qty=qty,
        price=price,
        side=side,
        order_id="ord-1",
    )


# ── Individual gate tests ──


class TestCorrelationCheckGate:
    def test_allows_uncorrelated(self):
        gate_mock = MagicMock()
        gate_mock.should_allow.return_value = SimpleNamespace(ok=True, violations=[])
        view_fn = lambda: {"positions": {}}
        gate = CorrelationCheckGate(gate_mock, view_fn)
        result = gate.check(_order_ev(), {})
        assert result.allowed is True

    def test_rejects_correlated(self):
        violation = SimpleNamespace(message="too correlated")
        gate_mock = MagicMock()
        gate_mock.should_allow.return_value = SimpleNamespace(ok=False, violations=[violation])
        view_fn = lambda: {"positions": {}}
        gate = CorrelationCheckGate(gate_mock, view_fn)
        result = gate.check(_order_ev(), {})
        assert result.allowed is False
        assert "correlated" in result.reason


class TestRiskSizeGate:
    def test_allows(self):
        risk = MagicMock()
        risk.check.return_value = SimpleNamespace(allowed=True, reason="")
        gate = RiskSizeGate(risk)
        result = gate.check(_order_ev(), {})
        assert result.allowed is True

    def test_rejects(self):
        risk = MagicMock()
        risk.check.return_value = SimpleNamespace(allowed=False, reason="too large")
        gate = RiskSizeGate(risk)
        result = gate.check(_order_ev(), {})
        assert result.allowed is False


class TestPortfolioRiskGate:
    def test_allows(self):
        agg = MagicMock()
        agg.evaluate_order.return_value = SimpleNamespace(ok=True, violations=[])
        gate = PortfolioRiskGate(agg)
        assert gate.check(_order_ev(), {}).allowed is True

    def test_rejects(self):
        v = SimpleNamespace(message="gross leverage exceeded")
        agg = MagicMock()
        agg.evaluate_order.return_value = SimpleNamespace(ok=False, violations=[v])
        gate = PortfolioRiskGate(agg)
        result = gate.check(_order_ev(), {})
        assert result.allowed is False

    def test_exception_passes(self):
        agg = MagicMock()
        agg.evaluate_order.side_effect = RuntimeError("boom")
        gate = PortfolioRiskGate(agg)
        assert gate.check(_order_ev(), {}).allowed is True


class TestAlphaHealthGate:
    def test_full_scale(self):
        ahm = MagicMock()
        ahm.position_scale.return_value = 1.0
        gate = AlphaHealthGate(ahm)
        result = gate.check(_order_ev(), {})
        assert result.allowed is True
        assert result.scale == 1.0

    def test_half_scale(self):
        ahm = MagicMock()
        ahm.position_scale.return_value = 0.5
        gate = AlphaHealthGate(ahm)
        result = gate.check(_order_ev(), {})
        assert result.allowed is True
        assert result.scale == 0.5

    def test_zero_rejects(self):
        ahm = MagicMock()
        ahm.position_scale.return_value = 0.0
        gate = AlphaHealthGate(ahm)
        result = gate.check(_order_ev(), {})
        assert result.allowed is False


class TestRegimeSizerGate:
    def test_scales(self):
        sizer = MagicMock()
        sizer.position_scale.return_value = 0.8
        gate = RegimeSizerGate(sizer)
        result = gate.check(_order_ev(), {})
        assert result.allowed is True
        assert result.scale == 0.8


class TestExecQualityGate:
    def test_no_hook(self):
        gate = ExecQualityGate(None)
        assert gate.check(_order_ev(), {}).allowed is True

    def test_rejects_zero(self):
        hook = SimpleNamespace(
            execution_quality=SimpleNamespace(should_reduce_size=lambda sym: 0.0),
        )
        gate = ExecQualityGate(hook)
        result = gate.check(_order_ev(), {})
        assert result.allowed is False

    def test_scales(self):
        hook = SimpleNamespace(
            execution_quality=SimpleNamespace(should_reduce_size=lambda sym: 0.7),
        )
        gate = ExecQualityGate(hook)
        result = gate.check(_order_ev(), {})
        assert result.allowed is True
        assert result.scale == 0.7


class TestWeightRecGate:
    def test_no_hook(self):
        gate = WeightRecGate(None)
        assert gate.check(_order_ev(), {}).allowed is True

    def test_zero_rejects(self):
        hook = SimpleNamespace(weight_recommendations={"BTCUSDT": 0.0})
        gate = WeightRecGate(hook)
        result = gate.check(_order_ev(), {})
        assert result.allowed is False

    def test_scales(self):
        hook = SimpleNamespace(weight_recommendations={"BTCUSDT": 0.6})
        gate = WeightRecGate(hook)
        result = gate.check(_order_ev(), {})
        assert result.allowed is True
        assert result.scale == 0.6


# ── Chain-level tests ──


class TestGateChain:
    def test_all_pass(self):
        g1 = MagicMock(name="G1")
        g1.name = "G1"
        g1.check.return_value = GateResult(allowed=True)
        g2 = MagicMock(name="G2")
        g2.name = "G2"
        g2.check.return_value = GateResult(allowed=True)

        chain = GateChain([g1, g2])
        ev = _order_ev()
        result = chain.process(ev, {})
        assert result is not None
        g1.check.assert_called_once()
        g2.check.assert_called_once()

    def test_early_reject(self):
        g1 = MagicMock(name="G1")
        g1.name = "G1"
        g1.check.return_value = GateResult(allowed=False, reason="blocked")
        g2 = MagicMock(name="G2")
        g2.name = "G2"
        g2.check.return_value = GateResult(allowed=True)

        chain = GateChain([g1, g2])
        result = chain.process(_order_ev(), {})
        assert result is None
        g2.check.assert_not_called()

    def test_cumulative_scaling(self):
        g1 = MagicMock(name="G1")
        g1.name = "G1"
        g1.check.return_value = GateResult(allowed=True, scale=0.5)
        g2 = MagicMock(name="G2")
        g2.name = "G2"
        g2.check.return_value = GateResult(allowed=True, scale=0.8)

        chain = GateChain([g1, g2])
        ev = _order_ev(qty=10.0)
        result = chain.process(ev, {})
        assert result is not None
        # 10.0 * 0.5 = 5.0, then 5.0 * 0.8 = 4.0
        assert abs(float(result.qty) - 4.0) < 1e-9


class TestApplyScaleDecimal:
    def test_apply_scale_preserves_decimal_type(self):
        from decimal import Decimal
        from runner.gate_chain import _apply_scale

        ev = SimpleNamespace(qty=Decimal("0.01"), symbol="BTCUSDT")
        _apply_scale(ev, 0.5, "TestGate")
        assert isinstance(ev.qty, Decimal), f"Expected Decimal, got {type(ev.qty)}"
        assert ev.qty == Decimal("0.005")

    def test_apply_scale_float_input_stays_float(self):
        from runner.gate_chain import _apply_scale

        ev = SimpleNamespace(qty=0.01, symbol="BTCUSDT")
        _apply_scale(ev, 0.5, "TestGate")
        assert isinstance(ev.qty, float)


class TestBuildGateChain:
    def test_minimal(self):
        chain = build_gate_chain(
            correlation_gate=MagicMock(),
            risk_gate=MagicMock(),
            get_state_view=lambda: {},
        )
        # At minimum: correlation + risk = 2 gates
        assert len(chain.gates) == 2

    def test_all_optional(self):
        chain = build_gate_chain(
            correlation_gate=MagicMock(),
            risk_gate=MagicMock(),
            get_state_view=lambda: {},
            portfolio_aggregator=MagicMock(),
            alpha_health_monitor=MagicMock(),
            regime_sizer=MagicMock(),
            portfolio_allocator=MagicMock(),
            hook=MagicMock(),
        )
        # 2 required + 4 optional + 2 from hook = 8 gates
        assert len(chain.gates) == 8
