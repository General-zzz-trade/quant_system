"""Tests for GateEvaluator."""
from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class MockGateResult:
    allowed: bool = True
    scale: float = 1.0
    reason: str = ""


class MockGate:
    """Configurable mock gate for testing."""
    def __init__(self, allowed=True, scale=1.0, reason=""):
        self._result = MockGateResult(allowed=allowed, scale=scale, reason=reason)

    def check(self, ev, ctx):
        return self._result


@pytest.fixture
def evaluator():
    from runner.gates.evaluator import GateEvaluator
    return GateEvaluator(
        liq_gate=MockGate(),
        mtf_gate=MockGate(),
        carry_gate=MockGate(),
        vpin_gate=MockGate(),
    )


class TestGateEvaluator:

    def test_zero_signal_returns_one(self, evaluator):
        assert evaluator.evaluate(0, {}, {}, "BTC_4h", "BTCUSDT") == 1.0

    def test_all_gates_pass(self, evaluator):
        scale = evaluator.evaluate(1, {}, {}, "BTC_4h", "BTCUSDT")
        assert scale == 1.0

    def test_liq_gate_blocks(self):
        from runner.gates.evaluator import GateEvaluator
        ev = GateEvaluator(
            liq_gate=MockGate(allowed=False, reason="cascade"),
            mtf_gate=MockGate(), carry_gate=MockGate(), vpin_gate=MockGate(),
        )
        assert ev.evaluate(1, {}, {}, "BTC_4h", "BTCUSDT") == 0.0

    def test_vpin_gate_blocks(self):
        from runner.gates.evaluator import GateEvaluator
        ev = GateEvaluator(
            liq_gate=MockGate(), mtf_gate=MockGate(), carry_gate=MockGate(),
            vpin_gate=MockGate(allowed=False, reason="toxic"),
        )
        assert ev.evaluate(-1, {}, {}, "ETH_1h", "ETHUSDT") == 0.0

    def test_scale_multiplication(self):
        from runner.gates.evaluator import GateEvaluator
        ev = GateEvaluator(
            liq_gate=MockGate(scale=0.8),
            mtf_gate=MockGate(scale=1.2),
            carry_gate=MockGate(scale=0.7),
            vpin_gate=MockGate(scale=1.0),
        )
        scale = ev.evaluate(1, {}, {}, "BTC_4h", "BTCUSDT")
        assert abs(scale - 0.8 * 1.2 * 0.7) < 1e-6

    def test_consensus_signal_injected(self):
        from runner.gates.evaluator import GateEvaluator
        calls = []
        class RecordingGate:
            def check(self, ev, ctx):
                calls.append(dict(ctx))
                return MockGateResult()
        ev = GateEvaluator(
            liq_gate=RecordingGate(), mtf_gate=MockGate(),
            carry_gate=MockGate(), vpin_gate=MockGate(),
        )
        consensus = {"BTCUSDT_4h": 1}
        ev.evaluate(1, {}, consensus, "BTCUSDT_1h", "BTCUSDT")
        assert calls[0].get("tf4h_model_signal") == 1

    def test_features_populated_in_context(self):
        from runner.gates.evaluator import GateEvaluator
        calls = []
        class RecordingGate:
            def check(self, ev, ctx):
                calls.append(dict(ctx))
                return MockGateResult()
        ev = GateEvaluator(
            liq_gate=RecordingGate(), mtf_gate=MockGate(),
            carry_gate=MockGate(), vpin_gate=MockGate(),
        )
        feat = {"funding_rate": 0.001, "basis": 0.05}
        ev.evaluate(1, feat, {}, "BTC_4h", "BTCUSDT")
        assert calls[0]["funding_rate"] == 0.001
        assert calls[0]["basis"] == 0.05

    def test_nan_features_excluded(self):
        from runner.gates.evaluator import GateEvaluator
        calls = []
        class RecordingGate:
            def check(self, ev, ctx):
                calls.append(dict(ctx))
                return MockGateResult()
        ev = GateEvaluator(
            liq_gate=RecordingGate(), mtf_gate=MockGate(),
            carry_gate=MockGate(), vpin_gate=MockGate(),
        )
        feat = {"funding_rate": float("nan"), "basis": 0.05}
        ev.evaluate(1, feat, {}, "BTC_4h", "BTCUSDT")
        assert "funding_rate" not in calls[0]
        assert calls[0]["basis"] == 0.05

    def test_last_scale_tracked(self, evaluator):
        evaluator.evaluate(1, {}, {}, "BTC_4h", "BTCUSDT")
        assert evaluator.last_scale == 1.0
