"""Parity tests: RustGateChain."""
import pytest

try:
    from _quant_hotpath import RustGateChain
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust not available")


class TestGateChainParity:
    def _base_context(self):
        return {
            "symbol": "ETHUSDT", "side": "buy", "signal": 1,
            "qty": 1.0, "price": 3000.0,
            "equity": 10000.0, "peak_equity": 10000.0,
            "drawdown_pct": 0.05, "z_score": 1.5,
            "avg_correlation": 0.3,
            "alpha_health_scale": 1.0,
            "staged_risk_scale": 1.0,
            "regime_scale": 1.0,
        }

    def test_empty_chain_allows(self):
        gc = RustGateChain()
        result = gc.process(self._base_context())
        assert result["allowed"] is True
        assert result["scale"] == 1.0

    def test_drawdown_rejects(self):
        gc = RustGateChain()
        gc.add_gate("drawdown", {"max_drawdown_pct": 0.1})
        ctx = self._base_context()
        ctx["drawdown_pct"] = 0.15
        result = gc.process(ctx)
        assert result["allowed"] is False

    def test_drawdown_passes(self):
        gc = RustGateChain()
        gc.add_gate("drawdown", {"max_drawdown_pct": 0.2})
        ctx = self._base_context()
        ctx["drawdown_pct"] = 0.05
        result = gc.process(ctx)
        assert result["allowed"] is True

    def test_equity_leverage_scaling(self):
        gc = RustGateChain()
        gc.add_gate("equity_leverage", {})
        ctx = self._base_context()
        ctx["z_score"] = 2.5  # extreme -> 1.5x z_scale
        result = gc.process(ctx)
        assert result["allowed"] is True
        assert result["scale"] > 1.0  # leverage x z_scale > 1

    def test_consensus_contrarian_boost(self):
        gc = RustGateChain()
        gc.add_gate("consensus_scaling", {})
        ctx = self._base_context()
        ctx["signal"] = 1
        ctx["consensus"] = {"BTCUSDT": -1, "SUIUSDT": -1}  # all disagree
        result = gc.process(ctx)
        assert result["allowed"] is True
        assert abs(result["scale"] - 1.3) < 0.01

    def test_correlation_rejects(self):
        gc = RustGateChain()
        gc.add_gate("correlation", {"max_avg_correlation": 0.5})
        ctx = self._base_context()
        ctx["avg_correlation"] = 0.8
        result = gc.process(ctx)
        assert result["allowed"] is False

    def test_chain_cumulative_scale(self):
        gc = RustGateChain()
        gc.add_gate("alpha_health", {})
        gc.add_gate("regime_sizer", {})
        ctx = self._base_context()
        ctx["alpha_health_scale"] = 0.8
        ctx["regime_scale"] = 0.5
        result = gc.process(ctx)
        assert result["allowed"] is True
        assert abs(result["scale"] - 0.4) < 0.01  # 0.8 x 0.5

    def test_chain_short_circuit(self):
        gc = RustGateChain()
        gc.add_gate("drawdown", {"max_drawdown_pct": 0.1})  # will reject
        gc.add_gate("alpha_health", {})  # never reached
        ctx = self._base_context()
        ctx["drawdown_pct"] = 0.15
        result = gc.process(ctx)
        assert result["allowed"] is False
        audit = result["audit"]
        assert len(audit) == 1  # short-circuited at first gate

    def test_staged_risk_blocks(self):
        gc = RustGateChain()
        gc.add_gate("staged_risk", {})
        ctx = self._base_context()
        ctx["staged_risk_scale"] = 0.0  # blocked
        result = gc.process(ctx)
        assert result["allowed"] is False

    def test_gate_names(self):
        gc = RustGateChain()
        gc.add_gate("drawdown", {"max_drawdown_pct": 0.2})
        gc.add_gate("equity_leverage", {})
        assert gc.gate_count() == 2
        names = gc.gate_names()
        assert "drawdown" in names
        assert "equity_leverage" in names
