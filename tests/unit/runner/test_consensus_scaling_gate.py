# tests/unit/runner/test_consensus_scaling_gate.py
"""Unit tests for ConsensusScalingGate."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from strategy.gates.consensus_scaling_gate import (
    ConsensusScalingGate,
    _consensus_scale,
)
from runner.gate_chain import GateResult


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------

class TestConsensusScale:
    def test_no_others_returns_1_0(self):
        """Only this symbol in the dict → no opinion."""
        assert _consensus_scale("ETH", 1, {"ETH": 1}) == 1.0

    def test_empty_consensus_returns_1_0(self):
        assert _consensus_scale("ETH", 1, {}) == 1.0

    def test_all_flat_returns_1_0(self):
        """Others have signal=0 → treated as inactive → no others."""
        consensus = {"ETH": 1, "BTC": 0, "SUI": 0}
        assert _consensus_scale("ETH", 1, consensus) == 1.0

    def test_all_agree_returns_1_0(self):
        """All others agree → scale=1.0 (no boost for crowd-following)."""
        consensus = {"ETH": 1, "BTC": 1, "SUI": 1, "AXS": 1}
        assert _consensus_scale("ETH", 1, consensus) == 1.0

    def test_all_disagree_returns_1_3_contrarian(self):
        """All others disagree → contrarian boost 1.3."""
        consensus = {"ETH": 1, "BTC": -1, "SUI": -1}
        assert _consensus_scale("ETH", 1, consensus) == 1.3

    def test_all_disagree_single_other(self):
        """Even a single opposing signal → contrarian 1.3."""
        consensus = {"ETH": 1, "BTC": -1}
        assert _consensus_scale("ETH", 1, consensus) == 1.3

    def test_contrarian_long_vs_bears(self):
        """Long against a bearish crowd → contrarian boost."""
        consensus = {"ETH": 1, "BTC": -1, "SUI": -1, "AXS": -1}
        assert _consensus_scale("ETH", 1, consensus) == 1.3

    def test_contrarian_short_vs_bulls(self):
        """Short against a bullish crowd → contrarian boost."""
        consensus = {"ETH": -1, "BTC": 1, "SUI": 1, "AXS": 1}
        assert _consensus_scale("ETH", -1, consensus) == 1.3

    def test_mixed_75pct_agree_returns_1_0(self):
        """3/4 others agree → ratio=0.75 → scale=1.0."""
        consensus = {"ETH": 1, "BTC": 1, "SUI": 1, "AXS": 1, "SOL": -1}
        # others: BTC=1, SUI=1, AXS=1, SOL=-1 → agree=3, total=4 → ratio=0.75
        assert _consensus_scale("ETH", 1, consensus) == 1.0

    def test_mixed_50pct_agree_returns_0_7(self):
        """2/4 agree → ratio=0.5 → scale=0.7."""
        consensus = {"ETH": 1, "BTC": 1, "SUI": 1, "AXS": -1, "SOL": -1}
        # others: BTC=1, SUI=1, AXS=-1, SOL=-1 → agree=2, total=4 → ratio=0.5
        assert _consensus_scale("ETH", 1, consensus) == 0.7

    def test_mixed_25pct_agree_returns_0_7(self):
        """1/4 agree → ratio=0.25 → scale=0.7."""
        consensus = {"ETH": 1, "BTC": 1, "SUI": -1, "AXS": -1, "SOL": -1}
        # others: BTC=1, SUI=-1, AXS=-1, SOL=-1 → agree=1, total=4 → ratio=0.25
        assert _consensus_scale("ETH", 1, consensus) == 0.7

    def test_below_25pct_returns_0_5(self):
        """1/5 agree → ratio=0.2 → scale=0.5."""
        consensus = {"ETH": 1, "BTC": 1, "SUI": -1, "AXS": -1, "SOL": -1, "DOT": -1}
        # others: BTC=1, SUI=-1, AXS=-1, SOL=-1, DOT=-1 → agree=1, total=5 → 0.2
        assert _consensus_scale("ETH", 1, consensus) == 0.5

    def test_flat_signal_not_counted(self):
        """signal=0 for self should not be called (gate handles it), but
        the pure function still works — flat others are excluded."""
        consensus = {"ETH": 1, "BTC": 0, "SUI": -1}
        # others: BTC(flat, excluded), SUI=-1 → agree=0, total=1 → contrarian 1.3
        assert _consensus_scale("ETH", 1, consensus) == 1.3


# ---------------------------------------------------------------------------
# Gate integration tests
# ---------------------------------------------------------------------------

def _make_event(signal: int = 1, symbol: str = "ETHUSDT") -> MagicMock:
    ev = MagicMock()
    ev.symbol = symbol
    ev.signal = signal
    return ev


class TestConsensusScalingGate:
    def test_always_allowed(self):
        gate = ConsensusScalingGate(consensus={"ETHUSDT": 1, "BTCUSDT": 1})
        ev = _make_event(signal=1)
        result = gate.check(ev, {})
        assert result.allowed is True

    def test_flat_signal_returns_1_0_no_op(self):
        gate = ConsensusScalingGate(consensus={"ETHUSDT": 0, "BTCUSDT": 1})
        ev = _make_event(signal=0)
        result = gate.check(ev, {})
        assert result.scale == pytest.approx(1.0)

    def test_all_agree_returns_1_0(self):
        consensus = {"ETHUSDT": 1, "BTCUSDT": 1, "SUIUSDT": 1}
        gate = ConsensusScalingGate(consensus=consensus)
        ev = _make_event(signal=1, symbol="ETHUSDT")
        result = gate.check(ev, {})
        assert result.scale == pytest.approx(1.0)

    def test_all_disagree_returns_1_3(self):
        consensus = {"ETHUSDT": 1, "BTCUSDT": -1, "SUIUSDT": -1}
        gate = ConsensusScalingGate(consensus=consensus)
        ev = _make_event(signal=1, symbol="ETHUSDT")
        result = gate.check(ev, {})
        assert result.scale == pytest.approx(1.3)

    def test_mixed_returns_0_7(self):
        consensus = {"ETHUSDT": 1, "BTCUSDT": 1, "SUIUSDT": -1}
        gate = ConsensusScalingGate(consensus=consensus)
        ev = _make_event(signal=1, symbol="ETHUSDT")
        # others: BTCUSDT=1 (agree), SUIUSDT=-1 (disagree) → agree=1/2 → 0.5 ratio → 0.7
        result = gate.check(ev, {})
        assert result.scale == pytest.approx(0.7)

    def test_no_others_returns_1_0(self):
        consensus = {"ETHUSDT": 1}
        gate = ConsensusScalingGate(consensus=consensus)
        ev = _make_event(signal=1, symbol="ETHUSDT")
        result = gate.check(ev, {})
        assert result.scale == pytest.approx(1.0)

    def test_empty_consensus_returns_1_0(self):
        gate = ConsensusScalingGate(consensus={})
        ev = _make_event(signal=1, symbol="ETHUSDT")
        result = gate.check(ev, {})
        assert result.scale == pytest.approx(1.0)

    def test_get_consensus_callable_takes_priority(self):
        """get_consensus callable is used when provided."""
        live_consensus = {"ETHUSDT": 1, "BTCUSDT": -1, "SUIUSDT": -1}
        gate = ConsensusScalingGate(
            get_consensus=lambda: live_consensus,
            consensus={"ETHUSDT": 1, "BTCUSDT": 1},  # ignored
        )
        ev = _make_event(signal=1, symbol="ETHUSDT")
        result = gate.check(ev, {})
        assert result.scale == pytest.approx(1.3)  # contrarian from live_consensus

    def test_get_consensus_exception_falls_back_gracefully(self):
        """If get_consensus() raises, gate is a no-op (scale=1.0)."""
        def bad():
            raise RuntimeError("network error")

        gate = ConsensusScalingGate(get_consensus=bad)
        ev = _make_event(signal=1, symbol="ETHUSDT")
        result = gate.check(ev, {})
        # consensus = {} after exception → no others → scale=1.0
        assert result.scale == pytest.approx(1.0)

    def test_consensus_from_context_fallback(self):
        """Falls back to context['consensus'] when no callable or static dict."""
        gate = ConsensusScalingGate()
        ev = _make_event(signal=1, symbol="ETHUSDT")
        ctx = {"consensus": {"ETHUSDT": 1, "BTCUSDT": -1, "SUIUSDT": -1}}
        result = gate.check(ev, ctx)
        assert result.scale == pytest.approx(1.3)

    def test_signal_from_context_fallback(self):
        """If event has no signal attr, reads from context."""
        class Ev:
            symbol = "ETHUSDT"

        gate = ConsensusScalingGate(
            consensus={"ETHUSDT": 1, "BTCUSDT": -1, "SUIUSDT": -1}
        )
        result = gate.check(Ev(), {"signal": 1})
        assert result.scale == pytest.approx(1.3)

    def test_gate_name(self):
        assert ConsensusScalingGate.name == "ConsensusScaling"

    def test_result_is_gate_result(self):
        gate = ConsensusScalingGate(consensus={"ETHUSDT": 1, "BTCUSDT": 1})
        ev = _make_event(signal=1, symbol="ETHUSDT")
        result = gate.check(ev, {})
        assert isinstance(result, GateResult)
