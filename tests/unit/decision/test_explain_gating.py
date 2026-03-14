"""Tests for decision/explain.py, decision/gating.py, and governance/feature_flags.py."""
from __future__ import annotations

import pytest
from types import SimpleNamespace

from decision.explain import DecisionExplanation, explain_decision
from decision.gating import FeatureFlagGate, SymbolBlacklistGate
from decision.governance.feature_flags import FeatureFlags


# ── DecisionExplanation ──────────────────────────────────────────────

class TestDecisionExplanation:
    def test_create_basic(self):
        exp = DecisionExplanation(
            symbol="BTC", action="buy", reason_codes=("momentum", "mean_revert")
        )
        assert exp.symbol == "BTC"
        assert exp.action == "buy"
        assert exp.reason_codes == ("momentum", "mean_revert")
        assert exp.risk_check_passed is True

    def test_summary_with_reasons(self):
        exp = DecisionExplanation(
            symbol="ETH", action="sell",
            reason_codes=("overbought",),
            risk_check_passed=True,
        )
        s = exp.summary()
        assert "ETH" in s
        assert "sell" in s
        assert "overbought" in s
        assert "passed" in s

    def test_summary_risk_blocked(self):
        exp = DecisionExplanation(
            symbol="SOL", action="buy",
            reason_codes=("breakout",),
            risk_check_passed=False,
        )
        s = exp.summary()
        assert "BLOCKED" in s

    def test_summary_empty_reasons(self):
        exp = DecisionExplanation(
            symbol="BTC", action="hold", reason_codes=()
        )
        s = exp.summary()
        assert "none" in s

    def test_frozen(self):
        exp = DecisionExplanation(
            symbol="BTC", action="buy", reason_codes=()
        )
        with pytest.raises(AttributeError):
            exp.symbol = "ETH"  # type: ignore[misc]


# ── explain_decision ─────────────────────────────────────────────────

class TestExplainDecision:
    def test_with_signal_and_risk(self):
        exp = explain_decision(
            symbol="BTC",
            action="buy",
            signal_results={"score": 0.9},
            risk_results={"allowed": True, "reason": "ok"},
            reason_codes=["momentum"],
        )
        assert exp.symbol == "BTC"
        assert exp.action == "buy"
        assert exp.risk_check_passed is True
        assert exp.signal_details == {"score": 0.9}
        assert exp.risk_details["reason"] == "ok"

    def test_risk_blocked(self):
        exp = explain_decision(
            symbol="ETH",
            action="sell",
            risk_results={"allowed": False},
        )
        assert exp.risk_check_passed is False

    def test_none_inputs(self):
        exp = explain_decision(symbol="X", action="flat")
        assert exp.risk_check_passed is True
        assert exp.signal_details == {}
        assert exp.risk_details == {}
        assert exp.reason_codes == ()

    def test_empty_reason_codes(self):
        exp = explain_decision(symbol="X", action="hold", reason_codes=[])
        assert exp.reason_codes == ()


# ── FeatureFlagGate ──────────────────────────────────────────────────

class TestFeatureFlagGate:
    def test_all_flags_present(self):
        gate = FeatureFlagGate(required=("alpha", "beta"))
        snap = SimpleNamespace(feature_flags={"alpha": True, "beta": True})
        result = gate.check(snap)
        assert result.allowed is True
        assert result.reasons == ()

    def test_missing_flag(self):
        gate = FeatureFlagGate(required=("alpha", "beta"))
        snap = SimpleNamespace(feature_flags={"alpha": True})
        result = gate.check(snap)
        assert result.allowed is False
        assert any("missing:beta" in r for r in result.reasons)

    def test_disabled_flag(self):
        gate = FeatureFlagGate(required=("alpha",))
        snap = SimpleNamespace(feature_flags={"alpha": False})
        result = gate.check(snap)
        assert result.allowed is False

    def test_no_required_flags(self):
        gate = FeatureFlagGate(required=())
        snap = SimpleNamespace()
        result = gate.check(snap)
        assert result.allowed is True

    def test_no_feature_flags_attr(self):
        gate = FeatureFlagGate(required=("x",))
        snap = SimpleNamespace()
        result = gate.check(snap)
        assert result.allowed is False
        assert "missing_feature_flags" in result.reasons

    def test_feature_flags_not_mapping(self):
        gate = FeatureFlagGate(required=("x",))
        snap = SimpleNamespace(feature_flags="not_a_dict")
        result = gate.check(snap)
        assert result.allowed is False


# ── SymbolBlacklistGate ──────────────────────────────────────────────

class TestSymbolBlacklistGate:
    def test_blacklisted(self):
        gate = SymbolBlacklistGate(blacklist=("DOGE", "SHIB"))
        snap = SimpleNamespace(symbol="DOGE")
        result = gate.check(snap)
        assert result.allowed is False
        assert "blacklisted:DOGE" in result.reasons

    def test_not_blacklisted(self):
        gate = SymbolBlacklistGate(blacklist=("DOGE",))
        snap = SimpleNamespace(symbol="BTC")
        result = gate.check(snap)
        assert result.allowed is True

    def test_empty_blacklist(self):
        gate = SymbolBlacklistGate(blacklist=())
        snap = SimpleNamespace(symbol="BTC")
        result = gate.check(snap)
        assert result.allowed is True

    def test_no_symbol_attr(self):
        gate = SymbolBlacklistGate(blacklist=("X",))
        snap = SimpleNamespace()
        result = gate.check(snap)
        assert result.allowed is True


# ── FeatureFlags ─────────────────────────────────────────────────────

class TestFeatureFlags:
    def test_enabled_key(self):
        ff = FeatureFlags(flags={"live_trading": True, "paper": False})
        assert ff.enabled("live_trading") is True

    def test_disabled_key(self):
        ff = FeatureFlags(flags={"live_trading": False})
        assert ff.enabled("live_trading") is False

    def test_missing_key(self):
        ff = FeatureFlags(flags={})
        assert ff.enabled("unknown") is False

    def test_empty_flags(self):
        ff = FeatureFlags(flags={})
        assert ff.enabled("anything") is False

    def test_frozen(self):
        ff = FeatureFlags(flags={"a": True})
        with pytest.raises(AttributeError):
            ff.flags = {}  # type: ignore[misc]
