# tests/unit/risk/test_risk_aggregator.py
"""RiskAggregator unit tests — covers rule chaining, fail-safe, stats, enable/disable."""
from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Any, Mapping

from event.types import IntentEvent, OrderEvent
from risk.aggregator import RiskAggregator, RiskAggregatorError, RiskEvalMetaBuilder, RuleStats
from risk.decisions import (
    RiskAction,
    RiskAdjustment,
    RiskCode,
    RiskDecision,
    RiskScope,
    RiskViolation,
)


# ---------------------------------------------------------------------------
# Stub rules
# ---------------------------------------------------------------------------

class AlwaysAllowRule:
    name = "always_allow"

    def evaluate_intent(self, intent: Any, *, meta: Mapping[str, Any]) -> RiskDecision:
        return RiskDecision.allow()

    def evaluate_order(self, order: Any, *, meta: Mapping[str, Any]) -> RiskDecision:
        return RiskDecision.allow()


class AlwaysRejectRule:
    name = "always_reject"

    def evaluate_intent(self, intent: Any, *, meta: Mapping[str, Any]) -> RiskDecision:
        v = RiskViolation(code=RiskCode.MAX_POSITION, message="rejected")
        return RiskDecision.reject((v,))

    def evaluate_order(self, order: Any, *, meta: Mapping[str, Any]) -> RiskDecision:
        v = RiskViolation(code=RiskCode.MAX_POSITION, message="rejected")
        return RiskDecision.reject((v,))


class AlwaysKillRule:
    name = "always_kill"

    def evaluate_intent(self, intent: Any, *, meta: Mapping[str, Any]) -> RiskDecision:
        v = RiskViolation(code=RiskCode.MAX_DRAWDOWN, message="kill")
        return RiskDecision.kill((v,))

    def evaluate_order(self, order: Any, *, meta: Mapping[str, Any]) -> RiskDecision:
        v = RiskViolation(code=RiskCode.MAX_DRAWDOWN, message="kill")
        return RiskDecision.kill((v,))


class ExplodingRule:
    name = "exploding"

    def evaluate_intent(self, intent: Any, *, meta: Mapping[str, Any]) -> RiskDecision:
        raise RuntimeError("boom")

    def evaluate_order(self, order: Any, *, meta: Mapping[str, Any]) -> RiskDecision:
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noop_meta_builder() -> RiskEvalMetaBuilder:
    return RiskEvalMetaBuilder(
        build_for_intent=lambda i: {},
        build_for_order=lambda o: {},
    )


@dataclass(frozen=True)
class _Header:
    event_type: str = "intent"
    ts: None = None
    event_id: str = "test-1"

@dataclass(frozen=True)
class FakeIntent:
    header: _Header = _Header(event_type="intent")
    symbol: str = "BTCUSDT"

@dataclass(frozen=True)
class FakeOrder:
    header: _Header = _Header(event_type="order")
    symbol: str = "BTCUSDT"


# ---------------------------------------------------------------------------
# Tests: basic aggregation
# ---------------------------------------------------------------------------

class TestBasicAggregation:
    def test_single_allow(self) -> None:
        agg = RiskAggregator(rules=[AlwaysAllowRule()], meta_builder=_noop_meta_builder())
        d = agg.evaluate_intent(FakeIntent())
        assert d.action == RiskAction.ALLOW

    def test_single_reject(self) -> None:
        agg = RiskAggregator(rules=[AlwaysRejectRule()], meta_builder=_noop_meta_builder())
        d = agg.evaluate_intent(FakeIntent())
        assert d.action == RiskAction.REJECT
        assert len(d.violations) == 1

    def test_kill_beats_allow(self) -> None:
        agg = RiskAggregator(
            rules=[AlwaysAllowRule(), AlwaysKillRule()],
            meta_builder=_noop_meta_builder(),
        )
        d = agg.evaluate_intent(FakeIntent())
        assert d.action == RiskAction.KILL

    def test_reject_beats_allow(self) -> None:
        agg = RiskAggregator(
            rules=[AlwaysAllowRule(), AlwaysRejectRule()],
            meta_builder=_noop_meta_builder(),
        )
        d = agg.evaluate_intent(FakeIntent())
        assert d.action == RiskAction.REJECT

    def test_violations_merged(self) -> None:
        agg = RiskAggregator(
            rules=[AlwaysRejectRule(), AlwaysKillRule()],
            meta_builder=_noop_meta_builder(),
        )
        d = agg.evaluate_intent(FakeIntent())
        assert d.action == RiskAction.KILL
        assert len(d.violations) == 2

    def test_empty_rules_raises(self) -> None:
        with pytest.raises(RiskAggregatorError, match="不能为空"):
            RiskAggregator(rules=[], meta_builder=_noop_meta_builder())

    def test_duplicate_names_raises(self) -> None:
        r1 = AlwaysAllowRule()
        r2 = AlwaysAllowRule()
        with pytest.raises(RiskAggregatorError, match="唯一"):
            RiskAggregator(rules=[r1, r2], meta_builder=_noop_meta_builder())


# ---------------------------------------------------------------------------
# Tests: fail-safe on rule exception
# ---------------------------------------------------------------------------

class TestFailSafe:
    def test_exception_defaults_to_reject(self) -> None:
        agg = RiskAggregator(
            rules=[ExplodingRule()],
            meta_builder=_noop_meta_builder(),
            fail_safe_action=RiskAction.REJECT,
        )
        d = agg.evaluate_intent(FakeIntent())
        assert d.action == RiskAction.REJECT
        assert "fail_safe" in d.tags

    def test_exception_can_kill(self) -> None:
        agg = RiskAggregator(
            rules=[ExplodingRule()],
            meta_builder=_noop_meta_builder(),
            fail_safe_action=RiskAction.KILL,
        )
        d = agg.evaluate_intent(FakeIntent())
        assert d.action == RiskAction.KILL

    def test_on_error_hook_called(self) -> None:
        errors = []
        def on_err(name: str, mode: str, exc: BaseException, meta: Mapping[str, Any]) -> None:
            errors.append((name, str(exc)))

        agg = RiskAggregator(
            rules=[ExplodingRule()],
            meta_builder=_noop_meta_builder(),
            on_error=on_err,
        )
        agg.evaluate_intent(FakeIntent())
        assert len(errors) == 1
        assert errors[0][0] == "exploding"
        assert "boom" in errors[0][1]


# ---------------------------------------------------------------------------
# Tests: stats tracking
# ---------------------------------------------------------------------------

class TestStats:
    def test_call_count(self) -> None:
        agg = RiskAggregator(rules=[AlwaysAllowRule()], meta_builder=_noop_meta_builder())
        agg.evaluate_intent(FakeIntent())
        agg.evaluate_intent(FakeIntent())
        snap = agg.snapshot()
        stats = snap.stats[0]
        assert stats.calls == 2
        assert stats.allow == 2
        assert stats.reject == 0

    def test_reject_counted(self) -> None:
        agg = RiskAggregator(rules=[AlwaysRejectRule()], meta_builder=_noop_meta_builder())
        agg.evaluate_intent(FakeIntent())
        snap = agg.snapshot()
        assert snap.stats[0].reject == 1

    def test_error_counted(self) -> None:
        agg = RiskAggregator(rules=[ExplodingRule()], meta_builder=_noop_meta_builder())
        agg.evaluate_intent(FakeIntent())
        snap = agg.snapshot()
        assert snap.stats[0].errors == 1


# ---------------------------------------------------------------------------
# Tests: enable / disable
# ---------------------------------------------------------------------------

class TestEnableDisable:
    def test_disabled_rule_skipped(self) -> None:
        agg = RiskAggregator(
            rules=[AlwaysAllowRule(), AlwaysRejectRule()],
            meta_builder=_noop_meta_builder(),
            disabled=["always_reject"],
        )
        d = agg.evaluate_intent(FakeIntent())
        assert d.action == RiskAction.ALLOW

    def test_enable_after_disable(self) -> None:
        agg = RiskAggregator(
            rules=[AlwaysRejectRule()],
            meta_builder=_noop_meta_builder(),
            disabled=["always_reject"],
        )
        d1 = agg.evaluate_intent(FakeIntent())
        assert d1.action == RiskAction.ALLOW  # disabled

        agg.enable("always_reject")
        d2 = agg.evaluate_intent(FakeIntent())
        assert d2.action == RiskAction.REJECT

    def test_disable_unknown_raises(self) -> None:
        agg = RiskAggregator(rules=[AlwaysAllowRule()], meta_builder=_noop_meta_builder())
        with pytest.raises(RiskAggregatorError, match="未知规则"):
            agg.disable("nonexistent")


# ---------------------------------------------------------------------------
# Tests: short-circuit
# ---------------------------------------------------------------------------

class TestShortCircuit:
    def test_stop_on_kill(self) -> None:
        """When stop_on_kill=True, remaining rules after KILL are not evaluated."""
        agg = RiskAggregator(
            rules=[AlwaysKillRule(), AlwaysAllowRule()],
            meta_builder=_noop_meta_builder(),
            stop_on_kill=True,
        )
        agg.evaluate_intent(FakeIntent())
        snap = agg.snapshot()
        kill_stats = [s for s in snap.stats if s.name == "always_kill"][0]
        allow_stats = [s for s in snap.stats if s.name == "always_allow"][0]
        assert kill_stats.calls == 1
        assert allow_stats.calls == 0  # skipped due to short-circuit

    def test_evaluate_order(self) -> None:
        agg = RiskAggregator(rules=[AlwaysAllowRule()], meta_builder=_noop_meta_builder())
        d = agg.evaluate_order(FakeOrder())
        assert d.action == RiskAction.ALLOW
