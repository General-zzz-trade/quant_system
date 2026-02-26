"""Property-based tests for risk decision merging."""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from risk.decisions import (
    RiskAction,
    RiskAdjustment,
    RiskCode,
    RiskDecision,
    RiskScope,
    RiskViolation,
    merge_decisions,
)


_PRIORITY = {
    RiskAction.ALLOW: 0,
    RiskAction.REDUCE: 1,
    RiskAction.REJECT: 2,
    RiskAction.KILL: 3,
}

risk_actions = st.sampled_from(list(RiskAction))


def _make_decision(action: RiskAction) -> RiskDecision:
    """Build a valid RiskDecision for the given action."""
    if action == RiskAction.ALLOW:
        return RiskDecision.allow()
    v = RiskViolation(code=RiskCode.UNKNOWN, message="test")
    if action == RiskAction.REDUCE:
        return RiskDecision.reduce((v,), adjustment=RiskAdjustment(max_qty=1.0))
    if action == RiskAction.REJECT:
        return RiskDecision.reject((v,))
    return RiskDecision.kill((v,))


@given(actions=st.lists(risk_actions, min_size=1, max_size=10))
@settings(max_examples=200)
def test_merge_decisions_highest_priority_wins(actions):
    """merge_decisions always returns the highest-priority action."""
    decisions = tuple(_make_decision(a) for a in actions)
    result = merge_decisions(decisions)
    expected = max(actions, key=lambda a: _PRIORITY[a])
    assert result.action == expected


@given(actions=st.lists(risk_actions, min_size=1, max_size=10))
@settings(max_examples=200)
def test_merge_decisions_aggregates_all_violations(actions):
    """All violations from all decisions appear in merged result."""
    decisions = tuple(_make_decision(a) for a in actions)
    result = merge_decisions(decisions)

    total_violations = sum(len(d.violations) for d in decisions)
    assert len(result.violations) == total_violations


def test_merge_empty_returns_allow():
    """Merging empty decisions returns ALLOW."""
    result = merge_decisions(())
    assert result.action == RiskAction.ALLOW


@given(n=st.integers(min_value=1, max_value=20))
@settings(max_examples=100)
def test_merge_all_allow_returns_allow(n):
    """Merging N ALLOW decisions returns ALLOW."""
    decisions = tuple(RiskDecision.allow() for _ in range(n))
    result = merge_decisions(decisions)
    assert result.action == RiskAction.ALLOW


@given(actions=st.lists(risk_actions, min_size=2, max_size=10))
@settings(max_examples=200)
def test_merge_is_idempotent(actions):
    """Merging the same set of decisions always gives the same result."""
    decisions = tuple(_make_decision(a) for a in actions)
    r1 = merge_decisions(decisions)
    r2 = merge_decisions(decisions)
    assert r1.action == r2.action
    assert len(r1.violations) == len(r2.violations)
