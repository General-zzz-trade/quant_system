"""ADR-0013 point 2 (S10 part 3): ``ModePolicy`` is the one list of the differences between paper and execution.

Each field is pinned for both modes; a new difference must change this list. What every field does is tested
where it acts (test_paper_replay.py, test_mode_differences.py and the paper runtime tests).
"""
from __future__ import annotations

import dataclasses

import pytest

from trading.application.mode_policy import ModePolicy
from trading.infrastructure.modes import EXECUTION, PAPER, policy_for


def test_the_two_policies_are_the_list_of_differences():
    assert dataclasses.asdict(PAPER) == {
        "name": "paper", "catch_up": "replay_all", "entry_ttl": "venue", "sizing_equity": "per_symbol_venue",
        "limiter_clock": "bar_time", "settle_new_bar_before_decision": True, "isolate_failures": False,
        "entry_beside_working_reduction": True}
    assert dataclasses.asdict(EXECUTION) == {
        "name": "execution", "catch_up": "latest_only", "entry_ttl": "lifecycle", "sizing_equity": "account_share",
        "limiter_clock": "monotonic", "settle_new_bar_before_decision": False, "isolate_failures": True,
        "entry_beside_working_reduction": False}


def test_every_field_differs_between_the_modes():
    differing = {f.name for f in dataclasses.fields(ModePolicy) if getattr(PAPER, f.name) != getattr(EXECUTION, f.name)}
    assert differing == {f.name for f in dataclasses.fields(ModePolicy)}


def test_the_mode_picks_the_policy_and_nothing_else_is_a_mode():
    assert policy_for("paper") is PAPER and policy_for("execution") is EXECUTION
    for other in ("live", "Paper", "", "dry-run"):
        with pytest.raises(ValueError, match="paper or execution"):
            policy_for(other)


def test_a_policy_cannot_be_changed_while_running():
    with pytest.raises(dataclasses.FrozenInstanceError):
        PAPER.isolate_failures = True              # type: ignore[misc]
