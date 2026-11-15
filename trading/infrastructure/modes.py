"""The two mode policies (ADR-0013 point 2): the one list of the differences between paper and execution.

The composition root picks the policy by the mode it was started in. Besides the fields of ``ModePolicy``,
paper differs only in what is plugged in, never in the code that runs:

- the venue is the simulated exchange (``trading.adapters.simulated``), built from the research brokers; no
  credentialed adapter is built and no signed request is sent (audit M6);
- the risk limits are the code defaults, and no account setting is read from the environment (S10 decision 3);
- there is no whole-account check (ADR-0010/0011) and no notifier;
- the order lifecycle does not wait after a placement: the simulated venue fills only when a bar is matched;
- the paper state (the simulated exchange and the cursors) is saved with the order journal.
"""
from __future__ import annotations

from trading.application.mode_policy import ModePolicy

PAPER = ModePolicy(name="paper", catch_up="replay_all", entry_ttl="venue", sizing_equity="per_symbol_venue",
                   limiter_clock="bar_time", settle_new_bar_before_decision=True, isolate_failures=False,
                   entry_beside_working_reduction=True)
EXECUTION = ModePolicy(name="execution", catch_up="latest_only", entry_ttl="lifecycle",
                       sizing_equity="account_share", limiter_clock="monotonic",
                       settle_new_bar_before_decision=False, isolate_failures=True,
                       entry_beside_working_reduction=False)


def policy_for(mode: str) -> ModePolicy:
    if mode == "paper":
        return PAPER
    if mode == "execution":
        return EXECUTION
    raise ValueError("mode must be paper or execution")


__all__ = ["EXECUTION", "PAPER", "ModePolicy", "policy_for"]
