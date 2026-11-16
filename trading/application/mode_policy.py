"""Every difference between paper and execution that the shared path acts on, as one value (ADR-0013 point 2).

The session, the market side, the exit side and the runtime read these fields; none of them asks "is this
paper?" (tests/architecture/test_paper_path.py). The composition root picks one of the two instances in
``trading.infrastructure.modes`` and translates the fields it owns (sizing, the gate clock, the lifecycle's
expiry, the journal's flip rule) into the parts it builds. The type lives here because the application layer may
not import infrastructure. A new difference is a new field, set in both instances, with a test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModePolicy:
    # The status file and the event log name the mode with it; nothing branches on it.
    name: Literal["paper", "execution"]
    # Bars that closed while market data was down, or while a paper process was stopped: paper replays every one
    # in time order across the symbols, as research does; execution decides on the newest only (ADR-0007).
    catch_up: Literal["replay_all", "latest_only"]
    # A resting maker entry: the simulated venue expires it after matching its last bar (research order, audit
    # H1-1); in execution the order lifecycle cancels it once ttl_bars hours have passed.
    entry_ttl: Literal["venue", "lifecycle"]
    # Sizing equity: paper sizes each symbol from its own simulated account, as research does (audit H1-3);
    # execution from the account equity times 1/number of symbols.
    sizing_equity: Literal["per_symbol_venue", "account_share"]
    # The clock of the entry gates: the order limiter's day and per-second window, "bar processed too late" and
    # the retry of entries the per-second limit deferred. Paper runs them on the bar's close and retries a
    # deferred entry 5 s later in simulated time (S10 decision 2, audit H1-5); execution on the runtime clock and
    # the monotonic clock.
    limiter_clock: Literal["bar_time", "monotonic"]
    # Paper: a new bar is matched by the simulated venue, its fills consumed and the symbols reconciled before the
    # decision (research order, audit H1-4). Execution consumes fills when the exit side reconciles.
    settle_new_bar_before_decision: bool
    # Execution: a side or a decision that raises is an issue that blocks entries, and the cycle goes on (S3c).
    # Paper raises: a paper record must not diverge from research silently.
    isolate_failures: bool
    # Paper: the simulated venue fills a market reduction at the next bar's open, so the entry of a flip is
    # reserved beside its acknowledged, still working reduction, as research places both on one bar (S10
    # decision 1). Execution keeps the stricter rule: no entry while any order of the symbol is unsettled.
    entry_beside_working_reduction: bool


__all__ = ["ModePolicy"]
