"""Fixes after the independent audit of the strategy loss stop (2026-11-16, PR #106).

- The stop's findings have their own issue kind (finding numbering as in the audit; see the execution log).
- The unrealized PnL is the strategy's own: ledger quantity x (venue mark - ledger entry), per ``target.*`` owner.
- Once reached, the stop stays reached for the life of the process.
- Funding that cannot be refreshed does not blind the stop while a total was read earlier in this process.
"""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from tests.unit.trading.test_strategy_loss_stop import _stop
from trading.domain.issues import Effect, IssueKind


def _raising(exc):
    def trip(reason):
        raise exc
    return trip


# -- point 5: the stop's own issue kind ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["unknown", "switch not set"])
def test_every_finding_of_the_stop_is_a_strategy_loss_stop_issue(tmp_path, case):
    stop, kill, _, _ = _stop(tmp_path, OSError("ledger locked") if case == "unknown" else D("-50"))
    if case == "switch not set":
        kill.trip = _raising(OSError("read-only file system"))
    (issue,) = stop.check()
    assert issue.kind is IssueKind.STRATEGY_LOSS_STOP and issue.kind.value == "strategy_loss_stop"
    assert issue.symbol is None
    assert issue.effects == frozenset({Effect.BLOCK_ALL_ENTRIES, Effect.CANCEL_RESTING_ENTRIES})
