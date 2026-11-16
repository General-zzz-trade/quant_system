"""Fixes after the independent audit of the strategy loss stop (2026-11-16, PR #106).

- The stop's findings have their own issue kind (finding numbering as in the audit; see the execution log).
- The unrealized PnL is the strategy's own: ledger quantity x (venue mark - ledger entry), per ``target.*`` owner.
- Once reached, the stop stays reached for the life of the process.
- Funding that cannot be refreshed does not blind the stop while a total was read earlier in this process.
"""
from __future__ import annotations

from decimal import Decimal as D
from types import SimpleNamespace as NS

import pytest

from tests.unit.trading.test_strategy_loss_stop import _stop
from trading.application.risk.loss_stop import strategy_unrealized
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


# -- point 1: the strategy's own unrealized PnL, from the ledger (audit findings 1 and 2) --------------------------
def _venue(**marks):
    """The venue's row per symbol; its quantity and entry price are never the strategy's (a manual position is
    netted into them). A symbol not named has no row."""
    asked = []

    def position(symbol):
        asked.append(symbol)
        if symbol not in marks:
            return None
        return NS(symbol=symbol, qty=D("999"), entry_price=D("1"), mark_price=marks[symbol])
    position.asked = asked
    return position


def test_unrealized_pnl_is_the_ledger_quantity_at_the_venue_s_mark_against_the_ledger_entry():
    ledger = [("BTCUSDT", D("0.01"), D("50000")), ("ETHUSDT", D("-2"), D("3000"))]
    venue = _venue(BTCUSDT=D("47000"), ETHUSDT=D("2900"))
    assert strategy_unrealized(ledger, venue) == D("-30") + D("200")
    assert strategy_unrealized([], venue) == 0


def test_a_ledger_quantity_of_zero_adds_nothing_whatever_the_venue_shows():
    # A person's own position on a strategy symbol: the venue holds 999, the ledger nothing.
    venue = _venue(BTCUSDT=D("47000"))
    assert strategy_unrealized([("BTCUSDT", D("0"), D("0")), ("ETHUSDT", D("0"), D("3000"))], venue) == 0
    assert venue.asked == []                                  # no row, no mark needed: ETHUSDT has none here


@pytest.mark.parametrize("mark", [None, D("nan"), D("inf"), D("0"), D("-1")])
def test_a_held_ledger_position_without_a_usable_mark_is_unknown(mark):
    with pytest.raises(RuntimeError, match="BTCUSDT.*mark"):
        strategy_unrealized([("BTCUSDT", D("0.01"), D("50000"))], _venue(BTCUSDT=mark))


def test_a_held_ledger_position_without_a_venue_row_is_unknown_never_zero():
    with pytest.raises(RuntimeError, match="BTCUSDT.*no row"):
        strategy_unrealized([("BTCUSDT", D("0.01"), D("50000"))], _venue(ETHUSDT=D("2900")))


@pytest.mark.parametrize("qty,entry", [("0.01", "0"), ("0.01", "-5"), ("0.01", "nan"), ("0.01", "inf"),
                                       ("nan", "50000"), ("inf", "50000")])
def test_a_ledger_position_that_cannot_be_used_is_unknown(qty, entry):
    with pytest.raises(RuntimeError, match="BTCUSDT.*ledger"):
        strategy_unrealized([("BTCUSDT", D(qty), D(entry))], _venue(BTCUSDT=D("47000")))


class Ledger:
    def __init__(self, rows):
        self.rows = rows

    def checkpoint(self, owner):
        return self.rows.get(owner)


def test_the_ledger_positions_are_each_owner_s_checkpoint():
    from trading.infrastructure.target import strategy_positions
    ledger = Ledger({"target.BTCUSDT": {"position": {"qty": "0.002", "entry_price": "30123.5", "realized_pnl": "1"}},
                     "target.ETHUSDT": {"position": {"qty": "0", "entry_price": "0", "realized_pnl": "-3"}}})
    assert strategy_positions(ledger, ["target.BTCUSDT", "target.ETHUSDT", "target.SOLUSDT"]) == [
        ("BTCUSDT", D("0.002"), D("30123.5")), ("ETHUSDT", D("0"), D("0"))]      # no checkpoint: nothing held
    broken = Ledger({"target.BTCUSDT": {"position": {"qty": "0.002"}}})
    with pytest.raises(KeyError):
        strategy_positions(broken, ["target.BTCUSDT"])                           # unknown, never 0

