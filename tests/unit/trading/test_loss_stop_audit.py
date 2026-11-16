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
@pytest.mark.parametrize("case", ["unknown", "reached", "switch not set"])
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
    def __init__(self, rows, owners=()):
        self.rows, self._owners = rows, list(owners)

    def checkpoint(self, owner):
        return self.rows.get(owner)

    def owners(self):
        return list(self._owners)


def test_the_ledger_positions_are_each_owner_s_checkpoint():
    from trading.infrastructure.target import strategy_positions
    ledger = Ledger({"target.BTCUSDT": {"position": {"qty": "0.002", "entry_price": "30123.5", "realized_pnl": "1"}},
                     "target.ETHUSDT": {"position": {"qty": "0", "entry_price": "0", "realized_pnl": "-3"}}})
    assert strategy_positions(ledger, ["target.BTCUSDT", "target.ETHUSDT", "target.SOLUSDT"]) == [
        ("BTCUSDT", D("0.002"), D("30123.5")), ("ETHUSDT", D("0"), D("0"))]      # no checkpoint: nothing held
    broken = Ledger({"target.BTCUSDT": {"position": {"qty": "0.002"}}})
    with pytest.raises(KeyError):
        strategy_positions(broken, ["target.BTCUSDT"])                           # unknown, never 0


# -- point 2: every target.* owner of the scope counts (audit finding 7) -------------------------------------------
def test_the_strategy_s_owners_are_the_spec_s_symbols_and_every_target_owner_in_the_ledger():
    from trading.infrastructure.target import strategy_owners
    ledger = Ledger({}, owners=["target.SOLUSDT", "target.BTCUSDT", "manual.BTCUSDT", "other"])
    assert strategy_owners(ledger, ["BTCUSDT", "ETHUSDT"]) == ["target.BTCUSDT", "target.ETHUSDT", "target.SOLUSDT"]

    class Unreadable:
        def owners(self):
            raise OSError("database is locked")
    with pytest.raises(OSError):
        strategy_owners(Unreadable(), ["BTCUSDT"])


def test_the_journal_lists_every_owner_its_scope_knows(tmp_path):
    from tests.support.state_db import state_db
    from trading.adapters.sqlite.order_journal import SqliteOrderJournal
    path = state_db(tmp_path / "state.sqlite")
    journal, other = SqliteOrderJournal(path, "scope"), SqliteOrderJournal(path, "another")
    assert journal.owners() == []
    journal.insert_fill(fill_id="f1", intent_id="i1", owner="target.BTCUSDT", payload={})     # a fill, not consumed
    journal.set_position_manually("target.SOLUSDT", {"qty": "0"}, venue={}, reason="ledger := venue", at=70.0)
    other.insert_fill(fill_id="f2", intent_id="i2", owner="target.XRPUSDT", payload={})       # another scope
    assert journal.owners() == ["target.BTCUSDT", "target.SOLUSDT"]
    with journal.connect() as db:                                                             # a checkpoint only
        db.execute("INSERT INTO execution_consumers VALUES ('scope', 'target.ETHUSDT', '{}')")
        db.execute("INSERT INTO manual_actions VALUES ('scope', 1.0, 'settle_order', 'intent-9', NULL, 'x', '{}', 'r')")
    assert journal.owners() == ["target.BTCUSDT", "target.ETHUSDT", "target.SOLUSDT"]


def test_funding_asks_for_the_symbols_it_is_given_at_each_refresh():
    from tests.unit.trading.test_strategy_capital import Clock, Venue
    from trading.application.equity import StrategyFunding
    venue, clock, symbols = Venue({"BTCUSDT": D("-1"), "SOLUSDT": D("-7")}), Clock(), ["BTCUSDT"]
    funding = StrategyFunding(venue, lambda: list(symbols), {"BTCUSDT": 10.0, "SOLUSDT": 5.0}.get, clock)
    assert funding.current() == D("-1")
    symbols.append("SOLUSDT")                               # an owner the ledger got since
    clock.now += 600
    assert funding.current() == D("-8")

    def unreadable():
        raise OSError("database is locked")
    never = StrategyFunding(venue, unreadable, {"BTCUSDT": 10.0}.get, Clock())
    with pytest.raises(RuntimeError, match="funding"):      # owners that cannot be listed: unknown, never 0
        never.current()


# -- point 4: once reached, reached for the life of the process (audit finding 6) ---------------------------------
def test_once_reached_every_check_blocks_entries_and_sets_a_cleared_switch_again_at_once(tmp_path):
    stop, kill, holder, clock = _stop(tmp_path, D("-45"))
    (reached,) = stop.check()
    assert reached.kind is IssueKind.STRATEGY_LOSS_STOP and "strategy loss stop reached" in reached
    assert kill.state().active and kill.state().flatten and stop.take_trip() is not None
    kill.write(active=False, reason="cleared by a person")
    assert stop.check() == [reached]                         # no minute's wait: the same cycle's check
    state = kill.state()
    assert state.active and state.flatten and "strategy loss stop" in state.reason
    assert stop.take_trip() is not None and stop.take_trip() is None       # each trip is reported once
    written = (tmp_path / "kill_switch.json").read_text()
    holder["pnl"] = D("25")                                  # the loss is gone (a person changed the ledger, say)
    for _ in range(3):
        clock.now += 61
        assert stop.check() == [reached]                     # still reached: this process never enters again
    assert (tmp_path / "kill_switch.json").read_text() == written and stop.take_trip() is None


def test_a_switch_that_cannot_be_set_after_a_clear_is_a_finding_until_it_is_set(tmp_path):
    stop, kill, _, _ = _stop(tmp_path, D("-50"))
    (reached,) = stop.check()
    assert stop.take_trip() is not None
    kill.write(active=False, reason="cleared by a person")
    real, kill.trip = kill.trip, _raising(OSError("read-only file system"))
    (failed,) = stop.check()
    assert "kill switch could not be set" in failed and failed.kind is IssueKind.STRATEGY_LOSS_STOP
    assert kill.state().active is False and stop.take_trip() is None
    kill.trip = real
    assert stop.check() == [reached] and kill.state().flatten is True


@pytest.mark.parametrize("content", [None, "{not json", '{"active": "yes"}'])
def test_a_missing_or_unreadable_kill_file_is_not_flatten_so_the_stop_writes_it(tmp_path, content):
    from trading.adapters.files.kill_switch import KillFile
    from trading.application.risk.loss_stop import StrategyLossStop
    path = tmp_path / "kill_switch.json"
    if content is not None:
        path.write_text(content)
    kill = KillFile(path, missing_is_active=True)                    # an initialized state directory
    before = kill.state()
    assert before.active and not before.flatten                      # blocks entries, closes nothing
    stop = StrategyLossStop(D("300"), D("0.15"), lambda: D("-50"), kill, clock=lambda: 1000.0)
    stop.check()
    state = kill.state()
    assert state.active and state.flatten and "strategy loss stop" in state.reason


def test_the_reason_names_the_limit_as_a_person_wrote_it(tmp_path):
    for limit, text in (("0.15", "limit 15 %"), ("0.075", "limit 7.5 %"), ("0.10", "limit 10 %")):
        stop, kill, _, _ = _stop(tmp_path / limit, D("-50"), limit=limit)
        stop.check()
        assert text in kill.state().reason, kill.state().reason


# -- point 6: funding that cannot be refreshed does not blind the stop (audit finding 4) --------------------------
def test_the_stop_gets_the_last_funding_total_read_in_this_process_whatever_its_age():
    from tests.unit.trading.test_strategy_capital import Clock, Venue
    from trading.application.equity import FUNDING_MAX_AGE, StrategyFunding
    venue, clock = Venue({"BTCUSDT": D("-4")}), Clock()
    funding = StrategyFunding(venue, ["BTCUSDT"], {"BTCUSDT": 10.0}.get, clock)
    assert funding.last_known() == D("-4") and len(venue.calls) == 1          # it refreshes when due
    venue.down = True
    clock.now += FUNDING_MAX_AGE + 1
    with pytest.raises(RuntimeError, match="funding"):
        funding.current()                                                     # the sizing base: the two-hour rule
    assert funding.last_known() == D("-4")                                    # the stop: the last total read
    clock.now += 30 * 86400
    assert funding.last_known() == D("-4") and len(venue.calls) == 3          # still tried, still kept
    venue.down, venue.income["BTCUSDT"] = False, D("-9")
    clock.now += 600
    assert funding.last_known() == D("-9")


def test_funding_never_read_in_this_process_is_unknown_for_the_stop_too():
    from tests.unit.trading.test_strategy_capital import Clock, Venue
    from trading.application.equity import StrategyFunding
    venue = Venue({"BTCUSDT": D("5")})
    venue.down = True
    funding = StrategyFunding(venue, ["BTCUSDT"], {"BTCUSDT": 10.0}.get, Clock())
    with pytest.raises(RuntimeError, match="funding"):
        funding.last_known()
    assert len(venue.calls) == 1


# -- point 8: gaps the auditor's surviving mutants showed ---------------------------------------------------------
@pytest.mark.parametrize("pnl", [D("Infinity"), D("-Infinity"), D("NaN"), float("inf"), float("nan")])
def test_a_pnl_that_is_not_a_finite_number_sizes_the_base_to_zero(pnl):
    from tests.unit.trading.test_strategy_capital import _account
    from trading.application.equity import CappedEquity
    assert CappedEquity(_account("10000"), D("300"), lambda: pnl).current() == 0


def test_the_first_fill_ignores_a_manual_position_of_another_owner(tmp_path):
    from tests.support.state_db import state_db
    from trading.adapters.sqlite.order_journal import SqliteOrderJournal
    journal = SqliteOrderJournal(state_db(tmp_path / "state.sqlite"), "scope")
    journal.set_position_manually("target.SOLUSDT", {"qty": "1"}, venue={}, reason="ledger := venue", at=70.0)
    assert journal.first_fill_at("target.BTCUSDT") is None            # never traded: no funding is asked for it
    journal.set_position_manually("target.BTCUSDT", {"qty": "1"}, venue={}, reason="ledger := venue", at=90.0)
    assert journal.first_fill_at("target.BTCUSDT") == 90.0            # its own, not the other owner's earlier one


def test_a_limit_of_exactly_15_per_cent_is_accepted():
    from tests.unit.trading.test_strategy_loss_stop import _load
    assert _load(STRATEGY_LOSS_STOP_PCT="15").strategy_loss_stop == D("0.15")
    assert _load(STRATEGY_LOSS_STOP_PCT="15.0").strategy_loss_stop == D("0.15")


# -- the stop's entry pause (PR #102): a mutant of the auditor survived in its store ------------------------------
def test_a_second_scheduled_pause_with_the_same_end_adds_no_row(tmp_path):
    from tests.unit.trading.test_stop_pause_auto_release import _rows, _store
    pauses, _ = _store(tmp_path)
    assert pauses.pause("BTCUSDT", "protective stop triggered", triggers=["7"], ends_at=5000.0) is True
    assert pauses.pause("BTCUSDT", "protective stop breached", triggers=["8"], ends_at=5000.0) is False
    assert len(_rows(tmp_path)) == 1 and pauses.covered("BTCUSDT") == {"7", "8"}
