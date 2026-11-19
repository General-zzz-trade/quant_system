"""Plan D2: the exit side sends requests of a symbol's own only for an active symbol.

Active (``ExitSupervisor.active``): the venue holds something in this reconciliation's position snapshot (or, without
a row there, held something in the last snapshot that had one), an order of the symbol is unsettled in the journal,
the strategy's ledger holds a position or a stop row is live (``ProtectiveStopManager.has_work``), or the session has
a working or deferred order. What cannot be read counts as active. An idle symbol reads neither its open orders (the
whole-account check reads every open order about once a minute) nor its position; its stop has nothing to do.
An emergency flatten takes the venue quantity from the snapshot: a flat row sends and reads nothing.

The world is the exit side as the runtime composes it over a fake venue (``exit_world``): BTCUSDT holds a position
with its stop, ETHUSDT and SOLUSDT are idle.
"""
from __future__ import annotations

import json
import sqlite3
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from tests.characterization.test_exit_budget_universe import cycles, exit_world
from trading.adapters.binance.rest import BinanceRetryableError
from trading.application.execution.flatten import Flattener
from trading.application.reconcile import reconcile_account
from trading.domain.bars import Bar
from trading.domain.issues import IssueKind as K

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
SCOPE = "binance-testnet-ut"


@pytest.fixture
def world(tmp_path, monkeypatch):
    return exit_world(tmp_path, SYMBOLS, ("BTCUSDT",), monkeypatch)


def _named(world) -> set[str]:
    """The symbols that requests since the last ``clear`` named."""
    return {params["symbol"] for _, _, params in world.client.requests if params.get("symbol")}


def _paths(world, symbol: str) -> list[str]:
    return [path for _, path, _ in world.client.of(symbol)]


def _account_wide_positions(world, answer):
    """Replace the venue's answer to the one read of every position (``positionRisk`` without a symbol):
    ``answer(rows)`` gets the real rows; None gives them back unchanged. A read of one symbol is answered as before."""
    world.answer = answer
    if getattr(world, "real_signed", None) is None:
        real = world.real_signed = world.fake.request_signed

        def signed(*, method, path, params=None):
            rows = real(method=method, path=path, params=params)
            if path == "/fapi/v2/positionRisk" and not (params or {}).get("symbol") and world.answer is not None:
                return world.answer(rows)
            return rows
        world.fake.request_signed = signed


def _fail_snapshot(world):
    def fail(rows):
        raise BinanceRetryableError("HTTP 503: unavailable")
    _account_wide_positions(world, fail)


def _issues(world, symbol=None) -> list[str]:
    return [i.kind.value for i in world.exits.last_issues[symbol or "BTCUSDT"]]


def _events(world, kind: str) -> list[dict]:
    return [row for row in map(json.loads, (world.state / "target_events.jsonl").read_text().splitlines())
            if row["kind"] == kind]


def _sql(world, statement, values):
    with sqlite3.connect(world.state / "state.sqlite") as db:
        db.execute(statement, values)


# -- normal cycles --------------------------------------------------------------------------------------------------
def test_only_the_active_symbol_sends_requests_of_its_own(world):
    world.client.clear()
    cycles(world, 1)
    assert _named(world) == {"BTCUSDT"}
    assert _paths(world, "BTCUSDT") == ["/fapi/v1/openOrders"]              # its open orders, nothing else alone
    assert [p for _, p, q in world.client.requests if p == "/fapi/v1/algoOrder"] == ["/fapi/v1/algoOrder"]
    assert [q for _, p, q in world.client.requests if p == "/fapi/v2/positionRisk"] == [{}]   # the one snapshot
    assert all(s.executor.ready for s in world.sessions.values())          # nothing blocked by the budget
    assert world.exits.last_issues == {s: [] for s in SYMBOLS}


def _venue(world, session):
    world.fake.positions["ETHUSDT"] = D("0.5")


def _ledger(world, session):
    _sql(world, "INSERT INTO execution_consumers VALUES (?, ?, ?)",
         (SCOPE, "target.ETHUSDT", '{"position": {"qty": "0.5", "entry_price": "101"}}'))


def _unsettled(world, session):          # an unsettled order of another owner on the symbol counts too
    _sql(world, "INSERT INTO submissions (scope, intent_id, symbol, client_id, order_id, settled, owner, side, qty, "
         "reducing) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
         (SCOPE, "m1", "ETHUSDT", "cid-m1", "", 0, "manual.ETHUSDT", "buy", "0.5", 0))


def _stop_row(world, session):
    session.executor.journal.add_stop(owner="target.ETHUSDT", client_algo_id="qsstopETH", symbol="ETHUSDT",
                                      revision="0.5@101", attempt=0, side="sell", trigger_price="93",
                                      created_at=world.clock["now"])


def _pending(world, session):
    session.pending["ETHUSDT:1:rebalance:0"] = False


def _deferred(world, session):
    session.deferred = SimpleNamespace(bar=None, reason="rebalance", items=[])


@pytest.mark.parametrize("make_active", [_venue, _ledger, _unsettled, _stop_row, _pending, _deferred])
def test_each_reason_to_be_active_gives_the_symbol_its_own_reads(world, make_active):
    session = world.sessions["ETHUSDT"]
    positions = world.adapter.get_position_snapshot()
    assert not world.exits.active("ETHUSDT", session, positions)
    make_active(world, session)
    assert world.exits.active("ETHUSDT", session, world.adapter.get_position_snapshot())
    world.client.clear()
    world.exits.reconcile()
    assert "/fapi/v1/openOrders" in _paths(world, "ETHUSDT")
    assert _named(world) <= {"BTCUSDT", "ETHUSDT"}                          # SOLUSDT stays quiet


def test_a_journal_that_cannot_be_read_makes_the_symbol_active(world, monkeypatch):
    session = world.sessions["ETHUSDT"]

    def unreadable(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(session.executor.journal, "orders", unreadable)
    assert world.exits.active("ETHUSDT", session, world.adapter.get_position_snapshot())


def test_the_stop_manager_has_work_only_with_a_ledger_position_or_a_live_stop_row(world):
    assert world.sessions["BTCUSDT"].stops.has_work()
    eth = world.sessions["ETHUSDT"]
    assert not eth.stops.has_work()
    _ledger(world, eth)
    assert eth.stops.has_work()
    sol = world.sessions["SOLUSDT"]
    sol.executor.journal.add_stop(owner="target.SOLUSDT", client_algo_id="qsstopSOL", symbol="SOLUSDT",
                                  revision="1@100", attempt=0, side="sell", trigger_price="92", created_at=0.0)
    assert sol.stops.has_work()
    sol.executor.journal.set_stop_status("qsstopSOL", "inactive")
    assert not sol.stops.has_work()


# -- the snapshot cannot be read, or has no row for a symbol ---------------------------------------------------------
def test_without_a_snapshot_only_active_symbols_read_their_position_and_entries_are_blocked(world):
    _fail_snapshot(world)
    world.client.clear()
    world.exits.reconcile()
    assert _named(world) == {"BTCUSDT"}
    assert _paths(world, "BTCUSDT").count("/fapi/v2/positionRisk") == 2    # reconciliation and stop, as before D2
    assert all(_issues(world, s) == ["positions_unread"] for s in SYMBOLS)
    assert not any(s.executor.ready for s in world.sessions.values())
    world.exits.reconcile()
    assert len([e for e in _events(world, "issue") if e["issue_kind"] == "positions_unread"]) == 1   # written once


def test_a_symbol_that_held_something_in_the_last_snapshot_still_reads_its_own(world):
    world.fake.positions["SOLUSDT"] = D("2")                                # someone's position, not the strategy's
    world.exits.reconcile()
    _fail_snapshot(world)
    world.client.clear()
    world.exits.reconcile()
    assert _named(world) == {"BTCUSDT", "SOLUSDT"}
    _account_wide_positions(world, None)
    world.fake.positions["SOLUSDT"] = D("0")
    world.exits.reconcile()                                                 # a good snapshot: SOL is flat again
    _fail_snapshot(world)
    world.client.clear()
    world.exits.reconcile()
    assert _named(world) == {"BTCUSDT"}


def test_a_row_whose_quantity_cannot_be_read_counts_as_held_later(world):
    session = world.sessions["SOLUSDT"]
    odd = {"SOLUSDT": SimpleNamespace(symbol="SOLUSDT", qty="not a number")}
    world.adapter.get_position_snapshot = lambda: odd
    world.exits.reconcile()
    assert world.exits.active("SOLUSDT", session, odd)
    assert world.exits.active("SOLUSDT", session, None)                   # the snapshot then fails: still active


def _exits_without_history(world, account):
    """A new exit side over the same sessions: what a restart has, after the start-up check passed."""
    from trading.application.exit_supervisor import ExitSupervisor
    from tests.characterization.test_exit_budget_universe import SPEC
    old = world.exits
    return ExitSupervisor(world.sessions, spec=SPEC, journal=old.journal, clock=old.clock, account=account,
                          equity=old.equity, kill=old.kill, drawdown=old.drawdown, pauses=old.pauses, notifier=None,
                          alert_prefix="", market_refused=lambda: None, decision_error=lambda symbol: None,
                          cycle_errors=lambda: [])


def test_before_any_snapshot_a_quiet_symbol_counts_as_flat(world):
    """The start-up check (ADR-0010) passed before the exit side was built: the account held nothing outside the
    ledger and the journal's orders, so a symbol with neither is flat until a snapshot says otherwise."""
    _fail_snapshot(world)
    exits = _exits_without_history(world, world.adapter)
    world.client.clear()
    exits.reconcile()
    assert _named(world) == {"BTCUSDT"}


def test_without_an_account_every_symbol_is_active():
    """No account, no snapshot: each symbol reads its own, as before D2 (a composition only tests make)."""
    from trading.application.exit_supervisor import ExitSupervisor
    from tests.characterization.test_exit_budget_universe import SPEC
    exits = ExitSupervisor({}, spec=SPEC, journal=None, clock=lambda: 0.0, account=None, equity=None, kill=None,
                           drawdown=None, pauses=None, notifier=None, alert_prefix="", market_refused=lambda: None,
                           decision_error=lambda symbol: None, cycle_errors=lambda: [])
    quiet = SimpleNamespace(pending={}, deferred=None, stops=SimpleNamespace(has_work=lambda: False),
                            executor=SimpleNamespace(journal=SimpleNamespace(orders=lambda **kw: [])))
    assert exits.active("BTCUSDT", quiet, None)
    exits.account = SimpleNamespace()
    assert not exits.active("BTCUSDT", quiet, None)


def test_an_idle_symbol_missing_from_the_snapshot_is_not_read_and_blocks_every_entry(world):
    _account_wide_positions(world, lambda rows: [r for r in rows if r["symbol"] != "ETHUSDT"])
    world.client.clear()
    world.exits.reconcile()
    assert _named(world) == {"BTCUSDT"} and "/fapi/v2/positionRisk" not in _paths(world, "BTCUSDT")
    assert all(_issues(world, s)[-1] == "positions_unread" for s in SYMBOLS)
    assert "ETHUSDT" in world.exits.snapshot_issue
    assert not any(s.executor.ready for s in world.sessions.values())


def test_an_active_symbol_missing_from_the_snapshot_reads_its_own_and_blocks_nothing_more(world):
    _account_wide_positions(world, lambda rows: [r for r in rows if r["symbol"] != "BTCUSDT"])
    world.client.clear()
    world.exits.reconcile()
    assert _paths(world, "BTCUSDT").count("/fapi/v2/positionRisk") == 2
    assert world.exits.snapshot_issue is None
    assert world.exits.last_issues == {s: [] for s in SYMBOLS}


# -- the emergency flatten -------------------------------------------------------------------------------------------
def test_a_kill_flatten_closes_the_held_symbol_and_the_idle_ones_send_nothing(world):
    world.kill.write(active=True, reason="test", flatten=True)
    world.client.clear()
    cycles(world, 1)
    assert world.fake.positions["BTCUSDT"] == 0
    assert _named(world) == {"BTCUSDT"}
    # one read of its position: the reduction's own sizing at the send (size_reduction), none for the decision
    assert _paths(world, "BTCUSDT").count("/fapi/v2/positionRisk") == 1
    world.client.clear()
    cycles(world, 3)                                                        # flat: its stop goes, then nothing
    assert world.fake.positions["BTCUSDT"] == 0
    world.client.clear()
    cycles(world, 1)
    assert _named(world) == set()


def test_a_flat_row_while_the_ledger_holds_reads_the_venue_and_closes(world):
    """The snapshot can be older than a fill (a resting entry filled since): the ledger then holds what the row does
    not show, and the venue is asked."""
    def flat_btc(rows):
        return [dict(r, positionAmt="0") if r["symbol"] == "BTCUSDT" else r for r in rows]
    _account_wide_positions(world, flat_btc)
    world.kill.write(active=True, reason="test", flatten=True)
    world.client.clear()
    cycles(world, 1)
    assert world.fake.positions["BTCUSDT"] == 0
    assert _paths(world, "BTCUSDT").count("/fapi/v2/positionRisk") >= 2    # the decision's read and the sizing's


def test_without_a_snapshot_a_kill_flatten_reads_only_symbols_that_could_hold_something(world):
    _fail_snapshot(world)
    world.kill.write(active=True, reason="test", flatten=True)
    world.client.clear()
    cycles(world, 1)
    assert world.fake.positions["BTCUSDT"] == 0
    assert _named(world) == {"BTCUSDT"}
    assert {e.get("symbol") for e in _events(world, "issue")} <= {None, "BTCUSDT"}      # no flatten issue for them


# -- the pieces on their own -------------------------------------------------------------------------------------------
class _Journal:
    def __init__(self, ledger="0", unresolved=(), fills=()):
        self.ledger, self.unresolved, self._fills = ledger, list(unresolved), list(fills)

    def checkpoint(self, owner):
        return {"position": {"qty": self.ledger}}

    def orders(self, owner=None, *, unresolved=False):
        return self.unresolved

    def fills(self, owner):
        return self._fills


class _Account:
    def __init__(self, held=D(0)):
        self.held, self.reads = held, []

    def get_positions(self, symbol="", *, strict=True):
        self.reads.append(("positions", symbol))
        return (SimpleNamespace(symbol=symbol, qty=self.held),) if self.held else ()

    def get_open_orders(self, symbol="", *, strict=True):
        self.reads.append(("orders", symbol))
        return (SimpleNamespace(order_id="web-1"),)


def test_reconciliation_without_venue_reads_sends_nothing_and_still_checks_the_journal():
    account = _Account(held=D(1))
    row = {"BTCUSDT": SimpleNamespace(symbol="BTCUSDT", qty=D(0))}
    assert reconcile_account(account, _Journal(), owner="o", symbol="BTCUSDT", positions=row,
                             venue_reads=False) == [] and account.reads == []
    issues = reconcile_account(account, _Journal(), owner="o", symbol="BTCUSDT", positions={}, venue_reads=False)
    assert issues == [] and account.reads == []                             # an unknown position is not read alone
    issues = reconcile_account(account, _Journal(ledger="1"), owner="o", symbol="BTCUSDT", positions={},
                               venue_reads=False)
    assert issues == [] and account.reads == []                             # nor compared as if it were flat
    held = {"BTCUSDT": SimpleNamespace(symbol="BTCUSDT", qty=D(2))}
    issues = reconcile_account(account, _Journal(ledger="1"), owner="o", symbol="BTCUSDT", positions=held,
                               venue_reads=False)
    assert [i.kind for i in issues] == [K.LEDGER_MISMATCH] and account.reads == []
    journal = _Journal(unresolved=[{"symbol": "BTCUSDT", "order_id": "1", "owner": "o"}], fills=[{"fill_id": "f"}])
    issues = reconcile_account(account, journal, owner="o", symbol="BTCUSDT", positions=row, venue_reads=False)
    assert [i.kind for i in issues] == [K.UNRESOLVED_ORDER, K.UNDELIVERED_FILL] and account.reads == []
    issues = reconcile_account(account, _Journal(), owner="o", symbol="BTCUSDT", positions={})
    assert account.reads == [("positions", "BTCUSDT"), ("orders", "BTCUSDT")]   # by default as before D2
    assert [i.kind for i in issues] == [K.LEDGER_MISMATCH, K.FOREIGN_HOLDING]


class _FlattenParts:
    def __init__(self, venue_qty=D(3)):
        self.sent, self.reads, self.issues = [], [], []
        self.venue_qty = venue_qty
        journal = SimpleNamespace(orders=lambda owner=None, **kw: [], intent_fills=lambda intent: [])
        self.executor = SimpleNamespace(journal=journal, owner="target.BTCUSDT", cancel_entries=lambda s: (),
                                        send_order=self.sent.append, take_refusal=lambda intent: None)
        self.account = SimpleNamespace(get_position=self._read)
        self.events = SimpleNamespace(write=lambda kind, **f: self.issues.append(f) if kind.value == "issue" else None)

    def _read(self, symbol):
        self.reads.append(symbol)
        return SimpleNamespace(symbol=symbol, qty=self.venue_qty)

    def flatten(self, *, ledger=D(0), **where):
        flattener = Flattener("BTCUSDT", self.executor, self.account, self.events, clock=lambda: 1000.0)
        bar = Bar(0, D(100), D(100), D(100), D(100), D(1))
        flattener.flatten("kill_switch", bar=bar, ledger=lambda: ledger, consume=lambda: None, **where)
        return self


def _row(qty):
    return SimpleNamespace(symbol="BTCUSDT", qty=D(qty))


def test_the_flatten_takes_the_venue_quantity_from_the_snapshot_row():
    parts = _FlattenParts().flatten(position=_row("2"))
    assert parts.reads == [] and len(parts.sent) == 1 and parts.sent[0].side == "sell"
    parts = _FlattenParts().flatten(position=_row("-2"), ledger=D(-1))
    assert parts.reads == [] and parts.sent[0].side == "buy"
    parts = _FlattenParts().flatten(position=_row("0"))
    assert parts.reads == [] and parts.sent == [] and parts.issues == []
    parts = _FlattenParts().flatten(position=_row("0"), ledger=D(1))       # the ledger holds what the row does not
    assert parts.reads == ["BTCUSDT"] and len(parts.sent) == 1


def test_without_a_row_the_flatten_reads_only_a_symbol_that_could_hold_something():
    parts = _FlattenParts().flatten(may_hold=False)
    assert parts.reads == [] and parts.sent == [] and parts.issues == []
    parts = _FlattenParts().flatten(may_hold=True)
    assert parts.reads == ["BTCUSDT"] and len(parts.sent) == 1
    parts = _FlattenParts().flatten()                                       # by default as before D2
    assert parts.reads == ["BTCUSDT"] and len(parts.sent) == 1
    parts = _FlattenParts(venue_qty=D(0)).flatten()
    assert parts.reads == ["BTCUSDT"] and parts.sent == []
