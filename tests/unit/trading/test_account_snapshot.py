"""R5 of the runtime extension (docs/arch/runtime-lab-strategies-plan.md): the exit side reads the account's positions
once per reconciliation for every symbol, instead of twice per symbol.

- The adapter's ``get_position_snapshot`` is one ``positionRisk`` read without a symbol: every row by symbol, flat rows
  included; a hedge-mode row, a repeated symbol, an unreadable quantity or answer raise. A symbol without a row is not
  in it: unknown, never flat - reconciliation and the stop manager then read that symbol alone, as before R5.
- The exit side consumes every symbol's fills and refreshes its orders first, then reads the snapshot, then reconciles
  and keeps the stop of each symbol with it. A snapshot that cannot be read leaves each symbol to read its own; one
  symbol whose fills cannot be consumed fails alone.
- Flattening and the manual tools are not changed: they read the symbol's position fresh.
"""
from __future__ import annotations

from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from trading.adapters.binance.adapter import BinanceAdapter, BinanceConfig
from trading.adapters.simulated.exchange import SimulatedExchange
from trading.application.exit_supervisor import ExitSupervisor
from trading.application.reconcile import reconcile_account
from trading.domain.issues import IssueKind as K
from trading.domain.strategy.overlay import StrategySpec


# -- the adapter ---------------------------------------------------------------------------------------------------
class Client:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def request_signed(self, *, method, path, params=None):
        self.calls.append((method, path, dict(params or {})))
        return self.rows


def _row(symbol, amt, side="BOTH"):
    return {"symbol": symbol, "positionAmt": amt, "entryPrice": "100", "markPrice": "101", "positionSide": side,
            "leverage": "10", "marginType": "isolated", "liquidationPrice": "50", "unRealizedProfit": "1"}


def _adapter(rows):
    client = Client(rows)
    return BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=client), client


def test_the_snapshot_is_one_read_of_every_row_flat_included():
    adapter, client = _adapter([_row("BTCUSDT", "0.5"), _row("ETHUSDT", "0"), _row("SOLUSDT", "-3")])
    snapshot = adapter.get_position_snapshot()
    assert client.calls == [("GET", "/fapi/v2/positionRisk", {})]
    assert {s: p.qty for s, p in snapshot.items()} == {"BTCUSDT": D("0.5"), "ETHUSDT": D(0), "SOLUSDT": D(-3)}
    assert snapshot["BTCUSDT"].entry_price == D(100) and snapshot["BTCUSDT"].margin_type == "isolated"
    assert "XRPUSDT" not in snapshot                     # no row: unknown, not flat


@pytest.mark.parametrize("rows,named", [
    ([_row("BTCUSDT", "1", side="LONG")], "Hedge"),
    ([_row("BTCUSDT", "1"), _row("BTCUSDT", "0")], "Hedge"),
    ([_row("BTCUSDT", "NaN")], "quantity"),
    ([{"positionAmt": "1"}], "snapshot"),
    ([{"symbol": "BTCUSDT", "positionSide": "BOTH", "entryPrice": "1"}], "quantity"),     # no quantity: unknown
    ({"code": -1}, "snapshot"),
])
def test_a_snapshot_that_cannot_be_trusted_raises(rows, named):
    adapter, _ = _adapter(rows)
    with pytest.raises(RuntimeError, match=named):
        adapter.get_position_snapshot()


def test_one_symbol_s_read_is_unchanged():
    adapter, client = _adapter([_row("BTCUSDT", "0")])
    assert adapter.get_position("BTCUSDT").qty == 0 and adapter.get_position("ETHUSDT") is None
    assert client.calls[0] == ("GET", "/fapi/v2/positionRisk", {"symbol": "BTCUSDT"})
    adapter, _ = _adapter([_row("BTCUSDT", "1", side="SHORT")])
    with pytest.raises(RuntimeError, match="Hedge"):
        adapter.get_position("BTCUSDT")


def test_the_simulated_venue_gives_every_symbol_flat_included():
    venue = SimulatedExchange.__new__(SimulatedExchange)
    broker = SimpleNamespace(position=SimpleNamespace(qty=D(0), entry_price=D(0)), mark=D(5))
    venue._brokers = {"BTCUSDT": broker, "ETHUSDT": broker}
    assert {s: p.qty for s, p in venue.get_position_snapshot().items()} == {"BTCUSDT": D(0), "ETHUSDT": D(0)}


# -- reconciliation with a snapshot -------------------------------------------------------------------------------
class Journal:
    def __init__(self, qty):
        self.qty = qty

    def checkpoint(self, owner):
        return {"position": {"qty": str(self.qty)}}

    def orders(self, owner=None, *, unresolved=False):
        return []

    def fills(self, owner):
        return []


class Account:
    def __init__(self, qty):
        self.qty, self.reads = qty, []

    def get_positions(self, symbol="", *, strict=True):
        self.reads.append(symbol)
        return (SimpleNamespace(symbol=symbol, qty=self.qty),) if self.qty else ()

    def get_open_orders(self, symbol="", *, strict=True):
        return ()


def test_reconciliation_uses_the_snapshot_row_and_reads_alone_without_one():
    account = Account(D(2))                              # what a read of the symbol alone would say
    snapshot = {"BTCUSDT": SimpleNamespace(symbol="BTCUSDT", qty=D(1))}
    assert reconcile_account(account, Journal(D(1)), owner="o", symbol="BTCUSDT", positions=snapshot) == []
    assert account.reads == []
    issues = reconcile_account(account, Journal(D(1)), owner="o", symbol="ETHUSDT", positions=snapshot)
    assert account.reads == ["ETHUSDT"] and [i.kind for i in issues] == [K.LEDGER_MISMATCH]
    issues = reconcile_account(account, Journal(D(2)), owner="o", symbol="BTCUSDT", positions=snapshot)
    assert [i.kind for i in issues] == [K.LEDGER_MISMATCH] and "venue qty=1, ledger qty=2" in issues[0]
    flat = {"BTCUSDT": SimpleNamespace(symbol="BTCUSDT", qty=D(0))}
    assert reconcile_account(account, Journal(D(0)), owner="o", symbol="BTCUSDT", positions=flat) == []
    assert reconcile_account(account, Journal(D(1)), owner="o", symbol="BTCUSDT", positions=None) != []


# -- the exit side ------------------------------------------------------------------------------------------------
class Session:
    def __init__(self, symbol, log, *, consume_fails=False):
        self.symbol, self.log, self.consume_fails = symbol, log, consume_fails
        self.executor = SimpleNamespace(notify_status=lambda: log.append(("notify", symbol)),
                                        reconcile_account=self._reconcile,
                                        block_entries=lambda: log.append(("blocked", symbol)))
        self.stops = SimpleNamespace(sync=self._sync, unprotected=lambda: False)

    def consume_fills(self):
        if self.consume_fails:
            raise RuntimeError("journal unreadable")
        self.log.append(("consume", self.symbol))

    def _reconcile(self, symbol, positions=None):
        self.log.append(("reconcile", symbol, positions))
        return []

    def _sync(self, positions=None):
        self.log.append(("stop", self.symbol, positions))
        return []


class SnapshotAccount:
    def __init__(self, log, fail=False):
        self.log, self.fail = log, fail

    def get_position_snapshot(self):
        self.log.append(("snapshot",))
        if self.fail:
            raise OSError("down")
        return {"AUSDT": "a", "BUSDT": "b"}


class Events:
    def __init__(self):
        self.rows = []

    def write(self, kind, **fields):
        self.rows.append((kind.value, fields))


def _exits(sessions, account):
    spec = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")
    return ExitSupervisor(sessions, spec=spec, journal=Events(), clock=lambda: 0.0, account=account, equity=None,
                          kill=None, drawdown=None, pauses=None, notifier=None, alert_prefix="",
                          market_refused=lambda: None, decision_error=lambda symbol: None, cycle_errors=lambda: [])


def test_the_exit_side_reads_the_positions_once_after_every_symbol_s_fills():
    log = []
    sessions = {s: Session(s, log) for s in ("AUSDT", "BUSDT")}
    _exits(sessions, SnapshotAccount(log)).reconcile()
    snap = {"AUSDT": "a", "BUSDT": "b"}
    assert log == [("consume", "AUSDT"), ("notify", "AUSDT"), ("consume", "BUSDT"), ("notify", "BUSDT"),
                   ("snapshot",),
                   ("reconcile", "AUSDT", snap), ("stop", "AUSDT", snap),
                   ("reconcile", "BUSDT", snap), ("stop", "BUSDT", snap)]


def test_without_a_snapshot_each_symbol_reads_its_own_and_every_entry_is_blocked():
    log = []
    sessions = {s: Session(s, log) for s in ("AUSDT", "BUSDT")}
    account = SnapshotAccount(log, fail=True)
    exits = _exits(sessions, account)
    exits.reconcile()
    assert [e for e in log if e[0] in ("reconcile", "stop")] == [("reconcile", "AUSDT", None), ("stop", "AUSDT", None),
                                                                ("reconcile", "BUSDT", None), ("stop", "BUSDT", None)]
    # R5 audit S2: not silent - the per-symbol reads weigh five times the snapshot each, so entries wait
    assert {s: [i.kind.value for i in found] for s, found in exits.last_issues.items()} == \
        {"AUSDT": ["positions_unread"], "BUSDT": ["positions_unread"]}
    assert {e[1] for e in log if e[0] == "blocked"} == {"AUSDT", "BUSDT"}
    exits.reconcile()
    assert len([1 for kind, _ in exits.journal.rows if kind == "issue"]) == 1               # written once
    account.fail = False
    exits.reconcile()
    assert exits.last_issues == {"AUSDT": [], "BUSDT": []} and exits.snapshot_issue is None


def test_a_symbol_whose_fills_cannot_be_consumed_fails_alone():
    log = []
    sessions = {"AUSDT": Session("AUSDT", log, consume_fails=True), "BUSDT": Session("BUSDT", log)}
    exits = _exits(sessions, SnapshotAccount(log))
    exits.reconcile()
    assert [i.kind for i in exits.last_issues["AUSDT"]] == [K.RECONCILE_FAILED]
    assert "journal unreadable" in exits.last_issues["AUSDT"][0] and exits.last_issues["BUSDT"] == []
    assert ("reconcile", "AUSDT", {"AUSDT": "a", "BUSDT": "b"}) not in log
    assert ("reconcile", "BUSDT", {"AUSDT": "a", "BUSDT": "b"}) in log


def test_without_an_account_there_is_no_snapshot():
    log = []
    _exits({"AUSDT": Session("AUSDT", log)}, None).reconcile()
    assert ("reconcile", "AUSDT", None) in log and ("snapshot",) not in log


# -- the stop manager ---------------------------------------------------------------------------------------------
def test_the_stop_manager_uses_the_snapshot_row_and_reads_alone_without_one():
    from trading.application.execution.stops import ProtectiveStopManager

    reads = []
    adapter = SimpleNamespace(get_position=lambda symbol: reads.append(symbol) or None)
    journal = SimpleNamespace(checkpoint=lambda owner: None, stop_rows=lambda owner, symbol, live_only=True: [])
    stops = ProtectiveStopManager(adapter, SimpleNamespace(journal=journal, owner="o"), "BTCUSDT",
                                  stop_pct=D("0.08"), clock=lambda: 0.0)
    seen = []
    stops._target = lambda ledger, venue: seen.append(venue)
    row = SimpleNamespace(symbol="BTCUSDT", qty=D(0), entry_price=D(0))
    assert stops.sync({"BTCUSDT": row}) == [] and reads == [] and seen == [row]
    assert stops.sync({"ETHUSDT": row}) == [] and reads == ["BTCUSDT"] and seen[-1] is None
    assert stops.sync() == [] and reads == ["BTCUSDT", "BTCUSDT"]
