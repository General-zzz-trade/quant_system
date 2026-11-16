"""ADR-0014 (S3a): a problem found during a cycle is a typed issue, and what it does to trading comes from one
fixed table. The text is for people; no trading decision reads it (tests/architecture/test_issue_rules.py)."""
from __future__ import annotations

from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from trading.application.alerts import ALERT_AFTER_SECONDS, IssueAlerts
from trading.application.issues import Issue
from trading.application.reconcile import reconcile_account
from trading.domain.issues import EFFECTS, Effect, IssueKind

E = Effect
BLOCK = frozenset({E.BLOCK_SYMBOL_ENTRIES})


def test_the_table_is_exactly_the_decided_one():
    # Adding, removing or changing any row fails here first (ADR-0014 acceptance).
    assert EFFECTS == {
        IssueKind.STOP_BREACHED: frozenset({E.FLATTEN_SYMBOL, E.PAUSE_SYMBOL, E.BLOCK_SYMBOL_ENTRIES}),
        IssueKind.STOP_TRIGGERED: frozenset({E.PAUSE_SYMBOL, E.BLOCK_SYMBOL_ENTRIES}),
        IssueKind.STOP_MISSING: BLOCK,
        IssueKind.STOP_UNCERTAIN: BLOCK,
        IssueKind.LEDGER_MISMATCH: BLOCK,
        IssueKind.FOREIGN_HOLDING: frozenset({E.BLOCK_ALL_ENTRIES, E.CANCEL_RESTING_ENTRIES}),    # ADR-0011 (S5a)
        IssueKind.UNRESOLVED_ORDER: BLOCK,
        IssueKind.UNDELIVERED_FILL: BLOCK,
        IssueKind.RECONCILE_FAILED: BLOCK,
        IssueKind.VENUE_BLOCKED: BLOCK,
        IssueKind.PAUSE_NOT_SAVED: BLOCK,
        IssueKind.MARKET_REFUSED: frozenset({E.BLOCK_ALL_ENTRIES}),
        IssueKind.MARKET_DATA: frozenset({E.ALERT_ONLY}),
        IssueKind.FLATTEN_FAILED: frozenset({E.ALERT_ONLY}),
        IssueKind.CANCEL_FAILED: frozenset({E.ALERT_ONLY}),
        IssueKind.DECISION_FAILED: BLOCK,                               # S3c: one symbol's bar failed
        IssueKind.CYCLE_FAILED: frozenset({E.BLOCK_ALL_ENTRIES}),       # S3c: a whole block of the cycle failed
        IssueKind.EVENT_LOG_FAILED: frozenset({E.BLOCK_ALL_ENTRIES}),   # S3 audit 3: the event log is not writable
        IssueKind.ACCOUNT_UNREADABLE: frozenset({E.BLOCK_ALL_ENTRIES, E.CANCEL_RESTING_ENTRIES}),  # ADR-0011
        IssueKind.ACCOUNT_SETTINGS: frozenset({E.BLOCK_ALL_ENTRIES, E.CANCEL_RESTING_ENTRIES}),    # ADR-0011
        IssueKind.FOREIGN_FILL: frozenset({E.ALERT_ONLY}),             # ADR-0012: a stop fill beyond the ledger
    }
    assert set(EFFECTS) == set(IssueKind)                         # every kind has a row
    assert all(EFFECTS[k] for k in IssueKind)                      # and at least one effect
    assert not any(E.ALERT_ONLY in v and len(v) > 1 for v in EFFECTS.values())


def test_an_issue_keeps_its_text_and_carries_its_kind_symbol_and_effects():
    issue = Issue(IssueKind.UNDELIVERED_FILL, "BTCUSDT", "BTCUSDT: undelivered fills")
    assert issue == "BTCUSDT: undelivered fills"                    # people (and the event log) see the same text
    assert (issue.kind, issue.symbol, issue.effects) == (IssueKind.UNDELIVERED_FILL, "BTCUSDT", BLOCK)
    with pytest.raises(TypeError):
        Issue("not a kind", "BTCUSDT", "x")                          # type: ignore[arg-type]


class Account:
    def __init__(self, *, positions=(), orders=(), fail=False):
        self.positions, self.orders, self.fail = positions, orders, fail

    def get_positions(self, symbol="", *, strict=True):
        if self.fail:
            raise ConnectionError("down")
        return self.positions

    def get_open_orders(self, symbol="", *, strict=True):
        return self.orders


class Journal:
    def __init__(self, *, unresolved=False, fills=()):
        self.unresolved, self._fills = unresolved, fills

    def orders(self, owner=None, *, unresolved=False):
        rows = [{"owner": "o", "symbol": "BTCUSDT", "order_id": "1", "intent_id": "i"}]
        return rows if (self.unresolved or not unresolved) else []

    def checkpoint(self, owner):
        return {"position": {"qty": "0", "entry_price": "100"}}

    def fills(self, owner=None):
        return list(self._fills)


def _pos(qty):
    return SimpleNamespace(symbol="BTCUSDT", qty=D(qty))


@pytest.mark.parametrize("account,journal,kind", [
    (Account(positions=(_pos("0.1"),)), Journal(), IssueKind.LEDGER_MISMATCH),
    (Account(positions=(_pos("0.1"), _pos("-0.1"))), Journal(), IssueKind.LEDGER_MISMATCH),
    (Account(), Journal(unresolved=True), IssueKind.UNRESOLVED_ORDER),
    (Account(orders=(SimpleNamespace(order_id="99"),)), Journal(), IssueKind.FOREIGN_HOLDING),
    (Account(), Journal(fills=({"fill_id": "f"},)), IssueKind.UNDELIVERED_FILL),
    (Account(fail=True), Journal(), IssueKind.RECONCILE_FAILED),
])
def test_reconciliation_names_the_kind_of_each_problem(account, journal, kind):
    issues = reconcile_account(account, journal, owner="o", symbol="BTCUSDT")
    assert issues and all(isinstance(i, Issue) and i.symbol == "BTCUSDT" for i in issues)
    assert {i.kind for i in issues} == {kind}


def test_alerts_follow_the_kind_not_the_wording():
    # The ledger quantity in the text changes while the mismatch lasts: it is still one problem.
    clock, sent = {"t": 0.0}, []
    notifier = SimpleNamespace(alert=lambda level, title, detail="": sent.append((level, title, detail)))
    alerts = IssueAlerts(notifier, lambda: clock["t"], prefix="p")
    for t, qty in ((0, "0.1"), (60, "0.2"), (ALERT_AFTER_SECONDS, "0.3")):
        clock["t"] = t
        text = f"BTCUSDT: venue qty={qty}, ledger qty=0"
        alerts.observe("BTCUSDT", [Issue(IssueKind.LEDGER_MISMATCH, "BTCUSDT", text)])
    assert sent == [("warning", "[p] BTCUSDT", "BTCUSDT: venue qty=0.3, ledger qty=0 (for 2 min)")]


def test_a_kind_missing_from_the_table_blocks_every_entry(monkeypatch):
    # ADR-0014 "state unknown": an unregistered kind fails the table test above, and in production blocks all.
    from trading.domain import issues as table
    monkeypatch.delitem(table.EFFECTS, IssueKind.LEDGER_MISMATCH)
    assert Issue(IssueKind.LEDGER_MISMATCH, "BTCUSDT", "x").effects == frozenset({E.BLOCK_ALL_ENTRIES})
