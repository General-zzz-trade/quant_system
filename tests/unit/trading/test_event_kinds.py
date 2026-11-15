"""ADR-0014 point 2 (S3b): event kinds are one enum and the event log accepts nothing else; every order and stop
status change is an event (data and state architecture decision, 2026-11-15)."""
from __future__ import annotations

import json

import pytest

from tests.support.state_db import state_db
from trading.adapters.files.event_log import JsonlEventLog
from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.application.execution.recorded_journal import RecordedOrderJournal
from trading.domain.events import EventKind

SCOPE = "binance-testnet-ut"


def test_the_event_kinds_keep_the_words_the_log_has_always_used():
    assert {k.name: k.value for k in EventKind} == {
        "WARMUP": "warmup", "INTENT": "intent", "FILL": "fill", "BLOCKED": "blocked", "ISSUE": "issue",
        "ALERT": "alert", "PAPER_EQUITY": "paper_equity", "PAPER_FILL": "paper_fill", "PAPER_RESET": "paper_reset",
        "ORDER": "order", "STOP": "stop", "SYMBOL_STATE": "symbol_state",      # S3d: every symbol state change
        "CONFIG": "config"}                                                  # S4: the configuration at the start


def test_the_event_log_writes_the_kind_word_and_refuses_anything_but_a_kind(tmp_path):
    log = JsonlEventLog(tmp_path / "events.jsonl")
    log.write(EventKind.INTENT, symbol="BTCUSDT", order_id="i1")
    with pytest.raises(TypeError):
        log.write("intent", symbol="BTCUSDT")                       # type: ignore[arg-type]
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [(r["kind"], r["order_id"]) for r in rows] == [("intent", "i1")]


class Events:
    def __init__(self, fail=False):
        self.rows, self.fail = [], fail

    def write(self, kind, **fields):
        if self.fail:
            raise OSError("disk full")
        self.rows.append((kind, fields))


def _journal(tmp_path, events):
    return RecordedOrderJournal(SqliteOrderJournal(state_db(tmp_path / "state.sqlite"), SCOPE), events)


def _reserve(journal, intent_id="i1"):
    return journal.reserve(intent_id=intent_id, symbol="BTCUSDT", client_id=f"c-{intent_id}", side="buy", qty="0.1",
                           reducing=False, owner="target.BTCUSDT", parent={"price": "100"}, order_type="market",
                           clock=lambda: 1000.0)


def test_every_order_status_change_is_an_event(tmp_path):
    events = Events()
    journal = _journal(tmp_path, events)
    assert _reserve(journal)[0] == ""
    journal.set_order_id("i1", "777")
    journal.update_status("i1", "partially_filled", settle=False)
    journal.update_status("i1", "partially_filled", settle=False)        # no change: no event
    journal.update_status("i1", "filled", settle=True)
    journal.update_status("i1", "filled", settle=True)                   # already settled: no change
    orders = [(f["order_id"], f["status"], f.get("previous"), f.get("settled")) for k, f in events.rows
              if k is EventKind.ORDER]
    assert orders == [("i1", "reserved", None, None), ("i1", "acknowledged", None, None),
                      ("i1", "partially_filled", "reserved", False), ("i1", "filled", "partially_filled", True)]
    assert all(k is EventKind.ORDER for k, _ in events.rows)
    assert journal.orders()[0]["status"] == "filled"                   # the journal itself is unchanged


def test_a_refused_reservation_writes_no_order_event(tmp_path):
    events = Events()
    journal = _journal(tmp_path, events)
    _reserve(journal)
    assert _reserve(journal, "i2")[0] != ""                            # an unresolved entry refuses the next
    assert [f["order_id"] for _, f in events.rows] == ["i1"]


def test_stop_status_changes_are_events(tmp_path):
    events = Events()
    journal = _journal(tmp_path, events)
    journal.add_stop(owner="target.BTCUSDT", client_algo_id="s1", symbol="BTCUSDT", revision="r", attempt=0,
                     side="sell", trigger_price="90", created_at=1000.0)
    journal.set_stop_status("s1", "active")
    journal.set_stop_status("s1", "triggered")
    assert [(k, f["client_algo_id"], f["status"]) for k, f in events.rows] == [
        (EventKind.STOP, "s1", "unknown"), (EventKind.STOP, "s1", "active"), (EventKind.STOP, "s1", "triggered")]


def test_a_failing_event_log_never_fails_the_order_path(tmp_path):
    journal = _journal(tmp_path, Events(fail=True))
    assert _reserve(journal)[0] == ""
    journal.set_order_id("i1", "777")
    assert journal.update_status("i1", "filled", settle=True) == "reserved"
    assert journal.orders()[0]["settled"] == 1


def test_settling_without_a_new_status_is_still_an_event(tmp_path):
    events = Events()
    journal = _journal(tmp_path, events)
    _reserve(journal)
    journal.update_status("i1", "filled", settle=False)
    journal.update_status("i1", "filled", settle=True)                   # only the settled flag changes
    assert [(f["status"], f["previous"], f["settled"]) for k, f in events.rows if "previous" in f] == [
        ("filled", "reserved", False), ("filled", "filled", True)]


def test_adopting_a_venue_order_is_an_event(tmp_path):
    # S3 audit finding 7: a stop that became an exchange order is adopted into the journal; that is a change too.
    events = Events()
    journal = _journal(tmp_path, events)
    journal.adopt(intent_id="stop-fill-1", symbol="BTCUSDT", client_id="c-adopt", order_id="888", side="sell",
                  owner="target.BTCUSDT", parent={"price": "0"}, created_at=1000.0)
    assert [(k, f["order_id"], f["status"], f["venue_order_id"]) for k, f in events.rows] == [
        (EventKind.ORDER, "stop-fill-1", "adopted", "888")]
