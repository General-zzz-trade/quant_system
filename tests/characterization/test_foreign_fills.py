"""S4-S5 audit (2026-11-16): a foreign fill is a record, never a fill the strategy waits for.

Findings 1, 2, 4 and 10, and the surviving mutants A2, A3 and A7:
- it is booked under its own intent id, so it never counts as the order's fill (the order settles on what the venue
  filled) and never as a fill waiting to be consumed (the symbol keeps taking entries);
- a fill consumed already is never split again;
- its issue and alert are written at least once: again after a crash before the ledger's part was consumed, but
  only once while one process retries a consumption that keeps failing;
- the fee is split by quantity, and an entry that would cross zero is cut at zero too.
"""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from runner.binance_target import TargetFillConsumer
from tests.characterization.test_stops_and_fills import OWNER, _drain, _order, _with_position
from tests.characterization.test_venue_sized_exits import _active, _venue
from trading.domain.events import EventKind


class Events:
    def __init__(self, fail_first: bool = False) -> None:
        self.rows: list[tuple[EventKind, dict]] = []
        self.fail_first = fail_first

    def write(self, kind, **fields):
        if self.fail_first:
            self.fail_first = False
            raise SystemExit("the process died while writing the event")
        self.rows.append((kind, fields))

    def issues(self):
        return [f for k, f in self.rows if k is EventKind.ISSUE and f.get("issue_kind") == "foreign_fill"]

    def alerts(self):
        return [f for k, f in self.rows if k is EventKind.ALERT]


def _stop_fill_beyond_the_ledger(tmp_path, events=None):
    """Ledger 0.1 long at 100, a manual 0.05 on top; the stop closes the whole 0.15 (consumed unless it fails)."""
    fake, executor, _, stops = _with_position(tmp_path)
    events = events if events is not None else Events()
    consumer = TargetFillConsumer(executor, events)
    _venue(fake, "0.15", "100")
    stops.sync()
    (client_id,) = _active(fake)
    fake.trigger_stop(client_id, D("95.9"))
    stops.sync()
    return fake, executor, consumer, events


def _stop_row(executor):
    (row,) = [r for r in executor.journal.orders(OWNER) if r["reducing"]]
    return row


def test_after_a_foreign_fill_the_symbol_still_takes_entries(tmp_path):
    fake, executor, consumer, events = _stop_fill_beyond_the_ledger(tmp_path)
    _drain(executor, consumer)
    assert consumer.position.qty == 0
    posts = len(fake.posts)
    executor.send_order(_order(5, "buy", "0.1"))
    assert len(fake.posts) == posts + 1           # until the audit: refused, "a fill of the symbol is not consumed"


def test_the_foreign_part_never_counts_as_the_order_s_fill(tmp_path):
    fake, executor, consumer, events = _stop_fill_beyond_the_ledger(tmp_path)
    _drain(executor, consumer)
    fills = executor.journal.intent_fills(_stop_row(executor)["intent_id"])
    assert sum(D(f["qty"]) for f in fills) == D("0.15")        # what the venue filled, counted once
    (foreign,) = executor.journal.fills("foreign:BTCUSDT")
    assert D(foreign["qty"]) == D("0.05")


def test_the_fee_is_split_by_quantity_and_the_issue_comes_with_an_alert(tmp_path):
    fake, executor, consumer, events = _stop_fill_beyond_the_ledger(tmp_path)
    _drain(executor, consumer)
    (whole,) = executor.journal.intent_fills(_stop_row(executor)["intent_id"])
    (foreign,) = executor.journal.fills("foreign:BTCUSDT")
    assert D(whole["fee"]) != 0
    assert D(foreign["fee"]) == D(whole["fee"]) * D("0.05") / D("0.15")
    assert len(events.issues()) == 1 and len(events.alerts()) == 1
    assert "foreign" in events.alerts()[0]["reason"]


def test_a_fill_consumed_already_is_never_split_again(tmp_path):
    fake, executor, _, stops = _with_position(tmp_path)
    events = Events()
    consumer = TargetFillConsumer(executor, events)
    fills = list(executor.send_order(_order(2, "sell", "0.1", reduce_only=True)))
    assert fills and all(consumer.consume(e) for e in fills) and consumer.position.qty == 0
    assert not any(consumer.consume(e) for e in fills)                   # delivered a second time
    assert executor.journal.fills("foreign:BTCUSDT") == [] and events.rows == []


def test_the_alert_is_written_again_after_a_crash_before_the_ledger_s_part_was_consumed(tmp_path):
    fake, executor, consumer, events = _stop_fill_beyond_the_ledger(tmp_path, Events(fail_first=True))
    with pytest.raises(SystemExit):
        _drain(executor, consumer)                     # the foreign part is stored, then the process dies
    after = Events()
    restarted = TargetFillConsumer(executor, after)
    _drain(executor, restarted)
    assert restarted.position.qty == 0 and len(executor.journal.fills("foreign:BTCUSDT")) == 1
    assert len(after.issues()) == 1 and len(after.alerts()) == 1


def test_a_consumption_that_keeps_failing_alerts_once_per_process(tmp_path, monkeypatch):
    fake, executor, consumer, events = _stop_fill_beyond_the_ledger(tmp_path)

    def locked(*args, **kwargs):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(executor.journal, "consume", locked)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            _drain(executor, consumer)
    assert len(events.issues()) == 1 and len(events.alerts()) == 1


def test_an_entry_that_would_cross_zero_is_cut_at_zero(tmp_path):
    fake, executor, _, stops = _with_position(tmp_path)                 # ledger 0.1 long
    events = Events()
    consumer = TargetFillConsumer(executor, events)
    for event in executor.send_order(_order(2, "sell", "0.15")):         # not a reduction
        consumer.consume(event)
    assert consumer.position.qty == 0
    (foreign,) = executor.journal.fills("foreign:BTCUSDT")
    assert D(foreign["qty"]) == D("0.05") and len(events.issues()) == 1
