"""Where two ``ModePolicy`` fields act below the session (S10 part 3): the order lifecycle's expiry of a resting
entry (``entry_ttl``) and the order journal's flip rule (``entry_beside_working_reduction``, S10 decision 1).
Execution's behaviour is the default of both and is unchanged."""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from tests.support.state_db import state_db
from tests.unit.trading.test_order_lifecycle import Lock, Venue
from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.application.execution.lifecycle import OrderLifecycle
from trading.domain.orders import ENTRY_UNRESOLVED

SCOPE = "binance-testnet-ut"


# -- entry_ttl: "venue" --------------------------------------------------------------------------------------------
def _resting_entry(tmp_path, **kw):
    events, clock = [], {"t": 1000.0}
    venue = Venue(events, state="new")
    lifecycle = OrderLifecycle(venue, SqliteOrderJournal(state_db(tmp_path / "state.sqlite"), SCOPE), SCOPE,
                               clock=lambda: clock["t"], sleeper=lambda s: None, lock=Lock(events),
                               owner="o", entry_order_type="limit", entry_ttl_seconds=60, **kw)
    lifecycle.submit(intent_id="i1", symbol="BTCUSDT", side="buy", qty=D("1"), limit_price=D("99"))
    return lifecycle, venue, events, clock


def test_when_the_venue_expires_entries_the_lifecycle_never_cancels_one_for_its_age(tmp_path):
    lifecycle, venue, events, clock = _resting_entry(tmp_path, venue_expires_entries=True)
    for t in (1060.0, 1000.0 + 10 * 3600):
        clock["t"] = t
        lifecycle.poll()
    assert "cancel" not in events and "i1" in lifecycle.pending
    venue.state = "canceled"                               # the venue expired it after matching its last bar
    lifecycle.poll()
    assert lifecycle.pending == {} and lifecycle.journal.orders("o")[0]["status"] == "canceled"


def test_by_default_the_lifecycle_cancels_a_resting_entry_once_its_lifetime_has_passed(tmp_path):
    lifecycle, venue, events, clock = _resting_entry(tmp_path)
    clock["t"] = 1060.0
    lifecycle.poll()
    assert "cancel" in events and lifecycle.pending == {}


def test_a_venue_expiry_does_not_stop_a_reduction_from_cancelling_the_entry(tmp_path):
    lifecycle, venue, events, clock = _resting_entry(tmp_path, venue_expires_entries=True)
    lifecycle.cancel_entries("BTCUSDT")                  # a reduction or an exit still takes the entry off
    assert "cancel" in events and lifecycle.pending == {}


# -- entry_beside_working_reduction ---------------------------------------------------------------------------------
def _journal(tmp_path, flag):
    return SqliteOrderJournal(state_db(tmp_path / "state.sqlite"), SCOPE, entry_beside_working_reduction=flag)


def _reserve(journal, intent_id, *, reducing, symbol="BTCUSDT"):
    return journal.reserve(intent_id=intent_id, symbol=symbol, client_id=f"c-{intent_id}", side="sell", qty="1",
                           reducing=reducing, owner="o", parent=None, order_type="market" if reducing else "limit",
                           clock=lambda: 1000.0)[0]


@pytest.mark.parametrize("flag,expected", [(True, ""), (False, ENTRY_UNRESOLVED)])
def test_the_entry_of_a_flip_is_reserved_beside_its_acknowledged_reduction_only_when_allowed(tmp_path, flag,
                                                                                               expected):
    journal = _journal(tmp_path, flag)
    assert _reserve(journal, "reduce", reducing=True) == ""
    journal.update_status("reduce", "submitted", settle=False)        # the venue acknowledged it; it rests
    assert _reserve(journal, "entry", reducing=False) == expected


@pytest.mark.parametrize("status", ["reserved", "unknown", "partially_filled"])
def test_a_reduction_the_venue_has_not_acknowledged_still_refuses_the_entry(tmp_path, status):
    journal = _journal(tmp_path, True)
    _reserve(journal, "reduce", reducing=True)
    if status != "reserved":
        journal.update_status("reduce", status, settle=False)
    assert _reserve(journal, "entry", reducing=False) == ENTRY_UNRESOLVED


def test_an_unsettled_entry_still_refuses_another_entry_with_the_flip_rule(tmp_path):
    journal = _journal(tmp_path, True)
    _reserve(journal, "entry1", reducing=False)
    journal.update_status("entry1", "submitted", settle=False)
    assert _reserve(journal, "entry2", reducing=False) == ENTRY_UNRESOLVED
    assert _reserve(journal, "other", reducing=False, symbol="ETHUSDT") == ""     # per symbol, as before


def test_the_flip_rule_does_not_let_a_second_reduction_overlap(tmp_path):
    journal = _journal(tmp_path, True)
    _reserve(journal, "reduce1", reducing=True)
    journal.update_status("reduce1", "submitted", settle=False)
    assert _reserve(journal, "reduce2", reducing=True) != ""
