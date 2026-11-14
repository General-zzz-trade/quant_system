"""Evidence for the order status table (step-6 audit M1, written before step 9 moves the executor).

The table is pinned as a literal, and each allowed transition is either produced by a scenario below
through the real ConfirmedExecutor, or listed in ALLOWED_WITHOUT_SCENARIO with the reason it is kept.
"""
from __future__ import annotations

from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from shared import confirmed_execution
from shared.confirmed_execution import ConfirmedExecutor
from trading.domain.orders import TRANSITIONS

SCOPE = "binance-testnet-ut"
SETTLED = {"filled", "canceled", "mmp_canceled", "rejected", "expired", "not_found"}
PINNED = {
    None: {"reserved", "submitted"},
    "reserved": {"submitted", "rejected_unconfirmed", "unknown"} | SETTLED,
    "submitted": {"unknown"} | SETTLED,
    "unknown": {"rejected_unconfirmed"} | SETTLED,
    "rejected_unconfirmed": {"unknown"} | SETTLED,
}
# Allowed because a venue may report any terminal state for any open order, but no scenario here
# drives that exact pair: (from, to) -> why it stays in the table.
ALLOWED_WITHOUT_SCENARIO = {
    (None, "reserved"): "row insert, checked by tests/contract/test_order_journal_sql.py",
    (None, "submitted"): "adopted stop insert, checked by tests/contract/test_order_journal_sql.py",
    ("reserved", "canceled"): "crash after placement; the venue later reports the order canceled",
    ("reserved", "expired"): "same, venue reports expired",
    ("reserved", "mmp_canceled"): "same, venue reports an MMP cancel",
    ("reserved", "rejected"): "crash after a rejected placement; the lookup confirms the rejection",
    ("submitted", "mmp_canceled"): "venue MMP cancel of a resting order",
    ("submitted", "rejected"): "venue reports a placed order as rejected",
    ("submitted", "not_found"): "only for an order without an order id; submitted rows normally have one",
    ("unknown", "mmp_canceled"): "venue MMP cancel after an ambiguous placement",
    ("unknown", "rejected"): "venue reports rejected after an ambiguous placement",
    ("rejected_unconfirmed", "filled"): "the venue rejected, but the order exists and filled",
    ("rejected_unconfirmed", "canceled"): "same, canceled",
    ("rejected_unconfirmed", "expired"): "same, expired",
    ("rejected_unconfirmed", "mmp_canceled"): "same, MMP cancel",
    ("rejected_unconfirmed", "not_found"): "a confirmed rejection always settles as rejected first",
}


class Venue:
    """Scripted venue: each method pops its next scripted result (an exception is raised)."""

    def __init__(self, *, send, lookups, fills=None):
        self.send, self.lookups, self.fills_script, self.client_id = list(send), list(lookups), fills or [], ""

    def _next(self, script):
        item = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(item, Exception):
            raise item
        return item

    def get_ticker(self, symbol):
        return {"bid1Price": "100", "ask1Price": "100"}

    def send_market_order(self, symbol, side, qty, *, reduce_only=False, client_order_id=""):
        self.client_id = client_order_id
        return self._next(self.send)

    send_limit_order = None

    def get_order(self, symbol, *, order_id="", client_order_id=""):
        state = self._next(self.lookups)
        if state is None:
            return None
        return SimpleNamespace(order_id="7", client_order_id=self.client_id, symbol="BTCUSDT", side="buy",
                               filled_qty=state[1], state=state[0])

    def get_recent_fills(self, *, symbol=None, order_id="", since_ms=0, limit=100):
        return tuple(SimpleNamespace(order_id="7", symbol="BTCUSDT", side="buy", trade_id=f"t{n}", qty=q,
                                     price="100", fee="0", fee_asset="USDT") for n, q in enumerate(self.fills_script))

    def cancel_order(self, symbol, order_id):
        return {}


SUBMITTED, REJECTED = {"status": "submitted", "orderId": "7"}, {"status": "rejected", "code": -2010}
DOWN = TimeoutError("venue unreachable")
SCENARIOS = {
    "fill": (dict(send=[SUBMITTED], lookups=[("filled", "1")], fills=["1"]), 1,
             {("reserved", "submitted"), ("submitted", "filled")}),
    "ambiguous then fill": (dict(send=[DOWN], lookups=[None, ("filled", "1")], fills=["1"]), 2,
                            {("reserved", "unknown"), ("unknown", "filled")}),
    "rejection confirmed": (dict(send=[REJECTED], lookups=[None]), 1,
                            {("reserved", "rejected_unconfirmed"), ("rejected_unconfirmed", "rejected")}),
    "lookup fails after submit, then cancel": (dict(send=[SUBMITTED], lookups=[DOWN, ("canceled", "0")]), 2,
                                                {("reserved", "submitted"), ("submitted", "unknown"),
                                                 ("unknown", "canceled")}),
    "rejected with an order id, lookups flap": (
        dict(send=[{"status": "rejected", "orderId": "7"}], lookups=[DOWN, None, DOWN, ("expired", "0")]), 4,
        {("reserved", "rejected_unconfirmed"), ("rejected_unconfirmed", "unknown"),
         ("unknown", "rejected_unconfirmed"), ("unknown", "expired")}),
    "submitted then expired": (dict(send=[SUBMITTED], lookups=[("expired", "0")]), 1,
                               {("reserved", "submitted"), ("submitted", "expired")}),
    "submitted then canceled": (dict(send=[SUBMITTED], lookups=[("canceled", "0")]), 1,
                                {("reserved", "submitted"), ("submitted", "canceled")}),
    "ambiguous, never placed": (dict(send=[DOWN], lookups=[None]), 2, {("reserved", "unknown"),
                                                                       ("unknown", "not_found")}),
    "ambiguous then canceled": (dict(send=[DOWN], lookups=[("canceled", "0")]), 1,
                                {("reserved", "unknown"), ("unknown", "canceled")}),
}


@pytest.fixture
def recorded(monkeypatch):
    seen: list[tuple[str | None, str]] = []
    original = confirmed_execution.check_transition

    def record(current, new):
        if current != new:
            seen.append((current, new))
        original(current, new)

    monkeypatch.setattr(confirmed_execution, "check_transition", record)
    return seen


def _run(tmp_path, venue_kw, polls):
    clock = {"t": 1000.0}
    executor = ConfirmedExecutor(Venue(**venue_kw), tmp_path / "orders.sqlite", SCOPE, wait_seconds=0, owner="o",
                                 entry_order_type="market", not_found_grace_seconds=90, clock=lambda: clock["t"])
    executor.submit(intent_id="i1", symbol="BTCUSDT", side="buy", qty=D("1"))
    for _ in range(polls - 1):
        clock["t"] += 100                                   # past the 90 s grace
        executor.poll()
    return executor


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario_produces_exactly_its_transitions(tmp_path, recorded, name):
    venue_kw, polls, expected = SCENARIOS[name]
    _run(tmp_path, venue_kw, polls)
    assert set(recorded) == expected


@pytest.mark.parametrize("lookup,expected", [(None, ("reserved", "not_found")),
                                             (("filled", "1"), ("reserved", "filled"))])
def test_restart_with_a_reserved_row(tmp_path, recorded, lookup, expected):
    """Crash after the reservation: absent after the grace -> not_found; placed and filled -> filled."""
    journal_path = tmp_path / "orders.sqlite"
    clock = {"t": 1000.0}

    def executor():
        return ConfirmedExecutor(Venue(send=[SUBMITTED], lookups=[lookup], fills=["1"] if lookup else []),
                                 journal_path, SCOPE, wait_seconds=0, owner="o", entry_order_type="market",
                                 not_found_grace_seconds=90, clock=lambda: clock["t"])

    executor().journal.reserve(intent_id="i1", symbol="BTCUSDT", client_id="c1", side="buy", qty="1",
                               reducing=False, owner="o", parent=None, order_type="market",
                               clock=lambda: clock["t"])          # the process dies here
    clock["t"] += 100
    restarted = executor()
    restarted.adapter.client_id = "c1"
    restarted.poll()
    assert recorded == [expected]


def test_the_table_is_pinned_and_every_transition_has_evidence_or_a_reason():
    assert {k: set(v) for k, v in TRANSITIONS.items()} == PINNED
    produced = set().union(*(expected for _, _, expected in SCENARIOS.values())) | {("reserved", "not_found"),
                                                                                    ("reserved", "filled")}
    allowed = {(k, v) for k, targets in PINNED.items() for v in targets}
    assert produced <= allowed
    assert allowed - produced == set(ALLOWED_WITHOUT_SCENARIO)
