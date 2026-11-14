"""OrderLifecycle driven by an injected clock, sleeper and lock (migration step 9 acceptance).

The legacy executor slept for real between placement and lookup; here the sleep is a recorded call, so
the order "place, wait, then look up" is asserted without waiting.
"""
from __future__ import annotations

import math
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.application.execution.lifecycle import OrderLifecycle

SCOPE = "binance-testnet-ut"


class Venue:
    def __init__(self, events, *, state="filled"):
        self.events, self.state, self.client_id = events, state, ""

    def get_ticker(self, symbol):
        self.events.append("ticker")
        return {"bid1Price": "100", "ask1Price": "101"}

    def send_market_order(self, symbol, side, qty, *, reduce_only=False, client_order_id=""):
        self.events.append(("market", reduce_only))
        self.client_id = client_order_id
        return {"status": "submitted", "orderId": "7"}

    def send_limit_order(self, symbol, side, qty, price, *, post_only=True, reduce_only=False, client_order_id=""):
        self.events.append(("limit", price))
        self.client_id = client_order_id
        return {"status": "submitted", "orderId": "7"}

    def get_order(self, symbol, *, order_id="", client_order_id=""):
        self.events.append("lookup")
        filled = "1" if self.state == "filled" else "0"
        return SimpleNamespace(order_id="7", client_order_id=self.client_id, symbol="BTCUSDT", side="buy",
                               state=self.state, filled_qty=filled)

    def get_recent_fills(self, *, symbol=None, order_id="", since_ms=0, limit=100):
        self.events.append("fills")
        if self.state != "filled":
            return ()
        return (SimpleNamespace(order_id="7", symbol="BTCUSDT", side="buy", trade_id="t1", qty="1", price="100",
                                fee="0", fee_asset="USDT"),)

    def cancel_order(self, symbol, order_id):
        self.events.append("cancel")
        self.state = "canceled"
        return {}


class Lock:
    def __init__(self, events):
        self.events, self.depth = events, 0

    def __enter__(self):
        self.depth += 1
        self.events.append("lock")
        return True

    def __exit__(self, *exc):
        self.depth -= 1
        self.events.append("unlock")
        return None


def _lifecycle(tmp_path, events, clock, *, wait_seconds=3.0, state="filled", **kw):
    lock = Lock(events)
    lifecycle = OrderLifecycle(Venue(events, state=state), SqliteOrderJournal(tmp_path / "orders.sqlite", SCOPE),
                               SCOPE, clock=lambda: clock["t"], sleeper=lambda s: events.append(("sleep", s)),
                               lock=lock, wait_seconds=wait_seconds, owner="o", **kw)
    return lifecycle, lock


def test_a_market_order_is_placed_then_waited_on_then_looked_up_under_the_lock(tmp_path):
    events, clock = [], {"t": 1000.0}
    lifecycle, lock = _lifecycle(tmp_path, events, clock, entry_order_type="market")
    facts = lifecycle.submit(intent_id="i1", symbol="BTCUSDT", side="buy", qty=D("1"))
    assert events == ["lock", ("market", False), ("sleep", 3.0), "lookup", "fills", "unlock"]
    assert [(f.trade_id, f.qty) for f in facts] == [("t1", D("1"))]
    assert lock.depth == 0 and lifecycle.pending == {}
    assert lifecycle.journal.orders("o")[0]["created_at"] == 1000.0      # the reservation used the injected clock


def test_no_wait_means_no_sleep_call(tmp_path):
    events, clock = [], {"t": 1000.0}
    lifecycle, _ = _lifecycle(tmp_path, events, clock, wait_seconds=0, entry_order_type="market")
    lifecycle.submit(intent_id="i1", symbol="BTCUSDT", side="buy", qty=D("1"))
    assert not [e for e in events if isinstance(e, tuple) and e[0] == "sleep"]


def test_a_resting_entry_is_canceled_only_once_its_ttl_has_passed_on_the_injected_clock(tmp_path):
    events, clock = [], {"t": 1000.0}
    lifecycle, lock = _lifecycle(tmp_path, events, clock, state="new", entry_order_type="limit",
                                 entry_ttl_seconds=60)
    lifecycle.submit(intent_id="i1", symbol="BTCUSDT", side="buy", qty=D("1"), limit_price=D("99"))
    assert events == ["lock", ("limit", 99.0), ("sleep", 3.0), "lookup", "fills", "unlock"]
    clock["t"] = 1059.9
    events.clear()
    lifecycle.poll()
    assert events == ["lock", "lookup", "fills", "unlock"] and "i1" in lifecycle.pending
    clock["t"] = 1060.0
    events.clear()
    lifecycle.poll()
    assert events == ["lock", "lookup", "cancel", "lookup", "fills", "unlock"]
    assert lifecycle.pending == {} and lock.depth == 0
    assert lifecycle.journal.orders("o")[0]["status"] == "canceled"


@pytest.mark.parametrize("scope,bad", [(SCOPE, {"wait_seconds": -1.0}), (SCOPE, {"wait_seconds": math.inf}),
                                       (SCOPE, {"entry_order_type": "stop"}),
                                       (SCOPE, {"not_found_grace_seconds": 9.0}), (SCOPE, {"entry_ttl_seconds": 0.0}),
                                       ("", {})])
def test_invalid_settings_are_refused_before_anything_is_built(scope, bad):
    settings = {"wait_seconds": 3.0, "entry_order_type": "limit", "not_found_grace_seconds": None,
                "entry_ttl_seconds": None, **bad}
    with pytest.raises(ValueError):
        OrderLifecycle.check_settings(scope, **settings)
