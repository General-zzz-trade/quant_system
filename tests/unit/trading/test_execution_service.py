"""ExecutionService driven by order intents, with every collaborator injected (migration step 9d)."""
from __future__ import annotations

import json
import threading
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.application.execution.lifecycle import OrderLifecycle
from trading.application.execution.service import ExecutionService, SendResult
from trading.domain.orders import OrderIntent

SCOPE, OWNER = "binance-testnet-ut", "target.BTCUSDT"


class Venue:
    """Market orders fill at 100 at once; post-only limits rest until canceled."""

    def __init__(self):
        self.calls, self.orders, self.position, self.blockers = [], {}, D(0), []

    def execution_blockers(self):
        return list(self.blockers)

    def get_positions(self, symbol="", *, strict=True):
        return (SimpleNamespace(symbol="BTCUSDT", qty=self.position),)

    def get_open_orders(self, symbol="", *, strict=True):
        return tuple(SimpleNamespace(order_id=k) for k, o in self.orders.items() if o["state"] == "new")

    def _place(self, kind, symbol, side, qty, client_order_id, state, trades):
        order_id = str(len(self.orders) + 1)
        self.orders[order_id] = {"client_order_id": client_order_id, "symbol": symbol, "side": side, "state": state,
                                 "filled_qty": str(qty) if trades else "0", "trades": trades}
        return {"status": "submitted", "orderId": order_id}

    def send_market_order(self, symbol, side, qty, *, reduce_only=False, client_order_id=""):
        self.calls.append(("market", side, qty, reduce_only))
        self.position += D(str(qty)) * (1 if side == "buy" else -1)
        return self._place("market", symbol, side, qty, client_order_id, "filled",
                           [(f"t{len(self.orders) + 1}", str(qty), "100")])

    def send_limit_order(self, symbol, side, qty, price, *, post_only=True, reduce_only=False, client_order_id=""):
        self.calls.append(("limit", side, qty, price))
        return self._place("limit", symbol, side, qty, client_order_id, "new", [])

    def get_order(self, symbol, *, order_id="", client_order_id=""):
        for key, o in self.orders.items():
            if key == order_id or (client_order_id and o["client_order_id"] == client_order_id):
                return SimpleNamespace(order_id=key, client_order_id=o["client_order_id"], symbol=o["symbol"],
                                       side=o["side"], state=o["state"], filled_qty=o["filled_qty"])
        return None

    def cancel_order(self, symbol, order_id):
        self.calls.append(("cancel", order_id))
        self.orders[order_id]["state"] = "canceled"
        return {}

    def get_recent_fills(self, *, symbol=None, order_id="", since_ms=0, limit=100):
        o = self.orders[order_id]
        return tuple(SimpleNamespace(order_id=order_id, symbol=o["symbol"], side=o["side"], trade_id=t, qty=q,
                                     price=p, fee="0", fee_asset="USDT") for t, q, p in o["trades"])

    def get_ticker(self, symbol):
        return {"bid1Price": "100", "ask1Price": "100"}


class Breaker:
    def __init__(self, allow=True):
        self.allow, self.successes, self.failures = allow, 0, 0

    def allow_request(self):
        return self.allow

    def record_success(self):
        self.successes += 1

    def record_failure(self):
        self.failures += 1


class Log:
    def __init__(self, fail=False):
        self.rows, self.fail = [], fail

    def write(self, kind, **fields):
        if self.fail:
            raise OSError("disk full")
        self.rows.append((kind, fields))


@pytest.fixture
def env(tmp_path):
    def build(*, entry_mode="market", breaker=None, log=None, venue=None, owner=OWNER):
        venue = venue or Venue()
        clocks = {"wall": 1000.0, "mono": 10.0}
        orders = OrderLifecycle(venue, SqliteOrderJournal(tmp_path / "orders.sqlite", SCOPE), SCOPE,
                                clock=lambda: clocks["wall"], sleeper=lambda s: clocks.update(mono=clocks["mono"] + s),
                                lock=threading.RLock(), wait_seconds=0.5, owner=owner,
                                entry_order_type="limit" if entry_mode == "maker" else "market",
                                not_found_grace_seconds=90,
                                entry_ttl_seconds=3 * 3600.0 if entry_mode == "maker" else None)
        service = ExecutionService(venue, orders, owner=owner, entry_mode=entry_mode, limit_offset=D("0.0002"),
                                   lock=threading.RLock(), clock=lambda: clocks["wall"],
                                   monotonic=lambda: clocks["mono"], breaker=breaker, event_log=log)
        statuses = []
        service.status_callback = lambda intent_id, status: statuses.append((intent_id, status))
        return service, venue, statuses, clocks
    return build


def _intent(n, side="buy", qty="1", *, reduce_only=False, price="100", source=OWNER):
    return OrderIntent(order_id=f"i{n}", source=source, symbol="BTCUSDT", side=side, qty=D(qty), price=D(price),
                       reduce_only=reduce_only, reason="rebalance")


def _parent(intent, calls=None):
    def build():
        if calls is not None:
            calls.append(intent.order_id)
        return {"order_id": intent.order_id, "price": str(intent.price), "note": "kept as given"}
    return build


def _own(service, fact, qty):
    """Consume a fill into the owner's checkpoint, as the fill consumer does."""
    service.journal.consume(OWNER, fact.fill_id, lambda: {
        "position": {"qty": qty, "entry_price": "100", "realized_pnl": "0", "fees": {}},
        "revision": fact.fill_id, "strategy": {}}, lambda: None, expected_revision=None)


def test_an_entry_needs_a_clean_reconciliation_and_a_closed_breaker(env):
    breaker, built = Breaker(), []
    service, venue, statuses, _ = env(breaker=breaker)
    assert service.ready is False                              # an owner starts blocked
    assert service.send(_intent(1), _parent(_intent(1), built)) == SendResult(False, ())
    assert statuses == [("i1", "blocked")] and venue.calls == [] and built == []
    assert service.reconcile("BTCUSDT") == [] and service.ready is True
    breaker.allow = False
    assert service.send(_intent(2), _parent(_intent(2), built)).submitted is False
    assert statuses[-1] == ("i2", "blocked") and venue.calls == [] and built == []
    breaker.allow = True
    result = service.send(_intent(3), _parent(_intent(3), built))
    assert result.submitted and [(f.qty, f.price) for f in result.facts] == [(D("1"), D("100"))]
    assert built == ["i3"] and breaker.successes == 1 and breaker.failures == 0


def test_a_repeated_intent_is_never_placed_again_and_builds_no_parent(env):
    service, venue, _, _ = env()
    service.ready = True
    built = []
    assert service.send(_intent(1), _parent(_intent(1), built)).submitted
    assert service.send(_intent(1), _parent(_intent(1), built)) == SendResult(False, ())
    assert len(venue.calls) == 1 and built == ["i1"]


@pytest.mark.parametrize("side,expected", [("buy", 99.98), ("sell", 100.02)])
def test_a_maker_entry_rests_two_basis_points_inside_the_reference(env, side, expected):
    service, venue, _, _ = env(entry_mode="maker")
    service.ready = True
    assert service.send(_intent(1, side), _parent(_intent(1, side))).submitted
    assert venue.calls == [("limit", side, 1.0, pytest.approx(expected))]


def test_a_maker_entry_without_a_positive_reference_price_fails_before_any_parent_is_built(env):
    service, venue, statuses, _ = env(entry_mode="maker")
    service.ready = True
    built = []
    intent = _intent(1, price="0")
    assert service.send(intent, _parent(intent, built)) == SendResult(False, ())
    assert built == [] and venue.calls == [] and service.ready is False and statuses == [("i1", "blocked")]


@pytest.mark.parametrize("qty", ["NaN", "-1", "Infinity"])
def test_an_invalid_quantity_fails_before_any_parent_is_built_or_order_sent(env, qty):
    service, venue, statuses, _ = env()
    service.ready = True
    built = []
    intent = _intent(1, qty=qty)
    assert service.send(intent, _parent(intent, built)) == SendResult(False, ())
    assert built == [] and venue.calls == [] and service.journal.orders() == []
    assert service.ready is False and statuses == [("i1", "blocked")]


def test_a_reduction_cancels_resting_entries_and_closes_only_what_the_owner_confirmed(env):
    service, venue, _, _ = env()
    service.ready = True
    (fact,) = service.send(_intent(1, qty="1"), _parent(_intent(1))).facts
    _own(service, fact, "1")
    venue.position += D("2")                                    # someone else's 2 on the same account
    close = _intent(2, "buy", "0", reduce_only=True)            # qty 0: the side comes from the venue position
    result = service.send(close, _parent(close))
    assert result.submitted and venue.calls[-1] == ("market", "sell", 1.0, True)
    stored = json.loads(next(r for r in service.journal.orders(OWNER) if r["intent_id"] == "i2")["parent"])
    assert stored == {"order_id": "i2", "price": "100", "note": "kept as given", "consumer_revision": fact.fill_id}


def test_a_reduction_cancels_a_resting_maker_entry_first(env):
    service, venue, _, _ = env(entry_mode="maker")
    service.ready = True
    service.send(_intent(1), _parent(_intent(1)))               # rests
    venue.position = D("1")
    close = _intent(2, "sell", "1", reduce_only=True)
    service.send(close, _parent(close))                          # no owned position: refused, but only after the cancel
    assert ("cancel", "1") in venue.calls and venue.orders["1"]["state"] == "canceled"
    assert not [c for c in venue.calls if c[0] == "market"]


def test_a_failure_blocks_entries_and_counts_on_the_breaker(env):
    breaker = Breaker()
    service, venue, statuses, _ = env(breaker=breaker)
    service.ready = True
    stranger = _intent(1, source="target.ETHUSDT")
    assert service.send(stranger, _parent(stranger)) == SendResult(False, ())
    assert service.ready is False and breaker.failures == 1 and statuses == [("i1", "blocked")]
    assert venue.calls == []


def test_a_parent_that_cannot_be_built_fails_before_anything_is_sent(env):
    service, venue, _, _ = env()
    service.ready = True

    def broken():
        raise KeyError("header")

    assert service.send(_intent(1), broken) == SendResult(False, ())
    assert venue.calls == [] and service.journal.orders() == [] and service.ready is False


def test_fills_reach_the_event_log_with_the_stored_reference_price_and_the_latency(env):
    log = Log()
    service, _, _, _ = env(log=log)
    service.ready = True
    intent = _intent(1, price="99.5")
    (fact,) = service.send(intent, _parent(intent)).facts
    assert log.rows == [("fill", {"symbol": "BTCUSDT", "side": "buy", "qty": 1.0, "ref_price": 99.5,
                                  "fill_price": 100.0, "latency_ms": 500.0, "order_id": fact.order_id,
                                  "fill_id": fact.fill_id, "ref_source": "decision_price"})]


def test_a_failing_event_log_never_fails_the_order(env):
    breaker = Breaker()
    service, _, _, _ = env(log=Log(fail=True), breaker=breaker)
    service.ready = True
    result = service.send(_intent(1), _parent(_intent(1)))
    assert result.submitted and len(result.facts) == 1
    assert service.ready is True and breaker.failures == 0 and breaker.successes == 1


def test_adopting_a_venue_order_is_idempotent_and_its_fills_are_collected(env):
    log = Log()
    service, venue, _, clocks = env(log=log)
    service.ready = True
    (fact,) = service.send(_intent(1), _parent(_intent(1))).facts
    _own(service, fact, "1")
    venue.send_market_order("BTCUSDT", "sell", "1", reduce_only=True, client_order_id="autoclose-1")   # the stop
    clocks["wall"] = 2000.0
    adopt = dict(intent_id="stop-1", symbol="BTCUSDT", side="sell", order_id="2", client_order_id="autoclose-1",
                 parent={"order_id": "stop-1", "price": "0", "reason": "protective_stop"})
    assert service.adopt(**adopt) is True and service.adopt(**adopt) is False
    row = next(r for r in service.journal.orders(OWNER) if r["intent_id"] == "stop-1")
    assert (row["status"], row["reducing"], row["created_at"], row["order_id"]) == ("submitted", 1, 2000.0, "2")
    fills = service.poll()
    assert [(f["intent_id"], f["side"], f["qty"]) for f in fills] == [("stop-1", "sell", "1")]   # i1 was consumed
    assert log.rows[-1][1]["ref_price"] == 0.0 and log.rows[-1][1]["fill_id"] == fills[-1]["fill_id"]
    assert service.fill_fact(fills[-1]["fill_id"])["intent_id"] == "stop-1"
    with pytest.raises(ValueError, match="Unknown fill"):
        service.fill_fact("no-such-fill")


def test_reconcile_reports_venue_blockers_first_and_blocks_until_clean(env):
    service, venue, statuses, _ = env()
    venue.blockers = ["adapter not connected"]
    venue.position = D("0.5")
    assert service.reconcile("BTCUSDT") == ["adapter not connected", "BTCUSDT: venue qty=0.5, ledger qty=0"]
    assert service.ready is False
    venue.blockers, venue.position = [], D(0)
    assert service.reconcile("BTCUSDT") == [] and service.ready is True
    service.send(_intent(1), _parent(_intent(1)))
    service.notify_status()
    assert statuses == [("i1", "filled")]


def test_the_reference_price_survives_a_restart(env):
    log = Log()
    service, venue, _, _ = env(entry_mode="maker")
    service.ready = True
    intent = _intent(1, price="250")
    service.send(intent, _parent(intent))                        # rests at the venue
    venue.orders["1"].update(state="filled", filled_qty="1", trades=[("t1", "1", "249.95")])
    restarted, _, _, _ = env(entry_mode="maker", log=log, venue=venue)
    assert [f["intent_id"] for f in restarted.poll()] == ["i1"]
    assert log.rows[0][1]["ref_price"] == 250.0 and log.rows[0][1]["fill_price"] == 249.95


@pytest.mark.parametrize("kw", [{"limit_offset": D("0.01")}, {"limit_offset": D("-0.0001")},
                                {"limit_offset": D("NaN")}, {"entry_mode": "stop"}])
def test_invalid_settings_are_refused(tmp_path, kw):
    venue = Venue()
    orders = OrderLifecycle(venue, SqliteOrderJournal(tmp_path / "orders.sqlite", SCOPE), SCOPE, clock=lambda: 0.0,
                            sleeper=lambda s: None, lock=threading.RLock(), owner=OWNER)
    settings = {"entry_mode": "maker", "limit_offset": D("0.0002"), **kw}
    with pytest.raises(ValueError):
        ExecutionService(venue, orders, owner=OWNER, lock=threading.RLock(), clock=lambda: 0.0,
                         monotonic=lambda: 0.0, **settings)


# --- step-9 audit F2 ---------------------------------------------------------------------------------

def test_an_entry_the_journal_refuses_is_reported_blocked_and_counts_no_success(env):
    breaker = Breaker()
    service, venue, statuses, _ = env(entry_mode="maker", breaker=breaker)
    service.ready = True
    first = service.send(_intent(1), _parent(_intent(1)))        # rests at the venue: no fill yet
    assert first == SendResult(True, ()) and breaker.successes == 0 and statuses == []
    second = service.send(_intent(2), _parent(_intent(2)))       # refused: an entry is still unresolved
    assert second == SendResult(True, ()) and statuses == [("i2", "blocked")]
    assert len(venue.calls) == 1 and breaker.successes == 0 and breaker.failures == 0


def test_a_repeated_intent_is_silent_even_when_entries_are_blocked(env):
    service, venue, statuses, _ = env()
    service.ready = True
    service.send(_intent(1), _parent(_intent(1)))
    service.ready = False
    assert service.send(_intent(1), _parent(_intent(1))) == SendResult(False, ())
    assert statuses == [] and len(venue.calls) == 1
