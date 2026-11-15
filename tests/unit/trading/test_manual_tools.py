"""Manual tools for a person (migration step 12b; ADR-0004 item 2 and the tech lead's 2026-11-15 decision).

Settling a stuck order and aligning a ledger to the venue: both show the venue first, need a reason and a
typed confirmation, refuse while the runtime runs, and leave an audit row. Nothing here is automatic.
"""
from __future__ import annotations

import ast
import io
import json
import threading
from decimal import Decimal as D
from pathlib import Path

import pytest

from tests.unit.binance.fake_venue import FakeBinance, make_bars
from trading.adapters.binance.adapter import BinanceAdapter
from trading.adapters.binance.config import BinanceConfig
from trading.adapters.files.lease import ExecutionLease
from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.application.execution.lifecycle import OrderLifecycle
from trading.application.reconcile import reconcile_account
from trading.infrastructure import manual

ROOT = Path(__file__).resolve().parents[3]
SCOPE, OWNER = "binance-testnet-ut", "target.BTCUSDT"
ENV = {"BINANCE_TESTNET": "1", "RISK_SCOPE": SCOPE}


@pytest.fixture
def venue(tmp_path):
    fake = FakeBinance(make_bars(10), clock=lambda: 1_800_000_000.0)
    fake.price["BTCUSDT"] = D("100")
    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)
    assert adapter.connect(symbols=("BTCUSDT",))
    journal = SqliteOrderJournal(tmp_path / "orders.sqlite", SCOPE)
    return fake, adapter, journal


def _lifecycle(adapter, journal, order_type="market"):
    return OrderLifecycle(adapter, journal, SCOPE, clock=lambda: 1_800_000_000.0, sleeper=lambda s: None,
                          lock=threading.RLock(), wait_seconds=0, owner=OWNER, entry_order_type=order_type,
                          not_found_grace_seconds=90, entry_ttl_seconds=3600.0 if order_type == "limit" else None)


def _run(tmp_path, adapter, *args, typed=""):
    out = io.StringIO()
    code = manual.main(["--state-dir", str(tmp_path), *args], stdin=io.StringIO(typed + "\n"), stdout=out, env=ENV,
                       account=adapter)
    return code, out.getvalue()


def _own(journal, qty="0.1", entry="100"):
    """A settled order and its consumed fill, as a real position leaves them."""
    journal.reserve(intent_id="i0", symbol="BTCUSDT", client_id="c0", side="buy", qty=qty.lstrip("-"),
                    reducing=False, owner=OWNER, parent=None, order_type="market", clock=lambda: 1.0)
    journal.update_status("i0", "filled", settle=True)
    journal.insert_fill(fill_id="f0", intent_id="i0", owner=OWNER, payload={"fill_id": "f0"})
    journal.consume(OWNER, "f0", lambda: {"position": {"qty": qty, "entry_price": entry, "realized_pnl": "1.5",
                                                       "fees": {"USDT": "0.2"}}, "revision": "f0", "strategy": {}},
                    lambda: None, expected_revision=None)


# -- settling a stuck order -------------------------------------------------------------------------------------
def test_an_order_the_venue_never_received_is_settled_as_not_found_after_confirmation(tmp_path, venue):
    fake, adapter, journal = venue
    fake.fault = "error_5xx"
    _lifecycle(adapter, journal).submit(intent_id="BTCUSDT:1:rebalance:0", symbol="BTCUSDT", side="buy", qty=D("1"))
    (row,) = journal.orders(unresolved=True)
    code, out = _run(tmp_path, adapter, "settle", "--intent", row["intent_id"], "--reason", "venue has no record",
                     typed="wrong")
    assert code == 1 and "not settled" in out and journal.orders(unresolved=True)
    code, out = _run(tmp_path, adapter, "settle", "--intent", row["intent_id"], "--reason", "venue has no record",
                     typed=row["intent_id"])
    assert code == 0 and "not_found" in out
    assert journal.orders(unresolved=True) == []
    (action,) = journal.manual_actions()
    assert (action["kind"], action["subject"], action["before"], action["after"], action["reason"]) == (
        "settle_order", row["intent_id"], "unknown", "not_found", "venue has no record")
    assert json.loads(action["venue"]) == {"state": "not found", "recorded_qty": "0"}


def test_an_order_still_working_at_the_venue_is_never_settled(tmp_path, venue):
    fake, adapter, journal = venue
    _lifecycle(adapter, journal, "limit").submit(intent_id="BTCUSDT:1:rebalance:0", symbol="BTCUSDT", side="buy",
                                                 qty=D("1"), limit_price=D("99"))
    (row,) = journal.orders(unresolved=True)
    code, out = _run(tmp_path, adapter, "settle", "--intent", row["intent_id"], "--reason", "r",
                     typed=row["intent_id"])
    assert code == 1 and "still working at the venue" in out
    assert journal.orders(unresolved=True) and journal.manual_actions() == []


def test_a_filled_order_whose_fills_the_journal_missed_is_settled_with_a_warning(tmp_path, venue):
    fake, adapter, journal = venue
    _lifecycle(adapter, journal, "limit").submit(intent_id="BTCUSDT:1:rebalance:0", symbol="BTCUSDT", side="buy",
                                                 qty=D("1"), limit_price=D("99"))
    (row,) = journal.orders(unresolved=True)
    fake.fill_limit(row["order_id"])                           # filled at the venue, never polled
    code, out = _run(tmp_path, adapter, "settle", "--intent", row["intent_id"], "--reason", "history truncated",
                     typed=row["intent_id"])
    assert code == 0 and "the venue filled 1, the journal recorded 0" in out and "position --symbol BTCUSDT" in out
    (action,) = journal.manual_actions()
    assert action["after"] == "filled" and json.loads(action["venue"]) == {
        "state": "filled", "filled_qty": "1", "recorded_qty": "0"}
    assert journal.intent_fills(row["intent_id"]) == []                       # no fill is ever made up


def test_an_order_the_venue_does_not_know_but_with_recorded_fills_is_never_settled(tmp_path, venue):
    fake, adapter, journal = venue
    journal.reserve(intent_id="i9", symbol="BTCUSDT", client_id="c9", side="buy", qty="1", reducing=False,
                    owner=OWNER, parent=None, order_type="market", clock=lambda: 1.0)
    journal.insert_fill(fill_id="f9", intent_id="i9", owner=OWNER, payload={"fill_id": "f9", "qty": "1"})
    code, out = _run(tmp_path, adapter, "settle", "--intent", "i9", "--reason", "r", typed="i9")
    assert code == 1 and "holds fills" in out and journal.manual_actions() == []


def test_the_journal_refuses_a_manual_settlement_that_is_not_final_or_has_no_reason(tmp_path, venue):
    fake, adapter, journal = venue
    journal.reserve(intent_id="i9", symbol="BTCUSDT", client_id="c9", side="buy", qty="1", reducing=False,
                    owner=OWNER, parent=None, order_type="market", clock=lambda: 1.0)
    for status, reason in (("unknown", "r"), ("submitted", "r"), ("not_found", " ")):
        with pytest.raises(ValueError):
            journal.settle_manually("i9", status, venue={}, reason=reason, at=1.0)
    assert journal.settle_manually("i9", "not_found", venue={}, reason="r", at=1.0) == "reserved"
    assert journal.settle_manually("i9", "not_found", venue={}, reason="r", at=2.0) is None     # already settled
    assert len(journal.manual_actions()) == 1


def test_listing_shows_every_open_order_with_the_venue_s_view(tmp_path, venue):
    fake, adapter, journal = venue
    _lifecycle(adapter, journal, "limit").submit(intent_id="BTCUSDT:1:rebalance:0", symbol="BTCUSDT", side="buy",
                                                 qty=D("1"), limit_price=D("99"))
    code, out = _run(tmp_path, adapter, "orders")
    (listed,) = json.loads(out)
    assert code == 0 and listed["intent_id"] == "BTCUSDT:1:rebalance:0" and listed["venue"]["state"] == "new"


# -- aligning a ledger ----------------------------------------------------------------------------------------
def test_a_ledger_is_set_to_the_venue_position_and_reconciliation_agrees_afterwards(tmp_path, venue):
    fake, adapter, journal = venue
    _own(journal, qty="0.1")
    fake.positions["BTCUSDT"], fake.entry["BTCUSDT"] = D("0.04"), D("101")
    assert reconcile_account(adapter, journal, owner=OWNER, symbol="BTCUSDT") == [
        "BTCUSDT: venue qty=0.04, ledger qty=0.1"]
    code, out = _run(tmp_path, adapter, "position", "--symbol", "BTCUSDT", "--reason", "stop fill lost", typed="ETH")
    assert code == 1 and journal.checkpoint(OWNER)["position"]["qty"] == "0.1"
    code, out = _run(tmp_path, adapter, "position", "--symbol", "BTCUSDT", "--reason", "stop fill lost",
                     typed="BTCUSDT")
    assert code == 0
    checkpoint = journal.checkpoint(OWNER)
    assert checkpoint["position"] == {"qty": "0.04", "entry_price": "101", "realized_pnl": "1.5",
                                      "fees": {"USDT": "0.2"}}
    assert checkpoint["revision"].startswith("manual-")
    assert reconcile_account(adapter, journal, owner=OWNER, symbol="BTCUSDT") == []
    (action,) = journal.manual_actions()
    assert (action["kind"], action["subject"], json.loads(action["before"])["qty"]) == ("set_position", OWNER, "0.1")


def test_a_flat_venue_sets_a_flat_ledger_without_an_entry_price(tmp_path, venue):
    fake, adapter, journal = venue
    _own(journal, qty="-0.1")
    code, _ = _run(tmp_path, adapter, "position", "--symbol", "BTCUSDT", "--reason", "closed by hand",
                   typed="BTCUSDT")
    assert code == 0 and journal.checkpoint(OWNER)["position"]["qty"] == "0"
    assert journal.checkpoint(OWNER)["position"]["entry_price"] == "0"


def test_a_ledger_is_never_aligned_while_orders_are_open_or_fills_are_unconsumed(tmp_path, venue):
    fake, adapter, journal = venue
    _own(journal)
    journal.insert_fill(fill_id="f1", intent_id="i1", owner=OWNER, payload={"fill_id": "f1"})
    code, out = _run(tmp_path, adapter, "position", "--symbol", "BTCUSDT", "--reason", "r", typed="BTCUSDT")
    assert code == 1 and "not consumed" in out and journal.manual_actions() == []
    journal.consume(OWNER, "f1", lambda: journal.checkpoint(OWNER), lambda: None, expected_revision="f0")
    fake.fault = "error_5xx"
    _lifecycle(adapter, journal).submit(intent_id="BTCUSDT:2:rebalance:0", symbol="BTCUSDT", side="sell",
                                        qty=D("0.1"), reduce_only=True)
    code, out = _run(tmp_path, adapter, "position", "--symbol", "BTCUSDT", "--reason", "r", typed="BTCUSDT")
    assert code == 1 and "settle the open orders" in out and journal.manual_actions() == []


# -- guards common to both ------------------------------------------------------------------------------------
def test_nothing_runs_while_the_runtime_holds_the_lease(tmp_path, venue):
    fake, adapter, journal = venue
    with ExecutionLease(tmp_path / "orders.sqlite", SCOPE):
        code, out = _run(tmp_path, adapter, "orders")
    assert code == 3 and "stop the runtime" in out


def test_a_reason_is_required(tmp_path, venue):
    fake, adapter, journal = venue
    _own(journal)
    code, out = _run(tmp_path, adapter, "position", "--symbol", "BTCUSDT", "--reason", "  ", typed="BTCUSDT")
    assert code == 2 and journal.manual_actions() == []


def test_a_missing_state_directory_or_journal_is_refused_and_never_created(tmp_path, venue):
    fake, adapter, journal = venue
    assert _run(tmp_path / "nowhere", adapter, "orders")[0] == 2 and not (tmp_path / "nowhere").exists()
    (tmp_path / "empty").mkdir()
    assert _run(tmp_path / "empty", adapter, "orders")[0] == 2 and not (tmp_path / "empty/orders.sqlite").exists()


def test_only_the_manual_command_calls_the_manual_journal_operations():
    names = {"settle_manually", "set_position_manually"}
    callers = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(("tests/", "docs/", ".", "target/")) or "/." in rel:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        if any(isinstance(n, ast.Attribute) and n.attr in names for n in ast.walk(tree)):
            callers.append(rel)
    assert callers == ["trading/infrastructure/manual.py"]


# -- audit after step 12b: finding 3 and the auditor's surviving mutants MA6, MA7, MA11 -----------------------
def _stop_row(journal, client_algo_id, status, revision="f0"):
    journal.ensure_stop_table()
    journal.add_stop(owner=OWNER, client_algo_id=client_algo_id, symbol="BTCUSDT", revision=revision, attempt=0,
                     side="buy", trigger_price="104", created_at=1.0)
    journal.set_stop_status(client_algo_id, status)


def test_a_ledger_is_never_aligned_while_a_stop_has_fired_and_the_runtime_has_not_seen_it(tmp_path, venue):
    fake, adapter, journal = venue
    _own(journal, qty="-0.1")
    _stop_row(journal, "qsstop1", "active")
    fake.algo["qsstop1"] = {"algoId": 1, "clientAlgoId": "qsstop1", "symbol": "BTCUSDT", "side": "BUY",
                            "algoStatus": "FINISHED", "triggerPrice": "104", "actualOrderId": "77",
                            "closePosition": "true"}                      # fired while the runtime was stopped
    code, out = _run(tmp_path, adapter, "position", "--symbol", "BTCUSDT", "--reason", "r", typed="BTCUSDT")
    assert code == 1 and "start the runtime once" in out and journal.manual_actions() == []
    fake.algo["qsstop1"]["algoStatus"] = "NEW"                            # a working stop is no reason to refuse
    code, _ = _run(tmp_path, adapter, "position", "--symbol", "BTCUSDT", "--reason", "r", typed="BTCUSDT")
    assert code == 0 and journal.checkpoint(OWNER)["position"]["qty"] == "0"


def test_a_ledger_is_never_aligned_while_a_stop_waits_for_its_order_or_cannot_be_looked_up(tmp_path, venue):
    fake, adapter, journal = venue
    _own(journal, qty="-0.1")
    _stop_row(journal, "qsstop1", "awaiting_order")
    code, out = _run(tmp_path, adapter, "position", "--symbol", "BTCUSDT", "--reason", "r", typed="BTCUSDT")
    assert code == 1 and "start the runtime once" in out
    journal.set_stop_status("qsstop1", "active")                          # unknown to the venue: cannot check
    original = fake.request_signed

    def down(*, method, path, params=None):
        if path == "/fapi/v1/algoOrder":
            from execution.adapters.binance.rest import BinanceRetryableError
            raise BinanceRetryableError("HTTP 503")
        return original(method=method, path=path, params=params)

    fake.request_signed = down
    code, out = _run(tmp_path, adapter, "position", "--symbol", "BTCUSDT", "--reason", "r", typed="BTCUSDT")
    assert code == 1 and "could not be checked" in out and journal.manual_actions() == []


def test_a_failed_venue_lookup_never_settles_an_order(tmp_path, venue):
    fake, adapter, journal = venue
    fake.fault = "error_5xx"
    _lifecycle(adapter, journal).submit(intent_id="BTCUSDT:1:rebalance:0", symbol="BTCUSDT", side="buy", qty=D("1"))

    def down(symbol, **kw):
        raise OSError("venue down")

    adapter.get_order = down
    code, out = _run(tmp_path, adapter, "settle", "--intent", "BTCUSDT:1:rebalance:0", "--reason", "r",
                     typed="BTCUSDT:1:rebalance:0")
    assert code == 1 and "could not be asked" in out and journal.orders(unresolved=True)


def test_several_venue_positions_are_never_aligned_and_a_failed_lookup_is_listed(tmp_path, venue):
    fake, adapter, journal = venue
    _own(journal)

    class Hedged:
        def get_positions(self, symbol="", *, strict=True):
            from trading.domain.venue import VenuePosition
            return (VenuePosition("binance", "BTCUSDT", D("0.1")), VenuePosition("binance", "BTCUSDT", D("-0.1")))

        def get_order(self, symbol, **kw):
            raise OSError("venue down")

        def get_algo_order(self, *, client_algo_id):
            return None

    code, out = _run(tmp_path, Hedged(), "position", "--symbol", "BTCUSDT", "--reason", "r", typed="BTCUSDT")
    assert code == 1 and "more than one venue position" in out
    fake.fault = "error_5xx"
    _lifecycle(adapter, journal).submit(intent_id="BTCUSDT:2:rebalance:0", symbol="BTCUSDT", side="buy", qty=D("1"))
    code, out = _run(tmp_path, Hedged(), "orders")
    (listed,) = json.loads(out)
    assert code == 0 and listed["venue"] == {"state": "lookup failed: OSError"}
