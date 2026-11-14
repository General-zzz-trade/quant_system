"""SqliteOrderJournal SQL details that no executor test pins (step-6 audit H2, written before step 9)."""
from __future__ import annotations

from decimal import Decimal as D
from types import SimpleNamespace

from shared.confirmed_execution import ConfirmedExecutor
from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.domain.orders import DUPLICATE, ENTRY_UNRESOLVED

SCOPE = "binance-testnet-ut"


def _journal(tmp_path) -> SqliteOrderJournal:
    return SqliteOrderJournal(tmp_path / "orders.sqlite", SCOPE)


def _reserve(journal, intent_id="i1", *, reducing=False, clock=lambda: 5.0):
    return journal.reserve(intent_id=intent_id, symbol="BTCUSDT", client_id=f"c-{intent_id}", side="buy", qty="1",
                           reducing=reducing, owner="o", parent=None, order_type="market" if reducing else "limit",
                           clock=clock)


def _row(journal, intent_id="i1"):
    return next(r for r in journal.orders() if r["intent_id"] == intent_id)


def test_status_writes_never_touch_a_settled_row_and_report_the_previous_status(tmp_path):
    journal = _journal(tmp_path)
    _reserve(journal)
    assert journal.update_status("i1", "submitted", settle=False) == "reserved"
    assert journal.update_status("i1", "filled", settle=True) == "submitted"
    assert (_row(journal)["status"], _row(journal)["settled"]) == ("filled", 1)
    assert journal.update_status("i1", "unknown", settle=False) is None     # settled rows are final
    assert (_row(journal)["status"], _row(journal)["settled"]) == ("filled", 1)


def test_settling_twice_keeps_the_row_settled_and_reports_nothing(tmp_path):
    journal = _journal(tmp_path)
    _reserve(journal)
    journal.update_status("i1", "canceled", settle=True)
    assert journal.update_status("i1", "canceled", settle=True) is None
    assert _row(journal)["settled"] == 1


def test_reservation_reads_the_clock_once_and_only_when_it_reserves(tmp_path):
    journal = _journal(tmp_path)
    calls = []

    def clock():
        calls.append(1)
        return 7.5

    assert _reserve(journal, clock=clock) == ("", 7.5)
    assert _row(journal)["created_at"] == 7.5 and calls == [1]
    assert _reserve(journal, clock=clock) == (DUPLICATE, 0.0)
    assert _reserve(journal, "i2", clock=clock) == (ENTRY_UNRESOLVED, 0.0)
    assert calls == [1]                                  # refusals never read the clock
    row = _row(journal)
    assert (row["status"], row["settled"], row["order_id"], row["reducing"]) == ("reserved", 0, "", 0)


def test_an_adopted_venue_order_is_an_open_market_reduction(tmp_path):
    journal = _journal(tmp_path)
    journal.adopt(intent_id="stop-1", symbol="BTCUSDT", client_id="auto", order_id="77", side="sell", owner="o",
                  parent={"reason": "stop"}, created_at=9.0)
    row = _row(journal, "stop-1")
    assert (row["status"], row["settled"], row["reducing"], row["order_type"], row["qty"]) == \
        ("submitted", 0, 1, "market", "0")


def test_a_new_stop_row_starts_unknown_and_live_rows_include_canceling(tmp_path):
    journal = _journal(tmp_path)
    journal.ensure_stop_table()
    for n, status in enumerate(["active", "unknown", "canceling", "triggered", "inactive", "rejected"]):
        journal.add_stop(owner="o", client_algo_id=f"s{n}", symbol="BTCUSDT", revision="r", attempt=n, side="sell",
                         trigger_price="96", created_at=float(n))
        if status != "unknown":
            journal.set_stop_status(f"s{n}", status)
    assert [r["status"] for r in journal.stop_rows("o", "BTCUSDT", live_only=False)][1] == "unknown"
    live = [r["status"] for r in journal.stop_rows("o", "BTCUSDT", live_only=True)]
    assert live == ["active", "unknown", "canceling"]


class _FilledVenue:
    def __init__(self):
        self.client_id = ""

    def send_market_order(self, symbol, side, qty, *, reduce_only=False, client_order_id=""):
        self.client_id = client_order_id
        return {"status": "submitted", "orderId": "7"}

    def get_order(self, symbol, *, order_id="", client_order_id=""):
        return SimpleNamespace(order_id="7", client_order_id=self.client_id, symbol="BTCUSDT", side="buy",
                               state="filled", filled_qty="1")

    def get_recent_fills(self, *, symbol=None, order_id="", since_ms=0, limit=100):
        return (SimpleNamespace(order_id="7", symbol="BTCUSDT", side="buy", trade_id="t1", qty="1", price="100",
                                fee="0", fee_asset="USDT"),)


def test_a_settled_order_leaves_the_executor_pending_set(tmp_path):
    executor = ConfirmedExecutor(_FilledVenue(), tmp_path / "orders.sqlite", SCOPE, wait_seconds=0, owner="o",
                                 entry_order_type="market")
    executor.submit(intent_id="i1", symbol="BTCUSDT", side="buy", qty=D("1"))
    assert executor.pending == {}
    assert executor.journal.orders("o", unresolved=True) == []
