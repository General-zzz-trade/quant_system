"""Only well-formed exchange facts about OUR order become fills (written before migration steps 5-6).

Kills the defects that survived the 2026-11-14 mutation run: the order identity check (M3) and the
per-field validation of trade facts (M4) in shared.confirmed_execution._observe.
"""
from __future__ import annotations

from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from shared.confirmed_execution import ConfirmedExecutor


class Venue:
    def __init__(self, *, order: dict | None = None, fills: list[dict] | None = None) -> None:
        self.order_patch, self.fill_rows, self.client_id, self.sent = order or {}, fills, "", 0

    def get_ticker(self, symbol):
        return {"bid1Price": "100", "ask1Price": "100"}

    def send_market_order(self, symbol, side, qty, *, reduce_only=False, client_order_id=""):
        self.client_id, self.sent = client_order_id, self.sent + 1
        return {"status": "submitted", "orderId": "7"}

    def get_order(self, symbol, *, order_id="", client_order_id=""):
        state = dict(order_id="7", client_order_id=self.client_id, symbol="BTCUSDT", side="buy", state="filled",
                     filled_qty="1")
        return SimpleNamespace(**{**state, **self.order_patch})

    def get_recent_fills(self, *, symbol=None, order_id="", since_ms=0, limit=100):
        good = dict(order_id="7", symbol="BTCUSDT", side="buy", trade_id="t1", qty="1", price="100", fee="0.02",
                    fee_asset="USDT")
        rows = [good] if self.fill_rows is None else [{**good, **row} for row in self.fill_rows]
        return tuple(SimpleNamespace(**row) for row in rows)

    def cancel_order(self, symbol, order_id):
        raise AssertionError("market orders are never canceled")


def _submit(tmp_path, venue):
    executor = ConfirmedExecutor(venue, tmp_path / "orders.sqlite", "binance-testnet-ut", wait_seconds=0,
                                 owner="o", entry_order_type="market")
    facts = executor.submit(intent_id="i1", symbol="BTCUSDT", side="buy", qty=D("1"))
    (row,) = executor.journal.orders("o")
    with executor.journal.connect() as db:
        stored = db.execute("SELECT COUNT(*) FROM execution_fills").fetchone()[0]
    return facts, row, stored


def test_a_well_formed_fill_of_our_order_is_recorded_and_settles(tmp_path):
    facts, row, stored = _submit(tmp_path, Venue())
    assert [(f.qty, f.price) for f in facts] == [(D("1"), D("100"))]
    assert (row["status"], row["settled"], stored) == ("filled", 1, 1)


@pytest.mark.parametrize("patch", [{"client_order_id": "someone-else"}, {"order_id": "8"}, {"symbol": "ETHUSDT"},
                                   {"side": "sell"}])
def test_an_order_lookup_that_is_not_our_order_records_nothing(tmp_path, patch):
    facts, row, stored = _submit(tmp_path, Venue(order=patch))
    assert facts == () and stored == 0
    assert (row["status"], row["settled"]) == ("unknown", 0)


@pytest.mark.parametrize("bad", [{"trade_id": ""}, {"price": "0"}, {"qty": "NaN"}, {"qty": "0"}, {"fee": "NaN"},
                                 {"side": "sell"}])
def test_an_invalid_trade_fact_is_never_recorded_and_keeps_the_order_open(tmp_path, bad):
    facts, row, stored = _submit(tmp_path, Venue(fills=[bad]))
    assert facts == () and stored == 0
    assert (row["status"], row["settled"]) == ("unknown", 0)


@pytest.mark.parametrize("other", [{"order_id": "9"}, {"symbol": "ETHUSDT"}])
def test_trades_of_other_orders_are_ignored_and_the_order_stays_open(tmp_path, other):
    facts, row, stored = _submit(tmp_path, Venue(fills=[other]))
    assert facts == () and stored == 0
    assert (row["status"], row["settled"]) == ("submitted", 0)   # filled_qty 1 is not yet confirmed by a trade
