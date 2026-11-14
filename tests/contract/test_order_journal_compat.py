"""An order journal written by the pre-migration code is read unchanged by the SQLite adapter (step 6)."""
from __future__ import annotations

import json
import sqlite3
from decimal import Decimal as D

from shared.confirmed_execution import ConfirmedExecutor
from trading.adapters.sqlite.order_journal import SqliteOrderJournal

# Tables exactly as shared/execution_journal.py and protective_stops.py created them at e3cd2d6.
BASELINE_DDL = [
    """CREATE TABLE submissions (scope TEXT, intent_id TEXT, symbol TEXT, client_id TEXT, order_id TEXT,
       settled INTEGER NOT NULL DEFAULT 0, side TEXT, qty TEXT, reducing INTEGER, owner TEXT NOT NULL DEFAULT '',
       parent TEXT, status TEXT NOT NULL DEFAULT 'unknown', order_type TEXT, created_at REAL,
       PRIMARY KEY(scope, intent_id))""",
    """CREATE TABLE execution_fills (scope TEXT NOT NULL, fill_id TEXT NOT NULL, intent_id TEXT NOT NULL,
       owner TEXT NOT NULL, payload TEXT NOT NULL, delivered INTEGER NOT NULL DEFAULT 0,
       PRIMARY KEY(scope, fill_id))""",
    """CREATE TABLE execution_consumers (scope TEXT NOT NULL, owner TEXT NOT NULL, checkpoint TEXT NOT NULL,
       PRIMARY KEY(scope, owner))""",
    """CREATE TABLE protective_stops (scope TEXT NOT NULL, owner TEXT NOT NULL, client_algo_id TEXT NOT NULL,
       symbol TEXT NOT NULL, revision TEXT NOT NULL, attempt INTEGER NOT NULL, side TEXT NOT NULL,
       trigger_price TEXT NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL,
       PRIMARY KEY(scope, client_algo_id))""",
]
SCOPE = "binance-testnet-ut"


class _Silent:
    def get_order(self, symbol, *, order_id="", client_order_id=""):
        raise TimeoutError("venue unreachable")


def test_baseline_journal_restores_open_orders_fills_and_stops(tmp_path):
    path = tmp_path / "orders.sqlite"
    payload = {"fill_id": f"{SCOPE}:BTCUSDT:7:t1", "intent_id": "i1", "symbol": "BTCUSDT", "order_id": "7",
               "trade_id": "t1", "side": "buy", "qty": "0.4", "price": "100", "fee": "0", "fee_asset": "USDT"}
    with sqlite3.connect(path) as db:
        for ddl in BASELINE_DDL:
            db.execute(ddl)
        db.execute("INSERT INTO submissions VALUES (?, 'i1', 'BTCUSDT', 'qsabc', '7', 0, 'buy', '1', 0, 'o', NULL,"
                   " 'submitted', 'limit', 1.0)", (SCOPE,))
        db.execute("INSERT INTO execution_fills VALUES (?, ?, 'i1', 'o', ?, 0)", (SCOPE, payload["fill_id"],
                                                                                 json.dumps(payload)))
        db.execute("INSERT INTO protective_stops VALUES (?, 'o', 'qsstop1', 'BTCUSDT', 'r1', 0, 'sell', '96',"
                   " 'active', 2.0)", (SCOPE,))
    journal = SqliteOrderJournal(path, SCOPE)
    journal.ensure_stop_table()
    assert [r["status"] for r in journal.stop_rows("o", "BTCUSDT", live_only=True)] == ["active"]
    assert journal.fills("o") == [payload]
    executor = ConfirmedExecutor(_Silent(), path, SCOPE, owner="o", wait_seconds=0)
    pending = executor.pending["i1"]
    assert (pending.order_id, pending.confirmed_qty, pending.seen) == ("7", D("0.4"), {"t1"})
    assert executor.submit(intent_id="i2", symbol="BTCUSDT", side="buy", qty=D("1"), limit_price=D("99")) == ()
    assert [r["intent_id"] for r in journal.orders("o")] == ["i1"]       # the open order still blocks entries
