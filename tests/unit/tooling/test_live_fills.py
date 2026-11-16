"""The pilot's real fills against the backtest's assumptions (tech lead 2026-11-16: the pilot exists to calibrate the
backtest). ``scripts/live_fills.py`` joins the execution state database's orders with the account's own trades (the
read-only history export) and the decision bar's open, and reports per kind of order what the fill cost against
what the backtest assumes: market orders at the next open, 3 bp worse; stops at the trigger, 10 bp worse; a 5 bp
taker fee. Synthetic data only."""
from __future__ import annotations

import json
import sqlite3
from decimal import Decimal as D

import pytest

from scripts import live_fills as lf
from tests.support.state_db import state_db

H = 3_600_000
T0 = 1_794_800_000_000 // H * H          # a decision bar's close


def _journal(path, scope="binance-mainnet-pilot"):
    db_path = state_db(path)
    rows = [  # intent id, symbol, order id, side, qty, created_at (s)
        (f"ETHUSDT:{T0}:rebalance:0", "ETHUSDT", "101", "buy", "0.05", T0 / 1000 + 4.0),
        (f"ETHUSDT:{T0 + H}:rebalance:0", "ETHUSDT", "102", "sell", "0.02", (T0 + H) / 1000 + 3.0),
        ("stop-ALGO-1", "ETHUSDT", "103", "sell", "0.03", (T0 + 2 * H) / 1000 + 1700),
        (f"BTCUSDT:{T0}:kill_switch:0", "BTCUSDT", "104", "sell", "0.001", T0 / 1000 + 9.0),
        (f"ETHUSDT:{T0 + 3 * H}:rebalance:0", "ETHUSDT", "105", "buy", "0.01", (T0 + 3 * H) / 1000 + 2.0),
    ]
    with sqlite3.connect(db_path) as db:
        for intent, symbol, order, side, qty, created in rows:
            db.execute("INSERT INTO submissions (scope, intent_id, symbol, client_id, order_id, side, qty, reducing, "
                       "owner, status, order_type, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (scope, intent, symbol, f"c-{order}", order, side, qty, 0, f"target.{symbol}", "filled",
                        "STOP_MARKET" if intent.startswith("stop-") else "MARKET", created))
        db.execute("INSERT INTO protective_stops (scope, owner, client_algo_id, symbol, revision, attempt, side, "
                   "trigger_price, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (scope, "target.ETHUSDT", "ALGO-1", "ETHUSDT", "r1", 1, "sell", "1910", "triggered", T0 / 1000))
    return db_path


def _history(tmp_path):
    d = tmp_path / "history"
    d.mkdir()

    def trade(order, price, qty, fee, at, side):
        return {"orderId": int(order), "price": str(price), "qty": str(qty), "quoteQty": str(D(price) * D(qty)),
                "commission": str(fee), "commissionAsset": "USDT", "time": at, "side": side, "maker": False}
    eth = [trade(101, "2000.2", "0.03", "0.0300030", T0 + 4100, "BUY"),      # two fills, average 2000.28
           trade(101, "2000.4", "0.02", "0.0200040", T0 + 4200, "BUY"),
           trade(102, "1999.0", "0.02", "0.0199900", T0 + H + 3100, "SELL"),
           trade(103, "1909.0", "0.03", "0.0286350", T0 + 2 * H + 1_700_100, "SELL")]
    btc = [trade(104, "70000", "0.001", "0.035", T0 + 9100, "SELL")]
    (d / "trades_ETHUSDT.json").write_text(json.dumps(eth))
    (d / "trades_BTCUSDT.json").write_text(json.dumps(btc))
    return d


OPENS = {("ETHUSDT", T0): D("2000"), ("ETHUSDT", T0 + H): D("2000"), ("BTCUSDT", T0): D("69990"),
         ("ETHUSDT", T0 + 3 * H): D("1950")}


def _report(tmp_path):
    db = _journal(tmp_path / "state.sqlite")
    return lf.compare(db, _history(tmp_path), open_price=lambda symbol, at: OPENS[(symbol, at)])


def test_each_filled_order_is_joined_with_its_trades_and_its_decision_bar(tmp_path):
    report = _report(tmp_path)
    by_order = {r.order_id: r for r in report.rows}
    buy = by_order["101"]
    assert (buy.kind, buy.symbol, buy.side, buy.decided_at) == ("rebalance", "ETHUSDT", "buy", T0)
    assert buy.qty == D("0.05") and buy.avg_price == D("2000.28") and buy.reference == D("2000")
    assert buy.slip_bp == pytest.approx(1.4)                            # (2000.28 / 2000 - 1) x 10^4, against us
    assert buy.fee_bp == pytest.approx(5.0) and buy.delay_ms == 4100
    sell = by_order["102"]
    assert sell.slip_bp == pytest.approx(5.0)                           # sold 1 USDT under the open: against us


def test_a_stop_is_measured_against_its_trigger(tmp_path):
    stop = {r.order_id: r for r in _report(tmp_path).rows}["103"]
    assert stop.kind == "stop" and stop.reference == D("1910")
    assert stop.slip_bp == pytest.approx((1910 - 1909) / 1910 * 1e4)


def test_a_flatten_is_its_own_kind_and_a_better_price_is_negative(tmp_path):
    flat = {r.order_id: r for r in _report(tmp_path).rows}["104"]
    assert flat.kind == "kill_switch" and flat.slip_bp == pytest.approx(-(70000 / 69990 - 1) * 1e4)   # sold above
    assert flat.slip_bp < 0


def test_an_order_with_no_trade_is_reported_as_missing_not_guessed(tmp_path):
    report = _report(tmp_path)
    assert "105" not in {r.order_id for r in report.rows}
    assert [m["order_id"] for m in report.missing] == ["105"]


def test_the_summary_compares_with_the_backtest_s_assumptions(tmp_path):
    summary = _report(tmp_path).summary()
    rebalance = summary["rebalance"]
    assert rebalance["orders"] == 2 and rebalance["assumed_slip_bp"] == 3.0
    assert rebalance["slip_bp_median"] == pytest.approx(3.2)
    assert summary["stop"]["assumed_slip_bp"] == 10.0 and summary["stop"]["orders"] == 1
    assert summary["fee_bp_median"] == pytest.approx(5.0) and summary["assumed_fee_bp"] == 5.0
    assert summary["missing_orders"] == 1


def test_the_database_is_opened_read_only(tmp_path):
    db = _journal(tmp_path / "state.sqlite")
    before = db.read_bytes()
    lf.compare(db, _history(tmp_path), open_price=lambda symbol, at: OPENS[(symbol, at)])
    assert db.read_bytes() == before
    with pytest.raises(sqlite3.OperationalError):
        lf.connect_read_only(db).execute("CREATE TABLE x (y)")


def test_the_report_text_names_the_assumptions_and_holds_no_secret(tmp_path):
    text = lf.render(_report(tmp_path))
    assert "rebalance" in text and "assumed 3" in text and "stop" in text and "assumed 10" in text
    assert "missing" in text
