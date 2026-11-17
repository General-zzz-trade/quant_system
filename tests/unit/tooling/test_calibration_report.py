"""The pilot's daily calibration report (task 3, 2026-11-17): per UTC day, fills against the backtest's references,
fees, order outcomes, the runtime's own refusals and funding against the backtest broker's charge. Synthetic data
only; built on the fixtures of ``test_live_fills``."""
from __future__ import annotations

import json
import sqlite3
from decimal import Decimal as D

import pytest

from scripts import live_fills as lf
from tests.unit.tooling.test_live_fills import H, OPENS, T0, _history, _journal

DAY0 = lf.day_of(T0)


def _db_with_outcomes(tmp_path):
    db = _journal(tmp_path / "state.sqlite")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE submissions SET settled=1 WHERE status='filled'")      # as the journal settles them
        for intent, status, settled in ((f"ETHUSDT:{T0 + 4 * H}:rebalance:0", "rejected", 1),
                                        (f"ETHUSDT:{T0 + 5 * H}:rebalance:0", "not_found", 1),
                                        (f"ETHUSDT:{T0 + 6 * H}:rebalance:0", "unknown", 0),
                                        ("ETHUSDT:7:foreign", "filled", 1)):           # not the strategy's order
            conn.execute("INSERT INTO submissions (scope, intent_id, symbol, client_id, order_id, side, qty, reducing, "
                         "owner, status, settled, order_type, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         ("binance-mainnet-pilot", intent, "ETHUSDT", f"c-{intent}", "", "buy", "0.01", 0,
                          "target.ETHUSDT", status, settled, "MARKET", T0 / 1000 + 60))
    return db


def _events(tmp_path):
    path = tmp_path / "events.jsonl"
    rows = [{"ts": T0 / 1000 + 10, "kind": "blocked", "symbol": "ETHUSDT", "reason": "drawdown_latched"},
            {"ts": T0 / 1000 + 20, "kind": "blocked", "symbol": "BTCUSDT", "reason": "drawdown_latched"},
            {"ts": T0 / 1000 + 30, "kind": "blocked", "symbol": "ETHUSDT", "refusal": "unresolved submission"},
            {"ts": T0 / 1000 + 40, "kind": "issue", "symbol": "ETHUSDT", "reason": "not a refusal"}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _funding_history(tmp_path):
    history = _history(tmp_path)
    income = [{"symbol": "ETHUSDT", "incomeType": "FUNDING_FEE", "income": "-0.0101", "asset": "USDT",
               "time": T0 + 1_800_000},
              {"symbol": "ETHUSDT", "incomeType": "FUNDING_FEE", "income": "-0.0118", "asset": "USDT",
               "time": T0 + 2 * H},
              {"symbol": "BTCUSDT", "incomeType": "FUNDING_FEE", "income": "0.0070", "asset": "USDT",
               "time": T0 + 1_800_000},
              {"symbol": "SOLUSDT", "incomeType": "FUNDING_FEE", "income": "-5", "asset": "USDT",   # not the strategy's
               "time": T0 + 1_800_000},
              {"symbol": "ETHUSDT", "incomeType": "COMMISSION", "income": "-1", "asset": "USDT", "time": T0}]
    (history / "income.json").write_text(json.dumps(income))
    (history / "summary.json").write_text(json.dumps({"to_ms": T0 + 4 * H}))
    return history


RATES = {"ETHUSDT": [(T0 + 1_800_000, D("0.0001"), D("2000")),     # position 0.05 long
                     (T0 + 2 * H, D("0.0002"), D("1950")),         # 0.03 long (sold 0.02 at T0+H+3.1 s)
                     (T0 + 3 * H, D("0.0001"), D("1900"))],        # flat (the stop sold 0.03 at T0+2H+1700 s)
         "BTCUSDT": [(T0 + 1_800_000, D("0.0001"), D("70000"))]}   # 0.001 short: a positive rate pays it


def _rates(symbol, start, end):
    return [r for r in RATES[symbol] if start <= r[0] <= end]


def test_fills_and_fees_are_grouped_by_the_day_of_the_first_trade(tmp_path):
    report = lf.compare(_journal(tmp_path / "state.sqlite"), _history(tmp_path),
                        open_price=lambda symbol, at: OPENS[(symbol, at)])
    day = lf.daily(report)[DAY0]
    assert day["entries"]["orders"] == 3 and day["entries"]["assumed_slip_bp"] == 3.0     # 101, 102 and the flatten
    assert day["entries"]["slip_bp_median"] == pytest.approx(1.4)
    assert day["stops"]["orders"] == 1 and day["stops"]["assumed_slip_bp"] == 10.0
    assert day["stops"]["slip_bp_median"] == pytest.approx((1910 - 1909) / 1910 * 1e4)
    assert day["fees"]["fee_bp_median"] == pytest.approx(5.0) and day["fees"]["assumed_fee_bp"] == 5.0
    expected_fee = 0.0300030 + 0.0200040 + 0.0199900 + 0.0286350 + 0.035
    assert day["fees"]["fee_usdt"] == pytest.approx(expected_fee, rel=1e-9)


def test_order_outcomes_give_the_rejection_rate_and_skip_foreign_orders(tmp_path):
    day = lf.outcomes(_db_with_outcomes(tmp_path))[DAY0]
    assert day["orders"] == 8                                       # 5 filled + rejected + not_found + unknown
    assert day["statuses"] == {"filled": 5, "rejected": 1, "not_found": 1, "unknown": 1}
    assert day["rejection_rate"] == pytest.approx(1 / 8) and day["unsettled"] == 1


def test_refusals_come_from_blocked_events_by_reason(tmp_path):
    assert lf.refusals(_events(tmp_path)) == {DAY0: {"drawdown_latched": 2, "unresolved submission": 1}}
    assert lf.refusals(None) == {}


def test_funding_is_compared_with_the_backtest_broker_charge_for_the_strategy_position(tmp_path):
    paid = lf.funding(_journal(tmp_path / "state.sqlite"), _funding_history(tmp_path), rates=_rates)[DAY0]
    # model: ETH -0.05 x 2000 x 0.0001 - 0.03 x 1950 x 0.0002 = -0.0217; BTC -(-0.001) x 70000 x 0.0001 = +0.007
    assert paid["model"] == pytest.approx(-0.0217 + 0.007)
    assert paid["actual"] == pytest.approx(-0.0101 - 0.0118 + 0.0070)          # SOLUSDT and commissions excluded
    assert paid["difference"] == pytest.approx(paid["actual"] - paid["model"])


def test_the_markdown_and_json_reports_hold_the_daily_numbers(tmp_path, monkeypatch):
    db = _db_with_outcomes(tmp_path)
    history = _funding_history(tmp_path)
    monkeypatch.setattr(lf, "binance_open", lambda symbol, at: OPENS[(symbol, at)])
    monkeypatch.setattr(lf, "binance_funding_rates", _rates)
    out_json, out_md = tmp_path / "report.json", tmp_path / "report.md"
    assert lf.main(["--state-db", str(db), "--history", str(history), "--events", str(_events(tmp_path)),
                    "--json-out", str(out_json), "--markdown-out", str(out_md)]) == 0
    data = json.loads(out_json.read_text())
    day = data["days"][DAY0]
    assert set(day) == {"entries", "stops", "fees", "outcomes", "refusals", "funding"}
    assert data["assumed"] == {"market": 3.0, "stop": 10.0, "fee": 5.0} and len(data["missing"]) == 1
    text = out_md.read_text()
    assert f"| {DAY0} | 3 | 1.40 | 1 |" in text and "12%" in text          # rejection rate 1/8
    assert "drawdown_latched 2" in text and "不猜" in text


def test_the_daily_report_does_not_change_the_state_database(tmp_path):
    db = _db_with_outcomes(tmp_path)
    before = db.read_bytes()
    lf.outcomes(db)
    lf.funding(db, _funding_history(tmp_path), rates=_rates)
    assert db.read_bytes() == before
