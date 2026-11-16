"""The read-only history export of the account check (tech lead 2026-11-16: check the backtest's fees, funding and
fill prices against the account's real records). GET requests on the allowlist only, raw records written to a
private directory, no credential anywhere. Runs against a fake transport; never contacts Binance."""
from __future__ import annotations

import io
import json
import stat
import urllib.parse

import pytest

from scripts import account_check as ac
from scripts import account_history as ah
from tests.unit.tooling.test_account_check import CANNED, KEY, SECRET

DAY = 86_400_000
NOW = 1_800_000_000_000


class Venue:
    """Income, trades and orders with times; answers windows and limits the way the export asks."""

    def __init__(self):
        self.calls = []
        self.income = [{"symbol": "ETHUSDT", "incomeType": "FUNDING_FEE", "income": "-0.01", "asset": "USDT",
                        "time": NOW - 80 * DAY + k * 8 * 3_600_000, "tranId": 1000 + k} for k in range(230)]
        self.income += [{"symbol": "DOGEUSDT", "incomeType": "COMMISSION", "income": "-0.02", "asset": "USDT",
                         "time": NOW - 3 * DAY, "tranId": 5000},
                        {"symbol": "", "incomeType": "TRANSFER", "income": "-4", "asset": "USDT",
                         "time": NOW - 2 * DAY, "tranId": 5001}]
        self.trades = {"ETHUSDT": [{"symbol": "ETHUSDT", "id": 1, "orderId": 9, "price": "2000", "qty": "0.1",
                                    "commission": "0.1", "commissionAsset": "USDT", "maker": False, "side": "SELL",
                                    "realizedPnl": "0", "time": NOW - 80 * DAY + 5}],
                       "DOGEUSDT": [{"symbol": "DOGEUSDT", "id": 2, "orderId": 10, "price": "0.1", "qty": "400",
                                     "commission": "0.02", "commissionAsset": "USDT", "maker": False, "side": "BUY",
                                     "realizedPnl": "0", "time": NOW - 3 * DAY}]}
        self.orders = {"ETHUSDT": [{"symbol": "ETHUSDT", "orderId": 9, "type": "MARKET", "side": "SELL",
                                    "avgPrice": "2000", "executedQty": "0.1", "status": "FILLED", "stopPrice": "0",
                                    "time": NOW - 80 * DAY, "updateTime": NOW - 80 * DAY + 5}],
                       "DOGEUSDT": []}

    def __call__(self, url, headers):
        parsed = urllib.parse.urlsplit(url)
        q = dict(urllib.parse.parse_qsl(parsed.query))
        self.calls.append((parsed.path, q, dict(headers)))
        if parsed.path in CANNED:
            return 200, json.dumps(CANNED[parsed.path]).encode()
        lo, hi, limit = int(q.get("startTime", 0)), int(q.get("endTime", NOW)), int(q.get("limit", 100))
        if hi - lo > 7 * DAY and parsed.path != "/fapi/v1/income":
            return 400, json.dumps({"code": -4165, "msg": "window too wide"}).encode()
        rows = {"/fapi/v1/income": self.income, "/fapi/v1/userTrades": self.trades.get(q.get("symbol", ""), []),
                "/fapi/v1/allOrders": self.orders.get(q.get("symbol", ""), [])}[parsed.path]
        hit = sorted((r for r in rows if lo <= r["time"] <= hi), key=lambda r: r["time"])
        return 200, json.dumps(hit[:limit]).encode()


def _export(tmp_path, venue, days=89):
    client = ac.ReadOnlyClient(KEY, SECRET, fetch=venue, clock=lambda: NOW / 1000)
    out = tmp_path / "history"
    summary = ah.export(client, out, days=days, now_ms=NOW)
    return out, summary


def test_the_export_writes_every_income_row_trade_and_order_of_every_symbol_seen(tmp_path):
    venue = Venue()
    out, summary = _export(tmp_path, venue)
    income = json.loads((out / "income.json").read_text())
    assert len(income) == 232 and len({r["tranId"] for r in income}) == 232      # paged, nothing twice
    assert summary["symbols"] == ["DOGEUSDT", "ETHUSDT"]                          # from the income rows
    assert [t["id"] for t in json.loads((out / "trades_ETHUSDT.json").read_text())] == [1]
    assert [t["id"] for t in json.loads((out / "trades_DOGEUSDT.json").read_text())] == [2]
    assert [o["orderId"] for o in json.loads((out / "orders_ETHUSDT.json").read_text())] == [9]
    assert summary["income_rows"] == 232 and summary["trades"] == 2 and summary["orders"] == 1
    assert summary["income_ascending"] is True


def test_only_get_requests_on_the_allowlist_are_made_and_windows_stay_within_a_week(tmp_path):
    venue = Venue()
    _export(tmp_path, venue)
    paths = {path for path, _, _ in venue.calls}
    assert paths == {"/fapi/v1/income", "/fapi/v1/userTrades", "/fapi/v1/allOrders"}
    assert all((ac.FAPI, path, True) in ac.ALLOWED for path in paths)
    for path, q, _ in venue.calls:
        assert int(q["endTime"]) - int(q["startTime"]) <= 7 * DAY
        assert int(q["startTime"]) >= NOW - 89 * DAY


def test_a_request_off_the_allowlist_is_refused_before_it_is_sent(tmp_path):
    venue = Venue()
    client = ac.ReadOnlyClient(KEY, SECRET, fetch=venue)
    with pytest.raises(PermissionError):
        client.get(ac.FAPI, "/fapi/v1/order", symbol="ETHUSDT")
    assert venue.calls == []


def test_the_files_are_private_and_hold_no_credential(tmp_path):
    venue = Venue()
    out, summary = _export(tmp_path, venue)
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    for path in out.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        text = path.read_text()
        assert KEY not in text and SECRET not in text and "signature" not in text
    assert KEY not in json.dumps(summary) and SECRET not in json.dumps(summary)


def test_a_full_page_is_followed_until_the_window_is_empty(tmp_path):
    venue = Venue()
    venue.income = [{"symbol": "ETHUSDT", "incomeType": "FUNDING_FEE", "income": "-0.01", "asset": "USDT",
                     "time": NOW - DAY + k, "tranId": k} for k in range(2500)]      # one window, three pages
    out, summary = _export(tmp_path, venue, days=5)
    assert summary["income_rows"] == 2500


def test_the_command_prints_a_summary_only(tmp_path):
    env = tmp_path / "credentials"
    env.write_text(f"BINANCE_API_KEY={KEY}\nBINANCE_API_SECRET={SECRET}\n")
    out = io.StringIO()
    code = ac.main(["--env-file", str(env), "--history", str(tmp_path / "h"), "--days", "89"], fetch=Venue(),
                   stdout=out, clock=lambda: NOW / 1000)
    text = out.getvalue()
    assert code == 0 and "income rows: 232" in text and "symbols: DOGEUSDT, ETHUSDT" in text
    assert KEY not in text and SECRET not in text and "-0.01" not in text          # no record values on the screen
    assert "open positions" not in text                                             # the history run does only that


def test_days_beyond_what_binance_keeps_are_refused(tmp_path):
    with pytest.raises(ValueError, match="89"):
        ah.export(ac.ReadOnlyClient(KEY, SECRET, fetch=Venue()), tmp_path / "x", days=120, now_ms=NOW)
