"""The read-only account check: GET requests on the allowlist only, and no credential in its output.

Runs against a fake transport; never contacts Binance.
"""
from __future__ import annotations

import io
import json
import urllib.parse

import pytest

from scripts import account_check as ac

KEY, SECRET = "fake-key-1234567890", "fake-secret-abcdefghij"

CANNED = {
    "/sapi/v1/account/apiRestrictions": {"ipRestrict": True, "enableReading": True, "enableFutures": True,
                                         "enableWithdrawals": False},
    "/fapi/v1/time": {"serverTime": 1_800_000_000_000},
    # plan D1: the contract list (public) and the margin tiers (signed) the pilot check reads
    "/fapi/v1/exchangeInfo": {"symbols": [
        {"symbol": s, "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT",
         "deliveryDate": 4133404800000} for s in ("BTCUSDT", "ETHUSDT")]},
    "/fapi/v1/leverageBracket": [
        {"symbol": s, "brackets": [{"bracket": 1, "initialLeverage": 125, "notionalCap": 50000, "notionalFloor": 0,
                                    "maintMarginRatio": 0.004, "cum": 0.0}]} for s in ("BTCUSDT", "ETHUSDT")],
    "/fapi/v2/account": {"canTrade": True, "totalWalletBalance": "167.5", "availableBalance": "150.1",
                         "assets": [{"asset": "USDT", "walletBalance": "167.5", "unrealizedProfit": "-1.2",
                                     "availableBalance": "150.1"},
                                    {"asset": "BNB", "walletBalance": "0", "unrealizedProfit": "0"}]},
    "/fapi/v1/positionSide/dual": {"dualSidePosition": False},
    "/fapi/v1/multiAssetsMargin": {"multiAssetsMargin": False},
    "/fapi/v1/feeBurn": {"feeBurn": False},
    "/fapi/v2/positionRisk": [
        {"symbol": "SOXLUSDT", "positionSide": "BOTH", "positionAmt": "-3", "entryPrice": "30", "markPrice": "31",
         "unRealizedProfit": "-3", "liquidationPrice": "80", "notional": "-93", "leverage": "5",
         "marginType": "cross", "isolatedMargin": "0"},
        {"symbol": "BTCUSDT", "positionSide": "BOTH", "positionAmt": "0", "leverage": "50", "marginType": "cross"},
        {"symbol": "ETHUSDT", "positionSide": "BOTH", "positionAmt": "0", "leverage": "10",
         "marginType": "isolated"}],
    "/fapi/v1/openOrders": [],
    "/fapi/v1/openAlgoOrders": [],
    "/fapi/v1/commissionRate": {"makerCommissionRate": "0.0002", "takerCommissionRate": "0.0005"},
    "/fapi/v1/apiTradingStatus": {"indicators": {}},
}


DAY = 86_400_000


def _income(query, *, descending=False):
    """The income history as Binance is assumed to answer: the symbol's funding rows from startTime to endTime, one
    every 8 hours, oldest first, at most ``limit``. ``descending`` answers newest first instead."""
    start, end, limit = int(query["startTime"][0]), int(query["endTime"][0]), int(query["limit"][0])
    times = list(range(start + 3_600_000, end + 1, 8 * 3_600_000))
    times = times[::-1][:limit] if descending else times[:limit]
    return [{"symbol": query["symbol"][0], "incomeType": query["incomeType"][0], "income": "-0.01", "asset": "USDT",
             "time": t, "tranId": t, "tradeId": ""} for t in times]


class Venue:
    """Answers each path from CANNED; ``answers`` replaces a path's answer with a function of the parsed query."""

    def __init__(self, errors=None, answers=None):
        self.calls, self.errors = [], errors or {}
        self.answers = {"/fapi/v1/income": _income, **(answers or {})}

    def __call__(self, url, headers):
        parsed = urllib.parse.urlsplit(url)
        self.calls.append((parsed.scheme + "://" + parsed.netloc, parsed.path, parsed.query, dict(headers)))
        if parsed.path in self.errors:
            return 401, json.dumps({"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."}).encode()
        if parsed.path in self.answers:
            return 200, json.dumps(self.answers[parsed.path](urllib.parse.parse_qs(parsed.query))).encode()
        return 200, json.dumps(CANNED[parsed.path]).encode()


def _env(tmp_path, text):
    path = tmp_path / "credentials"
    path.write_text(text)
    return path


def _run(tmp_path, venue, text=None):
    text = text if text is not None else (
        f"TELEGRAM_BOT_TOKEN=tg-not-read\nexport BINANCE_API_KEY='{KEY}'\nBINANCE_API_SECRET=\"{SECRET}\"\n")
    out = io.StringIO()
    code = ac.main(["--env-file", str(_env(tmp_path, text))], fetch=venue, stdout=out)
    return code, out.getvalue()


def test_only_the_two_key_names_are_taken_from_the_file(tmp_path):
    path = _env(tmp_path, f"# comment\nTELEGRAM_BOT_TOKEN=x\nexport BINANCE_API_KEY='{KEY}'\n"
                          f"BINANCE_API_SECRET=\"{SECRET}\"\nBINANCE_TESTNET_API_KEY=t\n")
    assert ac.read_credentials(path) == {"BINANCE_API_KEY": KEY, "BINANCE_API_SECRET": SECRET}


def test_every_request_is_an_allowlisted_read_and_nothing_secret_is_printed(tmp_path):
    venue = Venue()
    code, out = _run(tmp_path, venue)
    assert code == 0
    assert {(base, path) for base, path, _, _ in venue.calls} <= {(b, p) for b, p, _ in ac.ALLOWED}
    for base, path, query, headers in venue.calls:
        signed = (base, path, True) in ac.ALLOWED
        assert ("signature=" in query) is signed and (headers.get("X-MBX-APIKEY") == KEY) is signed
        assert SECRET not in query
    signatures = [urllib.parse.parse_qs(q)["signature"][0] for _, _, q, _ in venue.calls if "signature=" in q]
    for secret_like in [KEY, SECRET, "tg-not-read", *signatures]:
        assert secret_like not in out
    assert "OPEN SOXLUSDT BOTH: qty -3" in out and "open positions: 1" in out
    assert "setting BTCUSDT BOTH: leverage 50x, margin cross" in out
    assert "enableWithdrawals: False" in out and "BTCUSDT: maker 0.0002, taker 0.0005" in out


def test_the_client_refuses_anything_off_the_allowlist_before_sending():
    venue = Venue()
    client = ac.ReadOnlyClient(KEY, SECRET, fetch=venue)
    for base, path in [(ac.FAPI, "/fapi/v1/order"), (ac.FAPI, "/fapi/v1/leverage"), (ac.FAPI, "/fapi/v1/marginType"),
                       (ac.FAPI, "/fapi/v1/algoOrder"), (ac.SAPI, "/sapi/v1/capital/withdraw/apply")]:
        with pytest.raises(PermissionError):
            client.get(base, path)
    assert venue.calls == []


def test_the_transport_only_ever_sends_get(monkeypatch):
    seen = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"{}"

    monkeypatch.setattr(ac.urllib.request, "urlopen", lambda request, timeout: seen.append(request.method)
                        or Response())
    assert ac._urlopen("https://fapi.binance.com/fapi/v1/time", {}) == (200, b"{}")
    assert seen == ["GET"]


def test_missing_credentials_make_no_request_and_name_only_the_variables(tmp_path):
    venue = Venue()
    code, out = _run(tmp_path, venue, text=f"BINANCE_API_KEY={KEY}\n")
    assert code == 2 and venue.calls == []
    assert "BINANCE_API_SECRET" in out and KEY not in out


def test_a_refused_section_is_reported_and_the_others_still_run(tmp_path):
    venue = Venue(errors={"/sapi/v1/account/apiRestrictions"})
    code, out = _run(tmp_path, venue)
    assert code == 0
    assert "unavailable: RuntimeError: HTTP 401, Binance code -2015" in out
    assert "totalWalletBalance: 167.5" in out


# -- the income history as the strategy's funding reads it (audit of the loss stop, 2026-11-16, finding 5) --------
NOW = 1_800_000_000.0


def _income_run(tmp_path, venue):
    out = io.StringIO()
    path = _env(tmp_path, f"BINANCE_API_KEY={KEY}\nBINANCE_API_SECRET={SECRET}\n")
    code = ac.main(["--env-file", str(path)], fetch=venue, stdout=out, clock=lambda: NOW)
    lines = out.getvalue().splitlines()
    start = lines.index("== funding income history (FUNDING_FEE, last 80 days)")
    section = []
    for line in lines[start + 1:]:
        if line.startswith("== "):
            break
        section.append(line.strip())
    return code, section


def test_the_income_history_is_asked_read_only_for_both_symbols_over_80_days_whole_and_with_a_page_of_5(tmp_path):
    venue = Venue()
    code, section = _income_run(tmp_path, venue)
    assert code == 0
    asked = [urllib.parse.parse_qs(query) for _, path, query, _ in venue.calls if path == "/fapi/v1/income"]
    now_ms = int(NOW * 1000)
    assert [(q["symbol"][0], q["incomeType"][0], int(q["startTime"][0]), int(q["endTime"][0]), q["limit"][0])
            for q in asked] == [(symbol, "FUNDING_FEE", now_ms - 80 * DAY, now_ms, limit)
                                for symbol in ("BTCUSDT", "ETHUSDT") for limit in ("1000", "5")]
    first, last = now_ms - 80 * DAY + 3_600_000, now_ms - 80 * DAY + 3_600_000 + 239 * 8 * 3_600_000
    assert section[0] == (f"BTCUSDT limit 1000: rows 240, first {first} (2026-10-27 09:00:00 UTC), "
                          f"last {last} (2027-01-15 01:00:00 UTC), ascending by time: True")
    assert section[1] == (f"BTCUSDT limit 5: rows 5, first {first} (2026-10-27 09:00:00 UTC), "
                          f"last {first + 4 * 8 * 3_600_000} (2026-10-28 17:00:00 UTC), ascending by time: True")
    assert section[2] == "BTCUSDT: the page of 5 starts at the oldest row of the page of 1000: True"
    assert section[3].startswith("ETHUSDT limit 1000: rows 240") and len(section) == 6


def test_an_income_history_answered_newest_first_is_shown_as_such(tmp_path):
    venue = Venue(answers={"/fapi/v1/income": lambda query: _income(query, descending=True)})
    code, section = _income_run(tmp_path, venue)
    assert code == 0
    assert "rows 240" in section[0] and section[0].endswith("ascending by time: False")
    assert "rows 5" in section[1] and section[1].endswith("ascending by time: False")
    assert section[2] == "BTCUSDT: the page of 5 starts at the oldest row of the page of 1000: False"


def test_an_empty_or_unreadable_income_history_is_reported_not_guessed(tmp_path):
    code, section = _income_run(tmp_path, Venue(answers={"/fapi/v1/income": lambda query: []}))
    assert code == 0
    assert section[0] == "BTCUSDT limit 1000: rows 0"
    assert section[2] == "BTCUSDT: the page of 5 starts at the oldest row of the page of 1000: no rows to compare"
    code, section = _income_run(tmp_path, Venue(answers={"/fapi/v1/income": lambda query: {"code": -1}}))
    assert section == ["unavailable: RuntimeError: the income history answer is not a list"]


def test_the_income_request_is_on_the_allowlist_as_a_signed_read():
    assert (ac.FAPI, "/fapi/v1/income", True) in ac.ALLOWED
    # audit of D1, L7: the pilot check also reads the public contract list (no key is sent with it)
    assert all(signed or path in ("/fapi/v1/time", "/fapi/v1/exchangeInfo") for _, path, signed in ac.ALLOWED)
